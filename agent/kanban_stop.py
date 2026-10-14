"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task

# Every tool that ends THIS worker's responsibility for the card, not just the two that close
# it out: ``kanban_request_review`` moves it to ``review`` (the injected worker lifecycle and
# goals.py's continuation/finalize prompts tell implementers to call it) and
# ``kanban_request_changes`` routes it back to the implementer (the sdlc-review skill tells
# reviewers to). Nudging after either asks a worker that did the right thing to
# ``kanban_complete`` a card it must not close — the destructive action this guard exists to
# prevent (t_6c8342e0), so the two handoff tools belong here beside the two close-out tools.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_schedule",
    "kanban_request_review",
    "kanban_request_changes",
})

#: Status of a card this worker still owns; anything else means it has been handed off.
_RUNNING_STATUS = "running"

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set for the dispatcher-owned worker, unless
    ``HERMES_KANBAN_STOP_NUDGE`` disables it. In-process delegate_task children and cron runs
    inherit the env var but own no board task and carry no kanban toolset."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(owned_kanban_task())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant" and any(
            _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS for tc in msg.get("tool_calls") or []
        ):
            return True
        if role == "tool" and str(msg.get("name") or "") in _TERMINAL_KANBAN_TOOLS:
            return True
    return False


def _card_left_this_worker(task_id: str) -> bool:
    """True only on positive evidence that the card is no longer THIS worker's live
    ``running`` task.

    Deliberately a bare read-only ``mode=ro`` open, NOT
    ``kanban_db_connect.connect``: this runs inside the agent's turn-stop path, where
    ``connect`` takes the cross-process init lock, runs migrations and creates a missing
    board — a write path a stop gate must never enter. ``mode=ro`` reads the live WAL board
    and refuses writes at the SQLite level (verified against the production board), and the
    connection is registered via ``connect_tracked`` for its short lifetime so a concurrent
    byte-level probe cannot cancel its locks.

    Both the status and this run's ownership are checked, because "the card is still
    ``running``" is only true under the claim that is doing the work: a card reclaimed by
    the dispatcher (``ready``), handed off (``review``), or re-claimed by a successor run
    (``running`` under a different ``current_run_id``) is not this worker's to close. An
    unresolvable path, unreadable DB, unknown row, absent ``HERMES_KANBAN_RUN_ID`` or any
    error returns False — FAIL OPEN, so a broken probe degrades to the transcript predicate
    instead of swallowing a legitimate nudge.
    """
    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not tid:
        return False
    expected_run_id = None
    raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    if raw_run_id:
        try:
            expected_run_id = int(raw_run_id)
        except ValueError:
            expected_run_id = None
    try:
        from hermes_cli.kanban_db import kanban_db_path
        from hermes_cli.sqlite_safe_read import connect_tracked

        path = Path(kanban_db_path()).expanduser()
        if not path.exists():
            return False
        conn = connect_tracked(
            path.resolve().as_uri() + "?mode=ro",
            tracking_path=path.resolve(),
            uri=True,
            timeout=1.0,
        )
        try:
            row = conn.execute(
                "SELECT status, current_run_id FROM tasks WHERE id = ?", (tid,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return False
        status = str(row[0]) if row[0] is not None else None
        if status is None:
            return False
        if status != _RUNNING_STATUS:
            return True
        current_run_id = int(row[1]) if row[1] is not None else None
        return expected_run_id is not None and current_run_id != expected_run_id
    except Exception:
        return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already handed off, budget exhausted).

    Two independent predicates, ORed into "do not fire": the transcript (did this session
    already make a terminal board call?) and the board row (has the card left ``running``
    under this claim?). The transcript check alone was the reported bug — a worker whose
    session *did* hand off, via a tool the set did not list, was still told it "is still
    running". The row check makes that sentence true by construction instead of by luck: any
    tool that moved the card out of ``running`` suppresses the nudge even when the message
    list is empty or unrecognized.
    """
    if not kanban_stop_nudge_enabled() or attempts >= max_attempts:
        return None
    if session_called_kanban_terminal(messages):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if _card_left_this_worker(tid):
        return None

    tid = tid or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not been handed off: this session made no terminal board "
        "call and the card is still `running`. Ending now causes a protocol "
        "violation (clean exit with `kanban_complete` / `kanban_request_review` / "
        "`kanban_block` unanswered).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, OR `kanban_block(reason=...)` if you are "
        "blocked. Reviewers approve with `kanban_complete` or send the card back with "
        "`kanban_request_changes(reason=...)`.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
