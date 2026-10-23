"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``, ``kanban_schedule``). Some models narrate
the next step and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` →
dispatcher ``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop
continues instead of exiting.

Terminality is a *landed* call, not an attempted one. A terminal call whose tool result is a
refusal — the executor's ``Error executing tool …`` prefix, or a board refusal carrying a
top-level ``error`` key — never moved the card off ``running``, so the guard must nudge the
worker to re-file rather than let the turn die into a booked protocol violation. Only those two
exact shapes count as refusals (fail-open): an unrecognised payload is treated as landed, so the
guard can never nudge a worker that genuinely handed off.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task


# Every tool that ends this worker's responsibility for the card, not just the two that
# close it out: ``kanban_request_review`` moves it to ``review`` (goals.py's continuation /
# finalize prompts tell builders to call it) and ``kanban_request_changes`` returns it to
# ``ready`` (the sdlc-review skill tells reviewers to). Nudging after either asks a worker
# that did the right thing to ``kanban_complete`` a card it must not close.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_schedule",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2

# The single prefix ``agent/tool_executor.py`` uses when a tool never reached its
# implementation: timeout, exception, shutdown and dead worker thread all take this path.
_EXECUTOR_REFUSAL_PREFIX = "Error executing tool "


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


def _as_text(content: Any) -> str:
    """Tool-result content as text: strings pass through, content-block lists are joined,
    anything else is JSON-encoded. Never raises."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                parts.append(text if isinstance(text, str) else json.dumps(block, sort_keys=True))
            elif block is not None:
                parts.append(str(block))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, sort_keys=True)
    return str(content)


def _result_is_refusal(content: Any) -> bool:
    """True only for the two measured refusal shapes; everything else is a landed call.

    1. the executor prefix ``Error executing tool `` (timeout / exception / shutdown), and
    2. a JSON object carrying a top-level ``error`` key (every board refusal goes through
       ``tools/registry.tool_error``).

    Deliberately fail-open: an unrecognised payload counts as landed so the guard never nudges
    a worker whose handoff succeeded through an envelope this predicate does not know.
    """
    text = _as_text(content)
    if text.lstrip().startswith(_EXECUTOR_REFUSAL_PREFIX):
        return True
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and "error" in payload


def session_terminal_call_landed(messages: Iterable[dict] | None) -> bool:
    """True if at least one terminal kanban call in this conversation *landed*.

    Only the tool-result rows decide: a terminal call with no result row counts as not landed,
    because the transcript carries no evidence the board accepted it and the card is still
    ``running`` — the guard's bounded nudge is the correct response. One landed call wins over
    any number of refused ones.
    """
    for msg in messages or ():
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        name = str(msg.get("name") or msg.get("tool_name") or "")
        if name in _TERMINAL_KANBAN_TOOLS and not _result_is_refusal(msg.get("content")):
            return True
    return False


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """Deprecated alias for :func:`session_terminal_call_landed`.

    Kept so older callers keep importing; the *meaning* is now "a terminal call landed",
    because an attempted-but-refused call is exactly the state the guard exists to catch.
    """
    return session_terminal_call_landed(messages)


def _terminal_tools_text() -> str:
    """The terminal tool set as worker-facing prose, derived so the contract cannot drift."""
    return ", ".join(f"`{name}`" for name in sorted(_TERMINAL_KANBAN_TOOLS))


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a landed terminal tool; ``None``
    when the guard should not fire (not a kanban worker, already handed off, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_terminal_call_landed(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # The transcript is the status source: this text is only reached when no terminal call
    # landed, so it never tells a worker to close a card it already sent to review.
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not been handed off: this session made no landed terminal board "
        f"call ({_terminal_tools_text()}). Ending now causes a protocol violation "
        "(clean exit with the card still `running`).\n\n"
        "A refused call (`{\"error\": …}` or `Error executing tool …`) is not a handoff — "
        "read the result, fix the cause, and call it again.\n\n"
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


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
    "session_terminal_call_landed",
]
