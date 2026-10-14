"""Tests for the kanban worker turn-end stop guard.

Two independent predicates decide whether the nudge fires, and both are covered here:

  * the TRANSCRIPT — did this session already make a terminal board call? A session
    whose last lifecycle call is ``kanban_request_review`` (implementer handing off to
    review) or ``kanban_request_changes`` (reviewer sending it back) has ended its turn
    correctly and must not be nudged. The set omitting these two was the reported defect
    (t_d28c22ca): every correct review handoff was told the card "is still ``running``".
  * the BOARD ROW — has the card left ``running`` under this claim? This makes the
    nudge's claim about the card's state true by construction rather than by luck, and
    it still suppresses the nudge when the message list is empty or unrecognized.

Fail-OPEN on the board probe is deliberate and pinned below: an absent/unreadable board
must never swallow a legitimate nudge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.kanban_stop import (
    _TERMINAL_KANBAN_TOOLS,
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _session(tool_name: str | None) -> list[dict]:
    """A transcript whose last lifecycle call is ``tool_name`` (or a narrated stop)."""
    messages: list[dict] = [{"role": "user", "content": "work kanban task"}]
    if tool_name is None:
        messages.append({"role": "assistant", "content": "Let me write the report next."})
        return messages
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": tool_name, "arguments": "{}"}}
        ],
    })
    messages.append({"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "ok"})
    return messages


@pytest.fixture
def board_env(monkeypatch, tmp_path):
    """A hermetic temp board plus the env pins a dispatcher-spawned worker gets.

    ``_card_left_this_worker`` resolves through ``HERMES_KANBAN_DB`` then a read-only open,
    so the fixture only needs the file on disk — but it builds the real schema through the
    real ``init_db``/``create_task`` path, so a schema or status-name change shows up here.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db_path = home / "kanban.db"
    kb.init_db(db_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    def _make(status: str) -> str:
        """Create a card, drive it to ``status`` through the real API, and pin
        ``HERMES_KANBAN_RUN_ID`` to the run this worker would own."""
        with kbc.connect(db_path) as conn:
            tid = kb.create_task(conn, title=f"{status} card", assignee="builder")
            claimed = kb.claim_task(conn, tid, claimer="host:1")
            assert claimed is not None
            run_id = claimed.current_run_id
            if status == "review":
                assert kb.request_review(conn, tid, summary="handed off", force=True) is True
            elif status == "done":
                assert kb.complete_task(conn, tid, summary="done", force=True) is True
            elif status == "blocked":
                assert kb.block_task(conn, tid, reason="waiting", force=True) is True
            elif status == "ready":
                assert kb.reclaim_task(conn, tid, reason="stale", force=True) is True
            elif status != "running":
                raise AssertionError(f"unhandled status {status!r}")
            assert kb.get_task(conn, tid).status == status
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        if run_id is not None:
            monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
        return tid

    return _make


def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_disabled_inside_delegated_child(clear_kanban_env):
    from agent.delegation_context import delegated_child_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with delegated_child_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_disabled_inside_non_dispatcher_context(clear_kanban_env):
    from agent.delegation_context import non_dispatcher_owned_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with non_dispatcher_owned_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True

def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = _session("kanban_heartbeat")
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = _session("kanban_complete")
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


@pytest.mark.parametrize(
    "tool_name,who",
    [
        ("kanban_request_review", "implementer handing the card to review"),
        ("kanban_request_changes", "reviewer routing the card back to the implementer"),
    ],
)
def test_no_nudge_after_handoff_tool(clear_kanban_env, tool_name, who):
    """The reported defect: both handoff tools end the worker's turn.

    ``kanban_request_review`` is what the injected worker lifecycle tells an implementer
    to call; ``kanban_request_changes`` is the sdlc-review skill's reviewer exit. Nudging
    after either orders a worker that did the right thing to ``kanban_complete`` a card it
    must not close — the destructive action this guard exists to prevent.
    """
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_handoff")
    messages = _session(tool_name)
    assert tool_name in _TERMINAL_KANBAN_TOOLS, who
    assert session_called_kanban_terminal(messages) is True, who
    assert build_kanban_stop_nudge(messages=messages) is None, who


@pytest.mark.parametrize("tool_name", ["kanban_comment", "kanban_heartbeat", "kanban_show"])
def test_non_terminal_kanban_tool_still_nudges(clear_kanban_env, tool_name):
    """Widening the set must not swallow the case the guard exists for."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = _session(tool_name)
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


def test_no_nudge_when_card_left_running_with_empty_messages(board_env):
    """The board row alone suppresses the nudge — no terminal call in the transcript.

    ``kanban_request_review`` clears the claim, so nothing in the message list has to be
    recognized for the guard to know the card is gone.
    """
    tid = board_env("review")
    assert build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid) is None
    assert build_kanban_stop_nudge(messages=_session(None), attempts=0, task_id=tid) is None


@pytest.mark.parametrize("status", ["done", "blocked"])
def test_no_nudge_for_any_non_running_status(board_env, status):
    """Every close-out lands outside ``running``; the row check is status-agnostic."""
    tid = board_env(status)
    assert build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid) is None


def test_no_nudge_after_the_card_is_reclaimed(board_env):
    """A dispatcher reclaim puts the card back to ``ready`` — the worker must not
    re-close it, and nothing in the transcript shows that happened."""
    tid = board_env("ready")
    assert build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid) is None


def test_no_nudge_when_a_successor_run_owns_the_card(board_env, monkeypatch):
    """Still ``running``, but under a DIFFERENT run: the earlier worker was superseded.

    This is the case a bare ``status == 'running'`` check would get wrong, and the reason
    the probe compares ``current_run_id`` against ``HERMES_KANBAN_RUN_ID``.
    """
    import os

    tid = board_env("running")
    live_run_id = int(os.environ["HERMES_KANBAN_RUN_ID"])
    assert build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid) is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(live_run_id + 999))
    assert build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid) is None


def test_nudge_still_fires_while_card_is_running(board_env):
    """The status consult is a suppression, not an off switch."""
    tid = board_env("running")
    nudge = build_kanban_stop_nudge(messages=[], attempts=0, task_id=tid)
    assert nudge is not None
    assert tid in nudge
    assert "still `running`" in nudge


def test_nudge_still_fires_when_board_is_unreadable(clear_kanban_env, tmp_path):
    """FAIL-OPEN: no board file means the probe knows nothing, so it must not suppress.

    A missing/unreadable board is exactly the state a broken worker should still be
    nudged out of, so the probe returning ``False`` must leave the transcript predicate
    in sole control.
    """
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_nowhere")
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path / "absent" / "kanban.db"))
    assert build_kanban_stop_nudge(messages=[], attempts=0) is not None
    # A board that exists but has no such row behaves the same...
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path / "empty" / "kanban.db"))
    (tmp_path / "empty").mkdir()
    Path(tmp_path / "empty" / "kanban.db").touch()
    assert build_kanban_stop_nudge(messages=[], attempts=0) is not None
    # ...as does a real board with the row absent and no run id pinned.
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path / "real" / "kanban.db"))
    (tmp_path / "real").mkdir()
    from hermes_cli import kanban_db as kb

    kb.init_db(tmp_path / "real" / "kanban.db")
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_missing_from_board")
    assert build_kanban_stop_nudge(messages=[], attempts=0) is not None


def test_nudge_offers_every_worker_exit(clear_kanban_env):
    """A card bound for review must never be steered to ``kanban_complete`` alone."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    nudge = build_kanban_stop_nudge(messages=_session("kanban_comment"))
    assert nudge is not None
    for exit_tool in (
        "kanban_complete", "kanban_request_review", "kanban_block", "kanban_request_changes",
    ):
        assert exit_tool in nudge, exit_tool


def test_budget_exhausted_never_nudges(clear_kanban_env):
    """The bounded retry is the whole point: attempts >= max_attempts stops the loop."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert build_kanban_stop_nudge(messages=[], attempts=2, max_attempts=2) is None


# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.
