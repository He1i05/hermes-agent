"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    _DEFAULT_MAX_ATTEMPTS,
    _TERMINAL_KANBAN_TOOLS,
    _result_is_refusal,
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
    session_terminal_call_landed,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


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
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.


@pytest.mark.parametrize(
    "tool_name,who",
    [
        ("kanban_request_review", "build worker handing off for same-card review"),
        ("kanban_request_changes", "review agent sending the card back"),
        ("kanban_schedule", "worker parking the card on a timed wait"),
    ],
)
def test_no_nudge_after_handoff_tool(clear_kanban_env, tool_name, who):
    """Handoff tools end the worker's turn just like complete/block.

    Both move the card out of ``running``, and the worker is told to call
    them — goals.py's continuation/finalize prompts name
    ``kanban_request_review``; the force-loaded sdlc-review skill names
    ``kanban_request_changes``. Nudging afterwards asks a worker that did
    the right thing to close a card it must not close.
    """
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_handoff")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is True, who
    assert build_kanban_stop_nudge(messages=messages) is None


def test_nudge_still_fires_for_non_terminal_kanban_tool(clear_kanban_env):
    """Widening the set must not swallow the case the guard exists for."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "Let me open the review next.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_comment", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_comment", "tool_call_id": "1", "content": "ok"},
    ]
    assert session_called_kanban_terminal(messages) is False
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None
    # The nudge offers every worker exit, not just close-out; a card that must go
    # through review must never be steered to ``kanban_complete`` alone.
    assert "kanban_request_review" in nudge and "kanban_block" in nudge


# ── T1–T6: terminality is a LANDED call, not an attempted one ────────────────
# A refused terminal call never moved the card off ``running``. Before the fix the
# guard suppressed its nudge on the tool *name* alone, so the turn died rc=0 and the
# dispatcher booked a protocol violation (measured: 71 sessions, 18.5% of recent
# terminal-call results were refusals).


def _terminal_turn(tool_name: str, content) -> list[dict]:
    """One assistant terminal tool call followed by its tool-role result."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": content},
    ]


def test_t1_executor_refused_terminal_call_nudges(clear_kanban_env):
    """T1 — the executor prefix (timeout / exception / dead thread) is a refusal."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_f2")
    messages = _terminal_turn(
        "kanban_block",
        "Error executing tool 'kanban_block': 'DaemonThreadPoolExecutor' object has no "
        "attribute '_initializer'",
    )
    assert _result_is_refusal(messages[1]["content"]) is True
    assert session_terminal_call_landed(messages) is False
    assert session_called_kanban_terminal(messages) is False  # the alias follows
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None
    assert "t_f2" in nudge


def test_t2_board_refused_terminal_call_nudges(clear_kanban_env):
    """T2 — a top-level JSON ``error`` is the board's own refusal envelope."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_f2")
    messages = _terminal_turn(
        "kanban_complete",
        '{"error": "unknown parameter(s): adoption_evidence"}',
    )
    assert _result_is_refusal(messages[1]["content"]) is True
    assert session_terminal_call_landed(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


def test_t3_landed_terminal_call_does_not_nudge(clear_kanban_env):
    """T3 — the guard's live traffic: a landed handoff must stay silent."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_ok")
    for tool, content in (
        ("kanban_request_review", '{"ok": true, "task_id": "t_ok", "status": "review"}'),
        ("kanban_complete", '{"ok": true, "task_id": "t_ok", "status": "done"}'),
        ("kanban_block", "ok"),
    ):
        messages = _terminal_turn(tool, content)
        assert session_terminal_call_landed(messages) is True, tool
        assert build_kanban_stop_nudge(messages=messages) is None, tool


def test_t4_refused_then_landed_does_not_nudge(clear_kanban_env):
    """T4 — any landed call wins over any number of refused ones."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_mixed")
    messages = _terminal_turn("kanban_complete", '{"error": "task is not running"}')
    messages += [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {"name": "kanban_request_review", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_request_review",
            "tool_call_id": "2",
            "content": '{"ok": true, "status": "review"}',
        },
    ]
    assert session_terminal_call_landed(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_t5_terminal_call_without_result_row_nudges_and_is_bounded(clear_kanban_env):
    """T5 — a call with no result row is NOT landed (the card is still running), and the
    nudge budget stays ``_DEFAULT_MAX_ATTEMPTS``."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_pending")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        }
    ]
    assert session_terminal_call_landed(messages) is False
    assert _DEFAULT_MAX_ATTEMPTS == 2
    for attempt in range(_DEFAULT_MAX_ATTEMPTS):
        assert build_kanban_stop_nudge(messages=messages, attempts=attempt) is not None
    assert build_kanban_stop_nudge(messages=messages, attempts=_DEFAULT_MAX_ATTEMPTS) is None


@pytest.mark.parametrize(
    "payload",
    [
        "Done.",  # plain text, no envelope at all
        "ok",  # the older tool stub's result
        [{"type": "text", "text": "the card is done"}],  # content-block provider
        '["not-an-object"]',
        "",  # absent body: fail-open, never nudge a worker whose result was dropped
    ],
)
def test_t6_unrecognised_payload_counts_as_landed(clear_kanban_env, payload):
    """T6 — fail-open: only the two exact refusal shapes suppress terminality."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_ok")
    assert _result_is_refusal(payload) is False
    messages = _terminal_turn("kanban_request_review", payload)
    assert session_terminal_call_landed(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_t6b_nudge_text_names_every_terminal_tool_and_the_refusal_rule(clear_kanban_env):
    """T6b — worker-facing text derived from the set, so it cannot under-describe it."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_text")
    nudge = build_kanban_stop_nudge(messages=[])
    assert nudge is not None
    for tool in _TERMINAL_KANBAN_TOOLS:
        assert f"`{tool}`" in nudge, tool
    assert "A refused call" in nudge
    assert "Error executing tool" in nudge
