"""Tests for ``hermes_cli.kanban_explain`` — the board's derived explanation.

Every field here is a claim the operator will act on, so each one is pinned against
the component that DECIDES it: the retry budget against the breaker's own
``effective_limit``, the cooldown against ``check_respawn_guard``, and the abandon
plan against the failure classifier.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_graph as kbg
from hermes_cli import kanban_explain as ke
from hermes_cli import kanban_failure_class as fc


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_INFRA_RETRY_LIMIT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_INFRA_COOLDOWN_SECONDS", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


OWNER_LOSS = (
    "pid 9 exited with code 1 Worker's last output: "
    "'Kanban owner refused: Gateway connection failed; no local fallback'"
)


def _book(conn, task_id, *, outcome="crashed", error=None, metadata=None, ended_at=None,
          status="blocked"):
    now = int(ended_at if ended_at is not None else time.time())
    cur = conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, error, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (task_id, outcome, now - 30, now, outcome, error, json.dumps(metadata or {})),
    )
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()
    return int(cur.lastrowid)


def _explain(conn, task_id, *, cfg=None, now=None):
    return ke.task_explain(conn, kb.get_task(conn, task_id), kanban_cfg=cfg if cfg is not None else {}, now=now)


# --- retry budget ------------------------------------------------------------

def test_retry_budget_limit_precedence_matches_the_breaker(board):
    """task > config > default, and ``remaining`` is the breaker's arithmetic."""
    conn = board
    task_override = kb.create_task(conn, title="task cap", max_retries=3)
    cfg_cap = kb.create_task(conn, title="config cap")
    default_cap = kb.create_task(conn, title="default cap")

    overridden = _explain(conn, task_override, cfg={"failure_limit": 9})["retry_budget"]
    assert (overridden["limit"], overridden["limit_source"]) == (3, "task")

    configured = _explain(conn, cfg_cap, cfg={"failure_limit": 9})["retry_budget"]
    assert (configured["limit"], configured["limit_source"]) == (9, "config")

    default_budget = _explain(conn, default_cap, cfg={})["retry_budget"]
    assert (default_budget["limit"], default_budget["limit_source"]) == (kbd.DEFAULT_FAILURE_LIMIT, "default")


@pytest.mark.parametrize("max_retries,used,expected", [(1, 3, (1, 0)), (0, 0, (0, 0)), (5, 2, (5, 3))])
def test_retry_budget_remaining_never_goes_negative(board, max_retries, used, expected):
    conn = board
    tid = kb.create_task(conn, title="card", max_retries=max_retries)
    conn.execute("UPDATE tasks SET consecutive_failures = ? WHERE id = ?", (used, tid))
    conn.commit()
    budget = _explain(conn, tid)["retry_budget"]
    assert (budget["limit"], budget["remaining"]) == expected


def test_explain_limit_equals_the_tripping_breaker_limit(board):
    """A real booking: the board must read the SAME effective_limit the breaker used."""
    conn = board
    tid = kb.create_task(conn, title="card", max_retries=2)
    for _ in range(2):
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
        conn.commit()
        kbd._record_task_failure(conn, tid, "boom", outcome="crashed", failure_limit=9)
    trip = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"][-1]
    assert trip.payload["effective_limit"] == 2
    assert trip.payload["limit_source"] == "task"
    budget = _explain(conn, tid)["retry_budget"]
    assert (budget["limit"], budget["limit_source"]) == (2, "task")
    assert budget["remaining"] == 0


def test_infra_allowance_is_read_from_the_run_history_not_a_stored_counter(board):
    """``infra_used`` counts the runs the LANDED classifier stamped, not a column.

    The classifier deliberately keeps no per-card infrastructure counter (it never
    parks a card the host killed), so the board must derive the count from
    ``task_runs.metadata.infrastructure`` — the one place the booking site writes.
    """
    conn = board
    tid = kb.create_task(conn, title="card")
    conn.execute("UPDATE tasks SET consecutive_failures = 1 WHERE id = ?", (tid,))
    conn.commit()
    _book(conn, tid, metadata={"infrastructure": True, "infrastructure_class": "gateway_disconnected"},
          status="ready", ended_at=int(time.time()) - 10)
    _book(conn, tid, metadata={"infrastructure": True}, status="ready",
          ended_at=int(time.time()) - 5)
    _book(conn, tid, outcome="rate_limited", status="ready", ended_at=int(time.time()) - 1)

    budget = _explain(conn, tid, cfg={"infrastructure_retry_limit": 3})["retry_budget"]
    assert (budget["infra_used"], budget["infra_limit"], budget["infra_remaining"]) == (2, 3, 1)
    # The unified budget is untouched by the host-caused bookings.
    assert (budget["used"], budget["remaining"]) == (1, kbd.DEFAULT_FAILURE_LIMIT - 1)
    # And the board says out loud that the allowance does not park anything.
    assert budget["infra_enforced"] is False


# --- cooldown parity with the guard -----------------------------------------

def test_infrastructure_cooldown_matches_check_respawn_guard(board, monkeypatch):
    conn = board
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "600")
    tid = kb.create_task(conn, title="card")
    _book(conn, tid, metadata={"infrastructure": True}, status="ready")

    guard = kbd.check_respawn_guard(conn, tid)
    explain = _explain(conn, tid)["cooldown"]
    assert guard == "infrastructure_cooldown"
    assert explain["kind"] == "infrastructure_cooldown"
    assert 0 < explain["seconds_remaining"] <= 600


def test_cooldown_renders_the_runs_own_stamped_spacing(board, monkeypatch):
    """A burst stamps a WIDER window on the run; the board renders that stamp.

    This is the parity that matters: the guard reads
    ``metadata.infrastructure_cooldown_seconds`` (clamped 60..3600), and the drawer
    must show the same number, not a re-derived default.
    """
    conn = board
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    tid = kb.create_task(conn, title="card")
    _book(conn, tid, metadata={"infrastructure": True, "infrastructure_cooldown_seconds": 900},
          status="ready", ended_at=int(time.time()) - 800)
    # 800s elapsed of a 900s stamped window: still cooling down, 100s left.
    assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"
    cooldown = _explain(conn, tid)["cooldown"]
    assert cooldown["kind"] == "infrastructure_cooldown"
    assert 90 <= cooldown["seconds_remaining"] <= 100


def test_elapsed_cooldown_reads_as_none_not_negative(board, monkeypatch):
    conn = board
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "60")
    tid = kb.create_task(conn, title="card")
    _book(conn, tid, metadata={"infrastructure": True}, status="ready",
          ended_at=int(time.time()) - 120)
    assert kbd.check_respawn_guard(conn, tid) is None
    cooldown = _explain(conn, tid)["cooldown"]
    assert cooldown["kind"] is None and cooldown["seconds_remaining"] == 0


def test_rate_limit_cooldown_matches_check_respawn_guard(board, monkeypatch):
    conn = board
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "600")
    tid = kb.create_task(conn, title="card")
    _book(conn, tid, outcome="rate_limited", status="ready")
    assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
    assert _explain(conn, tid)["cooldown"]["kind"] == "rate_limit_cooldown"


# --- owner loss --------------------------------------------------------------

def test_owner_loss_only_from_infrastructure_runs(board):
    conn = board
    infra = kb.create_task(conn, title="infra")
    _book(conn, infra, error=OWNER_LOSS, metadata={"worker_output": OWNER_LOSS})
    loss = _explain(conn, infra)["owner_loss"]
    assert loss["variant"] == "gateway_connection_failed"
    assert "gateway connection failed" in loss["text"]
    assert loss["run_id"]

    card = kb.create_task(conn, title="card failure")
    _book(conn, card, error="protocol violation: no handoff", metadata={"protocol_violation": True})
    assert _explain(conn, card)["owner_loss"] is None


# --- fail closed -------------------------------------------------------------

def test_unclassified_failure_is_not_abandonable_and_says_why(board):
    conn = board
    root = kb.create_task(conn, title="root quest", triage=True)
    kids = kbg.decompose_triage_task(
        conn, root, root_assignee="builder",
        children=[{"title": "child", "assignee": "builder"}],
    )
    _book(conn, kids[0], error="pid 1 exited with code 1 Worker's last output: 'Traceback'")
    explain = _explain(conn, root)
    assert explain["blocking_children"][0]["failure_class"] is None
    assert explain["abandon"]["eligible"] is False
    assert any(reason.startswith("unclassified_descendant:") for reason in explain["abandon"]["blocked_by"])
    assert explain["classifier"]["deployed"] is True


def test_board_still_renders_budget_and_cooldown_when_nothing_is_classified(board):
    conn = board
    tid = kb.create_task(conn, title="quiet card")
    explain = _explain(conn, tid)
    assert explain["failure_class"] is None
    assert explain["failure_evidence"] is None
    assert explain["cooldown"]["kind"] is None
    assert explain["retry_budget"]["limit"] >= 1


# --- the invariant -----------------------------------------------------------

def test_classification_and_abandon_eligibility_can_never_disagree(board):
    """One call path: a card-class/unclassified descendant is BLOCKED, an infra one is in SCOPE."""
    conn = board
    root = kb.create_task(conn, title="root quest", triage=True)
    kids = kbg.decompose_triage_task(
        conn, root, root_assignee="builder",
        children=[{"title": f"child {i}", "assignee": "builder"} for i in range(3)],
    )
    _book(conn, kids[0], error=OWNER_LOSS)
    _book(conn, kids[1], error="protocol violation: no handoff", metadata={"protocol_violation": True})
    _book(conn, kids[2], error="pid 2 exited with code 1")

    plan = _explain(conn, root)["abandon"]
    assert plan["eligible"] is False
    assert plan["scope"] == [kids[0]]
    blocked_ids = {reason.split(":", 1)[1] for reason in plan["blocked_by"]}
    assert blocked_ids == {kids[1], kids[2]}
    assert set(plan["scope"]).isdisjoint(blocked_ids)

    for kid in kids:
        kid_explain = _explain(conn, kid)
        # A child's own plan is its root's plan — same scope, same refusals.
        assert kid_explain["abandon"]["root"] == root
        assert kid_explain["abandon"]["scope"] == plan["scope"]
        if kid_explain["failure_class"] == fc.FAILURE_INFRA:
            assert kid in plan["scope"]
        else:
            assert kid not in plan["scope"]


def test_the_cards_own_class_and_the_abandon_gate_agree(board):
    """The invariant covers the card BEING abandoned, not only its descendants.

    Review round 1 found ``abandon_plan`` classifying descendants only, so a
    card-class root/leaf read ``eligible: true`` and a host-flavoured reason could
    bury genuinely failed work. The plan now classifies the root's own newest
    booking with the same function that produces ``explain.failure_class``.
    """
    conn = board
    card_leaf = kb.create_task(conn, title="card that genuinely failed")
    _book(conn, card_leaf, error="protocol violation: no handoff", metadata={"protocol_violation": True})
    card_explain = _explain(conn, card_leaf)
    assert card_explain["failure_class"] == fc.FAILURE_CARD
    assert card_explain["abandon"]["eligible"] is False
    assert card_explain["abandon"]["blocked_by"] == [f"card_class_root:{card_leaf}"]

    infra_leaf = kb.create_task(conn, title="card the host killed")
    _book(conn, infra_leaf, error=OWNER_LOSS, metadata={"infrastructure": True})
    infra_explain = _explain(conn, infra_leaf)
    assert infra_explain["failure_class"] == fc.FAILURE_INFRA
    assert infra_explain["abandon"]["eligible"] is True

    # A child's drawer shows its root's plan, so it must explain the ROOT's refusal.
    root = kb.create_task(conn, title="root quest", triage=True)
    kids = kbg.decompose_triage_task(
        conn, root, root_assignee="builder",
        children=[{"title": "infra child", "assignee": "builder"}],
    )
    _book(conn, kids[0], error=OWNER_LOSS, metadata={"infrastructure": True})
    _book(conn, root, error="protocol violation: no handoff", metadata={"protocol_violation": True},
          status="blocked")
    child_explain = _explain(conn, kids[0])
    assert child_explain["abandon"]["root"] == root
    assert child_explain["abandon"]["blocked_by"] == [f"card_class_root:{root}"]
    assert child_explain["abandon"]["eligible"] is False


def test_complete_child_is_already_terminal_not_a_blocker(board):
    conn = board
    root = kb.create_task(conn, title="root quest", triage=True)
    kids = kbg.decompose_triage_task(
        conn, root, root_assignee="builder",
        children=[{"title": "done child", "assignee": "builder"},
                  {"title": "infra child", "assignee": "builder"}],
    )
    conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (kids[0],))
    conn.commit()
    _book(conn, kids[1], error=OWNER_LOSS)
    plan = _explain(conn, root)["abandon"]
    assert plan["eligible"] is True
    assert plan["already_terminal"] == [kids[0]]
    assert plan["scope"] == [kids[1]]


# --- cost --------------------------------------------------------------------

def test_board_explain_query_count_is_constant(board):
    """Bulk loader: the same number of statements for 3 cards and for 60."""
    conn = board

    def count_statements() -> int:
        seen: list[str] = []
        conn.set_trace_callback(seen.append)
        try:
            ke.board_explain(conn, kb.list_tasks(conn), kanban_cfg={})
        finally:
            conn.set_trace_callback(None)
        return len(seen)

    for i in range(2):
        kb.create_task(conn, title=f"small {i}")
    small = count_statements()
    for i in range(58):
        kb.create_task(conn, title=f"bulk {i}")
    large = count_statements()
    assert large == small, (small, large)
