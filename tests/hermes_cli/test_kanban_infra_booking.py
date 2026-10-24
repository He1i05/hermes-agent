"""The abandonment read model against REAL bookings from the landed classifier.

This is the seam between the two cards: ``56683f8fe2`` decides ``infrastructure`` at
booking time and stamps ``task_runs.metadata``; ``abandon`` consumes those stamps and
adds nothing of its own. So the properties pinned here are:

* a host-caused booking charges NO retry budget and never becomes sticky (the sibling
  fix's invariant — re-asserted here because abandon depends on it), and
* the very same run makes the card read ``infrastructure`` in the explain payload and
  puts it in its root's abandon frontier, while
* a card-class crash still charges exactly one failure and is NEVER in the frontier.

Positive control and the failing case live in one file on purpose: an over-broad
classifier would pass the first half and fail the second.
"""

from __future__ import annotations

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
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


OWNER_LOSS = (
    "pid 9 exited with code 1 Worker's last output: "
    "'Kanban owner refused: Gateway connection failed; no local fallback'"
)


def _ready(conn, task_id):
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    conn.commit()


def _booked_run(conn, task_id, *, error, infrastructure, extra=None):
    """Book one failure through the REAL breaker entry point, then stamp the run the
    way the crash sweep does (``_end_run`` metadata carries the classification)."""
    _ready(conn, task_id)
    tripped = kbd._record_task_failure(
        conn, task_id, error, outcome="crashed", infrastructure=infrastructure,
    )
    now = int(time.time())
    metadata = {"infrastructure": True} if infrastructure else {"protocol_violation": True}
    metadata.update(extra or {})
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, error, metadata) "
        "VALUES (?, 'crashed', ?, ?, 'crashed', ?, ?)",
        (task_id, now - 30, now, error, kbd._kb._json_or_null(metadata)),
    )
    conn.commit()
    return tripped


def _root_with_two_children(conn):
    root = kb.create_task(conn, title="root quest", triage=True)
    kids = kbg.decompose_triage_task(
        conn, root, root_assignee="builder",
        children=[{"title": "host killed", "assignee": "builder"},
                  {"title": "genuinely broken", "assignee": "builder"}],
    )
    return root, kids


def test_host_caused_booking_is_not_charged_and_makes_the_child_infra(board):
    conn = board
    host, _broken = _root_with_two_children(conn)

    tripped = _booked_run(conn, host, error=OWNER_LOSS, infrastructure=True,
                          extra={"infrastructure_class": "gateway_connection_failed"})
    assert tripped is False, "an infrastructure booking must never trip the breaker"
    child = kb.get_task(conn, host)
    assert child.consecutive_failures == 0          # budget untouched
    assert child.status == "ready"                  # retried, not parked
    trip_events = [e for e in kb.list_events(conn, host) if e.kind == "gave_up"]
    assert trip_events == []
    assert fc.blocking_class(conn, host) == fc.FAILURE_INFRA


def test_genuine_crash_still_charges_one_and_reads_card(board):
    """Positive control: the classifier must not be so wide that a broken card
    looks like a host problem."""
    conn = board
    _root, kids = _root_with_two_children(conn)
    _host, broken = kids[0], kids[1]

    tripped = _booked_run(conn, broken, error="protocol violation: no handoff",
                          infrastructure=False)
    assert tripped is False                          # 1 of the default 2
    child = kb.get_task(conn, broken)
    assert child.consecutive_failures == 1           # charged exactly once
    assert fc.blocking_class(conn, broken) == fc.FAILURE_CARD


def test_the_root_is_abandonable_for_the_infra_child_only(board):
    conn = board
    root, (host, broken) = _root_with_two_children(conn)
    _booked_run(conn, host, error=OWNER_LOSS, infrastructure=True)
    _booked_run(conn, broken, error="protocol violation: no handoff", infrastructure=False)

    explain = ke.task_explain(conn, kb.get_task(conn, root), kanban_cfg={})
    plan = explain["abandon"]
    # One card-class child ⇒ the root is NOT silently abandonable, and the refusal
    # names the child instead of hiding it inside a cascade.
    assert plan["eligible"] is False
    assert plan["blocked_by"] == [f"card_class_descendant:{broken}"]
    assert plan["scope"] == [host]

    # Clear the genuine failure out of the way (the operator's decision, not the
    # cascade's) and the same root becomes abandonable by default.
    conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (broken,))
    conn.commit()
    plan = ke.task_explain(conn, kb.get_task(conn, root), kanban_cfg={})["abandon"]
    assert plan["eligible"] is True
    assert plan["scope"] == [host]
