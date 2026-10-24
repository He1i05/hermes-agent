"""Tests for ``hermes_cli.kanban_failure_class`` — the ONE classifier.

The abandon gate and the board's ``explain`` badge both read this module, so the
tests here pin (a) the closed vocabulary and its card-first precedence, (b) the
run-level and task-level entry points against real ``task_runs`` rows, and (c) the
fan-in edge direction that decides the abandon scope.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_graph as kbg
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
    "pid 3613818 exited with code 1 Worker's last output: "
    "'Kanban owner refused: Gateway connection failed; no local fallback'"
)


def _book(conn, task_id, *, outcome="crashed", error=None, metadata=None, ended_at=None,
          status="blocked"):
    """Insert one ended run the way the crash sweep does (metadata = event payload)."""
    now = int(ended_at if ended_at is not None else time.time())
    cur = conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, error, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (task_id, outcome, now - 30, now, outcome, error, json.dumps(metadata or {})),
    )
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()
    return int(cur.lastrowid)


# --- the predicate -----------------------------------------------------------

def test_vocabulary_tables_match_the_landed_classifier():
    """Drift guard: this module must never carry a second copy of the rule.

    ``host_loss_variant`` delegates to ``kanban_db_dispatch._host_loss_class`` (the
    landed classifier, 56683f8fe2). The constants kept here are the shared test
    vector, so they are asserted equal to the landed tables — if either side gains a
    variant the other does not know, this test fails instead of the board quietly
    disagreeing with the booking site about a real run.
    """
    from hermes_cli import kanban_db_dispatch as kdd

    assert fc.HOST_LOSS_WRAPPER == kdd._HOST_LOSS_WRAPPER
    assert fc.HOST_LOSS_TAIL_CHARS == kdd._INFRASTRUCTURE_TAIL_CHARS
    assert tuple(fc.WRAPPED_CLASSES) == tuple(kdd._WRAPPED_CLASSES)
    assert tuple(v for _, v in fc.BARE_TOKENS) == tuple(kdd._HOST_LOSS_BARE_TOKENS)
    assert fc._PROVIDER_WALL_RE.pattern == kdd._PROVIDER_WALL_RE.pattern
    assert fc.classifier_deployed() is True


@pytest.mark.parametrize("text,expected", fc.ERROR_VECTORS)
def test_error_vector_table(text, expected):
    """The shared contract table: the classifier PR and this module agree on it."""
    assert fc.classify_error_text(text) == expected


@pytest.mark.parametrize("variant,text", [
    ("gateway_connection_failed", "Kanban owner refused: Gateway connection failed; no local fallback"),
    ("gateway_timeout", "Kanban owner refused: Gateway connection timed out; no local fallback"),
    ("gateway_disconnected", "Kanban owner refused: Gateway disconnected"),
    ("gateway_protocol_mismatch", "Kanban owner refused: Gateway protocol mismatch (v2 vs v3)"),
    ("execution_unknown", "Kanban owner refused: Kanban execution unknown; retained by owner"),
    ("owner_unavailable", "WorkerPersistenceError('owner_unavailable')"),
    ("owner_refused_timeout", "Kanban owner refused: the RPC timed out"),
    ("owner_refused_other", "Kanban owner refused: something else entirely"),
])
def test_host_loss_variants(variant, text):
    assert fc.host_loss_variant(text) == variant
    assert fc.classify_error_text(text) == fc.FAILURE_INFRA


def test_wrapper_must_be_in_the_tail():
    """The anchor is the tail window: gateway vocabulary far from the death is prose."""
    prose = "I could not reach the Gateway connection failed endpoint earlier. " + ("x" * 400)
    assert fc.host_loss_variant(prose) is None
    assert fc.classify_error_text(prose) is None


@pytest.mark.parametrize("text", [
    "worker exited cleanly (rc=0) without a terminal kanban call — protocol violation.",
    "protocol_violation: no handoff landed",
    "booked work_present: artifacts landed, handoff missing",
    "terminal_provider: model gpt-x is gone (404)",
    "provider wall: HTTP 429 Too Many Requests",
    "out of credits",
])
def test_card_classes(text):
    assert fc.classify_error_text(text) == fc.FAILURE_CARD


def test_card_precedence_beats_host_vocabulary():
    """A worker that printed gateway words and then failed its own protocol is CARD."""
    mixed = (
        "Kanban owner refused: Gateway connection failed; no local fallback — "
        "and the card itself exited cleanly without a terminal kanban call (protocol violation)"
    )
    assert fc.classify_error_text(mixed) == fc.FAILURE_CARD


def test_no_evidence_is_no_classification():
    assert fc.classify_error_text("") is None
    assert fc.classify_error_text(None) is None
    assert fc.classify_error_text("pid 1 exited with code 1") is None


# --- run-level ---------------------------------------------------------------

def test_classify_run_prefers_the_booking_flag(board):
    conn = board
    tid = kb.create_task(conn, title="card")
    run = _book(conn, tid, metadata={"infrastructure": True})
    row = conn.execute("SELECT * FROM task_runs WHERE id = ?", (run,)).fetchone()
    assert fc.classify_run(row) == fc.FAILURE_INFRA


def test_classify_run_falls_back_to_error_text_for_legacy_rows(board):
    """Every 2026-10-24 owner-loss run predates the flag: text must classify them."""
    conn = board
    tid = kb.create_task(conn, title="card")
    _book(conn, tid, error=OWNER_LOSS, metadata={"worker_output": "Kanban owner refused: Gateway connection failed"})
    assert fc.blocking_class(conn, tid) == fc.FAILURE_INFRA
    evidence = fc.failure_evidence(conn, tid)
    assert evidence["source"] == "error_text"
    assert evidence["variant"] == "gateway_connection_failed"


def test_classify_run_card_metadata_markers(board):
    conn = board
    for marker in ("protocol_violation", "work_present", "terminal_provider"):
        tid = kb.create_task(conn, title=f"card {marker}")
        _book(conn, tid, metadata={marker: True})
        assert fc.blocking_class(conn, tid) == fc.FAILURE_CARD, marker


def test_classify_run_success_and_neutral_are_not_bookings(board):
    conn = board
    tid = kb.create_task(conn, title="card")
    _book(conn, tid, outcome="completed", status="ready")
    assert fc.blocking_class(conn, tid) is None

    other = kb.create_task(conn, title="quota")
    _book(conn, other, outcome="rate_limited", error=OWNER_LOSS, status="ready")
    assert fc.blocking_class(conn, other) is None


def test_blocking_class_skips_neutral_quota_requeue(board):
    """A rate-limited requeue says nothing; the previous real booking still decides."""
    conn = board
    tid = kb.create_task(conn, title="card")
    _book(conn, tid, error=OWNER_LOSS, ended_at=int(time.time()) - 100)
    _book(conn, tid, outcome="rate_limited", status="ready", ended_at=int(time.time()) - 10)
    assert fc.blocking_class(conn, tid) == fc.FAILURE_INFRA


def test_blocking_class_none_without_runs(board):
    conn = board
    tid = kb.create_task(conn, title="never ran")
    assert fc.blocking_class(conn, tid) is None
    assert fc.latest_booking(conn, tid) is None


# --- graph direction ---------------------------------------------------------

def _fan_in(conn, n=2):
    root = kb.create_task(conn, title="root quest", triage=True)
    kids = kbg.decompose_triage_task(
        conn, root, root_assignee="builder",
        children=[{"title": f"child {i}", "assignee": "builder"} for i in range(n)],
    )
    return root, kids


def test_fan_in_edges_point_into_the_root(board):
    conn = board
    root, kids = _fan_in(conn, 2)
    rows = conn.execute("SELECT parent_id, child_id FROM task_links").fetchall()
    assert {(r["parent_id"], r["child_id"]) for r in rows} == {(k, root) for k in kids}
    assert fc.is_fan_in_root(conn, root) is True
    assert all(fc.is_fan_in_root(conn, k) is False for k in kids)
    assert fc.fan_in_children(conn, root) == sorted(kids)


def test_subtree_walks_the_fan_in_direction_not_the_dependents(board):
    """``WHERE child_id = root`` — reading ``parent_id`` would invert the scope."""
    conn = board
    root, kids = _fan_in(conn, 2)
    # A grandchild that WAITS ON kid 0 (edge grandchild -> kid 0): still in the
    # root's subtree, because the root waits for the whole graph.
    grand = kb.create_task(conn, title="grandchild")
    kb.link_tasks(conn, grand, kids[0])
    assert sorted(fc.subtree_ids(conn, root)) == sorted([kids[0], kids[1], grand])
    assert fc.subtree_ids(conn, kids[0]) == [grand]
    # A card that DEPENDS ON the root (the inverted direction) is not its subtree.
    dependent = kb.create_task(conn, title="dependent", parents=[root])
    assert dependent not in fc.subtree_ids(conn, root)


def test_nearest_fan_in_root(board):
    conn = board
    root, kids = _fan_in(conn, 2)
    assert fc.nearest_fan_in_root(conn, root) == root
    assert fc.nearest_fan_in_root(conn, kids[0]) == root
    leaf = kb.create_task(conn, title="standalone")
    assert fc.nearest_fan_in_root(conn, leaf) is None
