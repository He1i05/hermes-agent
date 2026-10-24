"""Tests for ``kanban_db.abandon_task`` and the ``hermes kanban abandon`` verb.

The behaviour under test is the operator escape hatch for a fan-in root whose
children were killed by the HOST: archive the root and the host-killed frontier
WITH a reason, never bury work that actually failed, never write to a terminal
card, and be idempotent.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_graph as kbg
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_explain as ke
from hermes_cli import kanban_failure_class as fc

OWNER_LOSS = (
    "pid 9 exited with code 1 Worker's last output: "
    "'Kanban owner refused: Gateway connection failed; no local fallback'"
)
REASON = "host killed the whole frontier on 2026-10-24"


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


def _book(conn, task_id, *, outcome="crashed", error=None, metadata=None, status="blocked"):
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, error, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (task_id, outcome, now - 30, now, outcome, error, json.dumps(metadata or {})),
    )
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()
    return int(cur.lastrowid)


def _tree(conn, *, infra=2, card=0, unclassified=0, done=0):
    root = kb.create_task(conn, title="root quest", triage=True)
    specs = ([{"title": f"infra {i}", "assignee": "builder"} for i in range(infra)]
             + [{"title": f"card {i}", "assignee": "builder"} for i in range(card)]
             + [{"title": f"unknown {i}", "assignee": "builder"} for i in range(unclassified)]
             + [{"title": f"done {i}", "assignee": "builder"} for i in range(done)])
    kids = kbg.decompose_triage_task(conn, root, root_assignee="builder", children=specs)
    order = {"infra": [], "card": [], "unknown": [], "done": []}
    for cid in kids:
        status_row = conn.execute("SELECT title FROM tasks WHERE id = ?", (cid,)).fetchone()
        order[status_row["title"].split()[0]].append(cid)
    for cid in order["infra"]:
        _book(conn, cid, error=OWNER_LOSS, metadata={"worker_output": OWNER_LOSS})
    for cid in order["card"]:
        _book(conn, cid, error="protocol violation: no handoff", metadata={"protocol_violation": True})
    for cid in order["unknown"]:
        _book(conn, cid, error="pid 2 exited with code 1")
    for cid in order["done"]:
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (cid,))
    conn.commit()
    return root, order


def _dump(conn) -> list:
    return [tuple(conn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall())
            for t in ("tasks", "task_links", "task_events", "task_runs")]


def _status(conn, task_id):
    return kb.get_task(conn, task_id).status


def _events(conn, task_id):
    return [e for e in kb.list_events(conn, task_id) if e.kind == "abandoned"]


# --- infra frontier ----------------------------------------------------------

def test_infra_child_failure_makes_the_root_abandonable(board):
    conn = board
    root, order = _tree(conn, infra=2)
    plan = ke.task_explain(conn, kb.get_task(conn, root), kanban_cfg={})["abandon"]
    assert plan["eligible"] is True
    assert plan["scope"] == sorted(order["infra"])
    assert plan["root"] == root

    result = kb.abandon_task(conn, root, reason=REASON, operator="juozas")
    assert result.ok is True, result.to_dict()
    assert _status(conn, root) == "archived"
    for cid in order["infra"]:
        assert _status(conn, cid) == "archived"
        task = kb.get_task(conn, cid)
        assert task.abandon_reason == REASON
        assert task.abandoned_by == "juozas"
        assert task.abandoned_at
        assert len(_events(conn, cid)) == 1
    root_events = _events(conn, root)
    assert len(root_events) == 1
    manifest = root_events[0].payload
    assert manifest["root_id"] == root
    assert manifest["reason"] == REASON
    assert sorted(manifest["scope"]) == sorted([root, *order["infra"]])
    assert manifest["forced"] is False


def test_genuine_child_failure_is_not_silently_abandoned(board):
    conn = board
    root, order = _tree(conn, infra=1, card=1)
    before = _status(conn, order["card"][0])

    plan = ke.task_explain(conn, kb.get_task(conn, root), kanban_cfg={})["abandon"]
    assert plan["eligible"] is False
    assert plan["blocked_by"] == [f"card_class_descendant:{order['card'][0]}"]
    assert plan["scope"] == order["infra"]

    result = kb.abandon_task(conn, root, reason=REASON, operator="juozas")
    assert result.ok is False
    assert result.blocked_by == [f"card_class_descendant:{order['card'][0]}"]
    assert result.archived == []
    # Zero writes: not one card moved, not one event written.
    assert _status(conn, root) == "todo"
    assert _status(conn, order["card"][0]) == before
    assert _status(conn, order["infra"][0]) == "blocked"
    for tid in (root, *order["infra"], *order["card"]):
        assert _events(conn, tid) == []


def test_unclassified_child_also_blocks_the_cascade(board):
    conn = board
    root, order = _tree(conn, infra=1, unclassified=1)
    result = kb.abandon_task(conn, root, reason=REASON)
    assert result.ok is False
    assert result.blocked_by == [f"unclassified_descendant:{order['unknown'][0]}"]


def test_force_archives_what_the_default_refuses(board):
    conn = board
    root, order = _tree(conn, infra=1, card=1)
    result = kb.abandon_task(conn, root, reason=REASON, force=True, operator="juozas")
    assert result.ok is True
    assert result.forced is True
    assert result.overridden_safeguards == [f"card_class_descendant:{order['card'][0]}"]
    assert _status(conn, order["card"][0]) == "archived"
    manifest = _events(conn, root)[0].payload
    assert manifest["forced"] is True
    assert manifest["overridden_safeguards"] == result.overridden_safeguards


def test_force_requires_a_specific_reason(board):
    conn = board
    root, _ = _tree(conn, infra=1, card=1)
    result = kb.abandon_task(conn, root, reason="oops", force=True)
    assert result.ok is False
    assert any("at least 20 characters" in error for error in result.errors)
    assert _status(conn, root) == "todo"


def test_missing_reason_is_refused(board):
    conn = board
    root, _ = _tree(conn, infra=1)
    result = kb.abandon_task(conn, root, reason="   ")
    assert result.ok is False
    assert result.errors and "reason is required" in result.errors[0]


# --- terminal / untouched cards ---------------------------------------------

def test_done_child_is_never_touched(board):
    conn = board
    root, order = _tree(conn, infra=1, done=1)
    done_id = order["done"][0]
    before = kb.get_task(conn, done_id)
    result = kb.abandon_task(conn, root, reason=REASON)
    assert result.ok is True
    assert done_id in result.already_terminal
    after = kb.get_task(conn, done_id)
    assert after == before
    assert _events(conn, done_id) == []


def test_keep_children_archives_only_the_root(board):
    conn = board
    root, order = _tree(conn, infra=2)
    result = kb.abandon_task(conn, root, reason=REASON, keep_children=True)
    assert result.ok is True
    assert result.archived == [root]
    for cid in order["infra"]:
        # No status write, no provenance, no event — the cascade never touched it.
        # (Its *status* may move because archiving the root releases the fan-in
        # edge and ``recompute_ready`` re-promotes it, which is the normal path.)
        task = kb.get_task(conn, cid)
        assert task.status != "archived"
        assert task.abandon_reason is None
        assert _events(conn, cid) == []


def test_leaf_card_degenerates_to_archive_with_a_reason(board):
    conn = board
    leaf = kb.create_task(conn, title="standalone card")
    result = kb.abandon_task(conn, leaf, reason=REASON)
    assert result.ok is True
    assert result.archived == [leaf]
    assert kb.get_task(conn, leaf).abandon_reason == REASON


# --- idempotency / dry run ---------------------------------------------------

def test_second_abandon_writes_nothing(board):
    conn = board
    root, _ = _tree(conn, infra=2)
    assert kb.abandon_task(conn, root, reason=REASON).ok is True
    dump = _dump(conn)
    again = kb.abandon_task(conn, root, reason=REASON)
    assert again.ok is True and again.already_abandoned is True
    assert again.archived == []
    assert _dump(conn) == dump


def test_dry_run_writes_nothing_but_names_the_plan(board):
    conn = board
    root, order = _tree(conn, infra=2)
    dump = _dump(conn)
    result = kb.abandon_task(conn, root, reason=REASON, dry_run=True)
    assert result.ok is True
    assert sorted(result.archived) == sorted([root, *order["infra"]])
    assert _dump(conn) == dump


# --- hard refusals -----------------------------------------------------------

def test_review_descendant_is_a_hard_refusal(board):
    conn = board
    root, order = _tree(conn, infra=1, card=1)
    review_id = order["card"][0]
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (review_id,))
    conn.commit()
    result = kb.abandon_task(conn, root, reason=REASON)
    assert result.ok is False
    assert f"review_descendant:{review_id}" in result.blocked_by
    assert _status(conn, review_id) == "review"


def test_running_descendant_is_a_hard_refusal(board):
    conn = board
    root, order = _tree(conn, infra=1, card=1)
    running_id = order["card"][0]
    conn.execute("UPDATE tasks SET status = 'running', claim_lock = 'other-host:1' WHERE id = ?",
                 (running_id,))
    conn.commit()
    result = kb.abandon_task(conn, root, reason=REASON)
    assert result.ok is False
    assert f"running_descendant:{running_id}" in result.blocked_by
    assert _status(conn, running_id) == "running"


def test_running_root_is_refused_without_force(board):
    conn = board
    root, _ = _tree(conn, infra=1)
    conn.execute("UPDATE tasks SET status = 'running', claim_lock = 'other-host:1' WHERE id = ?", (root,))
    conn.commit()
    result = kb.abandon_task(conn, root, reason=REASON)
    assert result.ok is False
    assert "root_running" in result.blocked_by


def test_unknown_root_is_refused(board):
    conn = board
    result = kb.abandon_task(conn, "t_deadbeef", reason=REASON)
    assert result.ok is False
    assert result.errors and "no such task" in result.errors[0]


# --- schema ------------------------------------------------------------------

def test_pre_migration_board_gains_the_columns_with_defaults(board, tmp_path):
    """Additive migration only: an existing board keeps its rows and gets NULL/0."""
    conn = board
    tid = kb.create_task(conn, title="legacy row")
    if sqlite3.sqlite_version_info < (3, 35, 0):
        pytest.skip("DROP COLUMN needs sqlite >= 3.35")
    for column in ("abandon_reason", "abandoned_at", "abandoned_by"):
        conn.execute(f"ALTER TABLE tasks DROP COLUMN {column}")
    conn.commit()
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("SELECT abandon_reason FROM tasks").fetchall()

    kb.init_db()
    task = kb.get_task(conn, tid)
    assert (task.abandon_reason, task.abandoned_at, task.abandoned_by) == (None, None, None)


# --- CLI ---------------------------------------------------------------------

def _ns(**overrides):
    base = dict(task_id=None, reason=REASON, force=False, keep_children=False,
                dry_run=False, json=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_abandon_refuses_with_a_reason_and_nonzero_exit(board, capsys):
    conn = board
    root, order = _tree(conn, infra=1, card=1)
    rc = kc._cmd_abandon(_ns(task_id=root))
    captured = capsys.readouterr()
    assert rc == 1
    assert "safeguard(s) would bury real work" in captured.err
    assert f"card_class_descendant:{order['card'][0]}" in captured.err
    assert _status(conn, root) == "todo"


def test_cli_abandon_dry_run_prints_the_plan_and_writes_nothing(board, capsys):
    conn = board
    root, _order = _tree(conn, infra=2)
    dump = _dump(conn)
    rc = kc._cmd_abandon(_ns(task_id=root, dry_run=True))
    captured = capsys.readouterr()
    assert rc == 0
    assert "Would abandon 3 card(s)" in captured.out
    assert _dump(conn) == dump


def test_cli_abandon_json_shape(board, capsys):
    conn = board
    root, order = _tree(conn, infra=1)
    rc = kc._cmd_abandon(_ns(task_id=root, json=True))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["root_id"] == root
    assert sorted(payload["archived"]) == sorted([root, *order["infra"]])
    assert payload["forced"] is False


def test_cli_abandon_is_denied_inside_a_delegated_child():
    assert "abandon" in kc._DELEGATED_CHILD_DENIED_ACTIONS


# --- audit trail / workspace -------------------------------------------------

def test_workspace_reap_is_recorded_from_what_was_actually_removed(board, monkeypatch, tmp_path):
    """The manifest must name the workspace loss, and only when it happened."""
    conn = board
    root, order = _tree(conn, infra=1)
    child = order["infra"][0]
    scratch = tmp_path / "kanban" / "workspaces" / child
    scratch.mkdir(parents=True)
    conn.execute("UPDATE tasks SET workspace_kind = 'scratch', workspace_path = ? WHERE id = ?",
                 (str(scratch), child))
    conn.commit()
    monkeypatch.setattr(kbw, "_is_managed_scratch_path", lambda p: True)
    monkeypatch.setattr(kbw, "_has_active_children", lambda c, t: False)
    monkeypatch.setattr(kbw, "_defer_shared_workspace_cleanup", lambda c, t, p: False)

    result = kb.abandon_task(conn, root, reason=REASON)
    assert result.ok is True
    assert result.reaped_workspace.get(child) == str(scratch)
    assert not scratch.exists()
    reap_events = [e for e in kb.list_events(conn, child) if e.kind == "abandon_workspace_reaped"]
    assert len(reap_events) == 1
    assert reap_events[0].payload["reaped_workspace"] == str(scratch)
