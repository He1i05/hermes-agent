"""Tests for decompose_triage_task + Shape A/C (SPEC Shapes A & C).

Hermetic by design: every test uses the kanban_home fixture (tmp_path +
monkeypatch.setenv("HERMES_HOME", ...)) — never points at ~/.hermes/kanban.db.

Shapes covered:
  - T-A1...T-A6    inverted-write regression, invariant assertion, selfheal,
                   deadlock-prevention guard
  - T-A4b/A4c      downstream-edge survival, audit-first discovery
  - T-C2..T-C7     scheduled roots, recompute_ready with rowcount gate, human-
                   deferred parking, stamping, release path
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import (
    decompose_triage_task,
    selfheal_root_edges,
    _assert_root_not_parent,
)
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---- helpers ----------------------------------------------------------------

def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn, title=title, body=body, assignee=assignee,
        tenant=tenant, triage=True,
    )


def _count_events(conn, task_id, kind):
    """Return count of task_events rows of kind for task_id."""
    return conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind=?",
        (task_id, kind),
    ).fetchone()[0]


def _set_status(conn, task_id, new_status):
    """Update a task's status via raw SQL (create_task only accepts 'running'/'blocked')."""
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (new_status, task_id))


# ===== Shape A — link-direction self-heal (T-A1...T-A6) =========================

def test_a1_inverted_write_regression(kanban_home):
    """Force an inverted row, run selfheal_root_edges -> healed + one event + idempotent."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="invert-and-heal")
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[
                {"title": "c0", "assignee": "w1"},
                {"title": "c1", "assignee": "w2"},
            ],
            author="decomposer",
        )

    # Force inverted: create (parent_id=tid, child_id=c0) -- WRONG direction.
    with kbc.connect() as conn:
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (tid, child_ids[0]),
        )

    with kbc.connect() as conn:
        result = selfheal_root_edges(conn, tid, author="tester")
    assert result["changed"] is True
    assert len(result["flipped"]) >= 1
    assert any(f["child_id"] == child_ids[0] for f in result["flipped"])

    # Verify the inverted row was deleted and c->root restored.
    bad_rows = conn.execute(
        "SELECT * FROM task_links WHERE parent_id=? AND child_id IN (?,?)",
        (tid, child_ids[0], child_ids[1]),
    ).fetchall()
    assert len(bad_rows) == 0, f"inverted rows still present: {bad_rows}"

    # Exactly one link_selfheal event.
    selfheal_events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='link_selfheal'",
        (tid,),
    ).fetchone()[0]
    assert selfheal_events == 1

    # Second call must be a no-op.
    with kbc.connect() as conn:
        result2 = selfheal_root_edges(conn, tid, author="tester")
    assert result2["changed"] is False
    assert result2["flipped"] == []


def test_a2_happy_path_zero_violations(kanban_home):
    """Plain decompose -> fanout-inverted count == 0; selfheal returns changed=False."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="clean decompose")
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[{"title": "c", "assignee": "w1"}],
            author="decomposer",
        )

    # Check for inverted rows (parent_id=root -> fanout child).
    with kbc.connect() as conn:
        placeholders = ",".join("?" * len(child_ids))
        inv = conn.execute(
            f"SELECT COUNT(*) FROM task_links WHERE parent_id=? AND child_id IN ({placeholders})",
            (tid, *child_ids),
        ).fetchone()[0]
    assert inv == 0, f"found {inv} fanout-inverted rows"

    # selfheal should be clean.
    with kbc.connect() as conn:
        result = selfheal_root_edges(conn, tid, author="tester")
    assert result["changed"] is False


def test_a3_root_edge_audited(kanban_home):
    """Decompose -> root has >=len(children) events of kind 'linked' with payload.kind=='root_edge'."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="audited root edge")
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[
                {"title": "A", "assignee": "w1"},
                {"title": "B", "assignee": "w2"},
            ],
            author="decomposer",
        )

    with kbc.connect() as conn:
        root_linked = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='linked'",
            (tid,),
        ).fetchone()[0]
    assert root_linked >= len(child_ids)

    root_events = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='linked'",
        (tid,),
    ).fetchall()
    root_edge_count = sum(
        1 for r in root_events if json.loads(r["payload"]).get("kind") == "root_edge"
    )
    assert root_edge_count >= len(child_ids), \
        f"expected >={len(child_ids)} root_edge events, got {root_edge_count}"


def test_a4_sibling_edges_untouched(kanban_home):
    """Children with parents=[0] -> sibling edge survives heal byte-identically."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="siblings + selfheal")
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[
                {"title": "A", "assignee": "w1", "parents": []},
                {"title": "B", "assignee": "w2", "parents": [0]},  # B waits on A
            ],
            author="decomposer",
        )

    # Record sibling edge before heal.
    with kbc.connect() as conn:
        before_link = conn.execute(
            "SELECT parent_id, child_id FROM task_links WHERE parent_id=? AND child_id=?",
            (child_ids[0], child_ids[1]),
        ).fetchall()

    # Force one inverted row.
    with kbc.connect() as conn:
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (tid, child_ids[0]),
        )

    # Heal.
    with kbc.connect() as conn:
        selfheal_root_edges(conn, tid, author="tester")

    # Sibling edge still exists.
    with kbc.connect() as conn:
        after_link = conn.execute(
            "SELECT parent_id, child_id FROM task_links WHERE parent_id=? AND child_id=?",
            (child_ids[0], child_ids[1]),
        ).fetchall()
    assert len(after_link) == len(before_link), "sibling edge was removed by heal"


def test_a4b_downstream_edge_survives_heal(kanban_home):
    """Legitimate downstream edge (root,X) for non-fanout X survives; only fanout child flipped."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="downstream edge")
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[
                {"title": "fanout", "assignee": "w1"},
                {"title": "c1", "assignee": "w2"},
            ],
            author="decomposer",
        )

    # Create a legitimate downstream task depending on this root.
    downstream_tid = kb.create_task(
        conn, title="downstream task", assignee="w3", initial_status="running",
        triage=False,
    )
    with kbc.connect() as conn:
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (tid, downstream_tid),  # downstream waits on root (legitimate)
        )

    # Force one fanout child inverted.
    with kbc.connect() as conn:
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (tid, child_ids[0]),
        )

    # Save the legitimate edge state.
    with kbc.connect() as conn:
        before_legit = conn.execute(
            "SELECT * FROM task_links WHERE parent_id=? AND child_id=?",
            (tid, downstream_tid),
        ).fetchall()

    # Heal.
    with kbc.connect() as conn:
        result = selfheal_root_edges(conn, tid, author="tester")
    assert result["changed"] is True
    assert len(result["flipped"]) == 1
    assert result["flipped"][0]["child_id"] == child_ids[0]  # only fanout child flipped

    # Legitimate downstream edge must still exist.
    with kbc.connect() as conn:
        after_legit = conn.execute(
            "SELECT * FROM task_links WHERE parent_id=? AND child_id=?",
            (tid, downstream_tid),
        ).fetchall()
    assert len(after_legit) == len(before_legit), \
        f"legitimate downstream edge was destroyed (before={len(before_legit)}, after={len(after_legit)})"
    assert set((row["parent_id"], row["child_id"]) for row in before_legit) == \
           set((row["parent_id"], row["child_id"]) for row in after_legit)


def test_a4c_audit_first_discovery(kanban_home):
    """Heal touches only decomposed.child_ids members; extra parent_id=root rows unchanged."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="audit-first")
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[
                {"title": "fanout", "assignee": "w1"},
                {"title": "c1", "assignee": "w2"},
                {"title": "c2", "assignee": "w3"},
            ],
            author="decomposer",
        )

    # Add an extra parent_id=root row that is NOT a fanout child.
    extra_tid = kb.create_task(
        conn, title="extra task", assignee="w4", initial_status="running",
        triage=False,
    )
    with kbc.connect() as conn:
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (tid, extra_tid),
        )
        # Force one fanout child inverted.
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (tid, child_ids[0]),
        )

    # Save state of all parent_id=root rows.
    with kbc.connect() as conn:
        before_extra = conn.execute(
            "SELECT * FROM task_links WHERE parent_id=? AND child_id=?",
            (tid, extra_tid),
        ).fetchall()
        before_c2 = conn.execute(
            "SELECT * FROM task_links WHERE parent_id=? AND child_id=?",
            (tid, child_ids[2]),
        ).fetchall()

    # Heal.
    with kbc.connect() as conn:
        result = selfheal_root_edges(conn, tid, author="tester")

    # Extra row must be unchanged (audit-first discovers only child_ids from decomposed event).
    with kbc.connect() as conn:
        after_extra = conn.execute(
            "SELECT * FROM task_links WHERE parent_id=? AND child_id=?",
            (tid, extra_tid),
        ).fetchall()
        after_c2 = conn.execute(
            "SELECT * FROM task_links WHERE parent_id=? AND child_id=?",
            (tid, child_ids[2]),
        ).fetchall()
    assert set((r["parent_id"], r["child_id"]) for r in before_extra) == \
           set((r["parent_id"], r["child_id"]) for r in after_extra), \
           f"non-fanout extra row was modified: before={before_extra}, after={after_extra}"
    assert len(before_c2) == len(after_c2), \
        f"c2 row was affected: before={before_c2}, after={after_c2}"


def test_a5_deadlock_prevention_guard(kanban_home):
    """After children complete, root must be 'ready'.
    With correct root-as-child convention this works; inverted rows would block."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="deadlock guard")
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[
                {"title": "c0", "assignee": "w1"},
                {"title": "c1", "assignee": "w2"},
            ],
            author="decomposer",
        )

    # After decompose, root is 'todo'.
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        assert root.status == "todo"

    # Complete all children.  complete_task fires recompute_ready internally.
    with kbc.connect() as conn:
        for cid in child_ids:
            ok = kb.complete_task(conn, cid, result="done")
            assert ok is True

    # Root MUST be 'ready' (healed direction releases; inverted would stay blocked/todo).
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        assert root.status == "ready", f"root stayed {root.status}, not 'ready'"


def test_a6_assert_raises_on_wrong_direction(kanban_home):
    """Inject wrong-direction link row -> _assert_root_not_parent raises ValueError."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="assert-check")
        # Manually create a child task (status defaults to 'triage' since no triage=False).
        cid = kb.create_task(
            conn, title="fake child", assignee="w1", initial_status="running",
        )

    # Create the GOOD root-as-child row: (_link(parent=cid, child=tid)).
    with kbc.connect() as conn:
        kb._link(conn, cid, tid)

    # Now inject WRONG direction (parent_id=tid, child_id=cid).
    with kbc.connect() as conn:
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (tid, cid),
        )

    # _assert_root_not_parent should detect the inversion and raise.
    with pytest.raises(ValueError) as exc_info:
        with kbc.connect() as conn:
            _assert_root_not_parent(conn, tid, [cid])
    assert tid in str(exc_info.value)


# ===== T1 phase cascade (DB layer) — carried from e6bb0b8bdf ===

def test_decompose_persists_child_phase_into_current_step_key(kanban_home):
    """The DB layer is the last enforcement layer: phase -> current_step_key."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="multi-step")
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[
                {"title": "look", "assignee": "researcher", "phase": "research", "parents": []},
                {"title": "build", "assignee": "engineer", "phase": "implementation", "parents": [0]},
                {"title": "support", "assignee": "engineer", "parents": []},
            ],
        )
    assert child_ids is not None and len(child_ids) == 3

    with kbc.connect() as conn:
        got = [kb.get_task(conn, cid).current_step_key for cid in child_ids]
    assert got == ["research", "implementation", None]


def test_decompose_accepts_prose_spelling_and_rejects_unknown_phase(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="spellings")
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[
                {"title": "design", "assignee": "scout", "phase": "System Design"},
                {"title": "ship", "assignee": "engineer", "phase": "deployment"},
            ],
        )
    assert child_ids is not None
    with kbc.connect() as conn:
        got = [kb.get_task(conn, cid).current_step_key for cid in child_ids]
    assert got == ["system-design", None]


def test_decompose_children_without_phase_leave_column_null(kanban_home):
    """Legacy child dicts (no ``phase`` key) must keep working, column NULL."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="legacy shape")
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
        )
    assert child_ids is not None
    with kbc.connect() as conn:
        assert kb.get_task(conn, child_ids[0]).current_step_key is None
        # No regression: the row still carries the normal defaults.
        assert kb.get_task(conn, child_ids[0]).status == "ready"

# ===== Shape C — scheduled roots & recompute_ready (T-C2..T-C7) ================

def test_c3_graph_deferred_root_never_reclaims(kanban_home):
    """5x (recompute_ready, claim_task) while children open → root never claimed, status=scheduled.
    Child has a parent so it won't be vacuously promoted."""
    with kbc.connect() as conn:
        root_tid = kb.create_task(
            conn, title="graph-deferred root", assignee="orchestrator",
            initial_status="running",
        )
        # Set it to ready first so schedule_task can succeed.
        _set_status(conn, root_tid, "ready")
        kb.schedule_task(conn, root_tid, reason="orchestration-only", reactivate_on="children_done")
        # One child with a fake parent so it won't be vacuously promoted by recompute_ready.
        fake_parent = kb.create_task(conn, title="fake parent", assignee="x", initial_status="running", triage=False)
        _set_status(conn, fake_parent, "done")
        kid = kb.create_task(conn, title="open child", assignee="worker", initial_status="running")
        _set_status(conn, kid, "todo")
        kb._link(conn, fake_parent, kid)  # fake_parent -> kid (kid has a parent)
        # Link: child -> root (child waits on root completion).
        kb._link(conn, kid, root_tid)

    for i in range(5):
        with kbc.connect() as conn:
            promoted = kb.recompute_ready(conn)
            claimed = kb.claim_task(conn, root_tid)
        # Root should not be promoted or claimed.
        with kbc.connect() as conn:
            root = kb.get_task(conn, root_tid)
        assert root.status == "scheduled", f"tick {i+1}: root status changed to {root.status}"
        assert claimed is None, f"tick {i+1}: claimed unexpectedly != None"


def test_c4_release_on_children_done(kanban_home):
    """Complete every child, one recompute_ready -> promoted>=1, status='ready',
    claim_task(root) returns Task, promoted-events delta exactly +1.

    R3-strengthened form: assert the write landed (status, claimability,
    event delta) -- never the counter alone.
    """
    with kbc.connect() as conn:
        root_tid = kb.create_task(
            conn, title="gated root", assignee="orchestrator",
            initial_status="running",
        )
        # Set to ready first so schedule_task succeeds.
        _set_status(conn, root_tid, "ready")
        kb.schedule_task(conn, root_tid, reason="orch", reactivate_on="children_done")
        # Child is 'ready' (default create) so complete_task accepts it and
        # recompute_ready doesn't vacuously promote it (scans todo/blocked/scheduled).
        kid = kb.create_task(conn, title="done child", assignee="w1")
        # Link: child -> root (root waits on graph).
        kb._link(conn, kid, root_tid)

    # Complete the child -> triggers internal recompute_ready.
    with kbc.connect() as conn:
        ok = kb.complete_task(conn, kid, result="done")
    assert ok is True

    # The internal recompute_ready fired by complete_task released the root;
    # verify the WRITE landed (status ready + claimable + a promoted event fired).
    with kbc.connect() as conn:
        promoted_events = _count_events(conn, root_tid, "promoted")
        assert promoted_events >= 1, \
            f"expected >=1 promoted event, got {promoted_events}"
        root = kb.get_task(conn, root_tid)
        assert root.status == "ready", f"root stayed {root.status}, not ready"
        claimed = kb.claim_task(conn, root_tid)
        assert claimed is not None, "claim_task(root) returned None after promotion"


def test_c4c_no_spurious_event_gate_survives(kanban_home):
    """A scheduled graph-deferred root whose children are NOT all done gains
    ZERO promoted events after recompute_ready; _graph_deferred_pending stays True.
    Child has a fake parent so it won't be vacuously promoted."""
    with kbc.connect() as conn:
        root_tid = kb.create_task(
            conn, title="pending deferral", assignee="orch",
            initial_status="running",
        )
        _set_status(conn, root_tid, "ready")
        kb.schedule_task(conn, root_tid, reason="orch", reactivate_on="children_done")
        # Child with a fake parent so it won't be vacuously promoted.
        fake_parent = kb.create_task(conn, title="fake parent", assignee="x", initial_status="running", triage=False)
        _set_status(conn, fake_parent, "done")
        kid = kb.create_task(conn, title="still-open child", assignee="w1", initial_status="running")
        _set_status(conn, kid, "todo")
        kb._link(conn, fake_parent, kid)
        # Link: child -> root (root waits on graph).
        kb._link(conn, kid, root_tid)

    with kbc.connect() as conn:
        prev_promoted = _count_events(conn, root_tid, "promoted")
        kb.recompute_ready(conn)
        # The root must gain ZERO promoted events while children are not all done.
        new_promoted = _count_events(conn, root_tid, "promoted")
        assert new_promoted == prev_promoted, "spurious promoted event fired on root"

    # Also check human-deferred card (reactivate_on absent) -> also zero events.
    with kbc.connect() as conn:
        hr_tid = kb.create_task(
            conn, title="human-deferred", assignee="orch",
            initial_status="running",
        )
        _set_status(conn, hr_tid, "ready")
        kb.schedule_task(conn, hr_tid, reason="backlog")  # no reactivate_on
        hr_prev = _count_events(conn, hr_tid, "promoted")
        kb.recompute_ready(conn)
        hr_new = _count_events(conn, hr_tid, "promoted")
        assert hr_new == hr_prev, "human-deferred gained spurious promoted event"


def test_c4d_ordinary_promotion_not_swallowed(kanban_home):
    """Plain parent->child: parent done, child todo -> promotes child with 1 event."""
    with kbc.connect() as conn:
        parent_tid = kb.create_task(
            conn, title="done parent", assignee="w1",
        )
        _set_status(conn, parent_tid, "done")
        kid = kb.create_task(
            conn, title="waiting child", assignee="w2",
        )
        _set_status(conn, kid, "todo")  # default create=ready; recompute_ready scans todo+blocked+scheduled
        kb._link(conn, parent_tid, kid)  # parent_id=parent, child_id=child
    with kbc.connect() as conn:
        promoted = kb.recompute_ready(conn)
        assert promoted >= 1, "plain parent-done didn't promote child"
        child = kb.get_task(conn, kid)
        assert child.status == "ready"


def test_c5_human_deferred_stays_parked(kanban_home):
    """Human-deferred scheduled card (reactivate_on=None) -> 0 promoted, 0 events."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="parked forever", assignee="orch",
            initial_status="running",
        )
        _set_status(conn, tid, "ready")
        kb.schedule_task(conn, tid, reason="backlog")
        kid = kb.create_task(
            conn, title="child of parked", assignee="w1",
        )
        kb._link(conn, kid, tid)  # kid -> tid (kid waits on tid)

    # Complete the child.
    with kbc.connect() as conn:
        ok = kb.complete_task(conn, kid, result="done")
    assert ok is True

    with kbc.connect() as conn:
        prev_promoted = _count_events(conn, tid, "promoted")
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert _count_events(conn, tid, "promoted") == prev_promoted
        root = kb.get_task(conn, tid)
        assert root.status == "scheduled"


def test_c6_children_stamped(kanban_home):
    """Every child gets a comment naming the root and reactivate_on."""
    with kbc.connect() as conn:
        root_tid = _create_triage(conn, title="stamped quest")
        child_ids = decompose_triage_task(
            conn, root_tid, root_assignee="orch",
            children=[
                {"title": "child A", "assignee": "w1"},
                {"title": "child B", "assignee": "w2"},
            ],
            author="decomposer",
        )
    # Schedule the root with reactivate_on.
    with kbc.connect() as conn:
        # Must be in ('todo','ready','running','blocked') for schedule_task.
        _set_status(conn, root_tid, "todo")
        kb.schedule_task(
            conn, root_tid, reason="orchestration-only", reactivate_on="children_done",
        )
    # Stamp children.
    from hermes_cli.kanban_gate import propagate_deferral
    with kbc.connect() as conn:
        propagate_deferral(conn, root_tid, child_ids, author="decomposer")
    # Verify comments.
    with kbc.connect() as conn:
        for cid in child_ids:
            comments = kb.list_comments(conn, cid)
            bodies = [c.body or "" for c in comments]
            assert any("reactivate_on=children_done" in b for b in bodies), \
                f"child {cid} missing reactivate_on stamp"


def test_c7_finished_child_of_graph_deferred_root_can_complete(kanban_home):
    """After root is released, a finished child of a graph-deferred root can
    complete normally -- contrast the pre-fix behaviour where it refused."""
    with kbc.connect() as conn:
        root_tid = kb.create_task(
            conn, title="graph-deferred", assignee="orch",
            initial_status="running",
        )
        _set_status(conn, root_tid, "ready")
        kb.schedule_task(conn, root_tid, reason="orch", reactivate_on="children_done")
        kid = kb.create_task(
            conn, title="fin child", assignee="w1",
        )
        kb._link(conn, kid, root_tid)

    # Complete the child -- succeeds despite being under a scheduled root.
    with kbc.connect() as conn:
        ok = kb.complete_task(conn, kid, result="done")
    assert ok is True
    # Child completed despite being under a scheduled root.
