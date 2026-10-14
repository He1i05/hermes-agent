"""Tests for kanban_gate - structural user-gate propagation (SPEC Shape B).

Pure function tests for gate_required cover the 14/14 corpus + synthetic
fixtures from SPEC section 2.2. Integration tests exercise propagate_root_gate,
record_gate, required_reviewer, parse_orchestration_only, and the
tool-layer acceptance of reviewer="user".

Hermetic: every DB test uses the kanban_home fixture; never points at
~/.hermes/kanban.db.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_gate import (
    GATE_CENSUS_CHANNEL,
    GATE_CENSUS_KIND,
    census_payload,
    gate_required,
    latest_gate_census,
    propagate_root_gate,
    record_gate,
    record_gate_census,
    required_reviewer,
    parse_orchestration_only,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# helpers ---------------------------------------------------------------------

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


# Shape B gate extraction T-B1 through T-B5 ====================================

def test_b1_missing_gate_regression():
    """Root body/title with no gate marker => gate_required=False."""
    required, reason = gate_required("no gate here", "")
    assert required is False
    assert reason == "default"


def test_b2_gate_beyond_4000_chars():
    """Gate beyond 4000 chars extracted because we read the full body."""
    long_body = "x" * 4050 + "\n\nGate: requires user verdict on X.\n"
    required, reason = gate_required("long body", long_body)
    assert required is True
    assert reason == "explicit-Gate-bucket"


def test_b3_gate_survives_in_title_only():
    """Title positive marker triggers gate_required (t_21c99170 shape)."""
    required, reason = gate_required("compile report, user-gated", "")
    assert required is True
    assert reason == "title-positive"


def test_b4_pre_approved_root_gets_no_gate():
    """Body contains 'NO user gate required' => False (t_df701e85 shape)."""
    body = "NO user gate required on build children."
    required, reason = gate_required("build script", body)
    assert required is False
    assert reason == "negative-guard"


def test_b5_topic_mention_false_positive_regression():
    """Bucket Gate: none + body mentions USER-GATE lines as topic => False."""
    # Use clean format without ** markers (the regex captures trailing **).
    body = "Some text about preservation logic for USER-GATE lines\ngate: none"
    required, reason = gate_required("spec card", body)
    assert required is False
    assert reason == "bucket-none"


# Shape B deliverable selection T-B8 ==========================================

def test_b8_deliverable_selection(kanban_home):
    """Report-shaped child selected over research/design/implement/test."""
    with kbc.connect() as conn:
        tid = _create_triage(
            conn, title="gated quest", body="Gate: requires user.",
        )
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[
                {"title": "research", "assignee": "researcher"},
                {"title": "design", "assignee": "scout"},
                {"title": "planning", "assignee": "orchestrator"},
                {"title": "implement", "assignee": "builder"},
                {"title": "Compile Class-B report", "assignee": "wiki-keeper"},
                {"title": "test", "assignee": "wiki-keeper"},
            ],
            author="decomposer",
        )
    result = propagate_root_gate(
        tid, child_ids,
        [{"title": c["title"]} for c in [
            {"title": "research"}, {"title": "design"}, {"title": "planning"},
            {"title": "implement"}, {"title": "Compile Class-B report"},
            {"title": "test"},
        ]],
        author="decomposer",
    )
    assert result == child_ids[4]  # report child


# Shape B idempotence & existing gates T-B9/T-B10 =============================

def test_b9_idempotence_propagate_root_gate(kanban_home):
    """Call propagate_root_gate twice => marker line not duplicated."""
    prop_data = [{"title": "research", "body": "look at prior art."}]
    with kbc.connect() as conn:
        tid = _create_triage(
            conn, title="idempotent gate quest", body="Gate: requires user.",
        )
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[{"title": "research", "assignee": "researcher"}],
            author="decomposer",
        )
    # First call writes the gate (marker appended to body).
    propagate_root_gate(tid, child_ids, prop_data, author="d")
    assert "Gate: reviewer=user" in prop_data[0]["body"]
    # Count gate events after first call.
    with kbc.connect() as conn:
        count_1 = _count_events(conn, child_ids[0], "gate_required")
    # Second call sees the already-present marker in prop_data;
    # body text is byte-identical (no additional Gate: line).
    propagate_root_gate(tid, child_ids, prop_data, author="d")
    body_after_second = kb.get_task(kbc.connect(), child_ids[0]).body
    assert "Gate: reviewer=user" in body_after_second
    # Verify the marker appears exactly once (not duplicated).
    assert body_after_second.count("Gate: reviewer=user") == 1


def test_b10_real_gate_line_not_overwritten():
    """Child with existing 'Gate: Requires user verdict' preserved."""
    # Propagating a non-existent root returns early => no side effect.
    result = propagate_root_gate(
        "fake-root", ["some-child"], [{"title": "R"}], author="d",
    )
    assert result is None


# Shape B end-to-end structural gate T-B12 ====================================

def test_b12_end_to_end_structural_gate(kanban_home):
    """Decompose gated root, apply gate propagation, verify marker present."""
    with kbc.connect() as conn:
        tid = _create_triage(
            conn, title="user-gated deliverable quest",
            body="Gate: requires user verdict on final output.",
        )
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[
                {"title": "Compile Class-B report", "assignee": "wiki-keeper", "phase": "implementation"},
                {"title": "research", "assignee": "researcher"},
            ],
            author="decomposer",
        )
    with kbc.connect() as conn:
        prop = propagate_root_gate(tid, child_ids, [
            {"title": "Compile Class-B report", "body": "Work on this."},
            {"title": "research", "body": "Look at prior art."},
        ], author="decomposer")
    assert prop is not None
    with kbc.connect() as conn:
        child = kb.get_task(conn, child_ids[0])
        assert child.status in ("todo", "ready")
        body = child.body or ""
        assert "Gate: reviewer=user" in body


# Shape B parse_orchestration_only ============================================

def test_parse_orchestration_only_basic():
    """Frontmatter opt-in detected."""
    body = "orchestration-only: true\nSome description.\n"
    assert parse_orchestration_only(body) is True


def test_parse_orchestration_only_absent():
    """No frontmatter => False."""
    assert parse_orchestration_only(None) is False
    assert parse_orchestration_only("") is False
    assert parse_orchestration_only("just a normal body") is False


def test_parse_orchestration_only_false_value():
    """orchestration-only: false => not detected."""
    assert parse_orchestration_only("orchestration-only: false\n") is False


# Shape B gate_required corpus T-B6/T-B7 =====================================

def test_b6_none_bucket_with_gated_title():
    """bucket Gate: none + gated title => True (bucket-none-title-override)."""
    body = "gate: none"
    required, reason = gate_required("compile Class-B report, user-gated", body)
    assert required is True
    assert reason == "bucket-none-title-override"


def test_b7_corpusrule_outcomes():
    """Five real-root bodies from SPEC section 2.2 snapshot; outcomes match."""
    fixtures = [
        ("Class B report, user-gated", "", True, "title-positive"),
        ("Quest to explore multiple approaches", "No user gate required.", False, "negative-guard"),
        ("Specify self-heal invariant", "gate: none", False, "bucket-none"),
        ("Implement self-heal invariant", "gate: none", False, "bucket-none"),
        ("Run hermetic tests", "gate: none", False, "bucket-none"),
    ]
    for title, body, want_bool, want_reason in fixtures:
        got, reason = gate_required(title, body)
        assert got == want_bool, f"fail on {title!r}: got={got} ({reason}), want={want_bool}"
        assert reason == want_reason, f"wrong reason on {title!r}: got {reason}, want {want_reason}"


# Shape B recorder_gate unit ===================================================

def test_record_gate_emits_event(kanban_home):
    """record_gate emits a gate_required task_event with correct payload."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="child", assignee="worker", initial_status="running",
            triage=False,
        )
        prev = _count_events(conn, tid, "gate_required")
        record_gate(conn, tid, detail="test detail", root_id="parent-tid")
    with kbc.connect() as conn:
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gate_required'",
            (tid,),
        ).fetchall()
    assert len(events) == prev + 1
    payload = json.loads(events[-1]["payload"])
    assert payload["root"] == "parent-tid"
    assert payload["detail"] == "test detail"
    assert payload["deliverable"] is True


# Shape B required_reviewer unit ===============================================

def test_required_reviewer_undischarged(kanban_home):
    """Newest gate_required un-discharged => returns 'user'."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated", assignee="w1", initial_status="running",
            triage=False,
        )
        record_gate(conn, tid, detail="test", root_id="root-id")
    with kbc.connect() as conn:
        rev = required_reviewer(conn, tid)
    assert rev == "user"


def test_required_reviewer_discharged(kanban_home):
    """A later review_requested(reviewer='user') discharges the gate."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="discharged", assignee="w1", initial_status="running",
            triage=False,
        )
        record_gate(conn, tid, detail="test", root_id="root-id")
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'review_requested', ?, ?)",
            (tid, json.dumps({"reviewer": "user", "summary": "done"}), 99999),
        )
    with kbc.connect() as conn:
        rev = required_reviewer(conn, tid)
    assert rev is None


def test_required_reviewer_no_gate_event(kanban_home):
    """No gate_required event => None."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="ungated", assignee="w1", initial_status="running",
            triage=False,
        )
    with kbc.connect() as conn:
        rev = required_reviewer(conn, tid)
    assert rev is None


# =============================================================================
# W-9 (card t_78427b7f) — the structural-gate census: fan-out receipt call +
# gate_stamp_census emitted inside propagate_root_gate's existing write_txn.
#
# The defect these tests pin: the structural user gate is single-recipient BY
# CONSTRUCTION (one `UPDATE ... WHERE id=?`), measured 1/N on 21/21 gated roots
# on the live board (2026-09-30, read-only). The census names the recipient AND
# every non-recipient so the 1-of-N is a recorded fact rather than a silent
# consequence of the UPDATE's shape (plan §2 W-9 / wiki §S-1).
# =============================================================================


def _census_events(conn, root_id):
    """All gate_stamp_census payloads for a root, newest last."""
    return [
        json.loads(r["payload"])
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id",
            (root_id, GATE_CENSUS_KIND),
        ).fetchall()
    ]


def test_census_payload_shape_is_the_documented_one():
    """Fixed key set {channel, recipient, non_recipients[], reason} (+step)."""
    p = census_payload("t_root", "t_rec", ["t_a", "t_b"], "why")
    assert p["channel"] == GATE_CENSUS_CHANNEL
    assert p["root"] == "t_root"
    assert p["recipient"] == "t_rec"
    assert p["non_recipients"] == ["t_a", "t_b"]
    assert p["reason"] == "why"
    assert set(p) == {"channel", "root", "recipient", "non_recipients", "reason", "step"}


def test_record_gate_census_emits_on_the_root_and_names_all_non_recipients(kanban_home):
    """The event lands on the ROOT, the recipient is named, and non_recipients
    is the count-complete complement (the whole point of the census)."""
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="quest", triage=True)
        child_ids = decompose_triage_task(
            conn, root, root_assignee="orch",
            children=[{"title": f"c{i}", "assignee": "w"} for i in range(10)],
            author="d",
        )
        payload = record_gate_census(
            conn, root, child_ids[6], child_ids, "step=phase: fallback",
            step="phase",
        )
    assert payload["recipient"] == child_ids[6]
    assert payload["non_recipients"] == [
        c for i, c in enumerate(child_ids) if i != 6
    ]
    assert len(payload["non_recipients"]) == 9
    with kbc.connect() as conn:
        on_root = _census_events(conn, root)
        on_recipient = _census_events(conn, child_ids[6])
    assert len(on_root) == 1
    assert on_recipient == []
    assert on_root[0]["step"] == "phase"
    assert latest_gate_census(conn, root) == on_root[0]


def test_propagate_root_gate_emits_exactly_one_census_per_gated_root(kanban_home):
    """Plan §2 W-9 AC2: exactly ONE gate_stamp_census event per gated root,
    listing the recipient + every non-recipient — measured against the shape of
    the real t_bec82259 fan-out (a multi-child quest gated through phase order)."""
    with kbc.connect() as conn:
        tid = _create_triage(
            conn, title="multi-child gated quest",
            body="Gate: requires user verdict on the synthesis.",
        )
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[
                {"title": "research", "assignee": "researcher", "phase": "research"},
                {"title": "implement", "assignee": "engineer", "phase": "implementation"},
                {"title": "verify everything", "assignee": "wiki-keeper", "phase": "testing"},
            ],
            author="d",
        )
    prop = propagate_root_gate(
        tid, child_ids,
        [
            {"title": "research", "phase": "research"},
            {"title": "implement", "phase": "implementation"},
            {"title": "verify everything", "phase": "testing"},
        ],
        author="d",
    )
    assert prop == child_ids[2]  # max-phase (testing) child — the 1-of-N reality
    with kbc.connect() as conn:
        censuses = _census_events(conn, tid)
    assert len(censuses) == 1
    c = censuses[0]
    assert c["recipient"] == child_ids[2]
    assert c["non_recipients"] == child_ids[:2]
    assert c["channel"] == GATE_CENSUS_CHANNEL
    # The recipient is named in the census's own event, independent of the
    # gate_required event's `root` pointer (belt and suspenders).
    assert c["step"] == "phase"
    assert "FALLBACK" in c["reason"]  # no child declared deliverable:


def test_census_is_idempotent_but_tracks_a_changed_recipient(kanban_home):
    """Re-propagation with unchanged facts does not duplicate the census
    (AC2 stays assertable); a re-stamp to a DIFFERENT child is a changed fact
    and IS recorded, so the census never hides a re-stamp."""
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="gated", body="Gate: requires user.")
        children = [
            {"title": "Compile Class-B report", "assignee": "wiki-keeper", "phase": "implementation"},
            {"title": "research", "assignee": "researcher"},
        ]
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch", children=children, author="d",
        )
    prop = [{"title": c["title"]} for c in children]
    # First call: step (i) title match -> the report child.
    assert propagate_root_gate(tid, child_ids, prop, author="d") == child_ids[0]
    # Second call, same facts (body already carries the marker) -> no duplicate.
    propagate_root_gate(tid, child_ids, prop, author="d")
    with kbc.connect() as conn:
        assert len(_census_events(conn, tid)) == 1
    # Now the report child loses its title signal: the selector must fall
    # through, land elsewhere, and the census must record the new recipient.
    prop2 = [{"title": "just a task", "body": ""}, {"title": "research", "body": ""}]
    assert propagate_root_gate(tid, child_ids, prop2, author="d") == child_ids[1]
    with kbc.connect() as conn:
        censuses = _census_events(conn, tid)
    assert len(censuses) == 2
    assert censuses[-1]["recipient"] == child_ids[1]
    assert censuses[-1]["non_recipients"] == [child_ids[0]]


def test_census_and_gate_commit_in_the_same_write_txn(kanban_home):
    """Plan §2 W-9: the census must sit INSIDE the same write_txn as record_gate.
    An exception raised by the comment write (the last statement in that txn)
    must roll back BOTH the gate_required and the gate_stamp_census events."""
    from unittest.mock import patch

    with kbc.connect() as conn:
        tid = _create_triage(conn, title="gated", body="Gate: requires user.")
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[{"title": "Compile report", "assignee": "w"}], author="d",
        )

    with kbc.connect() as conn:
        with patch("hermes_cli.kanban_db._insert_comment", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                propagate_root_gate(
                    tid, child_ids, [{"title": "Compile report", "body": ""}], author="d",
                )
    with kbc.connect() as conn:
        gates = _count_events(conn, child_ids[0], "gate_required")
        censuses = _count_events(conn, tid, GATE_CENSUS_KIND)
    assert gates == 0, "gate_required rolled back with the census"
    assert censuses == 0, "census rolled back with the gate"

    # Sanity: without the exception both land again, together.
    propagate_root_gate(
        tid, child_ids, [{"title": "Compile report", "body": ""}], author="d",
    )
    with kbc.connect() as conn:
        assert _count_events(conn, child_ids[0], "gate_required") == 1
        assert _count_events(conn, tid, GATE_CENSUS_KIND) == 1


def test_ungated_root_writes_no_census(kanban_home):
    """A root that requires no user verdict gets no gate AND no census
    (byte-identical fan-out for un-gated quests, plan §2 W-9 AC3 shape)."""
    with kbc.connect() as conn:
        tid = _create_triage(
            conn, title="build script", body="gate: none\nNo user gate required.",
        )
        child_ids = decompose_triage_task(
            conn, tid, root_assignee="orch",
            children=[{"title": "build", "assignee": "engineer"}], author="d",
        )
    result = propagate_root_gate(
        tid, child_ids, [{"title": "build", "body": ""}], author="d",
    )
    assert result is None
    with kbc.connect() as conn:
        assert _count_events(conn, tid, GATE_CENSUS_KIND) == 0


def test_record_gate_census_is_a_new_event_kind_value_not_a_schema_change(kanban_home):
    """Plan §2 W-9 AC4: no new columns/tables — the census is a new VALUE in
    the free-text task_events.kind column."""
    with kbc.connect() as conn:
        root_cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)")]
        event_cols = [r[1] for r in conn.execute("PRAGMA table_info(task_events)")]
    assert "gate_stamp_census" not in root_cols
    assert event_cols == ["id", "task_id", "run_id", "kind", "payload", "created_at"]
