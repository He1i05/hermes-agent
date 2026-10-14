"""W-10: the DECLARED DELIVERABLE marker + robust selector (card t_82eafe14).

The defect under test
---------------------
``kanban_gate._select_deliverable_child`` routes the structural USER GATE to
exactly ONE child per root (1-of-N *by construction*). The recipient used to be
chosen by a heuristic whose phases were ordered by the SDLC chain, so the
terminal chain phase (``testing``) always won: on ``t_bec82259`` (10 children)
the gate landed on ``t_f4ecdef7``, a *verification* card, not the synthesis.
Measured read-only on the live board 2026-09-30 (code pin 26b000cc0f), 21/21
gated roots reproduce that heuristic exactly — but it is a FALLBACK, and it was
silent.

Two things are asserted here:

* the tie-break operator is corrected to LAST-wins (``>=``) as the docstring
  always claimed — a real change with a measured blast radius (the FIRST vs the
  LAST of equal-max-phase siblings), and
* the real cure is a DECLARATION: a child that writes ``deliverable: true`` or
  ``delivers: <root-id>`` is chosen on purpose, and a heuristic-decided
  recipient is REPORTED as a fallback rather than passing silently.

Hermetic: every board is a throwaway ``HERMES_HOME`` under ``tmp_path`` (the
repo's ``kanban_home`` fixture) and nothing here opens the live ``kanban.db``.
Under ``scripts/run_tests.sh`` the whole file additionally runs inside ``env -i``.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_gate as gate
from hermes_cli import kanban_streams as streams


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# --------------------------------------------------------------------------- #
# fixtures — the measured shapes, not invented ones
# --------------------------------------------------------------------------- #

#: ``t_bec82259``'s ORIGINAL 10-child fan-out in emission order, with each
#: child's canonical phase (frozen from the read-only child-list probe).
_BEC_CHILDREN = [
    ("t_13573906", "Audit decomposer scope loss & release-protocol handoff failures", "grounding"),
    ("t_acf927d3", "Propose decomposer completeness rules & stamp propagation architecture", "research"),
    ("t_17784ba2", "Architect release watchdog cron & decomposer routing patch", "system-design"),
    ("t_6fa2f4da", "Plan implementation steps for decomposer patch & watchdog cron", "engineering-planning"),
    ("t_09110241", "User review & authorization for decomposer & release protocol changes", None),
    ("t_49ae5a1c", "Implement decomposer completeness logger, stamp propagator & watchdog cron", "implementation"),
    ("t_f4ecdef7", "Verify atomic handoff, stamp propagation & decomposer stream mapping", "testing"),
    ("t_29933582", "Update kanban orchestration wiki with new decomposer rules & release protocol", None),
    ("t_a14757c9", "Draft rollout plan & user notification for decomposer & release protocol", None),
    ("t_04c12d1e", "Land wiki page for: Propose decomposer completeness rules & stamp propagation", None),
]

#: ``t_26d8eb94``'s 4-child fan-out (grounding + three ``research`` siblings —
#: the tie-break tie that decides the recipient).
_D26_CHILDREN = [
    ("t_6ae16538", "Adjudicate D-Lane Dirty Tree & Create Read-Only Snapshot", "grounding"),
    ("t_e1cea019", "Structural & Meta-Harness Audit of D Pipeline", "research"),
    ("t_89ff2358", "Analysis-Quality & Signal-Economics Audit of D Pipeline", "research"),
    ("t_21082df6", "Adjudicate D-Lane Cron State & Wiki Staleness", "research"),
]

#: ``t_6ae16538`` — the root where the ``>` -> ``>=`` change was MEASURED to
#: move the recipient (4 children, grounding + three ``research``).
_TIE_CHILDREN = [
    ("t_6ae16538", "Adjudicate D-Lane Dirty Tree", "grounding"),
    ("t_e1cea019", "Structural & Meta-Harness Audit of D Pipeline", "research"),
    ("t_89ff2358", "Analysis-Quality Audit of D Pipeline", "research"),
    ("t_21082df6", "Adjudicate D-Lane Cron State & Wiki Staleness", "research"),
]


def _pairs(rows, bodies: dict | None = None):
    """``(children, ids)`` from a frozen ``(id, title, phase)`` fixture."""
    children = [
        {"title": title, "phase": phase, "body": (bodies or {}).get(cid, "")}
        for cid, title, phase in rows
    ]
    return children, [cid for cid, _, _ in rows]


def _legacy_strict_gt(children, ids):
    """The PRE-W-10 selector step (ii): strict ``>``, so the FIRST max-phase
    child wins. Kept here as the measured negative control (a fix nobody can
    "simplify" back without this test failing)."""
    best, best_idx = None, -1
    for cid, child in zip(ids, children):
        phase = gate.normalize_phase(child.get("phase"))
        if phase is None or phase not in gate.PHASE_INDEX:
            continue
        idx = gate.PHASE_INDEX[phase]
        if idx > best_idx:
            best_idx, best = idx, cid
    return best


# --------------------------------------------------------------------------- #
# 1 — the declaration parser (W-10 literals, single source of truth in streams)
# --------------------------------------------------------------------------- #

def test_declaration_literals_are_shared_with_the_streams_contract():
    """The selector keys off the SAME literals the prompt contract renders."""
    assert gate.DELIVERABLE_TRUE_KEY == streams.DELIVERABLE_KEY == "deliverable"
    assert gate.DELIVERS_KEY == streams.DELIVERS_KEY == "delivers"
    assert gate.declared_deliverable is streams.declared_deliverable


@pytest.mark.parametrize("body, want", [
    ("deliverable: true", "true"),
    ("- **deliverable**: true", "true"),
    ("Deliverable: YES\nGate: none", "true"),
    ("deliverable: 1", "true"),
    ("delivers: t_abcd1234", "t_abcd1234"),
    ("- delivers: t_abcd1234", "t_abcd1234"),
    ("phase: implementation\nwork\ndelivers: t_root\nAdoption: none", "t_root"),
    # Non-declarations:
    ("deliverable: false", None),
    ("deliverable: maybe", None),
    ("delivers:", None),
    ("Primary deliverable: FEASIBILITY-REPORT.md", None),
    ("the deliverable is the root synthesis", None),
    ("## DELIVERABLE SHAPE\n- One consolidated packet.", None),
    ("", None),
    (None, None),
])
def test_declared_deliverable_is_declaration_only(body, want):
    assert streams.declared_deliverable(body) == want


def test_deliverable_line_is_rendered_idempotently_and_never_fabricated():
    body = "phase: implementation\ncode it\n"
    once = streams.ensure_deliverable_line(body, deliverable=True)
    assert once == f"{body.rstrip()}\n\ndeliverable: true"
    twice = streams.ensure_deliverable_line(once, deliverable=True, delivers="t_root")
    assert twice == once, "an existing declaration is authoritative (never duplicated)"
    assert streams.ensure_deliverable_line(body) == body, "no declaration -> byte-identical"
    assert streams.deliverable_lines(deliverable=True, delivers="t_ab12") == [
        "delivers: t_ab12", "deliverable: true",
    ]
    assert streams.deliverable_lines(deliverable=False) == []


# --------------------------------------------------------------------------- #
# 2 — the selector order: declared > title > phase(last-wins) > last child
# --------------------------------------------------------------------------- #

def test_selector_prefers_the_declared_marker_over_every_heuristic():
    children, ids = _pairs(_BEC_CHILDREN)
    bodies = {"t_29933582": "phase: implementation\nUpdate the wiki.\ndeliverable: true\n"}
    for child, (cid, _, _) in zip(children, _BEC_CHILDREN):
        child["body"] = bodies.get(cid, "")
    sel = gate.select_deliverable(children, ids, root_id="t_bec82259")
    assert sel.child_id == "t_29933582"
    assert sel.step == "declared"
    assert sel.fallback is False
    assert sel.declared_marker == "true"


def test_delivers_for_another_root_is_not_this_roots_deliverable():
    children, ids = _pairs(_D26_CHILDREN)
    children[1]["body"] = "delivers: t_somewhere_else\n"
    sel = gate.select_deliverable(children, ids, root_id="t_26d8eb94")
    assert sel.step == "phase", "a foreign `delivers:` must not claim this root's gate"


def test_selector_title_hit_still_wins_when_there_is_no_declaration():
    children, ids = _pairs([
        ("c1", "research", "research"),
        ("c2", "Compile Class-B report", "implementation"),
        ("c3", "test", "testing"),
    ])
    sel = gate.select_deliverable(children, ids)
    assert (sel.child_id, sel.step, sel.fallback) == ("c2", "title", True)


def test_phase_fallback_is_LAST_wins_and_reported_as_a_fallback():
    children, ids = _pairs(_TIE_CHILDREN)
    sel = gate.select_deliverable(children, ids)
    assert sel.child_id == ids[-1], "LAST max-phase child wins (the docstring's contract)"
    assert sel.step == "phase"
    assert sel.fallback is True, "a heuristic choice must never be silent (W-9 census reads this)"


def test_last_child_fallback_and_empty_selection():
    children, ids = _pairs([("c1", "a", None), ("c2", "b", None)])
    sel = gate.select_deliverable(children, ids)
    assert (sel.child_id, sel.step, sel.fallback) == ("c2", "last-child", True)
    empty = gate.select_deliverable([], [])
    assert empty.child_id is None and empty.step == "none"


def test_backcompat_wrapper_returns_the_same_child():
    children, ids = _pairs(_TIE_CHILDREN)
    assert gate._select_deliverable_child(children, ids) == ids[-1]


# --------------------------------------------------------------------------- #
# 3 — AC1: the tie-break operator delta (measured, frozen fixture)
# --------------------------------------------------------------------------- #

def test_tie_break_operator_changes_the_recipient_on_the_measured_root():
    """AC1: `>=` vs `>` is a REAL change — on ``t_6ae16538`` the strict `>` picks
    the FIRST of three ``research`` siblings and ``>=`` picks the LAST. The
    change is asserted, not dropped."""
    children, ids = _pairs(_TIE_CHILDREN)
    legacy = _legacy_strict_gt(children, ids)
    current = gate.select_deliverable(children, ids).child_id
    assert legacy == "t_e1cea019"
    assert current == "t_21082df6"
    assert legacy != current, "the tie-break fix must visibly move the recipient here"


# --------------------------------------------------------------------------- #
# 4 — AC2/AC3: the incident roots (documented bias + declared cure)
# --------------------------------------------------------------------------- #

def test_ac2_bec82259_fallback_is_the_verification_card_and_the_marker_cures_it():
    """AC2: without a declaration the gate lands on ``t_f4ecdef7`` (a testing
    card) under BOTH operators — the bias is documented. With the marker it
    lands on the intended deliverable."""
    children, ids = _pairs(_BEC_CHILDREN)
    for operator_name, pick in (
        ("'>' (legacy)", _legacy_strict_gt(children, ids)),
        ("'>=' (current)", gate.select_deliverable(children, ids).child_id),
    ):
        assert pick == "t_f4ecdef7", f"{operator_name} must reproduce the bias"

    children[7]["body"] = "deliverable: true\n"  # t_29933582
    sel = gate.select_deliverable(children, ids, root_id="t_bec82259")
    assert sel.child_id == "t_29933582"
    assert sel.step == "declared" and not sel.fallback


def test_ac3_t26d8eb94_declared_marker_beats_the_phase_pick():
    children, ids = _pairs(_D26_CHILDREN)
    fallback = gate.select_deliverable(children, ids).child_id
    assert fallback == "t_21082df6"  # last of three research siblings
    children[1]["body"] = "delivers: t_26d8eb94\n"  # t_e1cea019
    sel = gate.select_deliverable(children, ids, root_id="t_26d8eb94")
    assert (sel.child_id, sel.step) == ("t_e1cea019", "declared")


# --------------------------------------------------------------------------- #
# 5 — the prompt contract (both card templates render it)
# --------------------------------------------------------------------------- #

def test_decomposer_prompt_carries_the_declared_deliverable_contract():
    prompt = decomp._SYSTEM_PROMPT
    assert "DECLARED DELIVERABLE" in prompt
    assert "deliverable: true" in prompt
    assert "delivers: <root task id>" in prompt
    assert '"deliverable": true' in prompt, "the structured reply key is documented"
    block = streams.deliverable_prompt_block()
    for literal in (streams.DELIVERABLE_KEY, streams.DELIVERS_KEY):
        assert literal in block


def test_undeclared_reply_is_not_touched_by_the_deliverable_pass():
    """Declaration-driven: a reply with no `deliverable` key adds no line."""
    assert streams.ensure_deliverable_line("phase: research\nsurvey\n") == "phase: research\nsurvey\n"
    assert decomp._declares_deliverable({}) is False
    assert decomp._declares_deliverable({"deliverable": False}) is False
    assert decomp._declares_deliverable({"deliverable": "maybe"}) is False
    assert decomp._declares_deliverable({"deliverable": True}) is True
    assert decomp._declares_deliverable({"delivers": "t_root"}) is True


# --------------------------------------------------------------------------- #
# 6 — hermetic end-to-end: the gate lands on the DECLARED child
# --------------------------------------------------------------------------- #

def _patch_list_profiles(names: list[str]):
    from types import SimpleNamespace
    fake = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name",
              return_value=names[0] if names else "default"),
    ]


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _decompose(kanban_home, *, root_body: str, llm_payload: dict, title: str = "gated quest"):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=title, body=root_body, triage=True)
    patches = _patch_list_profiles(["orchestrator", "engineer", "researcher"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_fake_aux_response(jsonlib.dumps(llm_payload))):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        children = [kb.get_task(conn, c) for c in (outcome.child_ids or [])]
    return tid, root, children, outcome


def _gate_recipients(conn, child_ids):
    """Gate recipients among ``child_ids`` (link-direction-agnostic).

    Decomposed children are linked root-as-CHILD (``kanban_db_graph.py``
    ``_link(conn, cid, task_id)``), so keying on ``task_links.parent_id`` would
    find nothing here; the authoritative fact is the ``gate_required`` event on
    the child.
    """
    if not child_ids:
        return []
    marks = ",".join("?" for _ in child_ids)
    rows = conn.execute(
        f"SELECT DISTINCT task_id FROM task_events WHERE kind = 'gate_required' "
        f"AND task_id IN ({marks})", tuple(child_ids),
    ).fetchall()
    return sorted(r["task_id"] for r in rows)


def test_end_to_end_gate_lands_on_the_declared_deliverable_not_the_phase_pick(kanban_home):
    """The whole point of W-10: a gated root whose reply marks one child as the
    deliverable routes the user gate THERE — not to the last ``testing`` card."""
    payload = {
        "fanout": True,
        "rationale": "synthesis + verification",
        "tasks": [
            {"title": "research the options", "body": "phase: research\nsurvey\n",
             "assignee": "researcher", "phase": "research", "parents": []},
            {"title": "Consolidate final synthesis report", "assignee": "orchestrator",
             "body": "phase: implementation\nwrite the report\n", "phase": "implementation",
             "parents": [0], "deliverable": True},
            {"title": "Verify the synthesis", "body": "phase: testing\nverify it\n",
             "assignee": "engineer", "phase": "testing", "parents": [1]},
        ],
    }
    tid, root, children, outcome = _decompose(
        kanban_home, root_body="Gate: requires user verdict on the final report.",
        llm_payload=payload,
    )
    declared = outcome.child_ids[1]
    with kbc.connect() as conn:
        recipients = _gate_recipients(conn, outcome.child_ids)
        # The declaration reached the PERSISTED body, not just the local dict.
        bodies = {c.id: (c.body or "") for c in children}
    assert recipients == [declared], f"gate went to {recipients}, expected [{declared}]"
    assert "deliverable: true" in bodies[declared]
    assert "deliverable: true" not in bodies[outcome.child_ids[2]], "no line is fabricated"


def test_end_to_end_undeclared_root_falls_back_to_last_wins_phase_order(kanban_home):
    """Legacy behaviour is preserved (and last-wins): with no declaration the
    gate goes to the LAST max-phase child, and the child bodies gain no
    deliverable line."""
    payload = {
        "fanout": True,
        "rationale": "two research + one testing",
        "tasks": [
            {"title": "research A", "body": "phase: research\nA\n", "assignee": "researcher",
             "phase": "research", "parents": []},
            {"title": "research B", "body": "phase: research\nB\n", "assignee": "researcher",
             "phase": "research", "parents": []},
            {"title": "check it", "body": "phase: testing\ncheck\n", "assignee": "engineer",
             "phase": "testing", "parents": [0, 1]},
        ],
    }
    tid, root, children, outcome = _decompose(
        kanban_home, root_body="Gate: requires user verdict.", llm_payload=payload,
    )
    with kbc.connect() as conn:
        recipients = _gate_recipients(conn, outcome.child_ids)
    assert recipients == [outcome.child_ids[-1]]
    for c in children:
        assert streams.declared_deliverable(c.body) is None


def test_end_to_end_foreign_delivers_does_not_steal_the_gate(kanban_home):
    """A `delivers: <other root>` line in a child body is ignored for this root."""
    payload = {
        "fanout": True,
        "rationale": "one research child with a foreign declaration",
        "tasks": [
            {"title": "research A", "body": "phase: research\ndelivers: t_deadbeef\n",
             "assignee": "researcher", "phase": "research", "parents": []},
            {"title": "check it", "body": "phase: testing\ncheck\n", "assignee": "engineer",
             "phase": "testing", "parents": [0]},
        ],
    }
    tid, root, children, outcome = _decompose(
        kanban_home, root_body="Gate: requires user verdict.", llm_payload=payload,
    )
    with kbc.connect() as conn:
        recipients = _gate_recipients(conn, outcome.child_ids)
    assert recipients == [outcome.child_ids[-1]], "phase fallback, not the foreign declaration"


def test_single_card_path_keeps_its_body_byte_identical_without_a_declaration(kanban_home):
    """fanout=false + no declaration: the single-card body is untouched."""
    payload = {
        "fanout": False,
        "rationale": "single unit",
        "title": "tightened title",
        "body": "phase: implementation\ndo the thing\nAdoption: none\n",
        "assignee": "engineer",
    }
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain", triage=True, body="Gate: none")
    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_fake_aux_response(jsonlib.dumps(payload))):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok
    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert streams.declared_deliverable(body) is None
