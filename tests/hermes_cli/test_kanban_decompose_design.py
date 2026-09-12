"""Tests for the T3 D1–D6 design contract on the DECOMPOSER path (t_c5f32661).

``hermes_cli/kanban_decompose`` assigns the ``system-design`` phase itself, so
it — not only the triage specifier — can produce a design card. Before this
card it produced design cards with ZERO D-sections, which is the exact F3.1
failure mode ("design card = prose dump, no invariants") T3 exists to prevent
and guaranteed rework under BA-P3.1.

Covered here:
  * ``kanban_design.ensure_design_contract`` — the idempotent, lossless fold
    the decomposer uses (as opposed to the specifier's payload-only path);
  * the decomposer prompt renders the contract from the shared module;
  * ``_clean_children`` folds the contract onto a design child and leaves every
    ordinary child byte-identical;
  * the fold ORDER is load-bearing (M2 classification, then buckets, then D1–D6);
  * the decomposer classifies on EXPLICIT signals only (no title heuristic).

The auxiliary LLM is mocked — no network, no real provider.
"""

from __future__ import annotations

import json as jsonlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_capability as capability
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_design as design
from hermes_cli import kanban_m2 as m2
from hermes_cli import kanban_specify as spec

DESIGN_CARD_TYPE = "system-design"

# A complete payload, mirroring the T3 reference instantiation.
VALID_PAYLOAD = {
    "D1": [{"name": "phase", "definition": "advisory routing tag on a card"},
           {"name": "design card", "definition": "a card carrying the D1-D6 contract"}],
    "D2": [{"invariant": "status authority stays with the dispatcher"},
           {"invariant": "a design card never blocks a fan-out"}],
    "D3": [{"failure": "design card is a prose dump", "mitigation": "review lane checks D1-D6 item by item"},
           {"failure": "designer review conflation", "mitigation": "separate reviewer run via the review lane"}],
    "D4": [{"surface": "hermes_cli/kanban_specify.py", "change": "design branch in the system prompt"},
           {"surface": "tasks.current_step_key", "change": "holds the phase tag"}],
    "D5": {"options": [
        {"label": "A", "name": "template-only contract", "pros": "+cheap, +no new profile",
         "cons": "- same agent authors and reviews"},
        {"label": "B", "name": "dedicated designer profile", "pros": "+fully independent",
         "cons": "- costs a slot and ~220MB RAM"},
    ]},
    "recommendation": "Option A - zero new profile; role separation via two cards",
    "D6": [{"step": "add the contract module"},
           {"step": "wire the specifier design branch"},
           {"step": "land the review checklist in scout SOUL"}],
}

HEADINGS = [design.DESIGN_SECTIONS[c].heading for c in design.DESIGN_SECTION_ORDER]


# ---------------------------------------------------------------------------
# Hermeticity: the worker shell pins HERMES_KANBAN_DB at the LIVE board and
# kanban_db.kanban_db_path() lets that pin WIN over HERMES_HOME. Drop the pin,
# then verify the resolved path is sandboxed — a dispatched worker would
# otherwise write real design-card rows into ~/.hermes/kanban.db.
# ---------------------------------------------------------------------------

def _escaped_the_sandbox(resolved: Path, live_pin: str | None, temp_roots: list[Path]) -> str | None:
    if not live_pin:
        return None
    resolved = Path(resolved).resolve()
    if any(str(resolved).startswith(str(Path(r).resolve())) for r in temp_roots):
        return None
    return (
        f"kanban DB escaped the test sandbox: {resolved} "
        f"(worker pin was {live_pin}) — refuse to write to the live board"
    )


@pytest.fixture(autouse=True)
def _design_decompose_db_off_the_live_board(monkeypatch, tmp_path):
    live_pin = os.environ.get("HERMES_KANBAN_DB")
    for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                 "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_LOGS_ROOT",
                 "HERMES_KANBAN_ATTACHMENTS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    kb_mod = sys.modules.get("hermes_cli.kanban_db")
    if kb_mod is not None:
        escaped = _escaped_the_sandbox(
            Path(kb_mod.kanban_db_path()), live_pin,
            [Path(tempfile.gettempdir()), tmp_path],
        )
        if escaped:
            pytest.fail(escaped)
    yield


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _patch_aux_client(content: str):
    return patch("agent.auxiliary_client.call_llm", return_value=_fake_aux_response(content))


def _patch_list_profiles(names: list[str]):
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


class _Routing:
    """Minimal routing stand-in for the ``_clean_children`` unit tests."""

    default_assignee = "fallback"
    valid_names = {"builder", "scout", "fallback", "wiki-keeper"}


def _clean(raw_tasks: list, *, routing=None):
    """Run ``_clean_children`` with the network-adjacent seams stubbed out."""
    with patch.object(decomp.capability, "_build_capability_table", return_value={}), \
         patch.object(decomp.wiki_landing, "emit_wiki_landing_children", return_value=[]):
        return decomp._clean_children("t_root", raw_tasks, routing or _Routing)


def _design_child(**overrides) -> dict:
    child = {
        "title": "Design the widget pipeline contract",
        "body": "phase: system-design\n**Goal**\nDecide how the widget pipeline is structured.",
        "assignee": "builder",
        "phase": DESIGN_CARD_TYPE,
        "design": VALID_PAYLOAD,
        "parents": [],
    }
    child.update(overrides)
    return child


# ---------------------------------------------------------------------------
# ensure_design_contract — the fold the decomposer uses
# ---------------------------------------------------------------------------

def test_ensure_contract_renders_all_six_sections_in_order():
    body = design.ensure_design_contract("phase: system-design\n**Goal**\nDecide.", VALID_PAYLOAD,
                                         phase=DESIGN_CARD_TYPE)
    positions = [body.index(h) for h in HEADINGS]
    assert positions == sorted(positions)


def test_ensure_contract_leads_with_the_design_marker():
    body = design.ensure_design_contract("**Goal**\nx", VALID_PAYLOAD, phase=DESIGN_CARD_TYPE)
    assert body.splitlines()[0] == design.DESIGN_CARD_TYPE_LINE


def test_ensure_contract_writes_the_phase_line_once():
    body = design.ensure_design_contract(
        "phase: system-design\n**Goal**\nx", VALID_PAYLOAD, phase=DESIGN_CARD_TYPE)
    assert [l for l in body.splitlines() if l.strip().startswith("phase:")] == ["phase: system-design"]


def test_ensure_contract_does_not_duplicate_an_existing_phase_line():
    body = design.ensure_design_contract(
        "phase: system-design\n**Goal**\nx", VALID_PAYLOAD, phase="implementation")
    assert design.body_phase(body) == "system-design", "the body's own phase wins"


def test_ensure_contract_is_idempotent():
    once = design.ensure_design_contract(
        "phase: system-design\n**Goal**\nx\n\nAdoption: none", VALID_PAYLOAD, phase=DESIGN_CARD_TYPE)
    assert design.ensure_design_contract(once, VALID_PAYLOAD, phase=DESIGN_CARD_TYPE) == once


def test_ensure_contract_is_lossless_on_a_section_the_body_already_has():
    """An existing section keeps its items verbatim — the fold must never
    discard content the author already wrote."""
    existing = (
        "phase: system-design\n**Goal**\nG.\n\n"
        "**D2. Invariants**\n- status authority stays with the dispatcher\n"
    )
    body = design.ensure_design_contract(existing, {}, phase=DESIGN_CARD_TYPE)
    assert "- status authority stays with the dispatcher" in body
    assert "**D2. Invariants**" in body


def test_ensure_contract_placeholders_a_missing_section_so_the_gap_is_reportable():
    existing = "phase: system-design\n**Goal**\nG.\n\n**D2. Invariants**\n- an invariant\n"
    body = design.ensure_design_contract(existing, {}, phase=DESIGN_CARD_TYPE)
    assert body.count("- none") == 5
    # Every heading is present, and the validator names the five gaps by section.
    assert all(h in body for h in HEADINGS)
    verdict = design.validate_design_card(body, is_design=True)
    assert verdict.ok is False
    assert design.DESIGN_SECTIONS["D1"].heading in "\n".join(verdict.violations) or \
        [v for v in verdict.violations if v.startswith("D1:")]


def test_ensure_contract_renders_from_an_empty_payload_without_raising():
    body = design.ensure_design_contract("phase: system-design\n**Goal**\nG.", None, phase=DESIGN_CARD_TYPE)
    assert all(h in body for h in HEADINGS)
    assert design.validate_design_card(body, is_design=True).ok is False


def test_ensure_contract_keeps_the_ordered_work_list_numbered():
    parsed = design.parse_design_sections(
        design.ensure_design_contract("phase: system-design\nx", VALID_PAYLOAD, phase=DESIGN_CARD_TYPE))
    assert parsed["D6"] == [
        "1. add the contract module",
        "2. wire the specifier design branch",
        "3. land the review checklist in scout SOUL",
    ]


def test_ensure_contract_keeps_the_recommendation_out_of_the_option_list():
    """A re-fold must not fold the RECOMMENDATION line into the last option."""
    once = design.ensure_design_contract("phase: system-design\nx", VALID_PAYLOAD, phase=DESIGN_CARD_TYPE)
    parsed = design.parse_design_sections(once)
    assert len(parsed["D5"]) == 3, parsed["D5"]
    assert parsed["D5"][-1].startswith("RECOMMENDATION:")
    assert "RECOMMENDATION" not in parsed["D5"][-2]


def test_ensure_contract_output_validates_clean_on_a_complete_payload():
    body = design.ensure_design_contract(
        "phase: system-design\n**Goal**\nDecide the shape.", VALID_PAYLOAD, phase=DESIGN_CARD_TYPE)
    verdict = design.validate_design_card(body, is_design=True)
    assert verdict.ok is True, verdict.violations


def test_ordinary_body_gains_only_headings_when_no_payload_is_supplied():
    """No payload + no D-sections => placeholders, never invented content."""
    body = design.ensure_design_contract("**Goal**\nAdd a --dry-run flag.", None, phase=None)
    assert all(h in body for h in HEADINGS)
    assert "Add a --dry-run flag." in body
    assert design.DESIGN_CARD_TYPE_LINE in body.splitlines()[0]


@pytest.mark.parametrize("payload", [{}, None, "not an object", 42, [1, 2]])
def test_ensure_contract_never_raises_on_hostile_payload(payload):
    body = design.ensure_design_contract("phase: system-design\n**Goal**\nx", payload, phase=DESIGN_CARD_TYPE)
    assert isinstance(body, str)
    assert all(h in body for h in HEADINGS)


# ---------------------------------------------------------------------------
# Explicit-signal-only classification (the decomposer path)
# ---------------------------------------------------------------------------

def test_decomposer_classifies_on_the_phase_it_assigned():
    assert design.is_design_card(phase=DESIGN_CARD_TYPE, title_heuristic=False) is True


def test_decomposer_classifies_on_an_emitted_card_type():
    assert design.is_design_card(card_type="design", title_heuristic=False) is True


def test_decomposer_classifies_on_two_existing_d_sections():
    body = design.ensure_design_contract("phase: system-design\nx", VALID_PAYLOAD, phase=DESIGN_CARD_TYPE)
    assert design.is_design_card(body=body, title_heuristic=False) is True


def test_decomposer_does_not_classify_on_a_title_that_merely_mentions_design():
    """The decomposer assigns the phase itself, so a title alone must never pull
    an ordinary child into the contract (this card's own title is the real
    example: "Wire the ... system-design ... contract into the DECOMPOSER path")."""
    assert design.is_design_card(title="Fix the system-design phase cascade", title_heuristic=False) is False
    assert design.is_design_card(title_heuristic=False) is False


@pytest.mark.parametrize("title", [
    # The live false positive this card found: a card ABOUT the phase name.
    "Wire the T3 D1–D6 contract into the DECOMPOSER path (system-design children get no design contract)",
    "Fix the system-design phase cascade",
    "Add current_step_key to system-design cards",
    "Update the system-design step key handling",
])
def test_a_title_merely_naming_the_phase_is_not_a_design_card(title):
    """The bare `system-design` phrase names the PHASE, not a design. Without
    the lookahead the phase vocabulary leaks into the heuristic — the same
    false-positive class as "implement the design-card template"."""
    assert design.is_design_card(title=title) is False


@pytest.mark.parametrize("title", [
    "System design for the kanban decomposer",
    "System-design for the ingest service",
])
def test_the_phase_phrase_still_counts_when_it_is_the_design_subject(title):
    assert design.is_design_card(title=title) is True


def test_the_specifier_still_uses_the_title_heuristic():
    """Dropping the heuristic is decomposer-only — the specifier keeps it."""
    assert design.is_design_card(title="Design the widget pipeline") is True


def test_decomposer_respects_an_explicit_card_type_task():
    assert design.is_design_card(card_type="task", phase=DESIGN_CARD_TYPE, title_heuristic=False) is False


# ---------------------------------------------------------------------------
# Decomposer prompt wiring
# ---------------------------------------------------------------------------

def test_decomposer_prompt_renders_the_contract_from_the_shared_module():
    assert all(h in decomp._SYSTEM_PROMPT for h in HEADINGS)
    assert "DESIGN CARDS" in decomp._SYSTEM_PROMPT


def test_both_templates_render_the_same_prompt_block():
    """One definition of the T3 text — the specifier and the decomposer append
    the same string (no hand-duplicated copy)."""
    assert spec._design_contract_block() == design.design_prompt_block()
    assert decomp._SYSTEM_PROMPT.endswith(design.design_prompt_block())


def test_decomposer_prompt_documents_design_children_as_optional():
    assert '"card_type"' in decomp._SYSTEM_PROMPT
    assert '"design"' in decomp._SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _clean_children: the design fold and the byte-identical ordinary child
# ---------------------------------------------------------------------------

def test_clean_children_folds_the_contract_onto_a_design_child():
    children, reason = _clean([_design_child()])
    assert reason == ""
    body = children[0]["body"]
    assert children[0]["design_card"] is True
    assert [body.index(h) for h in HEADINGS] == sorted(body.index(h) for h in HEADINGS)
    assert body.splitlines()[0] == design.DESIGN_CARD_TYPE_LINE
    assert design.body_phase(body) == DESIGN_CARD_TYPE
    assert design.validate_design_card(body, is_design=True).ok is True


def test_clean_children_reports_the_payload_verdict():
    children, _ = _clean([_design_child()])
    meta = children[0]["design_metadata"]
    assert meta["design_payload_ok"] is True
    assert meta["missing_sections"] == []


def test_clean_children_flags_a_thin_payload_but_still_renders_every_heading():
    children, _ = _clean([_design_child(design={"D1": [{"name": "x", "definition": "y"}]})])
    body = children[0]["body"]
    assert all(h in body for h in HEADINGS)
    assert children[0]["design_metadata"]["design_payload_ok"] is False
    assert children[0]["design_metadata"]["missing_sections"] == ["D2", "D3", "D4", "D5", "D6"]


def test_clean_children_classifies_on_the_phase_alone_with_no_payload():
    """The live failure: the decomposer assigns phase=system-design and emits
    prose with no structured payload. The card must still carry all six."""
    children, _ = _clean([
        {"title": "spec_guard_and_watchdog_architecture",
         "body": "phase: system-design\nDesign the architecture for two components:\n1. write guard\n2. watchdog",
         "assignee": "builder", "phase": DESIGN_CARD_TYPE, "parents": []},
    ])
    body = children[0]["body"]
    assert children[0]["design_card"] is True
    assert all(h in body for h in HEADINGS)
    # The prose survives; the gaps are reported as placeholders, not silently absent.
    assert "Design the architecture for two components" in body
    assert design.validate_design_card(body, is_design=True).ok is False


def test_clean_children_detects_a_design_body_without_a_phase_field():
    """A child that wrote the phase only as body frontmatter still gets it."""
    children, _ = _clean([_design_child(phase=None)])
    assert children[0].get("design_card") is True
    assert design.DESIGN_SECTIONS["D6"].heading in children[0]["body"]


def test_clean_children_leaves_an_ordinary_child_byte_identical():
    plain = "**Goal**\nAdd a flag.\n\n**Approach**\ncode it"
    children, _ = _clean([
        {"title": "Add a dry-run flag", "body": plain, "assignee": "builder", "parents": []},
    ])
    assert "design_card" not in children[0]
    assert "design_metadata" not in children[0]
    assert "**D1." not in children[0]["body"]
    assert design.DESIGN_CARD_TYPE_LINE not in children[0]["body"]
    # Byte-identical to the pre-T3 output: the bucket headings and nothing else.
    expected = [{"body": plain, "title": "Add a dry-run flag"}]
    capability.append_bucket_headings(expected)
    assert children[0]["body"] == expected[0]["body"]


def test_clean_children_leaves_a_design_MENTIONING_ordinary_child_untouched():
    """A card whose title says "system-design" but that has no explicit signal
    is not a design card — the title heuristic is off on this path."""
    plain = "**Goal**\nFix the phase cascade."
    children, _ = _clean([
        {"title": "Fix the system-design phase cascade", "body": plain,
         "assignee": "builder", "phase": "implementation", "parents": []},
    ])
    assert "design_card" not in children[0]
    assert "**D1." not in children[0]["body"]


def test_clean_children_mixes_design_and_plain_children_independently():
    children, reason = _clean([
        _design_child(),
        {"title": "Add a dry-run flag", "body": "code it", "assignee": "builder", "parents": []},
    ])
    assert reason == ""
    assert children[0].get("design_card") is True
    assert "design_card" not in children[1]
    assert "**D1." in children[0]["body"] and "**D1." not in children[1]["body"]


def test_bucket_headings_land_above_d1_not_inside_d6():
    """The bucket appender is routing boilerplate; if the design fold ran first,
    its trailing "Adoption: none" would be absorbed into D6's last item."""
    children, _ = _clean([_design_child(body="phase: system-design\n**Goal**\nx")])
    body = children[0]["body"]
    assert body.index("Adoption: none") < body.index(design.DESIGN_SECTIONS["D1"].heading)
    assert design.parse_design_sections(body)["D6"][-1] == "3. land the review checklist in scout SOUL"


def test_a_design_child_never_gets_a_spurious_m2_block():
    """The design fold's own vocabulary ("scout", "evaluate", "compare") makes
    the M2 research classifier fire — so the design fold MUST run after it."""
    folded = design.ensure_design_contract(
        "phase: system-design\n**Goal**\nG.", VALID_PAYLOAD, phase=DESIGN_CARD_TYPE)
    assert m2.is_research_card("Design the widget pipeline contract", folded) is True, \
        "precondition: the folded body would classify research if M2 ran last"
    children, _ = _clean([_design_child()])
    assert "web_searches:" not in children[0]["body"]


# ---------------------------------------------------------------------------
# End-to-end through decompose_task (aux LLM mocked)
# ---------------------------------------------------------------------------

def test_decompose_fanout_gives_a_system_design_child_the_full_contract(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="harden the board", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "design then build",
        "tasks": [
            {"title": "spec_guard_and_watchdog_architecture",
             "body": "phase: system-design\nDesign the write guard and the watchdog.",
             "assignee": "builder", "phase": DESIGN_CARD_TYPE,
             "card_type": "design", "design": VALID_PAYLOAD, "parents": []},
            {"title": "implement the write guard",
             "body": "phase: implementation\nCode it.", "assignee": "builder",
             "phase": "implementation", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "builder"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert len(outcome.child_ids) == 2
    with kbc.connect() as conn:
        design_child = kb.get_task(conn, outcome.child_ids[0])
        plain_child = kb.get_task(conn, outcome.child_ids[1])

    assert design_child.current_step_key == DESIGN_CARD_TYPE
    body = design_child.body or ""
    assert [body.index(h) for h in HEADINGS] == sorted(body.index(h) for h in HEADINGS)
    assert design.validate_design_card(body, is_design=True).ok is True
    assert design.body_card_type(body) == "design"

    # The ordinary implementation child is unchanged by T3.
    assert "**D1." not in (plain_child.body or "")
    assert design.DESIGN_CARD_TYPE_LINE not in (plain_child.body or "")


def test_decompose_single_path_applies_the_contract(kanban_home):
    """fanout=false can also classify `system-design` — same contract."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="harden the board", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "one design unit",
        "title": "Design the write guard",
        "body": "phase: system-design\n**Goal**\nDecide the guard shape.",
        "phase": DESIGN_CARD_TYPE,
        "card_type": "design",
        "design": VALID_PAYLOAD,
    })

    patches = _patch_list_profiles(["orchestrator", "builder"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.design_card is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    body = task.body or ""
    assert task.current_step_key == DESIGN_CARD_TYPE
    assert all(h in body for h in HEADINGS)
    assert design.validate_design_card(body, is_design=True).ok is True


def test_decompose_single_path_leaves_an_ordinary_card_untouched(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="add a flag", triage=True)

    body = "**Goal**\nAdd a --dry-run flag."
    llm_payload = jsonlib.dumps({
        "fanout": False, "rationale": "one unit", "title": "Add a dry-run flag",
        "body": body, "phase": "implementation",
    })

    patches = _patch_list_profiles(["orchestrator", "builder"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.design_card is False
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body == body
    assert "**D1." not in (task.body or "")


# ---------------------------------------------------------------------------
# Acceptance: no live or decomposed card carries the phase without the contract
# ---------------------------------------------------------------------------

def test_no_decomposed_design_card_lacks_the_contract_on_a_scratch_board(kanban_home):
    """The board-wide acceptance criterion, asserted on a hermetic board: every
    card carrying `phase: system-design` carries all six D-sections."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True, "rationale": "cascade",
        "tasks": [
            {"title": "design it", "body": "phase: system-design\nDecide.",
             "assignee": "builder", "phase": DESIGN_CARD_TYPE, "parents": []},
            {"title": "build it", "body": "phase: implementation\nCode.",
             "assignee": "builder", "phase": "implementation", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "builder"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        rows = conn.execute(
            "SELECT id, body, current_step_key FROM tasks"
        ).fetchall()
    offenders = []
    for row in rows:
        text = row["body"] or ""
        if design.body_phase(text) != DESIGN_CARD_TYPE and row["current_step_key"] != DESIGN_CARD_TYPE:
            continue
        missing = [h for h in HEADINGS if h not in text]
        if missing:
            offenders.append((row["id"], missing))
    assert offenders == [], f"design cards without the contract: {offenders}"
