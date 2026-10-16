"""Tests for the T3 designer lens: the D1–D6 design-card contract.

Covers ``hermes_cli.kanban_design`` (contract, detection, rendering, parsing,
item-by-item validation) and its wiring into the specifier
(``kanban_specify.specify_task``), including the guarantee that an ordinary
card is byte-identical to the pre-T3 behaviour.

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

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_design as design
from hermes_cli import kanban_specify as spec


def _escaped_the_sandbox(resolved: Path, live_pin: str | None, temp_roots: list[Path]) -> str | None:
    """Return a failure message when ``resolved`` (a kanban DB path) escapes the
    test sandbox while a live-board pin was present; ``None`` when it is safe.

    Split out of the autouse fixture so the predicate itself is unit-tested —
    a guard nobody exercises is a guard that rots.
    """
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
def _design_db_off_the_live_board(monkeypatch, tmp_path):
    """Fail closed against the live-board pin.

    A dispatched worker shell exports ``HERMES_KANBAN_DB`` pointing at the REAL
    ``~/.hermes/kanban.db``, and ``kanban_db.kanban_db_path()`` lets that pin
    WIN over ``HERMES_HOME``. The per-test ``kanban_home`` fixture (and the
    suite-wide ``_hermetic_environment``) are therefore not sufficient on their
    own: an unhermetic probe run wrote design-card rows into the live board on
    2026-09-12. Drop the pin, then verify the resolved path is sandboxed.
    """
    live_pin = os.environ.get("HERMES_KANBAN_DB")
    for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                 "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_LOGS_ROOT",
                 "HERMES_KANBAN_ATTACHMENTS_ROOT"):
        monkeypatch.delenv(name, raising=False)

    kb_mod = sys.modules.get("hermes_cli.kanban_db")
    if kb_mod is not None:
        escaped = _escaped_the_sandbox(
            Path(kb_mod.kanban_db_path()),
            live_pin,
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


def _mock_call_llm(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return MagicMock(return_value=resp)


def _patch_aux(content: str):
    return patch("agent.auxiliary_client.call_llm", _mock_call_llm(content)), _mock_call_llm


# ---------------------------------------------------------------------------
# A complete, valid design payload — the reference instantiation
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "D1": [
        {"name": "phase", "definition": "advisory routing tag on a card"},
        {"name": "design card", "definition": "a card carrying the D1-D6 contract"},
    ],
    "D2": [
        {"invariant": "status authority stays with the dispatcher"},
        {"invariant": "a design card never blocks a fan-out"},
    ],
    "D3": [
        {"failure": "design card is a prose dump", "mitigation": "review lane checks D1-D6 item by item"},
        {"failure": "designer review conflation", "mitigation": "separate reviewer run via the review lane"},
    ],
    "D4": [
        {"surface": "hermes_cli/kanban_specify.py", "change": "design branch in the system prompt"},
        {"surface": "tasks.current_step_key", "change": "holds the phase tag"},
    ],
    "D5": {
        "options": [
            {"label": "A", "name": "template-only contract", "pros": "+cheap, +no new profile", "cons": "- same agent authors and reviews"},
            {"label": "B", "name": "dedicated designer profile", "pros": "+fully independent", "cons": "- costs a slot and ~220MB RAM"},
        ],
    },
    "recommendation": "Option A — zero new profile; role separation via two cards",
    "D6": [
        {"step": "add the contract module"},
        {"step": "wire the specifier design branch"},
        {"step": "land the review checklist in scout SOUL"},
    ],
}


def _design_reply(**overrides) -> str:
    payload = {
        "title": "Design the widget pipeline contract",
        "card_type": "design",
        "body": "**Goal**\nDecide how the widget pipeline is structured.",
        "design": VALID_PAYLOAD,
    }
    payload.update(overrides)
    return jsonlib.dumps(payload)


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------


def test_section_order_is_d1_through_d6():
    assert design.DESIGN_SECTION_ORDER == ("D1", "D2", "D3", "D4", "D5", "D6")
    assert sorted(design.DESIGN_SECTIONS) == list(design.DESIGN_SECTION_ORDER)


def test_every_section_declares_a_heading_purpose_and_requirement():
    for code in design.DESIGN_SECTION_ORDER:
        section = design.DESIGN_SECTIONS[code]
        assert section.code == code
        assert section.heading.startswith(f"{code}.")
        assert section.purpose and section.requirement


# ---------------------------------------------------------------------------
# Detection — precision matters most
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("title", [
    "Design the widget pipeline contract",
    "System design for the kanban decomposer",
    "Decide how to build the ingest service",
    "Architecture design for the sync engine",
    "Write a design document for the API",
])
def test_design_titles_are_detected(title):
    assert design.is_design_card(title=title) is True


@pytest.mark.parametrize("title", [
    # An engineering-PLANNING card, not a design card.
    "Design a migration plan",
    # A card ABOUT the design-card template — implementation, not design.
    "Implement D1-D6 fields in kanban_specify.py design-card template",
    "Update the design-card review checklist",
    "Implement current_step_key persistence in child insertion",
    "Review implementation and doctrine updates",
    "Add capability manifest lines to profiles",
    "Fix the cron job request_overrides bug",
])
def test_non_design_titles_are_not_detected(title):
    assert design.is_design_card(title=title) is False


def test_explicit_card_type_wins_over_title_heuristics():
    assert design.is_design_card(card_type="design", title="Fix a bug") is True
    assert design.is_design_card(card_type="task", title="Design the widget pipeline contract") is False


def test_phase_system_design_marks_the_card():
    assert design.is_design_card(phase="system-design") is True
    assert design.is_design_card(phase="implementation") is False
    assert design.is_design_card(body="phase: system-design\n\n**Goal**\nx") is True


def test_card_type_frontmatter_line_is_detected():
    assert design.body_card_type("card_type: design\n\n**Goal**\nx") == "design"
    assert design.body_card_type("**Goal**\nx") is None
    assert design.is_design_card(body="card_type: DESIGN\nx") is True


def test_two_or_more_existing_d_sections_count_as_a_design_card():
    body = "**D1. Named semantics**\n- `a` — b\n\n**D2. Invariants**\n- x\n"
    assert design.is_design_card(body=body) is True


# ---------------------------------------------------------------------------
# Rendering — all six sections, always, in order
# ---------------------------------------------------------------------------


def test_render_emits_all_six_headings_in_order():
    block = design.render_design_sections(VALID_PAYLOAD)
    positions = [block.index(design.DESIGN_SECTIONS[c].heading) for c in design.DESIGN_SECTION_ORDER]
    assert positions == sorted(positions)
    for code in design.DESIGN_SECTION_ORDER:
        assert design.DESIGN_SECTIONS[code].heading in block


def test_render_skeleton_carries_every_heading_with_placeholder():
    block = design.render_design_skeleton()
    for code in design.DESIGN_SECTION_ORDER:
        assert design.DESIGN_SECTIONS[code].heading in block
    assert block.count(f"- {design.DESIGN_PLACEHOLDER}") == len(design.DESIGN_SECTION_ORDER)


def test_render_accepts_heading_keys_and_short_keys():
    block = design.render_design_sections({
        "D1. Named semantics": "**D1**\n- only a string body\n",
        "invariants": [{"invariant": "x holds"}],
        "planning": ["do the thing"],
    })
    assert "only a string body" in block
    assert "x holds" in block
    assert "1. do the thing" in block


def test_render_d1_pairs_name_and_definition():
    block = design.render_design_sections({"D1": [{"name": "phase", "definition": "a routing tag"}]})
    assert "- `phase` — a routing tag" in block


def test_render_d3_keeps_mitigation_adjacent_to_its_risk():
    block = design.render_design_sections(
        {"D3": [{"failure": "it breaks", "mitigation": "it is guarded"}]}
    )
    assert "- it breaks — Mitigation: it is guarded" in block


def test_render_d5_labels_options_and_appends_recommendation():
    block = design.render_design_sections({"D5": VALID_PAYLOAD["D5"], "recommendation": "Option A — cheap"})
    assert "- Option A: template-only contract" in block
    assert "- Option B: dedicated designer profile" in block
    assert "RECOMMENDATION: Option A — cheap" in block


def test_render_d5_normalises_a_sign_the_model_already_wrote():
    """The renderer owns the +/- markers: a payload that writes its own sign
    must not render the doubled `+ +cheap` a reviewer has to re-read."""
    block = design.render_design_sections({"D5": {"options": [
        {"label": "A", "name": "x", "pros": "+cheap", "cons": "-costly"},
        {"label": "B", "name": "y", "pros": "quick", "cons": "fragile"},
    ]}, "recommendation": "Option A — cheap"})
    assert "+ cheap; - costly" in block
    assert "+ +cheap" not in block and "- -costly" not in block
    assert "+ quick; - fragile" in block
    assert design.validate_design_card(block, is_design=True).sections["D5"].ok is True


def test_render_d6_numbers_the_work_list():
    block = design.render_design_sections({"D6": [{"step": "one"}, {"step": "two"}]})
    assert "1. one" in block and "2. two" in block


def test_render_is_deterministic():
    assert design.render_design_sections(VALID_PAYLOAD) == design.render_design_sections(VALID_PAYLOAD)


# ---------------------------------------------------------------------------
# Parsing round-trip
# ---------------------------------------------------------------------------


def test_parse_recovers_render_output():
    parsed = design.parse_design_sections(design.render_design_sections(VALID_PAYLOAD))
    assert parsed["D1"] == ["`phase` — advisory routing tag on a card", "`design card` — a card carrying the D1-D6 contract"]
    assert parsed["D2"] == ["status authority stays with the dispatcher", "a design card never blocks a fan-out"]
    assert len(parsed["D5"]) == 3  # two options + the recommendation line
    assert parsed["D6"] == ["1. add the contract module", "2. wire the specifier design branch", "3. land the review checklist in scout SOUL"]


def test_parse_drops_placeholders():
    parsed = design.parse_design_sections(design.render_design_skeleton())
    assert all(items == [] for items in parsed.values())


def test_parse_ignores_prose_mentioning_d_numbers():
    body = "**Goal**\nImplement the D1-D6 contract (see D2 below).\n\n**D1. Named semantics**\n- `x` — y\n"
    parsed = design.parse_design_sections(body)
    assert parsed["D1"] == ["`x` — y"]
    assert parsed["D2"] == []


def test_parse_stops_a_section_at_the_next_non_d_heading():
    body = "**D1. Named semantics**\n- `x` — y\n\n**Goal**\nnot a D1 item\n"
    assert design.parse_design_sections(body)["D1"] == ["`x` — y"]


def test_merge_replaces_existing_sections_and_is_idempotent():
    body = "**Goal**\nkeep me\n\n**D1. Named semantics**\n- stale\n"
    once = design.merge_design_sections(body, design.render_design_sections(VALID_PAYLOAD))
    twice = design.merge_design_sections(once, design.render_design_sections(VALID_PAYLOAD))
    assert once == twice
    assert "keep me" in once
    assert "stale" not in once
    assert once.count("**D1. Named semantics**") == 1


def test_merge_on_empty_body_returns_the_block():
    block = design.render_design_sections({})
    assert design.merge_design_sections("", block) == block.strip()
    assert design.merge_design_sections(None, block) == block.strip()


# ---------------------------------------------------------------------------
# Validation — item by item, never raises
# ---------------------------------------------------------------------------


def test_valid_design_card_passes_every_section():
    body = design.merge_design_sections(
        "**Goal**\nx\n\n**Acceptance criteria**\n- y\n", design.render_design_sections(VALID_PAYLOAD),
    )
    result = design.validate_design_card(body, is_design=True)
    assert result.ok is True
    assert result.violations == []
    for code in design.DESIGN_SECTION_ORDER:
        assert result.sections[code].present is True
        assert result.sections[code].ok is True
        assert result.sections[code].items


def test_skeleton_fails_every_section():
    result = design.validate_design_card(design.render_design_skeleton(), is_design=True)
    assert result.ok is False
    assert len(result.violations) == len(design.DESIGN_SECTION_ORDER)
    assert not any(r.ok for r in result.sections.values())


@pytest.mark.parametrize("code,mutate,needle", [
    ("D1", {"D1": [{"name": "phase only, no definition"}]}, "lacks a definition"),
    ("D2", {"D2": []}, "no invariants"),
    ("D3", {"D3": [{"failure": "a risk with no guard"}]}, "no mitigation"),
    ("D4", {"D4": [{"surface": "something vague", "change": "changes"}]}, "no concrete file"),
    ("D5", {"D5": {"options": [{"label": "A", "name": "only one", "pros": "+p", "cons": "-c"}]}}, ">=2 genuine options"),
    ("D5", {"D5": {"options": [
        {"label": "A", "name": "a", "pros": "+p", "cons": "-c"},
        {"label": "B", "name": "b", "pros": "+p", "cons": "-c"},
    ]}, "recommendation": "go with the second one"}, "does not name one of the option labels"),
])
def test_item_level_violations_are_reported(code, mutate, needle):
    payload = {k: v for k, v in VALID_PAYLOAD.items()}
    payload.update(mutate)
    result = design.validate_design_card(design.render_design_sections(payload), is_design=True)
    issues = " ".join(result.sections[code].issues)
    assert needle in issues, (code, issues)
    assert result.sections[code].ok is False


@pytest.mark.parametrize("surface", [
    "hermes_cli/kanban_specify.py",
    "tasks.current_step_key",
    "config key widget_pipeline.stages",
    "envelope field idempotency_key",
    "env var WIDGET_PIPELINE_CONTRACT_VERSION",
    "column widget_events.last_completed_stage",
    "`~/knowledge/wiki/widget-pipeline-contract.md`",
    "`design` object key",
])
def test_d4_accepts_concrete_surfaces(surface):
    """D4 exists to force EXACT surfaces. A schema field the design introduces
    ("envelope field idempotency_key") is exact even without a dot-path — the
    live-LLM probe produced those and must not be marked non-concrete."""
    block = design.render_design_sections({"D4": [{"surface": surface, "change": "c"}]})
    report = design.validate_design_card(block, is_design=True).sections["D4"]
    assert report.ok is True, report.issues


@pytest.mark.parametrize("surface", ["something vague", "the pipeline", "several places", "the database"])
def test_d4_still_rejects_vague_surfaces(surface):
    block = design.render_design_sections({"D4": [{"surface": surface, "change": "c"}]})
    report = design.validate_design_card(block, is_design=True).sections["D4"]
    assert report.ok is False
    assert "names no concrete file/column/config key" in " ".join(report.issues)


def test_d6_renders_an_ordered_list_even_from_unordered_input():
    """D6 is the ORDERED work list: the template numbers whatever it is given."""
    block = design.render_design_sections({"D6": ["first thing", "second thing"]})
    assert "1. first thing" in block and "2. second thing" in block
    assert design.validate_design_card(block, is_design=True).sections["D6"].ok is True


def test_d6_rejects_a_hand_written_unordered_list():
    body = design.render_design_sections(VALID_PAYLOAD).replace(
        "1. add the contract module\n2. wire the specifier design branch\n3. land the review checklist in scout SOUL",
        "- add the contract module\n- wire the specifier design branch",
    )
    report = design.validate_design_card(body, is_design=True).sections["D6"]
    assert "not ordered" in " ".join(report.issues)
    assert report.present is True  # the items are there, the ORDER is what is missing


def test_validation_never_raises_on_hostile_input():
    for body in [None, "", 0, [], {"a": 1}, "**D1.", "**D5. x**\n" + "y" * 5000]:
        design.validate_design_card(body, is_design=True)


def test_review_metadata_is_item_by_item():
    body = design.merge_design_sections("**Goal**\nx\n", design.render_design_sections(VALID_PAYLOAD))
    meta = design.design_review_metadata(body, is_design=True)
    assert meta["design_contract"] == "D1-D6"
    assert meta["design_ok"] is True
    assert meta["is_design_card"] is True
    for code in design.DESIGN_SECTION_ORDER:
        entry = meta[code]
        assert entry["heading"].startswith(f"{code}.")
        assert entry["present"] is True and entry["ok"] is True
        assert entry["items"] and entry["issues"] == []


def test_review_metadata_flags_missing_sections_individually():
    meta = design.design_review_metadata("**Goal**\nx\n", is_design=True)
    assert meta["design_ok"] is False
    assert meta["D1"]["present"] is False
    assert any("D1" in v for v in meta["violations"])


# ---------------------------------------------------------------------------
# Structured-payload validation
# ---------------------------------------------------------------------------


def test_valid_payload_validates():
    result = design.validate_design_payload(VALID_PAYLOAD)
    assert result.ok is True
    assert result.missing == []
    assert result.item_counts["D5"] == 3  # 2 options + recommendation


def test_payload_validation_counts_each_section():
    result = design.validate_design_payload({"D1": [{"name": "a", "definition": "b"}]})
    assert result.ok is False
    assert result.missing == ["D2", "D3", "D4", "D5", "D6"]
    assert result.item_counts["D1"] == 1


def test_payload_validation_rejects_non_object():
    for bad in [None, "x", [], 3]:
        result = design.validate_design_payload(bad)
        assert result.ok is False
        assert result.missing == list(design.DESIGN_SECTION_ORDER)


def test_payload_validation_flags_single_option_and_missing_recommendation():
    result = design.validate_design_payload({
        "D5": {"options": [{"label": "A", "name": "only", "pros": "+p", "cons": "-c"}]},
    })
    assert "D5: fewer than 2 options" in result.violations
    assert "D5: no named recommendation in the payload" in result.violations


# ---------------------------------------------------------------------------
# Specifier wiring — design cards
# ---------------------------------------------------------------------------


def test_specify_renders_all_six_sections(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="design the widget pipeline", triage=True)

    patcher, _ = _patch_aux(_design_reply())
    with patcher:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    assert outcome.design_card is True
    assert outcome.design_metadata is not None

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    body = task.body or ""
    assert task.status == "ready"
    # The four standard sections survive, and D1-D6 land after them.
    assert "**Goal**" in body
    assert body.splitlines()[0] == design.DESIGN_CARD_TYPE_LINE
    positions = [body.index(design.DESIGN_SECTIONS[c].heading) for c in design.DESIGN_SECTION_ORDER]
    assert positions == sorted(positions)
    # And the rendered card validates clean.
    assert design.validate_design_card(body, is_design=True).ok is True


def test_specify_design_outcome_carries_payload_metadata(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="design the widget pipeline", triage=True)
    patcher, _ = _patch_aux(_design_reply())
    with patcher:
        outcome = spec.specify_task(tid, author="ace")
    assert outcome.design_metadata["design_payload_ok"] is True
    assert outcome.design_metadata["missing_sections"] == []
    assert outcome.design_metadata["item_counts"]["D2"] == 2


def test_specify_fills_missing_sections_with_placeholders(kanban_home):
    """A thin payload must not produce a half-empty card: every heading is
    rendered, and the metadata names exactly which sections were missing."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="design the widget pipeline", triage=True)
    patcher, _ = _patch_aux(_design_reply(design={"D1": [{"name": "x", "definition": "y"}]}))
    with patcher:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.design_card is True
    assert outcome.design_metadata["design_payload_ok"] is False
    assert outcome.design_metadata["missing_sections"] == ["D2", "D3", "D4", "D5", "D6"]

    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    for code in design.DESIGN_SECTION_ORDER:
        assert design.DESIGN_SECTIONS[code].heading in body


def test_specify_design_detected_by_card_type_without_title_signal(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="do the thing", triage=True)
    patcher, _ = _patch_aux(_design_reply(title="Do the thing"))
    with patcher:
        outcome = spec.specify_task(tid, author="ace")
    assert outcome.design_card is True
    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert "**D6. Planning readiness**" in body


def test_design_payload_alone_triggers_the_contract(kanban_home):
    """An LLM that emits `design` but forgets `card_type` still gets the card."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ordinary sounding title", triage=True)
    reply = jsonlib.dumps({"title": "T", "body": "**Goal**\nx", "design": VALID_PAYLOAD})
    patcher, _ = _patch_aux(reply)
    with patcher:
        outcome = spec.specify_task(tid, author="ace")
    assert outcome.design_card is True


# ---------------------------------------------------------------------------
# Specifier wiring — existing flows are untouched
# ---------------------------------------------------------------------------


def test_ordinary_card_is_byte_identical_to_pre_t3_behaviour(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    body = "**Goal**\nA concrete goal.\n\n**Approach**\n- do it"
    patcher, _ = _patch_aux(jsonlib.dumps({"title": "Refined rough", "body": body}))
    with patcher:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    assert outcome.design_card is False
    assert outcome.design_metadata is None
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body == body                      # not a byte added
    assert design.DESIGN_CARD_TYPE_LINE not in (task.body or "")
    assert task.status == "ready"
    assert task.title == "Refined rough"


def test_non_json_reply_still_becomes_the_body(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    patcher, _ = _patch_aux("just prose, no json at all")
    with patcher:
        outcome = spec.specify_task(tid, author="ace")
    assert outcome.ok is True
    assert outcome.design_card is False
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == "just prose, no json at all"


def test_card_type_task_suppresses_the_contract(kanban_home):
    """Explicit `card_type: task` must win over a design-shaped title."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="design the widget pipeline", triage=True)
    body = "**Goal**\nnot a design card\n"
    patcher, _ = _patch_aux(jsonlib.dumps({"title": "T", "body": body, "card_type": "task"}))
    with patcher:
        outcome = spec.specify_task(tid, author="ace")
    assert outcome.design_card is False
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == body


def test_design_prompt_block_renders_from_the_contract_module():
    """The prompt must be generated from kanban_design, not hand-duplicated."""
    assert spec._DESIGN_PROMPT_BLOCK == spec._design_contract_block()
    for code in design.DESIGN_SECTION_ORDER:
        assert design.DESIGN_SECTIONS[code].heading in spec._SYSTEM_PROMPT
    assert "DESIGN CARDS" in spec._SYSTEM_PROMPT


def test_specify_outcome_defaults_keep_design_fields_absent():
    outcome = spec.SpecifyOutcome("t_x", False, "not in triage")
    assert outcome.design_card is False
    assert outcome.design_metadata is None


# ---------------------------------------------------------------------------
# Hermeticity guard — the fixture that keeps these tests off the LIVE board
# ---------------------------------------------------------------------------


def test_guard_flags_a_live_pin_that_escapes_the_sandbox(tmp_path):
    """The exact 2026-09-12 leak: a worker's HERMES_KANBAN_DB pin (live board)
    resolving outside the temp sandbox must be reported, not silently written.

    The escaping path is derived from ``tmp_path`` rather than spelled as the
    real ``~/.hermes`` path: resolving a literal real-home path IS I/O against
    production state, which the conftest home I/O guard refuses (reads as much
    as writes), so the literal form tripped that guard before the predicate
    could report the escape. The predicate branch exercised is the same.
    """
    live = tmp_path.parent / "live-board" / "kanban.db"
    msg = _escaped_the_sandbox(live, str(live), [tmp_path])
    assert msg and "escaped the test sandbox" in msg
    assert str(live.resolve()) in msg


def test_guard_allows_a_sandboxed_path(tmp_path):
    assert _escaped_the_sandbox(tmp_path / "kanban.db", "/home/juozas/.hermes/kanban.db", [tmp_path]) is None


def test_guard_is_inert_without_a_live_pin(tmp_path):
    """A plain dev-shell run (no worker pin) must never trip the guard."""
    assert _escaped_the_sandbox(Path("/home/juozas/.hermes/kanban.db"), None, [tmp_path]) is None


# ---------------------------------------------------------------------------
# Specifier wiring — T3 composes with the T1 phase cascade (the rebase seam)
# ---------------------------------------------------------------------------


def _phase_line_of(body: str) -> str:
    for line in body.splitlines():
        if line.strip().startswith("phase:"):
            return line.strip()
    return ""


def test_design_card_keeps_its_phase_tag(kanban_home):
    """A design card is normally tagged `system-design`; the design fold must not
    eat the phase (it prepends `card_type:` to the body front matter)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="design the widget pipeline", triage=True)
    reply = jsonlib.dumps({
        "title": "Design the widget pipeline",
        "phase": "system-design",
        "body": "phase: system-design\n**Goal**\nDecide the pipeline shape.",
        "card_type": "design",
        "design": VALID_PAYLOAD,
    })
    patcher, _ = _patch_aux(reply)
    with patcher:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    assert outcome.design_card is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.current_step_key == "system-design"
    # The marker line leads, the phase tag survives right after it.
    assert task.body.splitlines()[0] == design.DESIGN_CARD_TYPE_LINE
    assert _phase_line_of(task.body) == "phase: system-design"
    assert design.body_phase(task.body) == "system-design"


def test_design_detected_from_the_phase_alone(kanban_home):
    """A `system-design` phase is itself the T3 signal, even with no card_type
    and an ordinary-looking title (kanban_design.DESIGN_PHASE)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="do the thing", triage=True)
    reply = jsonlib.dumps({
        "title": "Do the thing",
        "phase": "system-design",
        "body": "phase: system-design\n**Goal**\nA goal.",
        "design": VALID_PAYLOAD,
    })
    patcher, _ = _patch_aux(reply)
    with patcher:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.design_card is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.current_step_key == "system-design"
    assert design.body_phase(task.body) == "system-design"
    for code in design.DESIGN_SECTION_ORDER:
        assert design.DESIGN_SECTIONS[code].heading in task.body


def test_ordinary_card_still_carries_its_phase(kanban_home):
    """Non-design replies are byte-identical for the body AND still persist the
    phase — the T3 fold is a no-op outside design cards."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    body = "phase: implementation\n**Goal**\nA concrete goal."
    patcher, _ = _patch_aux(jsonlib.dumps({"title": "R", "body": body, "phase": "implementation"}))
    with patcher:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.design_card is False
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body == body
    assert task.current_step_key == "implementation"


def test_design_card_does_not_duplicate_a_written_phase_line(kanban_home):
    """The model writing its own `phase:` inside the body must yield exactly one
    such line after the design fold — no duplication from a re-render."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="design the widget pipeline", triage=True)
    reply = jsonlib.dumps({
        "title": "Design the widget pipeline",
        "phase": "system-design",
        "body": "phase: system-design\n**Goal**\nDecide the shape.",
        "card_type": "design",
        "design": VALID_PAYLOAD,
    })
    patcher, _ = _patch_aux(reply)
    with patcher:
        spec.specify_task(tid, author="ace")

    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    phase_lines = [l for l in body.splitlines() if l.strip().startswith("phase:")]
    assert phase_lines == ["phase: system-design"], phase_lines
