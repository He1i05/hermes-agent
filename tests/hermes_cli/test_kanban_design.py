"""Contract tests for ``hermes_cli.kanban_design`` on the deployed trunk.

The module is the review lane's D1–D6 checker (``validate_design_card``); it is
shipped additively so a reviewer imports the contract instead of re-deriving
D1–D6 from memory. Provenance: byte-identical copy of the T3 revision
(sha256 ``cf1c15adc281577b6860617bf0abea84bce2cd2290671c7c819ef015dc9bb5f2``)
from ``t_b1a3333f`` (commit 4be8b882c7). The specifier-wiring tests of that
worktree file are intentionally NOT carried over: this trunk has no T3 specifier
or decomposer fold, and the module is dependency-light on purpose.

The last test is trunk-native: it is the end-to-end proof for the worker-authored
design card — a worker puts a D1–D6 body on its own card with ``kanban_edit``,
requests review, and the review lane's checker passes the body it will read.

No network, no real provider, no live board.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_design as design


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



# ---------------------------------------------------------------------------
# End-to-end: a worker-authored design card carries its D1–D6 body to review
# ---------------------------------------------------------------------------


@pytest.fixture
def authored_design_card(tmp_path, monkeypatch):
    """A running card claimed by a dispatcher worker, on a scratch board."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "designer")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="Design the widget pipeline contract",
                             assignee="designer")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid, run_id


def test_worker_authors_design_body_that_the_review_lane_can_validate(authored_design_card):
    """The gap this closes: a worker could not put D1–D6 on its own card.

    ``hermes kanban edit`` is fenced for every spawned worker shell, so before
    the tool existed the design landed as an attachment/comment and the gate on
    ``tasks.body`` could never pass. Here the body is authored through the tool,
    the card is handed to review, and the checker the review lane runs
    (``validate_design_card``) passes on the body the reviewer will read.
    """
    from tools import kanban_tools as kt

    tid, _run_id = authored_design_card
    body = design.merge_design_sections(
        "**Goal**\nDecide how the widget pipeline is structured.\n\n"
        "**Acceptance criteria**\n- the reviewer can validate this body\n",
        design.render_design_sections(VALID_PAYLOAD),
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

    authored = json.loads(kt._handle_edit({"body": body}))
    assert authored["ok"] is True, authored
    assert authored["fields"] == ["body"]
    assert authored["body_bytes"] == len(body.encode("utf-8"))
    assert authored["body_sha256"] == digest

    reviewed = json.loads(kt._handle_request_review({"summary": "design body authored in-card"}))
    assert reviewed.get("ok") is True, reviewed
    assert reviewed["status"] == "review"

    conn = kbc.connect()
    try:
        task = kb.get_task(conn, tid)
        edited = [e for e in kb.list_events(conn, tid) if e.kind == "edited"]
    finally:
        conn.close()

    assert task.status == "review"
    assert task.body == body, "the reviewer must read the authored bytes, not a re-render"
    assert edited and edited[-1].payload["fields"] == ["body"]
    assert edited[-1].payload["author"] == "designer"

    result = design.validate_design_card(task.body, is_design=True)
    assert result.ok is True, result.violations
    assert result.violations == []
    for code in design.DESIGN_SECTION_ORDER:
        assert result.sections[code].ok is True
    metadata = result.review_metadata()
    assert metadata["design_ok"] is True
    assert metadata["design_contract"] == "D1-D6"
