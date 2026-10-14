"""Tests for the T4 M2 research-card acceptance block (design §6 C4.1/C4.3).

Pins the five EXACT field names + render order, the >=3 web_searches floor,
tolerant parsing/validation, research-vs-build classification, and the
backward-compatibility contract: non-research card generation is unchanged.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_m2 as m2
from hermes_cli import kanban_specify as spec

EXACT = ["web_searches", "lead_ledger", "anchor_note", "vendor_caveat", "wiki_handoff_disposition"]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _patch_aux_client(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return patch("agent.auxiliary_client.call_llm", return_value=resp)


# ---------- field contract ----------


def test_field_names_and_order_match_spec_exactly():
    assert list(m2.M2_FIELDS) == EXACT
    assert m2.M2_MIN_WEB_SEARCHES == 3


def test_render_emits_all_five_in_canonical_order():
    lines = m2.render_m2_block(web_searches=5).splitlines()
    assert [line.split(":")[0] for line in lines] == EXACT
    assert len(lines) == 5


def test_render_defaults_and_enforces_the_floor():
    assert m2.render_m2_block().splitlines()[0] == "web_searches: 3"
    assert m2.render_m2_block(web_searches=0).splitlines()[0] == "web_searches: 3"
    assert m2.render_m2_block(web_searches=2).splitlines()[0] == "web_searches: 3"
    assert m2.render_m2_block(web_searches=9).splitlines()[0] == "web_searches: 9"


def test_render_keeps_filled_values_verbatim():
    block = m2.render_m2_block(web_searches=4, anchor_note="anchors: A, B; search C",
                               wiki_handoff_disposition="no wiki value (internal-only)")
    assert "anchor_note: anchors: A, B; search C" in block
    assert "wiki_handoff_disposition: no wiki value (internal-only)" in block


def test_lead_ledger_renders_attached_placeholder():
    assert "lead_ledger: attached" in m2.render_m2_block()


# ---------- parsing / validation ----------


def test_parse_reads_all_five_and_tolerates_markers():
    text = "- web_searches: 4\n**lead_ledger: attached**\nanchor_note: x\nvendor_caveat: y\nwiki_handoff_disposition: z"
    found = m2.parse_m2_block(text)
    assert set(found) == set(EXACT)
    assert found["web_searches"] == "4"


def test_parse_ignores_unrelated_lines_and_empty_input():
    assert m2.parse_m2_block("**Goal**\nDo it.") == {}
    assert m2.parse_m2_block(None) == {}
    assert m2.parse_m2_block("") == {}


def test_has_m2_block_requires_all_five():
    assert m2.has_m2_block(m2.render_m2_block()) is True
    assert m2.has_m2_block("web_searches: 4\nlead_ledger: attached") is False


def test_missing_m2_fields_lists_in_canonical_order():
    assert m2.missing_m2_fields("web_searches: 4\n") == EXACT[1:]
    assert m2.missing_m2_fields(m2.render_m2_block()) == []


def test_validate_flags_missing_nonint_and_below_floor():
    assert m2.validate_m2_block(m2.render_m2_block(web_searches=4)) == []
    problems = m2.validate_m2_block("web_searches: 2")
    assert len(problems) == 5  # 4 missing fields + the below-floor web_searches
    assert sum(p.startswith("missing field:") for p in problems) == 4
    assert "must be an integer" in m2.validate_m2_block(
        m2.render_m2_block().replace("web_searches: 3", "web_searches: many"))[0]
    assert any("below the floor" in p for p in m2.validate_m2_block(
        m2.render_m2_block().replace("web_searches: 3", "web_searches: 1")))


def test_validate_never_raises_on_junk():
    for junk in (None, 7, ["x"], "web_searches: !!", "lead_ledger: \n"):
        assert isinstance(m2.validate_m2_block(junk), list)


# ---------- classification ----------


@pytest.mark.parametrize("title,body,expected", [
    ("Survey agent frameworks", "", True),
    ("Research vendors", "", True),
    ("Evaluate auth options", "", True),
    ("Comparison of harnesses", "", True),
    ("Do the thing", "phase: research\nbody", True),
    ("Implement the research index", "research everything", False),
    ("Implement login", "Add a form.", False),
    ("Build the survey tool", "survey", False),
    ("Fix the crawler", "", False),
])
def test_is_research_card_classification(title, body, expected):
    assert m2.is_research_card(title, body) is expected


def test_is_research_card_handles_non_string_input():
    assert m2.is_research_card(None, None) is False


# ---------- ensure_m2_block ----------


def test_ensure_inserts_block_after_phase_line():
    body = "phase: research\n\n**Goal**\nSurvey."
    out = m2.ensure_m2_block(body)
    lines = out.splitlines()
    assert lines[0] == "phase: research"
    assert [l.split(":")[0] for l in lines[1:6]] == EXACT
    assert "**Goal**" in out


def test_ensure_prepends_when_no_phase_line():
    out = m2.ensure_m2_block("**Goal**\nSurvey.")
    assert out.splitlines()[0] == "web_searches: 3"
    assert "**Goal**" in out


def test_ensure_preserves_caller_values_and_is_idempotent():
    partial = "web_searches: 7\n**Goal**\nSurvey."
    once = m2.ensure_m2_block(partial)
    assert m2.parse_m2_block(once)["web_searches"] == "7"
    assert once.count("web_searches:") == 1
    assert m2.ensure_m2_block(once) == once
    assert m2.ensure_m2_block(m2.ensure_m2_block(once)) == once


def test_ensure_handles_empty_body():
    assert m2.has_m2_block(m2.ensure_m2_block(""))


# ---------- specifier integration ----------


def test_specify_renders_block_on_research_card(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    content = jsonlib.dumps({
        "title": "Survey agentic frameworks",
        "body": "**Goal**\nSurvey.\n\n**Acceptance criteria**\n- [ ] cited",
    })
    with _patch_aux_client(content):
        outcome = spec.specify_task(tid, author="t")
    assert outcome.ok
    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert m2.has_m2_block(body)
    assert [l.split(":")[0] for l in body.splitlines() if l.split(":")[0] in EXACT] == EXACT


def test_specify_leaves_non_research_card_untouched(kanban_home):
    """Backward compatibility: the pre-change body must survive byte-for-byte."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    plain = "**Goal**\nAdd a form.\n\n**Acceptance criteria**\n- [ ] works"
    with _patch_aux_client(jsonlib.dumps({"title": "Implement login", "body": plain})):
        assert spec.specify_task(tid, author="t").ok
    with kbc.connect() as conn:
        assert (kb.get_task(conn, tid).body or "") == plain


def test_specify_completes_a_partial_llm_block(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    partial = "web_searches: 7\n**Goal**\nSurvey."
    with _patch_aux_client(jsonlib.dumps({"title": "Research the landscape", "body": partial})):
        assert spec.specify_task(tid, author="t").ok
    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert m2.validate_m2_block(body) == []
    assert body.count("web_searches:") == 1
    assert m2.parse_m2_block(body)["web_searches"] == "7"


def test_specify_raw_fallback_body_also_gets_the_block(kanban_home):
    """A malformed (non-JSON) reply becomes the body — research ones still get the block."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    with _patch_aux_client("Survey the framework landscape in depth"):
        outcome = spec.specify_task(tid, author="t")
    assert outcome.ok
    with kbc.connect() as conn:
        assert m2.has_m2_block(kb.get_task(conn, tid).body or "")


# ---------- decomposer integration (T4 C4.1/C4.3) ----------


def _patch_profiles(names: list[str]):
    from types import SimpleNamespace

    fakes = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fakes),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0]),
    ]


def _decompose(tid: str, payload: str, names: list[str]):
    from hermes_cli import kanban_decompose as decomp

    patchers = _patch_profiles(names)
    for p in patchers:
        p.start()
    try:
        with _patch_aux_client(payload), patch(
            "hermes_cli.config.load_config_readonly", return_value={},
        ):
            return decomp.decompose_task(tid, author="t")
    finally:
        for p in patchers:
            p.stop()


def test_decompose_fanout_gives_research_child_the_block_not_the_build_child(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest", triage=True)
    payload = jsonlib.dumps({
        "fanout": True, "rationale": "x",
        "tasks": [
            {"title": "Survey the framework landscape", "body": "**Goal**\nsurvey",
             "assignee": None, "parents": []},
            {"title": "Implement the integration", "body": "**Goal**\ncode it",
             "assignee": None, "parents": [0]},
        ],
    })
    outcome = _decompose(tid, payload, ["default"])
    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        kids = [kb.get_task(conn, cid) for cid in (outcome.child_ids or [])]
    research_kid = next(k for k in kids if "Survey" in k.title)
    build_kid = next(k for k in kids if "Implement" in k.title)
    assert m2.has_m2_block(research_kid.body)
    assert m2.validate_m2_block(research_kid.body) == []
    assert not m2.has_m2_block(build_kid.body)
    # C2.3 bucket headings must survive the M2 insertion.
    assert "Adoption:" in research_kid.body


def test_decompose_single_path_also_carries_the_block(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest2", triage=True)
    payload = jsonlib.dumps({
        "fanout": False, "rationale": "y", "title": "Survey auth options",
        "body": "**Goal**\nsurvey auth", "assignee": None,
    })
    outcome = _decompose(tid, payload, ["default"])
    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert m2.has_m2_block(body)
    assert m2.validate_m2_block(body) == []


def test_decompose_single_build_card_is_untouched(kanban_home):
    """Backward compatibility on the decompose path: no block on a build card."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest3", triage=True)
    plain = "**Goal**\nAdd a form."
    payload = jsonlib.dumps({
        "fanout": False, "rationale": "y", "title": "Implement login",
        "body": plain, "assignee": None,
    })
    assert _decompose(tid, payload, ["default"]).ok
    with kbc.connect() as conn:
        body = kb.get_task(conn, tid).body or ""
    assert "web_searches:" not in body
    # C2.3 bucket headings are appended by capability routing on all children.
    assert body.startswith(plain)


def test_decompose_verbless_build_child_gains_no_spurious_block(kanban_home):
    """Regression: the C2.3 heading 'Research-feedback: none' is routing
    boilerplate, not intent. A child whose title carries no build verb must not
    be reclassified as research by its own bucket headings."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest4", triage=True)
    payload = jsonlib.dumps({
        "fanout": True, "rationale": "y",
        "tasks": [
            {"title": "Polish the landing page", "body": "**Goal**\ntighten spacing",
             "assignee": None, "parents": []},
            {"title": "Write the ops runbook", "body": "**Goal**\ndocument the steps",
             "assignee": None, "parents": []},
        ],
    })
    assert _decompose(tid, payload, ["default"]).ok
    with kbc.connect() as conn:
        kids = kb.list_tasks(conn, status=None)
    for kid in kids:
        if kid.title in ("Polish the landing page", "Write the ops runbook"):
            assert not m2.has_m2_block(kid.body), kid.title
            # The bucket headings still landed.
            assert "Research-feedback: none" in (kid.body or "")


def test_bucket_boilerplate_scrub_leaves_real_research_signals_alone():
    """Scrubbing only removes the EMPTY bucket lines."""
    scrub = m2._BUCKET_BOILERPLATE_RE.sub
    body = (
        "**Goal**\nDo it.\nAdoption: none\nDocs: none\n"
        "Research-feedback: run a framework survey\nGate: none"
    )
    assert "Adoption: none" not in scrub("", body)
    assert "Research-feedback: run a framework survey" in scrub("", body)
    assert m2.is_research_card("Polish the landing page", body) is True

    for none_form in ("Adoption: none", "Docs: NONE", "- **Gate: n/a**", "* Research-feedback: -"):
        assert scrub("", none_form).strip() == "", none_form


# The bucket headings are answered in prose, so the *placeholder* frequently
# arrives sentence-final ("Research-feedback: none.") — 6 such lines were live
# on the board when this was fixed. A placeholder is a placeholder whether or
# not it is punctuated; an end-anchored tail that requires the bare token lets
# the scrub silently miss, and every verbless child then gets a spurious M2
# block stapled on.
BASE = "## Goal\nExecute wiki updates based on t_1 policy decision and clean stale pages.\n"
_PLACEHOLDER_TITLE = "Wiki Update: Division of Labor & Staleness Cleanup"


@pytest.mark.parametrize("heading", ["Adoption", "Docs", "Research-feedback", "Gate"])
@pytest.mark.parametrize("value", ["None.", "none.", "N/A.", "n/a.", "-.", "none.  "])
def test_punctuated_placeholder_scrubs_under_every_bucket_heading(heading, value):
    """AC1: `None.` (and friends) are placeholder forms, not intent — under all
    four headings, with trailing whitespace and with emphasis wrappers."""
    for wrapper in ("{h}: {v}", "- **{h}: {v}**", "# {h}: {v}", "> {h}: {v}", "{h}: `{v}`"):
        line = wrapper.format(h=heading, v=value)
        assert m2._BUCKET_BOILERPLATE_RE.sub("", line).strip() == "", line
        assert m2.is_research_card(
            _PLACEHOLDER_TITLE, BASE + line + "\n"
        ) is False, line


def test_punctuated_placeholder_regression_is_research_card():
    """The exact live defect: the decomposer's dotted boilerplate flipped a
    non-research card to research and stapled an unsatisfiable M2 block on."""
    assert m2.is_research_card(_PLACEHOLDER_TITLE, BASE + "Research-feedback: None.\n") is False
    assert m2.is_research_card(_PLACEHOLDER_TITLE, BASE) is False


def test_regression_positive_control_same_line_with_content_still_counts():
    """Positive control for the regression above: the SAME heading and the SAME
    sentence-final period, but carrying content, must survive the scrub. This
    passes at both revisions — so a "fix" that widens the tail to swallow
    everything after the heading cannot make the target assertion pass."""
    content = "Research-feedback: two follow-on surveys needed.\n"
    assert m2._BUCKET_BOILERPLATE_RE.sub("", BASE + content) == BASE + content
    assert m2.is_research_card(_PLACEHOLDER_TITLE, BASE + content) is True


def test_scrub_keeps_bucket_lines_that_carry_content_after_the_placeholder():
    """Positive control: only the empty/placeholder forms are scrubbed — a
    bucket line that answers the heading with real prose survives byte-for-byte
    (live shape: `Adoption: none — evidence only; the queue is downstream.`)."""
    scrub = m2._BUCKET_BOILERPLATE_RE.sub
    for line in (
        "Adoption: none — evidence only; the adoption queue is produced downstream.",
        "Docs: none of the existing pages cover this; a new page is needed.",
        "Research-feedback: none of the vendor numbers were third-party verified.",
        "Gate: none of the options can ship without the user's verdict.",
        "Docs: none. A separate wiki-keeper landing card receives the findings.",
    ):
        assert scrub("", line) == line, line
    assert m2.is_research_card(
        _PLACEHOLDER_TITLE,
        BASE + "Research-feedback: none of the surveys covered this.\n",
    ) is True


# ---------- cross-feature: main's Step 0 body block (wiki-index rollout) ----------

# Verbatim body block the wiki-index rollout injects into wiki-owning cards.
STEP0_BODY_BLOCK = """## Step 0 — KNOWLEDGE LOOKUP (verb-first, MANDATORY)
This card touches the knowledge wiki. Before any other action, query the index:
  python3 ~/.hermes/scripts/wiki_index.py search "<your query>"
(verbs: search | backlinks | related | outline | related_skills | goals_query)
- <=2 BM25 phrasing retries (strip qualifiers, use single tokens/aliases) BEFORE any grep.
- grep/search_files/read_file over ~/knowledge/wiki is legal only after 2 failed verb
  attempts AND with a concrete file target — record which attempts failed.
- Structure questions: `outline <slug>` first, not a whole-page read.
- Provenance ("what grounds X"): `related` + `backlinks`, not grep.
- Do NOT use `execute_code`, `python3 -c`, or `python3 - <<'PY'` — all three are
  auto-denied in a worker session. `python3 <scriptfile>` is permitted.
Then record `wiki_lookup: {verbs:[...], fallback_reason: null|<text>}` in completion metadata."""

BUCKETS = "Adoption: none\nDocs: none\nResearch-feedback: none\nGate: none"


def test_step0_body_block_does_not_flip_a_build_card_into_research():
    """The Step 0 block contains "lookup"/"retrieval"-shaped prose and lands in
    wiki-owning BUILD card bodies too — it must not trigger the M2 block."""
    assert m2.is_research_card("Implement the crawler", STEP0_BODY_BLOCK) is False
    assert m2.is_research_card(
        "Update the wiki index loader", STEP0_BODY_BLOCK + "\n" + BUCKETS) is False
    assert m2.is_research_card("Survey agent frameworks", STEP0_BODY_BLOCK) is True
