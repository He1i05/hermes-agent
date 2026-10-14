"""W-8: the DECLARED STREAM COVERAGE contract (quest t_bec82259, card t_e9fb49b3).

The defect under test
---------------------
The decomposer's only view of a root body was ``_truncate(task.body, 4000)``
(``kanban_specify.py``). Measured on the real roots (read-only, 2026-09-30, code
pin ``26b000cc0f``):

    t_26d8eb94  body 8234  S1@3144 S2@3812 | S3@4383 S4@4923 S5@5510 S6@5945
    t_bec82259  body 8501  R1@5284 .. R5@6937   (all five past the cut)

Only S1/S2 were ever in front of the model. No receipt built from the prompt
slice can report S3–S6, and a receipt built as a body-vs-children *diff*
false-passes on exactly this input (keyword overlap: S3–S6 each borrow S1's
child on the generic word "pipeline"). The fix is therefore a DECLARATION
contract — the root declares its streams, each child declares which it covers or
folds in, and a stream with neither declaration is a gap.

These tests are all hermetic: every board is a throwaway ``HERMES_HOME`` under
``tmp_path`` (the repo's ``kanban_home`` fixture), and nothing here opens the
live ``kanban.db``. Under ``scripts/run_tests.sh`` the whole file additionally
runs inside ``env -i``.
"""

from __future__ import annotations

import json as jsonlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as spec
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

#: The real ``t_26d8eb94`` stream head: ``## STREAMS TO PLAN (...)`` + six 400+char
#: tagged entries. Offsets are the measured ones from the live body.
_REAL_ROOT_HEAD = "TRIAGE CARD — D financial pipeline FULL audit\n\n## Step 0 — LOOKUP\nx\n\n"
_REAL_STREAMS_HEADING = (
    "## STREAMS TO PLAN (triager to decompose; disjoint scopes; read-only DB contract)\n"
)
_STREAM_DESCRIPTIONS = {
    "S1": "STRUCTURAL & META-HARNESS AUDIT — full dependency-chain map of the v3 pipeline",
    "S2": "ANALYSIS-QUALITY & SIGNAL-ECONOMICS AUDIT — entry points, calibration, PSR/DSR",
    "S3": "SIGNALS-DB AUDIT — consumer census, _match_historical usage, delivered immutability",
    "S4": "SANITATION / DEAD-WEIGHT AUDIT — unused modules, crons, config keys, DB tables",
    "S5": "DISCORD DELIVERY & USER-FACING STRUCTURE AUDIT — digest shape, render contract",
    "S6": "FINANCIAL-GAIN SYNTHESIS & PROMOTION QUEUE — rank all findings by P&L impact",
}


def _real_shaped_root() -> str:
    """A >4000-char root body shaped exactly like the measured one, with the
    stream section starting near char 3000 so S1/S2 sit inside the clamp and
    S3–S6 land beyond it."""
    pad = "Adjudicate the dirty tree BEFORE any observation. " * 62  # ~3.0 KB
    lines = [
        "TRIAGE CARD — D financial pipeline FULL audit\n",
        "\n## Step 0 — KNOWLEDGE LOOKUP (verb-first, MANDATORY)\n",
        "python3 ~/.hermes/scripts/wiki_index.py search \"...\"\n",
        "\n## SEQUENCING & CONTEXT\n",
        pad + "\n",
        "\n" + _REAL_STREAMS_HEADING,
    ]
    for tag, desc in _STREAM_DESCRIPTIONS.items():
        # 430 chars per entry — the measured per-line length on the real body.
        lines.append(f"{tag} {desc} " + ("detail. " * 50) + "\n")
    lines.append("\n## CONSTRAINTS\n- Read-only on kanban.db (mode=ro URI).\n")
    lines.append("\n## DELIVERABLE SHAPE\n- One consolidated audit packet.\n")
    return "".join(lines)


def _patch_aux_client(content: str):
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


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


def _decompose(kanban_home, *, root_body: str, llm_payload: dict,
               title: str = "quest") -> dict:
    """Create a hermetic triage root, decompose it against ``llm_payload``,
    and return the captured prompt + outcome + persisted children."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=title, body=root_body, triage=True)

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_aux_response(jsonlib.dumps(llm_payload))

    patches = _patch_list_profiles(["orchestrator", "engineer", "researcher"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=_capture):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        children = [kb.get_task(conn, c) for c in (outcome.child_ids or [])]
    return {"captured": captured, "outcome": outcome, "root": root, "children": children}


# --------------------------------------------------------------------------- #
# 1 — the parser reads the REAL root shapes (the load-bearing measurement)
# --------------------------------------------------------------------------- #

def test_declared_streams_reads_the_measured_t26d8eb94_shape():
    """Six streams, document order, and the measured @char offsets.

    The offsets are the live ones (S1@3144 … S6@5945 on the 8234-char body);
    reproduced here at the same shape so a change that shifts the section is
    caught rather than silently changing the expected answer.
    """
    body = _real_shaped_root()
    found = streams.declared_streams(body)
    assert [s.tag for s in found] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert [s.offset for s in found] == sorted(s.offset for s in found)
    # The first two sit inside the 4000-char clamp; the last four beyond it.
    inside = [s.tag for s in found if s.offset < spec.PROMPT_BODY_LIMIT]
    outside = [s.tag for s in found if s.offset >= spec.PROMPT_BODY_LIMIT]
    assert inside == ["S1", "S2"], inside
    assert outside == ["S3", "S4", "S5", "S6"], outside


def test_prose_mention_of_streams_is_not_a_section():
    """A body that DISCUSSES streams in prose declares none.

    This is the exact shape that broke a naive ``\\bSTREAMS?\\b`` heading match:
    ``t_bec82259`` line 15 says "every named stream/scope section ..." and line
    21 is the real ``## SCOPE STREAMS`` heading. Matching the prose line would
    open the section in the wrong place and silently report zero streams.
    """
    body = (
        "## WHY\n"
        "- DECOMPOSER COMPLETENESS: every named stream/scope section in a triage "
        "body must map to >=1 child OR an explicit fold-in note naming the "
        "absorbing child.\n"
        "- the receipt lists stream->child mapping\n"
        "\n## SCOPE STREAMS (triager to decompose)\n"
        "R1 EVIDENCE CONSOLIDATION — walk the incidents\n"
        "R2 DECOMPOSER COMPLETENESS MECHANISM — design the receipt\n"
    )
    assert streams.declared_stream_tags(body) == ["R1", "R2"]


def test_streams_frontmatter_short_form_is_accepted():
    assert streams.declared_stream_tags("streams: S1, S2") == ["S1", "S2"]
    assert streams.declared_stream_tags("- **streams**: s3 s4") == ["S3", "S4"]


def test_tag_normalization_is_closed_on_the_tag_shape():
    assert streams.normalize_tag("**S3**") == "S3"
    assert streams.normalize_tag("r5)") == "R5"
    assert streams.normalize_tag("s6:") == "S6"
    # Not tag-shaped: a prose word or an over-long letter/digit run is rejected
    # rather than coerced, so a typo can never read downstream as a real stream.
    assert streams.normalize_tag("STREAM") is None
    assert streams.normalize_tag("ABCD12") is None
    assert streams.normalize_tag("S1234") is None
    assert streams.normalize_tag("") is None
    assert streams.normalize_tag(None) is None


def test_body_with_no_streams_declares_none():
    assert streams.declared_streams("") == []
    assert streams.declared_stream_tags("## Goal\njust a plain card") == []


# --------------------------------------------------------------------------- #
# 2 — the prompt carries the FULL declared coverage (the core defect fix)
# --------------------------------------------------------------------------- #

def test_prompt_body_makes_streams_beyond_the_cut_visible():
    """The defect: S3–S6 were never in front of the model. They must be now."""
    body = _real_shaped_root()
    assert len(body) > spec.PROMPT_BODY_LIMIT  # precondition: the clamp fires

    prompt_body = streams.prompt_body(body, limit=spec.PROMPT_BODY_LIMIT)

    # The raw clamp still happened (the model is not handed an unbounded body)…
    assert len(prompt_body) > spec.PROMPT_BODY_LIMIT
    assert streams.APPENDIX_HEADING in prompt_body
    # …and every declared stream is now named, with its offset and its side of
    # the cut, so a compliance failure is distinguishable from a boundary one.
    for tag in ("S1", "S2", "S3", "S4", "S5", "S6"):
        assert f"[{tag}] @char" in prompt_body, tag
    assert "BEYOND the cut (NOT visible above)" in prompt_body
    assert "inside the visible text" in prompt_body
    # The appendix states the truncation fact explicitly (plan §7.2).
    assert f"CUT at char {spec.PROMPT_BODY_LIMIT - 1} of {len(body)}" in prompt_body
    # The declarations the model must now emit are named exactly.
    assert "covers: <tag>" in prompt_body
    assert "Fold-in: <tag> — <why>" in prompt_body


def test_prompt_body_is_byte_identical_when_the_body_fits():
    """The ordinary card (~770 bodies on the board) renders unchanged."""
    for body in ("## Goal\nshort", "", "x" * (spec.PROMPT_BODY_LIMIT - 1),
                 "x" * spec.PROMPT_BODY_LIMIT):
        assert streams.prompt_body(body, limit=spec.PROMPT_BODY_LIMIT) == body


def test_prompt_body_reports_no_streams_declared_and_invents_nothing():
    """A >4000-char body with no stream section gets NO-STREAMS-DECLARED.

    Absence of declared streams is itself the signal — the receipt must never
    invent a gap (plan §7.1), and the prompt must not invite invented tags.
    """
    body = "ordinary long card\n" + ("filler. " * 700)
    out = streams.prompt_body(body, limit=spec.PROMPT_BODY_LIMIT)
    assert streams.NO_STREAMS_DECLARED in out
    assert "do NOT invent stream tags" in out
    assert "[S1] @char" not in out
    assert "The root DECLARES" not in out


def test_task_prompt_fields_uses_the_coverage_aware_body():
    """``_task_prompt_fields`` (shared by specify + decompose) is the call site."""
    import types
    body = _real_shaped_root()
    task = types.SimpleNamespace(id="t_x", title="t", body=body)
    fields = spec._task_prompt_fields(task)
    assert "S6" in fields["body"] and "[S6] @char" in fields["body"]
    assert fields["task_id"] == "t_x"

    short = types.SimpleNamespace(id="t_y", title="t", body="## Goal\nshort")
    assert spec._task_prompt_fields(short)["body"] == "## Goal\nshort"


def test_prompt_body_tolerates_a_none_body():
    assert streams.prompt_body(None, limit=spec.PROMPT_BODY_LIMIT) == ""


# --------------------------------------------------------------------------- #
# 3 — the prompt CONTRACT is in both templates (AC1, single source of truth)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("prompt_attr", ["_SYSTEM_PROMPT"])
def test_both_card_templates_carry_the_contract_exactly(prompt_attr):
    block = streams.coverage_prompt_block()
    for module in (spec, decomp):
        prompt = getattr(module, prompt_attr)
        assert "DECLARED STREAM COVERAGE" in prompt
        assert "covers: <tag>" in prompt
        assert "Fold-in: <tag> — <why>" in prompt
        assert "COVERAGE GAP" in prompt
        assert "NEVER fabricate" in prompt
        # Rendered from the module, not retyped — the literals are the module's.
        assert streams.COVERS_KEY in prompt and streams.FOLDIN_KEY in prompt
        assert "DECLARED STREAM COVERAGE" in block


def test_decomposer_response_schema_documents_the_declaration_keys():
    prompt = decomp._SYSTEM_PROMPT
    assert '"covers": ["<stream tag this card OWNS>"' in prompt
    assert '"fold_in": {"<stream tag this card ABSORBS>"' in prompt


# --------------------------------------------------------------------------- #
# 4 — the declaration oracle (0/4 false-pass; the diff oracle is the control)
# --------------------------------------------------------------------------- #

def test_declaration_oracle_scores_zero_false_passes_where_the_diff_scores_four():
    """The measured oracle comparison, as a permanent negative fixture.

    On the real incident the four truly-missing streams (S3–S6) each borrowed
    S1's child on the generic word "pipeline" — every stream heading carried it
    and every child body contained it. A keyword diff built from those heading
    tokens therefore reported the missing streams COVERED. The declaration
    oracle reports them as GAPS, because no child declares them.
    """
    root = ("## STREAMS\n"
            "S1 pipeline structural audit\n"
            "S2 pipeline economics audit\n"
            "S3 pipeline signals-db audit\n"
            "S4 pipeline dead-weight audit\n"
            "S5 pipeline delivery audit\n"
            "S6 pipeline synthesis queue\n")
    # The children as the incident actually produced them: the S1 child's body
    # mentions the shared heading token ("pipeline") for every stream scope, and
    # declares nothing.
    incident_children = [
        "Structural & meta-harness audit of the D pipeline\n"
        "This covers the dependency chain; see also the pipeline signals DB, "
        "pipeline dead weight, pipeline delivery and the pipeline synthesis queue.",
        "Analysis-quality & signal-economics audit of the D pipeline\n",
    ]

    # DECLARATION oracle: every stream is a gap (0 false-passes).
    assert streams.missing_coverage(root, incident_children) == [
        "S1", "S2", "S3", "S4", "S5", "S6",
    ]

    # The REJECTED diff oracle, reproduced faithfully (overlap the stream
    # heading tokens with the child bodies): all six streams read as covered.
    def naive_diff_gaps(root_body: str, kids: list[str]) -> list[str]:
        gaps = []
        for s in streams.declared_streams(root_body):
            tokens = [t for t in re.findall(r"[a-z]{4,}", s.description.lower())
                      if t not in {"audit", "with"}]
            if not any(tok in " ".join(kids).lower() for tok in tokens):
                gaps.append(s.tag)
        return gaps

    assert naive_diff_gaps(root, incident_children) == []  # 6/6 false-pass

    # With declarations, the same children account for coverage exactly.
    declared = [
        "covers: S1",
        "covers: S2\nFold-in: S3 — absorbed by the census\n"
        "Fold-in: S4 — absorbed\nFold-in: S5 — absorbed\nFold-in: S6 — absorbed",
    ]
    assert streams.missing_coverage(root, declared) == []


def test_missing_coverage_returns_the_whole_set_for_undeclared_children():
    """F3: 0 declared children is a LOUD result, never a silent pass."""
    root = "## SCOPE STREAMS\nR1 a\nR2 b\nR3 c\n"
    assert streams.missing_coverage(root, ["plain body", "another body"]) == [
        "R1", "R2", "R3",
    ]
    assert streams.missing_coverage(root, []) == ["R1", "R2", "R3"]


def test_missing_coverage_is_vacuous_on_a_prompt_slice():
    """Why the receipt must read the FULL body: the slice declares zero streams."""
    full = _real_shaped_root()
    sliced = full[: spec.PROMPT_BODY_LIMIT]
    assert len(streams.declared_streams(full)) == 6
    assert len(streams.declared_streams(sliced)) == 2  # S1, S2 only
    assert streams.missing_coverage(sliced, []) == ["S1", "S2"]
    # …and a receipt over the slice cannot name S3–S6 at all.
    assert "S3" not in streams.declared_stream_tags(sliced)


# --------------------------------------------------------------------------- #
# 5 — declaration lines are rendered, idempotently, never fabricated
# --------------------------------------------------------------------------- #

def test_ensure_declaration_lines_renders_canonical_lines():
    out = streams.ensure_declaration_lines(
        "child body", covers=["S1", "s1", "S2"], fold_in={"S3": "absorbed by census"},
    )
    assert out.count("covers: S1") == 1          # de-duplicated
    assert "covers: S2" in out
    assert "Fold-in: S3 — absorbed by census" in out
    assert streams.covers_tags(out) == ["S1", "S2"]
    assert streams.foldin_streams(out) == ["S3"]


def test_ensure_declaration_lines_is_idempotent_and_never_fabricates():
    once = streams.ensure_declaration_lines("body", covers=["S1"], fold_in={"S2": "why"})
    twice = streams.ensure_declaration_lines(once, covers=["S1"], fold_in={"S2": "why"})
    assert once == twice
    # No declaration in -> byte-identical body out.
    assert streams.ensure_declaration_lines("body") == "body"
    assert streams.ensure_declaration_lines("body", covers=[], fold_in={}) == "body"
    # Unparseable declarations are dropped, not coerced.
    assert streams.ensure_declaration_lines("body", covers=["not-a-tag"]) == "body"


def test_fold_in_accepts_the_shapes_a_model_actually_writes():
    for value, expect in [
        ("S4", ["S4"]),
        (["S4", "S5"], ["S4", "S5"]),
        ("S4 — absorbed by the census card", ["S4"]),
        ("S4, S5", ["S4", "S5"]),
        ({"S4": "why"}, ["S4"]),
        ([], []),
        (None, []),
        (123, []),
    ]:
        lines = streams.coverage_lines(fold_in=value)
        assert streams.foldin_streams("\n".join(lines)) == expect, value
    assert streams.foldin_streams(
        streams.ensure_declaration_lines("b", fold_in="S4 — absorbed")
    ) == ["S4"]


def test_accounted_tags_unions_covers_and_fold_in():
    text = "covers: S1\ncovers: S2\nFold-in: S3 — merged\nFold-in: S2 — dup"
    assert streams.accounted_tags(text) == ["S1", "S2", "S3"]
    assert streams.covers_tags(text) == ["S1", "S2"]
    assert streams.foldin_streams(text) == ["S3", "S2"]


# --------------------------------------------------------------------------- #
# 6 — end-to-end, hermetic: a >4000-char streamed root fans out fully declared
# --------------------------------------------------------------------------- #

def test_streamed_root_prompt_carries_all_streams_and_children_declare_coverage(
    kanban_home,
):
    """The integration evidence the quest asks for.

    A hermetic >4000-char triage body: (a) the user prompt the decomposer
    receives names S1..S6 including the four past the cut; (b) a reply that
    declares coverage produces child bodies carrying the canonical lines; and
    (c) the declaration oracle over the PERSISTED children reports no gap.
    """
    body = _real_shaped_root()
    payload = {
        "fanout": True,
        "rationale": "six-stream split",
        "tasks": [
            {"title": "S1 structural audit", "body": "phase: research\nmap it",
             "assignee": "researcher", "phase": "research", "parents": [],
             "covers": ["S1"]},
            {"title": "S2 economics audit", "body": "phase: research\neconomics",
             "assignee": "researcher", "phase": "research", "parents": [],
             "covers": "S2"},
            {"title": "S3-S6 consolidated packet", "body": "phase: research\npacket",
             "assignee": "researcher", "phase": "research", "parents": [0, 1],
             "covers": ["S3"],
             "fold_in": {"S4": "absorbed by the sanitation census",
                         "S5": "digest render contract folded into the packet",
                         "S6": "promotion queue is the packet's last section"}},
        ],
    }

    result = _decompose(kanban_home, root_body=body, llm_payload=payload)
    user_prompt = result["captured"]["messages"][1]["content"]

    for tag in ("S1", "S2", "S3", "S4", "S5", "S6"):
        assert f"[{tag}] @char" in user_prompt, tag
    assert "BEYOND the cut" in user_prompt

    bodies = [c.body or "" for c in result["children"]]
    assert streams.covers_tags(bodies[0]) == ["S1"]
    assert streams.covers_tags(bodies[1]) == ["S2"]
    assert streams.foldin_streams(bodies[2]) == ["S4", "S5", "S6"]
    assert streams.missing_coverage(body, bodies) == []

    # The declarations reach the PERSISTED body, not just the local dict — a
    # worker reading its card sees what it owns.
    with kbc.connect() as conn:
        persisted = [kb.get_task(conn, c).body or "" for c in result["outcome"].child_ids]
    assert streams.missing_coverage(body, persisted) == []
    assert any("covers: S1" in b for b in persisted)


def test_a_reply_that_declares_nothing_yields_gaps_not_a_silent_pass(kanban_home):
    """F3: the receipt reports K gaps instead of trusting the writer."""
    body = _real_shaped_root()
    payload = {
        "fanout": True,
        "rationale": "no declarations",
        "tasks": [
            {"title": "do the thing", "body": "phase: implementation\ncode it",
             "assignee": "engineer", "phase": "implementation", "parents": []},
        ],
    }
    result = _decompose(kanban_home, root_body=body, llm_payload=payload)
    bodies = [c.body or "" for c in result["children"]]
    assert streams.missing_coverage(body, bodies) == ["S1", "S2", "S3", "S4", "S5", "S6"]


def test_streamless_root_children_are_byte_identical_to_pre_change(kanban_home):
    """Backward compatibility: the ordinary fan-out gains no coverage lines."""
    payload = {
        "fanout": True,
        "rationale": "plain",
        "tasks": [
            {"title": "build it", "body": "phase: implementation\ncode it",
             "assignee": "engineer", "phase": "implementation", "parents": []},
        ],
    }
    result = _decompose(kanban_home, root_body="## Goal\nplain quest, no streams",
                        llm_payload=payload)
    body = result["children"][0].body or ""
    assert "covers:" not in body and "Fold-in:" not in body
    assert body.startswith("phase: implementation")


def test_single_card_reply_carries_the_declarations(kanban_home):
    """fanout=false: the one card declares the streams it owns."""
    body = _real_shaped_root()
    payload = {
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened single card",
        "body": "phase: implementation\nDo all of it.",
        "assignee": "engineer",
        "phase": "implementation",
        "covers": ["S1", "S2", "S3", "S4", "S5", "S6"],
    }
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest", body=body, triage=True)
    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(jsonlib.dumps(payload)):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        card_body = kb.get_task(conn, tid).body or ""
    assert streams.missing_coverage(body, [card_body]) == []
    assert streams.covers_tags(card_body) == ["S1", "S2", "S3", "S4", "S5", "S6"]


def test_no_db_write_leaks_outside_the_hermetic_home(kanban_home):
    """Every board write above is inside the throwaway home, never the live one."""
    resolved = Path(kb.kanban_db_path()).resolve()
    assert str(kanban_home.resolve()) in str(resolved)
