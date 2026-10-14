"""W-7: the post-decompose coverage RECEIPT (quest t_bec82259, card t_6a91f797).

The defect these tests pin
--------------------------
The decomposer's only view of a root body was ``_truncate(task.body, 4000)``.
Measured read-only on the live board (2026-09-30, pin ``26b000cc0f``):

    t_26d8eb94  body 8234  S1@3144 S2@3812 | S3@4383 S4@4923 S5@5510 S6@5945
    t_bec82259  body 8501  R1@5284 R2@5685 R3@6173 R4@6662 R5@6937  (all past)

A receipt built from the prompt slice reports ZERO declared streams on
``t_bec82259`` and passes VACUOUSLY; a receipt built as a body-vs-children
keyword diff reports every missing stream COVERED (measured: 6/6 on
``t_26d8eb94`` and 5/5 on ``t_bec82259`` against the REAL child bodies, both
frozen below). Both rejected shapes are asserted here as permanent negative
fixtures so nobody can "simplify" the receipt back into either.

The two root fixtures below reproduce the measured ``@char`` offsets exactly
(constructed, asserted) rather than approximately: the plan's acceptance
criteria are stated in those offsets, so a body edit that shifts the stream
section must break the fixture rather than silently change the expected answer
(plan §4 Layer 3).

Hermetic: every board is a throwaway ``HERMES_HOME`` under ``tmp_path`` (the
repo's ``kanban_home`` fixture) and nothing here opens the live ``kanban.db``.
Under ``scripts/run_tests.sh`` the whole file additionally runs inside ``env -i``.
"""

from __future__ import annotations

import json as jsonlib
import re

import pytest

from hermes_cli import kanban_coverage as cov
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_specify as spec
from hermes_cli import kanban_streams as streams


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb.init_db()
    return home


# --------------------------------------------------------------------------- #
# fixtures — the MEASURED shapes and offsets, not invented ones
# --------------------------------------------------------------------------- #

_S_DESCS = {
    "S1": "STRUCTURAL & META-HARNESS AUDIT — full dependency-chain map of the v3 pipeline",
    "S2": "ANALYSIS-QUALITY & SIGNAL-ECONOMICS AUDIT — entry points, calibration, PSR/DSR",
    "S3": "SIGNALS-DB AUDIT — consumer census, _match_historical usage, delivered immutability",
    "S4": "SANITATION / DEAD-WEIGHT AUDIT — unused modules, crons, config keys, DB tables",
    "S5": "DISCORD DELIVERY & USER-FACING STRUCTURE AUDIT — digest shape, render contract",
    "S6": "FINANCIAL-GAIN SYNTHESIS & PROMOTION QUEUE — rank all findings by P&L impact",
}
#: Measured on the live ``t_26d8eb94`` body (8234 chars).
_S_OFFSETS = {"S1": 3144, "S2": 3812, "S3": 4383, "S4": 4923, "S5": 5510, "S6": 5945}

_R_DESCS = {
    "R1": "EVIDENCE CONSOLIDATION — walk the two live incidents end to end",
    "R2": "DECOMPOSER COMPLETENESS MECHANISM — design the declared coverage receipt",
    "R3": "RELEASE WATCHDOG DESIGN — build the H1 freeze detector",
    "R4": "CONSTRAINTS & SEQUENCING — carry the root constraints to every child",
    "R5": "SYNTHESIS — consolidate the class-A promotion queue",
}
#: Measured on the live ``t_bec82259`` body (8501 chars) — ALL beyond 3999.
_R_OFFSETS = {"R1": 5284, "R2": 5685, "R3": 6173, "R4": 6662, "R5": 6937}

_HEAD = (
    "TRIAGE CARD — measured-shape fixture\n"
    "\n## Step 0 — KNOWLEDGE LOOKUP (verb-first, MANDATORY)\n"
    "python3 ~/.hermes/scripts/wiki_index.py search \"...\"\n"
    "\n## SEQUENCING & CONTEXT\n"
)
_STREAMS_HEADING = "\n## SCOPE STREAMS (triager to decompose; disjoint scopes)\n"
_TAIL = "\n## CONSTRAINTS\n- Read-only on kanban.db (mode=ro URI).\n"


def _exact_offset_root(descs: dict[str, str], offsets: dict[str, int]) -> str:
    """A root body whose declared streams sit at EXACTLY ``offsets``.

    Deterministic filler (no randomness), so the fixture is reproducible and the
    asserted offsets are the measured ones.
    """
    tags = list(descs)
    first = offsets[tags[0]]
    filler_target = first - len(_HEAD) - len(_STREAMS_HEADING)
    assert filler_target > 0, "fixture head does not fit before the first stream"
    filler = "Adjudicate the dirty tree BEFORE any observation. "
    pad = filler * (filler_target // len(filler))
    pad += "x" * (filler_target - len(pad))
    out = _HEAD + pad + _STREAMS_HEADING
    assert len(out) == first, (len(out), first)
    for i, tag in enumerate(tags):
        nxt = offsets[tags[i + 1]] if i + 1 < len(tags) else None
        line = f"{tag} {descs[tag]}"
        if nxt is None:
            line += " " + "detail. " * 60
        else:
            needed = nxt - offsets[tag] - 1
            assert len(line) <= needed, (tag, len(line), needed)
            while len(line) + len(" detail.") <= needed:
                line += " detail."
            line += "x" * (needed - len(line))
            assert len(line) == needed, (tag, len(line), needed)
        out += line + "\n"
    out += _TAIL.strip("\n")
    return out


def _real_shaped_root() -> str:
    """The ``t_26d8eb94`` shape: S1/S2 inside the cut, S3–S6 beyond it."""
    return _exact_offset_root(_S_DESCS, _S_OFFSETS)


def _r_shaped_root() -> str:
    """The ``t_bec82259`` shape: ALL five R-streams beyond the cut."""
    return _exact_offset_root(_R_DESCS, _R_OFFSETS)


#: FROZEN VERBATIM bodies of the real children, dumped read-only from the live
#: board 2026-09-30 (ids in the decomposer's `decomposed` event order). These
#: are the inputs the measured false-pass was taken on: the keyword diff scores
#: 0 gaps over them while the declaration oracle reports every stream a gap.
#: Reproduce with: select body from tasks where id in (<the ids below>).

_S_REAL_CHILDREN = [  # verbatim, read-only dump of t_26d8eb94
    {"id": "t_6ae16538", "body": "phase: grounding\nweb_searches: 3\nlead_ledger: attached\nanchor_note: <anchors named in the card body + the out-of-list angles to search>\nvendor_caveat: <evaluator (vendor-reported vs third-party) for any vendor number cited>\nwiki_handoff_disposition: <wiki-worthy -> wiki-keeper landing child | no wiki value (internal-only)>\n\n## Objective\nAdjudicate the current working tree against HEAD to settle residue vs. remediation, ensuring auditors evaluate a fly-forward stable state.\n\n## Scope\n- Inspect uncommitted diffs in /home/juozas/trading-project-v3 against HEAD.\n- Verify `lane_unfreeze_D.flag` status and gitignore rules (ensure transient lane state is not committed by backups).\n- Validate `lane_unfreeze.py --status` reliability.\n- Commit the adjudicated state to HEAD.\n- Create a read-only snapshot (copy to /tmp or mode=ro URI) for downstream workers.\n\n## Acceptance Criteria\n- Working tree matches HEAD; no dirty residue.\n- Read-only snapshot created and accessible.\n- Waiver state (`dd8506e2`) confirmed present.\n\nAdoption: none\nDocs: none\nResearch-feedback: none\nGate: none"},
    {"id": "t_e1cea019", "body": "phase: research\nweb_searches: 3\nlead_ledger: attached\nanchor_note: Anchors from t_26d8eb94, D-lane audit, signals DB, structural/meta-harness scope.\nvendor_caveat: N/A (internal tooling)\nwiki_handoff_disposition: internal-only\n\n## Step 0 \u2014 KNOWLEDGE LOOKUP (verb-first, MANDATORY)\nThis card touches the knowledge wiki. Before any other action, query the index:\n  python3 ~/.hermes/scripts/wiki_index.py search \"pipeline-recent-versions D-lane audit signals\"\n(verbs: search | backlinks | related | outline | related_skills | goals_query)\n- Required reads via verbs: system/pipeline-recent-versions.md, system/vix-percentile-rank.md, systems/d-lane-shadow-chop-regime.md, systems/financial-advisor-pipeline (if present).\n- <=2 BM25 phrasing retries before any grep; grep only after 2 failed verb attempts AND with a concrete file target; record fallback reason.\n\n## Objective\nPerform a read-only structural and meta-harness audit of the D pipeline using the adjudicated snapshot.\n\n## Scope (S1)\n- Full dependency-chain map of v3 pipeline (scripts/, scanner/, analysts/, meta pipeline run_meta_pipeline.py, improvement_engine.py, b_decision_engine.py).\n- Theatre-component scan: identify hardcoded lookups, day-seeded RNG, fixed confidence values.\n- config.yaml utilization: verify every referenced key is present and valid.\n- Confidence-flow verification: ensure scores are consumed logically, not decorative.\n- Meta-loop health: check family census n_obs accumulation; verify prospective-watermark discard fault pattern (last_eval == created_at) for D families.\n- Waiver-state verification: confirm unfreeze flag, actuator 23:31 behavior, and backup-commit risk mitigation.\n\n## Acceptance Criteria\n- Dependency graph produced.\n- List of theatre components identified.\n- Config utilization report generated.\n- Meta-loop health verdict issued.\n- Waiver-state risks documented.\n\nAdoption: none\nDocs: none\nResearch-feedback: Report structural anomalies to t_26d8eb94 for potential refinement.\nGate: none\nGate: reviewer=user \u2014 D financial pipeline FULL audit: structural/meta-harness/analysis/signals-DB/sanitation/Discord + Class-A promotion queue (user pre-authorized, RE-SEQUENCED 2026-09-28: fires NOW, Lane-4 refinement parked)"},
    {"id": "t_89ff2358", "body": "phase: research\nweb_searches: 3\nlead_ledger: attached\nanchor_note: Anchors from t_26d8eb94, D-lane audit, signal-economics scope.\nvendor_caveat: N/A (internal tooling)\nwiki_handoff_disposition: internal-only\n\n## Step 0 \u2014 KNOWLEDGE LOOKUP (verb-first, MANDATORY)\nThis card touches the knowledge wiki. Before any other action, query the index:\n  python3 ~/.hermes/scripts/wiki_index.py search \"pipeline-recent-versions D-lane audit signals\"\n(verbs: search | backlinks | related | outline | related_skills | goals_query)\n- Required reads via verbs: system/pipeline-recent-versions.md, system/vix-percentile-rank.md, systems/d-lane-shadow-chop-regime.md, systems/financial-advisor-pipeline (if present).\n- <=2 BM25 phrasing retries before any grep; grep only after 2 failed verb attempts AND with a concrete file target; record fallback reason.\n\n## Objective\nPerform a read-only analysis-quality and signal-economics audit of the D pipeline using the adjudicated snapshot.\n\n## Scope (S2)\n- Evaluate trade signal production: entry points accuracy and timeliness.\n- Confidence calibration analysis: Brier score and Expected Calibration Error (ECE) assessment.\n- PSR/DSR (Price Support/Resistance) signal quality verification.\n- Calls/Puts selection symmetry analysis.\n- Review n>=30 gate compliance and resolved signal count (target 62).\n\n## Acceptance Criteria\n- Signal quality metrics calculated.\n- Calibration report generated.\n- Symmetry analysis completed.\n- Gate compliance verified.\n\nAdoption: none\nDocs: none\nResearch-feedback: Report signal degradation patterns to t_26d8eb94 for potential refinement.\nGate: none"},
    {"id": "t_21082df6", "body": "phase: research\nweb_searches: 3\nlead_ledger: attached\nanchor_note: Anchors from t_26d8eb94, wiki contradiction resolution, cron state.\nvendor_caveat: N/A (internal tooling)\nwiki_handoff_disposition: emit a wiki-keeper landing child\n\n## Step 0 \u2014 KNOWLEDGE LOOKUP (verb-first, MANDATORY)\nThis card touches the knowledge wiki. Before any other action, query the index:\n  python3 ~/.hermes/scripts/wiki_index.py search \"pipeline-recent-versions D-lane audit signals\"\n(verbs: search | backlinks | related | outline | related_skills | goals_query)\n- Required reads via verbs: system/pipeline-recent-versions.md, system/vix-percentile-rank.md, systems/d-lane-shadow-chop-regime.md, systems/financial-advisor-pipeline (if present).\n- <=2 BM25 phrasing retries before any grep; grep only after 2 failed verb attempts AND with a concrete file target; record fallback reason.\n\n## Objective\nResolve the known wiki contradiction regarding D-lane cron state and update documentation.\n\n## Scope\n- Read `jobs.json` live to establish AUTHORITATIVE cron state (enabled=True vs parked).\n- Compare against wiki claims in pipeline-recent-versions.md:20 and vix-percentile-rank.md:114.\n- Disposition stale wiki lines: update or annotate with correction.\n- Verify daily-full-sweep-D cron schedule (13:10 daily) matches live config.\n\n## Acceptance Criteria\n- Authoritative cron state documented.\n- Wiki pages updated to reflect live state.\n- Staleness contradiction resolved.\n\nAdoption: none\nDocs: Updated wiki pages reflecting live cron state.\nResearch-feedback: None\nGate: none"},
]

_R_REAL_CHILDREN = [  # verbatim, read-only dump of t_bec82259
    {"id": "t_13573906", "body": "phase: grounding\nConduct a forensic audit of the current decomposer logic and release-protocol handoff mechanism. Specifically: trace how triage bodies are parsed into child tasks, identify where stream sections (S1\u2013S6) are silently dropped, and verify why user-gate/sequencing stamps fail to propagate to all children. Simultaneously, audit the release-protocol session lifecycle to confirm the exact point where the archive-to-successor handoff fails when the orchestrator session terminates. Deliverables: a mapped failure tree showing exactly where scope loss and stamp leakage occur, plus a timeline of the dropped handoff incident. Acceptance criteria: clear identification of code/config roots for both incidents; no speculative fixes yet.\nAdoption: none\nDocs: none\nResearch-feedback: none\nGate: none"},
    {"id": "t_acf927d3", "body": "phase: research\nweb_searches: 3\nlead_ledger: attached\nanchor_note: Anchors from t_bec82259 incident logs; searches cover ephemeral orchestrator state persistence, atomic multi-step handoffs in CI/CD, and reliable fan-out decomposition patterns.\nvendor_caveat: When citing vendor-reported numbers, note they are vendor-reported.\nwiki_handoff_disposition: wiki-worthy -> emit a wiki-keeper landing child\n\nSynthesize findings from the grounding audit. Propose concrete rules for decomposer completeness: mandate that every named stream/scope section maps to \u22651 child or an explicit fold-in note. Define the stamp propagation contract: how user-gate and sequencing stamps must reach every child atomically, or trigger a per-child inheritance report. Recommend the architectural pattern for the release watchdog (cron vs event-driven) to ensure atomic archive->successor creation survives session drops. Deliverables: a structured proposal document outlining the new routing rules, stamp propagation mechanism, and watchdog architecture. Acceptance criteria: proposal explicitly addresses silent drops, stamp leakage, and session durability; ready for system-design.\nAdoption: none\nDocs: none\nResearch-feedback: none\nGate: none"},
    {"id": "t_17784ba2", "body": "phase: system-design\nTranslate the research proposal into a technical architecture. Design the decomposer patch: specify how the parser will validate stream coverage, generate a receipt comment mapping streams to children, and flag missing streams. Design the stamp propagator: specify how metadata (reviewer=user, BLOCKED-ON, supersede notes) will be injected into every child task object during fan-out. Design the release watchdog cron: specify the schedule, the atomic script that checks for pending archives, creates the successor triage card from the preserved body file, and arms the D-audit trigger. Include error handling, idempotency guarantees, and integration points with the existing orchestrator. Deliverables: architecture diagram/specification, data flow for stamp propagation, cron job definition, and decomposer validation logic. Acceptance criteria: design covers all three components, specifies interfaces, and ensures atomicity for the release handoff.\nAdoption: none\nDocs: none\nResearch-feedback: none\nGate: none"},
    {"id": "t_6fa2f4da", "body": "phase: engineering-planning\nBreak the system-design specification into executable implementation tickets. Map each component (decomposer validator, stamp propagator, watchdog cron) to specific files/modules in the repository. Define the step-by-step coding sequence, test scenarios, and deployment order. Ensure the plan accounts for backward compatibility with existing triage bodies and graceful degradation if the watchdog fails. Deliverables: a prioritized implementation backlog with clear acceptance criteria per sub-task, dependency graph, and rollback strategy. Acceptance criteria: plan is granular enough for direct execution; all critical paths identified; test coverage strategy defined.\nAdoption: none\nDocs: none\nResearch-feedback: none\nGate: none"},
    {"id": "t_09110241", "body": "phase: null\nReview the complete design and engineering plan for the decomposer hardening and release watchdog. Verify that the proposed stamp propagation mechanism covers all edge cases (e.g., partial fan-outs, concurrent updates), that the decomposer completeness checker prevents silent scope loss, and that the watchdog cron provides true session-independent atomicity. Approve or request revisions before implementation begins. Acceptance criteria: explicit user sign-off or documented revision requests; no hidden risks in the design.\nAdoption: none\nDocs: none\nResearch-feedback: none\nGate: requires-user-verdict on design approval before proceeding to implementation."},
    {"id": "t_49ae5a1c", "body": "phase: implementation\nExecute the engineering plan. Code the decomposer completeness validator: implement stream-to-child mapping logic, receipt comment generation, and missing-stream flagging. Code the stamp propagator: inject reviewer, blocking, and supersede metadata into every child task object during fan-out. Code the release watchdog cron: build the atomic archive-to-successor script, schedule it, and integrate it with the D-audit trigger arming logic. Write unit tests for each component. Deliverables: merged PRs containing the decomposer patch, stamp propagator, and watchdog cron, plus associated tests. Acceptance criteria: all components pass CI; decomposer logs stream mappings; stamps appear on all children; watchdog successfully creates successors without session dependency.\nAdoption: none\nDocs: none\nResearch-feedback: none\nGate: none"},
    {"id": "t_f4ecdef7", "body": "phase: testing\nValidate the implemented changes against the original incident scenarios. Run controlled decomposer fan-outs with multi-stream triage bodies to verify that no streams are silently dropped and that receipt comments are generated accurately. Inject user-gate and sequencing stamps into root tasks and verify 100% propagation to all children. Trigger the release watchdog cron manually and via schedule to confirm atomic archive->successor creation and D-audit arming. Document results and fix any regressions. Acceptance criteria: zero silent scope loss in test runs; 100% stamp propagation; watchdog executes flawlessly across simulated session drops; all tests green.\nAdoption: none\nDocs: none\nResearch-feedback: none\nGate: none\nGate: reviewer=user \u2014 Kanban orchestration hardening round 2: decomposer completeness + release-protocol automation (two live incidents 2026-09-27/28: overnight idle from dropped handoff, decomposer scope loss + stamp non-propagation)"},
    {"id": "t_29933582", "body": "## Step 0 \u2014 KNOWLEDGE LOOKUP (verb-first, MANDATORY)\nThis card touches the knowledge wiki. Before any other action, query the index:\n  python3 ~/.hermes/scripts/wiki_index.py search \"kanban orchestration baseline decomposer release watchdog\"\n(verbs: search | backlinks | related | outline | related_skills | goals_query)\n- <=2 BM25 phrasing retries before any grep; grep only after 2 failed verb attempts AND with a concrete file target; record fallback reason.\nUpdate the kanban orchestration wiki pages to reflect the new decomposer completeness rules, stamp propagation contract, and release watchdog protocol. Document the expected behavior for triage bodies, the automatic receipt comments, the metadata injection process, and the cron schedule/integration details. Ensure the wiki accurately describes the guardrails against silent scope loss and session-dropped handoffs. Deliverables: updated wiki markdown files reflecting all new operational rules and architecture. Acceptance criteria: wiki pages fully synchronized with the implemented system; clear runbooks for operators; no stale references to old session-dependent protocols.\nAdoption: none\nDocs: commit updated wiki pages to the knowledge repository.\nResearch-feedback: none\nGate: none"},
    {"id": "t_a14757c9", "body": "phase: null\nDraft the rollout plan and user notification for the decomposer hardening and release watchdog changes. Outline the migration path for existing workflows, explain the new automatic behaviors (stream mapping receipts, stamp propagation, cron arming), and define success metrics for monitoring. Prepare a concise announcement for the team/user detailing what changed, why it matters, and how to verify the improvements. Deliverables: rollout checklist, user-facing announcement draft, and monitoring dashboard queries. Acceptance criteria: rollout plan includes rollback steps; announcement clearly communicates the new deterministic behavior; monitoring setup verified.\nAdoption: draft and distribute rollout plan and user notification.\nDocs: none\nResearch-feedback: none\nGate: none"},
    {"id": "t_04c12d1e", "body": "## Step 0 \u2014 KNOWLEDGE LOOKUP (verb-first, MANDATORY)\nThis card touches the knowledge wiki. Before any other action, query the index:\n  python3 ~/.hermes/scripts/wiki_index.py search \"<your query>\"\n(verbs: search | backlinks | related | outline | related_skills | goals_query)\n- <=2 BM25 phrasing retries (strip qualifiers, use single tokens/aliases) BEFORE any grep.\n- grep/search_files/read_file over ~/knowledge/wiki is legal only after 2 failed verb\n  attempts AND with a concrete file target \u2014 record which attempts failed.\n- Structure questions: `outline <slug>` first, not a whole-page read.\n- Provenance (\"what grounds X\"): `related` + `backlinks`, not grep.\n- Do NOT use `execute_code`, `python3 -c`, or `python3 - <<'PY'` \u2014 all three are\n  auto-denied in a worker session. `python3 <scriptfile>` is permitted.\nThen record `wiki_lookup: {verbs:[...], fallback_reason: null|<text>}` in completion metadata.\n\n**Goal**\nLand the wiki page(s) for this card's gated parent research card (\"Propose decomposer completeness rules & stamp propagation architecture\") and wire them into the knowledge graph.\n\n**Approach**\n- Read the parent card's handoff (`kanban_show` on the parent: summary + artifacts) instead of re-running the research.\n- Query the index verb-first (Step 0 above); seed the search with the parent's `anchor_note` vocabulary.\n- Ingest into `~/knowledge/wiki` with typed frontmatter and wikilinks to the related pages; run `python3 ../wiki-validate.py` before committing.\n- Commit + push `He1i05/knowledge-wiki` per the vault convention (never edit `raw/`).\n\n**Acceptance criteria**\n- [ ] at least one wiki page created/updated under `~/knowledge/wiki` for this research output\n- [ ] page passes `wiki-validate.py`\n- [ ] wikilinks connect it to the existing graph (not an orphan page)\n- [ ] wiki commit sha recorded in completion metadata\n\n**Out of scope**\n- Re-doing the research (the parent card owns it); editing the parent card's own artifacts.\n\nDocs: wiki landing for the gated parent research card (this card)\nResearch-feedback: parent declared wiki-handoff disposition: wiki-worthy -> emit a wiki-keeper landing child\n\nAdoption: none\nGate: none"},
]


def _naive_diff_gaps(root_body: str, kids: list) -> list[str]:
    """The REJECTED oracle, reproduced faithfully: overlap the stream-heading
    tokens with the child bodies. Kept only to assert it false-passes."""
    joined = " ".join(
        (k.get("body") or "") if isinstance(k, dict) else str(k) for k in kids
    ).lower()
    gaps = []
    for s in streams.declared_streams(root_body):
        tokens = [
            t for t in re.findall(r"[a-z]{4,}", s.description.lower())
            if t not in {"audit", "with"}
        ]
        if not any(tok in joined for tok in tokens):
            gaps.append(s.tag)
    return gaps


def _create(conn, title, body, parents=()):
    return kb.create_task(conn, title=title, body=body, parents=list(parents))


# --------------------------------------------------------------------------- #
# 0 — the fixtures themselves: exact measured offsets (plan §4 Layer 3)
# --------------------------------------------------------------------------- #

def test_fixtures_reproduce_the_measured_offsets_and_sides_of_the_cut():
    s = _real_shaped_root()
    assert {x.tag: x.offset for x in streams.declared_streams(s)} == _S_OFFSETS
    assert streams.declared_stream_tags(s[: spec.PROMPT_BODY_LIMIT]) == ["S1", "S2"]

    r = _r_shaped_root()
    assert {x.tag: x.offset for x in streams.declared_streams(r)} == _R_OFFSETS
    # EVERY R-stream is beyond the cut: the slice declares nothing.
    assert streams.declared_stream_tags(r[: spec.PROMPT_BODY_LIMIT]) == []
    assert all(off > spec.PROMPT_BODY_LIMIT for off in _R_OFFSETS.values())


# --------------------------------------------------------------------------- #
# 1 — AC1: the t_26d8eb94 shape, S1/S2 covered -> exactly 4 gaps for S3-S6
# --------------------------------------------------------------------------- #

def test_ac1_four_gap_lines_for_s3_to_s6_when_s1_s2_are_covered():
    """Plan §2 W-7 AC1, fixture from G7."""
    body = _real_shaped_root()
    children = [
        {"id": "t_c1", "body": "phase: research\ncovers: S1\nmap the chain"},
        {"id": "t_c2", "body": "phase: research\ncovers: S2\nthe economics"},
    ]
    receipt = cov.fanout_receipt(body, children, root_id="t_26d8eb94")
    text = receipt.text()

    assert [g.tag for g in receipt.gaps] == ["S3", "S4", "S5", "S6"]
    assert text.count(cov.GAP_LINE_PREFIX) == 4
    assert cov.LOUD_FAIL in text
    # The mapping is real, not just a count: the covered streams name their child.
    assert "COVERED by `t_c1`" in text
    assert "COVERED by `t_c2`" in text
    # The boundary fact is stated (2 inside the slice, 4 beyond it).
    assert "CUT at char 3999 of" in text
    assert "4 BEYOND it (S3, S4, S5, S6)" in text


def test_ac1_real_incident_children_declare_nothing_so_all_six_are_gaps():
    """The MEASURED state of t_26d8eb94: 4 real children, 0 declarations.

    This is the honest answer the incident demanded and the naive diff denies
    (it reports 0 gaps on the same input — see AC3).
    """
    receipt = cov.fanout_receipt(
        _real_shaped_root(), _S_REAL_CHILDREN, root_id="t_26d8eb94",
    )
    assert [g.tag for g in receipt.gaps] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert receipt.text().count(cov.GAP_LINE_PREFIX) == 6
    assert receipt.declaring_children == 0
    assert receipt.children_total == 4


# --------------------------------------------------------------------------- #
# 2 — AC2: the t_bec82259 shape, ALL five streams are gaps
# --------------------------------------------------------------------------- #

def test_ac2_all_five_r_streams_are_gaps():
    """Plan §2 W-7 AC2, fixture from G8 (8501-char body, every stream past 3999)."""
    body = _r_shaped_root()
    assert len(body) > spec.PROMPT_BODY_LIMIT

    children = [
        {"id": f"t_k{i}", "body": f"phase: research\nplain child {i} — no declarations"}
        for i in range(10)
    ]
    receipt = cov.fanout_receipt(body, children, root_id="t_bec82259")
    text = receipt.text()

    assert [g.tag for g in receipt.gaps] == ["R1", "R2", "R3", "R4", "R5"]
    assert text.count(cov.GAP_LINE_PREFIX) == 5
    # ALL five are beyond the cut: 0 declared inside the slice.
    assert receipt.inside_cut_tags == []
    assert receipt.beyond_cut_tags == ["R1", "R2", "R3", "R4", "R5"]
    assert "0 declared stream(s) inside the slice" in text
    assert "10 child(ren) emitted" in text


def test_ac2_real_incident_children_all_five_are_gaps():
    """Against the FROZEN real 10-children set (0/10 declare coverage)."""
    receipt = cov.fanout_receipt(
        _r_shaped_root(), _R_REAL_CHILDREN, root_id="t_bec82259",
    )
    assert [g.tag for g in receipt.gaps] == ["R1", "R2", "R3", "R4", "R5"]
    assert receipt.declaring_children == 0
    assert receipt.children_total == 10


# --------------------------------------------------------------------------- #
# 3 — AC3: the REJECTED diff oracle is asserted at false-pass (negative fixture)
# --------------------------------------------------------------------------- #

def test_ac3_naive_diff_false_passes_on_the_real_t26d8eb94_children():
    """Plan §2 W-7 AC3 + §4 Layer 4 — the regression guard against a future
    "simplification" back into a keyword diff.

    Measured on the REAL children: the diff reports every one of the six
    missing streams COVERED. The plan's headline number is the four streams it
    names explicitly (S3–S6) — asserted exactly, so the published measurement
    keeps reproducing.
    """
    body = _real_shaped_root()
    bodies = [c["body"] for c in _S_REAL_CHILDREN]

    gaps = [g.tag for g in cov.fanout_receipt(body, bodies).gaps]
    assert gaps == ["S1", "S2", "S3", "S4", "S5", "S6"]

    diff_gaps = _naive_diff_gaps(body, bodies)
    assert diff_gaps == []                       # 6/6 false-pass
    missing = {"S3", "S4", "S5", "S6"}
    assert missing <= set(gaps)                  # the oracle catches all four
    assert not (missing & set(diff_gaps))         # the diff catches none
    assert len(missing) == 4                      # the 4/4 false-pass number


def test_ac3b_naive_diff_false_passes_on_the_real_tbec82259_children():
    """The same comparison on the second root's REAL children: 5/5 false-pass."""
    body = _r_shaped_root()
    bodies = [c["body"] for c in _R_REAL_CHILDREN]

    gaps = [g.tag for g in cov.fanout_receipt(body, bodies).gaps]
    assert gaps == ["R1", "R2", "R3", "R4", "R5"]
    assert _naive_diff_gaps(body, bodies) == []  # 5/5 false-pass


# --------------------------------------------------------------------------- #
# 4 — AC4: the FULL body, and why the prompt slice passes vacuously
# --------------------------------------------------------------------------- #

def test_ac4_prompt_slice_reports_zero_streams_vacuous_pass():
    """Plan §2 W-7 AC4 (rule C-2). A receipt built from the slice is vacuous."""
    full = _r_shaped_root()
    sliced = full[: spec.PROMPT_BODY_LIMIT]

    # The slice declares NOTHING on this root — so a receipt over it cannot
    # report a single gap, no matter what the fan-out did.
    assert streams.declared_stream_tags(sliced) == []
    vacuous = cov.fanout_receipt(sliced, [{"id": "t_k", "body": "plain"}], root_id="t_bec82259")
    assert vacuous.declared_total == 0
    assert vacuous.gaps == []
    assert cov.NO_STREAMS_LINE in vacuous.text()
    assert cov.LOUD_FAIL not in vacuous.text()

    # The FULL body (what read_root_body returns) declares all five.
    honest = cov.fanout_receipt(full, [{"id": "t_k", "body": "plain"}], root_id="t_bec82259")
    assert honest.declared_total == 5
    assert [g.tag for g in honest.gaps] == ["R1", "R2", "R3", "R4", "R5"]


def test_read_root_body_returns_the_full_body_not_the_slice(kanban_home):
    """``read_root_body`` is the call shape that guarantees rule C-2."""
    body = _r_shaped_root()
    with kbc.connect() as conn:
        tid = _create(conn, "root", body)
        got = cov.read_root_body(conn, tid)
    assert got == body
    assert len(got) > spec.PROMPT_BODY_LIMIT
    assert streams.declared_stream_tags(got) == ["R1", "R2", "R3", "R4", "R5"]
    # …and a vanished root is None, never a fabricated body.
    with kbc.connect() as conn:
        assert cov.read_root_body(conn, "t_does_not_exist") is None


def test_prompt_visible_end_matches_the_streams_appendix_cut():
    """The receipt's inside/beyond boundary is the SAME boundary the prompt
    appendix states — asserted against ``kanban_streams``, not re-derived."""
    body = _r_shaped_root()
    appendix = streams.coverage_appendix(body, cut=spec.PROMPT_BODY_LIMIT - 1)
    assert f"CUT at char {spec.PROMPT_BODY_LIMIT - 1} of {len(body)}" in appendix

    receipt = cov.fanout_receipt(body, [])
    assert receipt.cut == spec.PROMPT_BODY_LIMIT - 1
    assert receipt.truncated is True
    # A body that fits is not "truncated" and nothing is hidden.
    short = "## SCOPE STREAMS\nS1 tiny\n"
    assert cov.fanout_receipt(short, []).truncated is False
    assert cov.fanout_receipt(short, []).inside_cut_tags == ["S1"]


# --------------------------------------------------------------------------- #
# 5 — fold-in accounting, and no invented gaps
# --------------------------------------------------------------------------- #

def test_fold_in_closes_a_gap_and_is_reported_as_such():
    body = (
        "## SCOPE STREAMS\n"
        "S1 structural audit\n"
        "S2 economics audit\n"
        "S3 signals-db audit\n"
        "S4 sanitation audit\n"
    )
    children = [
        {"id": "t_a", "body": "covers: S1\n"},
        {"id": "t_b", "body": "covers: S2\nFold-in: S3 — absorbed by the S2 census\n"},
        {"id": "t_c", "body": "Fold-in: S4 (into t_a) — descoped by the user at triage\n"},
    ]
    receipt = cov.fanout_receipt(body, children, root_id="t_r")
    text = receipt.text()

    assert receipt.gaps == []
    assert receipt.covered_total == 4
    assert cov.LOUD_FAIL not in text
    assert "FOLDED IN by `t_b`" in text
    assert "FOLDED IN by `t_c`" in text
    assert "No gaps." in text
    assert receipt.declaring_children == 3


def test_a_root_with_no_declared_streams_gets_no_invented_gap():
    """Rule C-1/plan §7.1: absence is the signal. The 274-card `Gate:`-line
    population must not be turned into fabricated gaps."""
    body = "## Goal\nan ordinary card with a Gate: line and no stream section at all\n"
    receipt = cov.fanout_receipt(body, [{"id": "t_x", "body": "child"}], root_id="t_plain")
    text = receipt.text()

    assert receipt.declared_total == 0
    assert receipt.gaps == []
    assert cov.NO_STREAMS_LINE in text
    assert "invents no gaps" in text
    assert cov.GAP_LINE_PREFIX not in text
    assert cov.LOUD_FAIL not in text
    assert "no streams declared (0 gaps)" in text


def test_streamless_root_text_states_the_boundary_fact_too():
    long_plain = "ordinary long card\n" + ("filler. " * 700)
    text = cov.fanout_receipt(long_plain, [], root_id="t_long").text()
    assert cov.NO_STREAMS_LINE in text
    assert "CUT at char" in text  # the boundary fact is still stated


# --------------------------------------------------------------------------- #
# 6 — the structured half (task_events payload; no schema change)
# --------------------------------------------------------------------------- #

def test_payload_carries_the_documented_keys_and_mapping_rows():
    body = "## SCOPE STREAMS\nS1 a\nS2 b\n"
    receipt = cov.fanout_receipt(
        body, [{"id": "t_a", "body": "covers: S1"}], root_id="t_r",
    )
    payload = receipt.payload()
    for key in ("root_id", "declared_streams", "children_emitted", "gaps", "outside_slice"):
        assert key in payload, key
    assert payload["root_id"] == "t_r"
    assert payload["declared_streams"] == ["S1", "S2"]
    assert payload["gaps"] == ["S2"]
    assert payload["covered"] == 1
    assert payload["children_emitted"] == 1
    assert payload["mappings"][0] == {
        "tag": "S1", "offset": payload["mappings"][0]["offset"],
        "seen_by_llm": True, "disposition": cov.DISPOSITION_COVERED, "child_id": "t_a",
    }


def test_receipt_writes_no_schema_change(kanban_home):
    """No new tables/columns — a new ``task_events.kind`` VALUE only."""
    with kbc.connect() as conn:
        before = {
            t: [r["name"] for r in conn.execute(f"PRAGMA table_info({t})")]
            for t in ("tasks", "task_events", "task_comments")
        }
    with kbc.connect() as conn:
        tid = _create(conn, "root", "## SCOPE STREAMS\nS1 a\n")
        outcome = cov.emit_receipt(tid, conn=conn)
    assert outcome.ok
    with kbc.connect() as conn:
        after = {
            t: [r["name"] for r in conn.execute(f"PRAGMA table_info({t})")]
            for t in ("tasks", "task_events", "task_comments")
        }
    assert after == before


# --------------------------------------------------------------------------- #
# 7 — board I/O, end-to-end and hermetic
# --------------------------------------------------------------------------- #

def test_emit_receipt_comments_the_root_and_emits_the_event(kanban_home):
    """The deliverable: a receipt comment on the ROOT + one structured event."""
    body = _real_shaped_root()
    with kbc.connect() as conn:
        root = _create(conn, "quest root", body)
        _create(conn, "S1 child", "covers: S1\nmap it", parents=[root])
        _create(conn, "S2 child", "covers: S2\nthe economics", parents=[root])
        _create(conn, "S3-S6 packet", "covers: S3\nFold-in: S4 — absorbed\n"
                                      "Fold-in: S5 — absorbed\nFold-in: S6 — absorbed",
                parents=[root])

    with kbc.connect() as conn:
        outcome = cov.emit_receipt(root, conn=conn)
    assert outcome.ok, outcome.reason
    assert outcome.declared_total == 6
    assert outcome.gaps == 0
    assert outcome.children_total == 3
    assert outcome.comment_id > 0

    with kbc.connect() as conn:
        comments = conn.execute(
            "SELECT author, body FROM task_comments WHERE task_id = ?", (root,),
        ).fetchall()
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
            (root, cov.RECEIPT_EVENT_KIND),
        ).fetchall()
    assert len(comments) == 1
    assert comments[0]["author"] == cov.RECEIPT_AUTHOR
    assert cov.RECEIPT_HEADING in comments[0]["body"]
    assert "Every declared stream is accounted for (6/6). No gaps." in comments[0]["body"]
    assert len(events) == 1
    payload = jsonlib.loads(events[0]["payload"])
    assert payload["gaps"] == []
    assert payload["declared_streams"] == ["S1", "S2", "S3", "S4", "S5", "S6"]


def test_emit_receipt_reports_gaps_loudly_on_the_incident_shape(kanban_home):
    body = _real_shaped_root()
    with kbc.connect() as conn:
        root = _create(conn, "quest root", body)
        _create(conn, "child with no declarations", "phase: research\njust do it",
                parents=[root])

    with kbc.connect() as conn:
        outcome = cov.emit_receipt(root, conn=conn)
    assert outcome.ok and outcome.gaps == 6
    with kbc.connect() as conn:
        text = conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ?", (root,),
        ).fetchone()["body"]
    assert text.count(cov.GAP_LINE_PREFIX) == 6
    assert cov.LOUD_FAIL in text


def test_emit_receipt_changes_nothing_else_on_the_board(kanban_home):
    """Advisory by construction: comment + event only. No status, body or link
    is touched (the shape W-9's AC3 asserts, pinned here too)."""
    body = _real_shaped_root()
    with kbc.connect() as conn:
        root = _create(conn, "quest root", body)
        c1 = _create(conn, "kid one", "covers: S1", parents=[root])
        c2 = _create(conn, "kid two", "plain", parents=[root])
        before = {
            cid: (
                kb.get_task(conn, cid).status,
                kb.get_task(conn, cid).body,
                kb.get_task(conn, cid).assignee,
            ) for cid in (root, c1, c2)
        }
        before_links = sorted(
            tuple(r) for r in conn.execute("SELECT parent_id, child_id FROM task_links")
        )

    with kbc.connect() as conn:
        cov.emit_receipt(root, conn=conn)

    with kbc.connect() as conn:
        after = {
            cid: (
                kb.get_task(conn, cid).status,
                kb.get_task(conn, cid).body,
                kb.get_task(conn, cid).assignee,
            ) for cid in (root, c1, c2)
        }
        after_links = sorted(
            tuple(r) for r in conn.execute("SELECT parent_id, child_id FROM task_links")
        )
    assert after == before
    assert after_links == before_links


def test_emit_receipt_is_advisory_on_a_missing_root(kanban_home):
    """Never raises into a committed fan-out (plan §2 W-9 AC3 / rule C-4)."""
    with kbc.connect() as conn:
        outcome = cov.emit_receipt("t_not_a_real_card", conn=conn)
    assert outcome.ok is False
    assert "not found" in outcome.reason
    assert outcome.gaps == 0


def test_emit_receipt_on_a_root_with_no_children_is_still_honest(kanban_home):
    body = _real_shaped_root()
    with kbc.connect() as conn:
        root = _create(conn, "quest root", body)
    with kbc.connect() as conn:
        outcome = cov.emit_receipt(root, conn=conn)
    assert outcome.ok
    assert outcome.children_total == 0
    assert outcome.gaps == 6  # every declared stream is unaccounted for


def test_building_twice_is_byte_identical():
    """Deterministic: the receipt is a pure function of (root body, children)."""
    body = _real_shaped_root()
    kids = [{"id": "t_a", "body": "covers: S1"}, {"id": "t_b", "body": "plain"}]
    assert cov.build_receipt(body, kids, root_id="t_r") == cov.build_receipt(
        body, kids, root_id="t_r",
    )


def test_fanout_child_ids_unions_the_decomposed_event_and_live_links(kanban_home):
    """Both sources matter on the measured roots: t_26d8eb94's event lists 4 ids
    while the board carries 3 more (S3/S4/S5) that the S-audit verdicts live on;
    t_bec82259's event lists 10 while the live board shows 1."""
    from hermes_cli.kanban_db import _append_event

    with kbc.connect() as conn:
        root = _create(conn, "quest root", "## SCOPE STREAMS\nS1 a\n")
        ev_child = _create(conn, "in the event only", "body")
        link_child = _create(conn, "in the links only", "body", parents=[root])
        _append_event(conn, root, "decomposed",
                      {"child_ids": [ev_child], "root_assignee": "default"})
        ids = cov.fanout_child_ids(conn, root)
    assert ids == [ev_child, link_child]           # event first, then links
    assert len(ids) == len(set(ids))               # de-duplicated


def test_fanout_child_ids_falls_back_to_links_when_the_event_is_absent(kanban_home):
    with kbc.connect() as conn:
        root = _create(conn, "quest root", "## SCOPE STREAMS\nS1 a\n")
        kid = _create(conn, "kid", "covers: S1", parents=[root])
        assert cov.fanout_child_ids(conn, root) == [kid]


def test_fanout_child_ids_drops_an_id_whose_row_is_gone(kanban_home):
    """A child row that no longer exists cannot be read; its streams stay gaps
    rather than being silently treated as declared."""
    body = "## SCOPE STREAMS\nS1 a\nS2 b\n"
    with kbc.connect() as conn:
        root = _create(conn, "quest root", body)
        kid = _create(conn, "kid", "covers: S1", parents=[root])
        conn.execute("DELETE FROM tasks WHERE id = ?", (kid,))
        children = cov.load_children(conn, cov.fanout_child_ids(conn, root))
    assert children == []
    assert [g.tag for g in cov.fanout_receipt(body, children).gaps] == ["S1", "S2"]


# --------------------------------------------------------------------------- #
# 8 — the oracle is the MODULE's (no drift between receipt and convention)
# --------------------------------------------------------------------------- #

def test_receipt_gaps_agree_with_kanban_streams_missing_coverage():
    """The receipt must never re-implement the oracle: for every fixture here,
    receipt gaps == ``kanban_streams.missing_coverage`` (the W-8 primitive)."""
    cases = [
        (_real_shaped_root(), [{"id": "t_a", "body": "covers: S1"}, {"id": "t_b", "body": "covers: S2"}]),
        (_real_shaped_root(), _S_REAL_CHILDREN),
        (_r_shaped_root(), _R_REAL_CHILDREN),
        (_r_shaped_root(), [{"id": "t_a", "body": "covers: R1\nFold-in: R2 — absorbed"}]),
        ("## SCOPE STREAMS\nS1 a\nS2 b\n", [{"id": "t_a", "body": "covers: S1\ncovers: S2"}]),
        ("plain body, no streams", [{"id": "t_a", "body": "covers: S1"}]),
    ]
    for body, kids in cases:
        receipt = cov.fanout_receipt(body, kids)
        expected = streams.missing_coverage(body, [k["body"] for k in kids])
        assert [g.tag for g in receipt.gaps] == expected, body[:40]


def test_receipt_uses_the_streams_vocabulary_literals():
    """No second spelling of the declaration keys anywhere in the receipt."""
    body = _real_shaped_root()
    text = cov.fanout_receipt(body, [], root_id="t_r").text()
    assert f"`{streams.COVERS_KEY}: <tag>`" in text
    assert f"`{streams.FOLDIN_KEY}: <tag> — <why>`" in text
    assert streams.NO_STREAMS_DECLARED in cov.fanout_receipt("plain", []).text()


# --------------------------------------------------------------------------- #
# 9 — integration: a real >4000-char fan-out, then the receipt on the root
# --------------------------------------------------------------------------- #

def _decompose(kanban_home, *, root_body: str, llm_payload: dict) -> str:
    """Create a hermetic triage root and fan it out through the REAL decomposer
    (only the LLM call is mocked). Returns the root id."""
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace

    from hermes_cli import kanban_decompose as decomp

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest", body=root_body, triage=True)

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = jsonlib.dumps(llm_payload)
    names = ["orchestrator", "engineer", "researcher"]
    fake = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    patches = [
        patch("hermes_cli.profiles.list_profiles", return_value=fake),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value="orchestrator"),
    ]
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", return_value=resp):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    return tid


def test_end_to_end_a_4000char_root_fans_out_and_the_receipt_lands_on_the_root(
    kanban_home,
):
    """The quest's deliverable shape, exercised through the REAL decomposer.

    A >4000-char triage body is decomposed with the production code path (only
    the LLM call is mocked), W-8 folds each child's declared coverage into its
    persisted body, and the receipt is then built from the PERSISTED rows —
    proving the full-body read plus the board I/O work against a real fan-out,
    not just hand-built dicts.
    """
    body = _real_shaped_root()
    assert len(body) > spec.PROMPT_BODY_LIMIT

    payload = {
        "fanout": True,
        "rationale": "six-stream split with a consolidation packet",
        "tasks": [
            {"title": "S1 structural audit", "body": "phase: research\nmap the chain",
             "assignee": "researcher", "phase": "research", "parents": [], "covers": ["S1"]},
            {"title": "S2 economics audit", "body": "phase: research\nthe economics",
             "assignee": "researcher", "phase": "research", "parents": [], "covers": ["S2"]},
            {"title": "S3-S6 consolidated packet", "body": "phase: research\npacket",
             "assignee": "researcher", "phase": "research", "parents": [0, 1],
             "covers": ["S3"],
             "fold_in": {"S4": "absorbed by the sanitation census",
                         "S5": "digest render contract folded in",
                         "S6": "promotion queue is the packet's last section"}},
        ],
    }
    root = _decompose(kanban_home, root_body=body, llm_payload=payload)

    # The receipt runs against the PERSISTED board state, exactly as W-9 will.
    with kbc.connect() as conn:
        outcome = cov.emit_receipt(root, conn=conn)
    assert outcome.ok, outcome.reason
    assert outcome.declared_total == 6
    # Children resolved from the root's own `decomposed` event + live links.
    assert outcome.children_total == 3
    assert outcome.gaps == 0, "every stream is declared (covers: or Fold-in:)"

    with kbc.connect() as conn:
        comment = conn.execute(
            "SELECT author, body FROM task_comments WHERE task_id = ? AND author = ? "
            "ORDER BY id DESC LIMIT 1", (root, cov.RECEIPT_AUTHOR),
        ).fetchone()
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
            (root, cov.RECEIPT_EVENT_KIND),
        ).fetchone()
    assert comment is not None
    assert "Every declared stream is accounted for (6/6). No gaps." in comment["body"]
    assert f"CUT at char {spec.PROMPT_BODY_LIMIT - 1} of {len(body)}" in comment["body"]
    payload_ev = jsonlib.loads(event["payload"])
    assert payload_ev["gaps"] == []
    assert payload_ev["outside_slice"] == ["S3", "S4", "S5", "S6"]
    assert payload_ev["mappings"][0]["child_id"].startswith("t_")


def test_end_to_end_a_fanout_that_declares_nothing_receipts_loud_gaps(kanban_home):
    """The same path with a non-compliant reply: the receipt reports the gaps
    (F3) instead of trusting the writer — and still never aborts anything."""
    body = _real_shaped_root()
    payload = {
        "fanout": True,
        "rationale": "no declarations at all",
        "tasks": [
            {"title": "do the thing", "body": "phase: implementation\ncode it",
             "assignee": "engineer", "phase": "implementation", "parents": []},
        ],
    }
    root = _decompose(kanban_home, root_body=body, llm_payload=payload)
    with kbc.connect() as conn:
        outcome = cov.emit_receipt(root, conn=conn)
    assert outcome.ok, outcome.reason
    assert outcome.declared_total == 6
    assert outcome.gaps == 6
    with kbc.connect() as conn:
        text = conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ? AND author = ? "
            "ORDER BY id DESC LIMIT 1", (root, cov.RECEIPT_AUTHOR),
        ).fetchone()["body"]
    assert text.count(cov.GAP_LINE_PREFIX) == 6
    assert cov.LOUD_FAIL in text


# --------------------------------------------------------------------------- #
# 10 — the leak guard: nothing here touches the live board
# --------------------------------------------------------------------------- #

def test_no_db_write_leaks_outside_the_hermetic_home(kanban_home):
    resolved = str(kb.kanban_db_path().resolve())
    assert str(kanban_home.resolve()) in resolved


def test_receipt_only_writes_the_comment_and_its_event(kanban_home):
    """Exactly one comment row and one receipt event per run — no extra rows."""
    body = _real_shaped_root()
    with kbc.connect() as conn:
        root = _create(conn, "quest root", body)
        _create(conn, "kid", "plain", parents=[root])
        before_events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        before_comments = conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0]
    with kbc.connect() as conn:
        cov.emit_receipt(root, conn=conn)
    with kbc.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == before_comments + 1
        # comment + its `commented` event + the receipt event
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == before_events + 2


# --------------------------------------------------------------------------- #
# 11 — W-9 (t_78427b7f): the call site inside _apply_fanout fires the receipt
# --------------------------------------------------------------------------- #

def test_w9_apply_fanout_files_the_receipt_on_the_root_without_an_extra_call(
    kanban_home,
):
    """W-9 AC1: after a real fan-out, the receipt comment is ALREADY on the root
    and names every declared stream — the fan-out itself filed it (the W-9 call
    site). Re-running `emit_receipt` by hand is NOT required and would be a
    second comment."""
    body = _real_shaped_root()
    payload = {
        "fanout": True,
        "rationale": "six-stream split with a consolidation packet",
        "tasks": [
            {"title": "S1 structural audit", "body": "phase: research\nmap the chain",
             "assignee": "researcher", "phase": "research", "parents": [], "covers": ["S1"]},
            {"title": "S2 economics audit", "body": "phase: research\nthe economics",
             "assignee": "researcher", "phase": "research", "parents": [], "covers": ["S2"]},
            {"title": "S3-S6 consolidated packet", "body": "phase: research\npacket",
             "assignee": "researcher", "phase": "research", "parents": [0, 1],
             "covers": ["S3"],
             "fold_in": {"S4": "absorbed", "S5": "folded", "S6": "last section"}},
        ],
    }
    root = _decompose(kanban_home, root_body=body, llm_payload=payload)

    with kbc.connect() as conn:
        comments = conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ? AND author = ?",
            (root, cov.RECEIPT_AUTHOR),
        ).fetchall()
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
            (root, cov.RECEIPT_EVENT_KIND),
        ).fetchall()
    # Exactly ONE receipt — filed by the fan-out, not one-per-reader.
    assert len(comments) == 1
    assert len(events) == 1
    text = comments[0]["body"]
    assert cov.RECEIPT_HEADING in text
    assert "Every declared stream is accounted for (6/6). No gaps." in text
    payload_ev = jsonlib.loads(events[0]["payload"])
    assert payload_ev["declared_streams"] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert payload_ev["gaps"] == []
    # The exact set THIS fan-out emitted — passed as `child_ids=`, not re-derived.
    assert payload_ev["children_emitted"] == 3


def test_w9_apply_fanout_receipt_reports_gaps_and_never_aborts_the_graph(kanban_home):
    """W-9 AC3: a non-compliant fan-out still commits; the receipt reports the
    gaps LOUDLY and the root still moved triage -> todo."""
    body = _real_shaped_root()
    payload = {
        "fanout": True,
        "rationale": "no coverage declarations",
        "tasks": [
            {"title": "do the thing", "body": "phase: implementation\ncode it",
             "assignee": "engineer", "phase": "implementation", "parents": []},
        ],
    }
    root = _decompose(kanban_home, root_body=body, llm_payload=payload)
    with kbc.connect() as conn:
        text = conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ? AND author = ? "
            "ORDER BY id DESC LIMIT 1", (root, cov.RECEIPT_AUTHOR),
        ).fetchone()["body"]
        root_status = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (root,)
        ).fetchone()["status"]
    assert text.count(cov.GAP_LINE_PREFIX) == 6
    assert cov.LOUD_FAIL in text
    # The root still committed (triage -> todo) — the receipt did not abort it.
    assert root_status == "todo"


def test_w9_apply_fanout_receipt_failure_leaves_the_graph_byte_identical(kanban_home):
    """W-9 AC3: an exception inside the receipt generator is advisory-only — the
    committed graph (child count, statuses, bodies, assignees) is byte-identical
    before and after the failed call.

    Proved by capturing the graph at the exact moment ``emit_receipt`` is invoked
    (it runs post-commit, inside ``_apply_fanout``) and comparing to the graph
    after ``decompose_task`` returns: the failure changed nothing."""
    from unittest.mock import patch

    body = _real_shaped_root()
    payload = {
        "fanout": True,
        "rationale": "two children",
        "tasks": [
            {"title": "a", "body": "phase: research\na", "assignee": "researcher",
             "phase": "research", "parents": [], "covers": ["S1"]},
            {"title": "b", "body": "phase: research\nb", "assignee": "researcher",
             "phase": "research", "parents": [], "covers": ["S2"]},
        ],
    }

    captured: dict = {}

    def _snapshot_then_boom(*args, **kwargs):
        with kbc.connect() as conn:
            captured["rows"] = [
                (r["id"], r["status"], r["body"], r["assignee"], r["current_step_key"])
                for r in conn.execute(
                    "SELECT id, status, body, assignee, current_step_key FROM tasks ORDER BY id"
                ).fetchall()
            ]
        raise RuntimeError("receipt exploded")

    with patch("hermes_cli.kanban_coverage.emit_receipt", side_effect=_snapshot_then_boom):
        root = _decompose(kanban_home, root_body=body, llm_payload=payload)

    # `_decompose` already asserts the outcome is ok — the receipt failure was
    # caught/logged, never surfaced as a fan-out failure.
    assert root
    with kbc.connect() as conn:
        after = [
            (r["id"], r["status"], r["body"], r["assignee"], r["current_step_key"])
            for r in conn.execute(
                "SELECT id, status, body, assignee, current_step_key FROM tasks ORDER BY id"
            ).fetchall()
        ]
    # NOTE: only the TASK rows are compared — the post-commit M3 capability and
    # wiki-landing audits legitimately append their own events after the receipt
    # call, so a global event count is not receipt-scoped. The graph is.
    assert after == captured["rows"], "the committed graph must be byte-identical"
    # The failed receipt wrote NOTHING: no receipt comment, no receipt event.
    with kbc.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM task_comments WHERE author = ?", (cov.RECEIPT_AUTHOR,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind = ?", (cov.RECEIPT_EVENT_KIND,)
        ).fetchone()[0] == 0
