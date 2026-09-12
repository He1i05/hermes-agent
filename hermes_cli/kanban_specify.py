"""Kanban triage specifier — flesh out a one-liner into a real spec.

``hermes kanban specify [task_id | --all]`` asks the auxiliary LLM for a
tightened title + concrete body for a Triage task, then flips it
``triage -> todo`` via ``kanban_db.specify_triage_task``.

Mirrors ``hermes_cli/goals.py``: same aux-client pattern, same "empty config
=> skip, don't crash" tolerance. One shot, no retry loop. JSON mode is not
requested (works on providers without it); the parse is lenient and falls
back to "whole reply is the body" so a malformed reply never strands a task.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_discovery as discovery
from hermes_cli.kanban_phases import CHAIN_ARROW, normalize_phase, parse_phase
from hermes_cli import kanban_m2 as m2
from hermes_cli import kanban_design as design
from hermes_cli import kanban_streams as streams

from utils import env_int

HERMES_KANBAN_SPECIFY_MAX_TOKENS = max(1500, env_int("HERMES_KANBAN_SPECIFY_MAX_TOKENS", 6000))

#: Prompt budgets for the task fields. The BODY budget is the measured defect
#: boundary (a root body past it lost S3–S6 / R1–R5 entirely), so it lives here
#: as one named constant that the coverage contract, the decomposer and (W-7)
#: the receipt all cite — never a second, drifting literal.
PROMPT_TITLE_LIMIT = 400
PROMPT_BODY_LIMIT = 4000

logger = logging.getLogger(__name__)


def _design_contract_block() -> str:
    """The D1–D6 contract rendered from ``kanban_design`` (single source of
    truth) for the specifier prompt: headings, purpose, requirement.

    Delegates to ``kanban_design.design_prompt_block`` so the T3 text has ONE
    definition shared with the decomposer prompt (which previously went
    without it entirely — that gap is t_c5f32661). The specifier's bytes are
    unchanged by the move.
    """
    return design.design_prompt_block()


_DESIGN_PROMPT_BLOCK = _design_contract_block()

_SYSTEM_PROMPT = """You are the Kanban triage specifier for the Hermes Agent board.
A user dropped a rough idea into the Triage column. Your job is to turn it
into a concrete, actionable task spec that an autonomous worker can pick up
and execute without further clarification.

Output a single JSON object with exactly three keys:

  {
    "title": "<tightened task title, <= 80 chars, imperative voice>",
    "body":  "<multi-line spec, see structure below>",
    "phase": "<one of the 7 phase names below, or null when none applies>"
  }

(For a DESIGN card, add the two keys `card_type` and `design` — see the
DESIGN CARDS section at the end. Never add them for an ordinary task.)

PHASE — tag which step of the SDLC chain this idea sits on
  The board's canonical chain is:

    grounding -> review -> research -> system-design -> engineering-planning -> implementation -> testing

  Classify the idea into exactly ONE of those seven names (lowercase,
  hyphenated, verbatim). Rules:
    - Use null only when the idea genuinely fits none of them (e.g. a pure
      question, a one-off ops chore with no deliverable).
    - Do NOT stretch the idea to a later phase than it is; an analysis or
      investigation is `research`, a "decide how to build X" is
      `system-design`, a "write/change code or infra" is `implementation`.
    - Phase is ADVISORY routing data, never a status, and never a substitute
      for acceptance criteria. A card tagged `testing` is still a card that
      gets reviewed like any other.
    - Also write the line `phase: <name>` as the FIRST line of the body, so
      the tag survives if the JSON field is dropped downstream.

(For a DESIGN card, add the two keys `card_type` and `design` — see the
DESIGN CARDS section at the end. Never add them for an ordinary task.)

The body MUST include these sections, each prefixed with a bold markdown
heading, in this order:

  **Step 0 — KNOWLEDGE LOOKUP (verb-first, MANDATORY)** — REQUIRED when the
      task is wiki-owning (title/body names a wiki page, slug, or
      ~/knowledge/wiki; OR it is research/design/audit/survey; OR its
      acceptance criteria mention doctrine, provenance, or "ground"). Emit this
      block verbatim, above every other section:

## Step 0 — KNOWLEDGE LOOKUP (verb-first, MANDATORY)
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
Then record `wiki_lookup: {verbs:[...], fallback_reason: null|<text>}` in completion metadata.

  - Use the Step 0 wording EXACTLY — do not paraphrase or shorten it.
  phase: <name>            (frontmatter line, omitted when the phase is null)

  **Goal** — one sentence, user-facing outcome.
  **Approach** — 2-5 bullets on how a worker should tackle it.
  **Acceptance criteria** — checklist of concrete, verifiable conditions.
  **Out of scope** — short list of things NOT to touch (omit if nothing
      obvious; never invent scope creep).

RESEARCH CARDS — M2 ACCEPTANCE BLOCK (mandatory when the idea is research):
  A card whose deliverable is research/reconnaissance/a landscape or
  framework survey (not building something) MUST carry this block, placed
  immediately after any `phase:` frontmatter line and before the body
  sections above:

      web_searches: <integer N of DISTINCT discovery queries the worker will run; floor 3>
      lead_ledger: attached
      anchor_note: <the anchors the card starts from + the out-of-list angles to search>
      vendor_caveat: <the evaluator to name whenever a vendor-reported number is cited>
      wiki_handoff_disposition: <wiki-worthy -> emit a wiki-keeper landing child | no wiki value (internal-only)>

  Those five field names are EXACT — never rename, reorder or drop one.
  `web_searches` must be an integer N >= 3 (distinct phrasings, not one
  query re-run). Non-research cards MUST NOT carry this block.

Rules:
  - Keep the tightened title close in meaning to the original idea — do
    NOT invent a different project.
  - If the original idea is already detailed, preserve its substance and
    just reformat into the sections above.
  - Never add invented requirements the user didn't hint at.
  - No preamble, no closing remarks, no code fences around the JSON.
  - Output only the JSON object and nothing else.
"""

_SYSTEM_PROMPT += _DESIGN_PROMPT_BLOCK
# T5 discovery-report convention, rendered from the single-source-of-truth
# module so the specifier prompt, the renderer and the review gate cannot drift
# apart. Declaration-driven: a card only carries the section when the reply
# declares `discovery:` itself.
_SYSTEM_PROMPT += discovery.discovery_prompt_block()
# W-8 declared-stream-coverage contract (quest t_bec82259 / card t_e9fb49b3),
# rendered from `kanban_streams` for the same anti-drift reason. Binding only
# when the root body declares streams; a streamless body gets no coverage lines.
_SYSTEM_PROMPT += streams.coverage_prompt_block()


_USER_TEMPLATE = """Task id: {task_id}
Current title: {title}
Current body:
{body}

Canonical phase chain (for the `phase` field): {phase_chain}
"""


@dataclass
class SpecifyOutcome:
    """Result of specifying a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    new_title: Optional[str] = None
    #: True when the card was specified under the D1–D6 design contract.
    design_card: bool = False
    #: ``kanban_design.validate_design_payload(...).as_metadata()`` — the
    #: structured-payload verdict (BA-P3.1 machine-readable half).
    design_metadata: Optional[dict] = None


def _truncate(text: str, limit: int) -> str:
    # Plain length clamp for LLM prompt fields; these never reach a terminal, so no escape stripping here.
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_blob(raw: str, fence_re: re.Pattern = _FENCE_RE) -> Optional[dict]:
    """Lenient JSON object extraction: strip code fences, take the first ``{``
    to the last ``}``. None if nothing parses to a dict."""
    if not raw:
        return None
    stripped = fence_re.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        val = json.loads(stripped[first : last + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return val if isinstance(val, dict) else None


def _nonblank(v) -> Optional[str]:
    return v if isinstance(v, str) and v.strip() else None


def _title_body(parsed: dict) -> tuple[Optional[str], Optional[str]]:
    """``(title, body)`` from an LLM reply: title stripped, body verbatim,
    either None when missing/blank."""
    title = _nonblank(parsed.get("title"))
    return (title.strip() if title else None), _nonblank(parsed.get("body"))


def _with_m2_block(title: Optional[str], body: Optional[str]) -> Optional[str]:
    """``body`` with the mandatory M2 acceptance block when the card is
    research; unchanged otherwise (T4 C4.1/C4.3).

    The specifier prompt ASKS for the block, but a model can drop or rename a
    field — the reviewer's gate then reads a card that never declared its
    contract. Rendering it here makes the five exact field names a property of
    generation, not of model compliance. Non-research cards and empty bodies
    pass through untouched, so existing card instantiation is unaffected.
    """
    if not body or not m2.is_research_card(title, body):
        return body
    return m2.ensure_m2_block(body)


def _with_discovery_report(title: Optional[str], body: Optional[str]) -> Optional[str]:
    """``body`` guaranteed to carry the declaration + section when the reply
    DECLARED discovery; unchanged otherwise (T5 P4).

    The prompt ASKS for the section, but a model can drop or rename a field —
    the orchestrator's evaluation step then reads a report that never declared
    its contract. Rendering it here makes the five exact field names a property
    of generation, not of model compliance. A card that did NOT declare
    ``discovery:`` is returned byte-identically, so ordinary and research cards
    are unaffected (the spurious-injection defect class fixed on t_71b6eca9 is
    designed out rather than patched later).
    """
    if not body:
        return body
    return discovery.ensure_discovery_section(body)


def _profile_author(default: str = "specifier") -> str:
    """Same identity contract as ``hermes_cli.kanban._profile_author``; ``$USER`` as the last
    resort for a human running the CLI outside any profile."""
    from hermes_cli.profiles import current_profile_name
    return current_profile_name() or os.environ.get("USER") or default


def _load_triage_task(task_id: str) -> tuple[Optional[kb.Task], str]:
    """``(task, "")`` when the task exists and is in triage, else ``(None, reason)``."""
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        return None, "unknown task id"
    if task.status != "triage":
        return None, f"task is not in triage (status={task.status!r})"
    return task, ""


def _task_prompt_fields(task: kb.Task) -> dict[str, str]:
    """Bounded ``task_id``/``title``/``body`` for the user prompt templates.

    The body is clamped to :data:`PROMPT_BODY_LIMIT`, but a clamped body is no
    longer a silent scope loss: ``streams.prompt_body`` appends the declared
    stream coverage read from the FULL body (W-8, quest t_bec82259 / card
    t_e9fb49b3), so the streams a cut would have hidden — S3–S6 on t_26d8eb94,
    R1–R5 on t_bec82259 — are still in front of the model. A body that fits the
    budget is returned byte-identically, so ordinary cards are unaffected.
    """
    return {
        "task_id": task.id,
        "title": _truncate(task.title or "", PROMPT_TITLE_LIMIT),
        "body": streams.prompt_body(task.body or "(no body)", limit=PROMPT_BODY_LIMIT),
    }


def _call_aux(verb: str, task_id: str, *, aux_task: str, system: str, user: str,
              max_tokens: int, timeout: int, log: logging.Logger = logger) -> tuple[Optional[str], str]:
    """One auxiliary LLM call; ``(reply_text, "")`` or ``(None, reason)``.

    ``call_llm`` applies all ``auxiliary.<aux_task>.*`` config (provider/model/
    base_url, extra_body, reasoning_effort, retries). Imported lazily so a
    missing aux client degrades to a skip instead of an import-time crash.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:  # pragma: no cover — import smoke test
        log.debug("%s: auxiliary client import failed: %s", verb, exc)
        return None, "auxiliary client unavailable"
    # Specify/decompose run outside any agent turn (CLI, dashboard route, gateway watcher), so no
    # conversation affinity scope is bound and the relay-affinity headers (x-opencode-session, the
    # OpenRouter/Portal sticky key) are omitted — the OpenCode Go relay rejects that with 400
    # MissingSessionID (#112043). Declare a per-task scope, but only when none is already bound so an
    # in-turn caller keeps its conversation's key.
    from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
    affinity_token = None if get_affinity_scope() else set_affinity_scope(f"kanban:{task_id}")
    try:
        # Route through call_llm so auxiliary.triage_specifier.* config (provider/model/base_url,
        # extra_body, reasoning_effort, retries) all apply — the direct-create path dropped extra_body
        # (#35566).
        resp = call_llm(
            task=aux_task,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as exc:
        suffix = " — skipping" if verb == "specify" else ""
        log.info("%s: API call failed for %s (%s)%s", verb, task_id, exc, suffix)
        return None, f"LLM error: {type(exc).__name__}"
    finally:
        if affinity_token is not None:
            reset_affinity_scope(affinity_token)
    try:
        return resp.choices[0].message.content or "", ""
    except Exception:
        return "", ""


def _apply_design_contract(
    parsed: dict, title: Optional[str], body: Optional[str],
) -> tuple[Optional[str], Optional[str], bool, Optional[dict]]:
    """Fold the D1–D6 design contract into a specifier reply.

    Returns ``(title, body, design_card, design_metadata)``. A design card
    gets the ``card_type: design`` frontmatter line, all six rendered sections
    merged into the body (structured payload first, the model's own body
    sections winning where present), and the structured-payload verdict for
    the review lane. Ordinary replies pass through byte-identically.
    """
    payload = parsed.get("design")
    is_design = design.is_design_card(
        card_type=parsed.get("card_type"), phase=parsed.get("phase"),
        title=title or parsed.get("title"), body=body,
    ) or isinstance(payload, dict)
    if not is_design:
        return title, body, False, None

    # Render from the structured payload; the model's own body sections are
    # preserved when it wrote them (merge replaces only the D-regions it has).
    block = design.render_design_sections(payload if isinstance(payload, dict) else {})
    merged = design.merge_design_sections(body or "", block)
    lines = merged.splitlines()
    if not any(design.body_card_type(line) for line in lines[:3]):
        merged = f"{design.DESIGN_CARD_TYPE_LINE}\n\n{merged}"
    metadata = design.validate_design_payload(payload).as_metadata() if isinstance(payload, dict) else (
        design.validate_design_payload(None).as_metadata()
    )
    return title, merged, True, metadata


def specify_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> SpecifyOutcome:
    """Specify one triage task and promote it to ``todo``. Expected failures
    (not in triage, no aux client, API error, malformed reply) surface as
    ``ok=False`` so an ``--all`` sweep continues."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        return SpecifyOutcome(task_id, False, reason)

    raw, reason = _call_aux(
        "specify", task_id, aux_task="triage_specifier", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(**_task_prompt_fields(task), phase_chain=CHAIN_ARROW),
        max_tokens=HERMES_KANBAN_SPECIFY_MAX_TOKENS, timeout=timeout or 120,
    )
    if raw is None:
        return SpecifyOutcome(task_id, False, reason)
    raw = raw.strip()

    parsed = _extract_json_blob(raw)
    phase: Optional[str] = None
    design_card = False
    design_metadata: Optional[dict] = None
    if parsed is None:
        # Whole reply becomes the body; the user can edit afterward.
        if not raw:
            return SpecifyOutcome(task_id, False, "LLM returned an empty response")
        new_title, new_body = None, raw
    else:
        new_title, new_body = _title_body(parsed)
        if new_body is None and new_title is None:
            return SpecifyOutcome(task_id, False, "LLM response missing title and body")
        # Structured ``phase`` field first, body frontmatter as the fallback.
        # Read the phase BEFORE the design fold: the fold may prepend the
        # ``card_type: design`` line, and the model's own first body line is the
        # authoritative place for ``phase: <name>``.
        phase = normalize_phase(parsed.get("phase")) or parse_phase(new_body)
        # T3: a design card gets the full D1–D6 contract folded into its body.
        new_title, new_body, design_card, design_metadata = _apply_design_contract(
            parsed, new_title, new_body,
        )
    # A phase already persisted on the card (re-specify) is never wiped by an
    # untagged reply; an unknown value is dropped rather than passed through.
    if phase is None:
        phase = normalize_phase(task.current_step_key)

    new_body = _with_m2_block(new_title, new_body)
    # T5: a reply that DECLARED `discovery:` gets the section completed to its
    # five exact field names; an undeclared body passes through untouched.
    new_body = _with_discovery_report(new_title, new_body)

    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn,
            task_id,
            title=new_title,
            body=new_body,
            current_step_key=phase,
            author=author or _profile_author(),
        )
    if not ok:
        # Race: promoted/archived between our read and the write.
        return SpecifyOutcome(task_id, False, "task moved out of triage before promotion")
    return SpecifyOutcome(
        task_id, True, "specified", new_title=new_title,
        design_card=design_card, design_metadata=design_metadata,
    )


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Task ids in the triage column; ``tenant`` narrows the sweep."""
    with kbc.connect_closing() as conn:
        tasks = kb.list_tasks(conn, status="triage", tenant=tenant, include_archived=False)
    return [t.id for t in tasks]
