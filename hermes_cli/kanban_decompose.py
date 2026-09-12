"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the gateway
dispatcher's auto-decompose path. Reads the profile roster (with
descriptions), asks the auxiliary LLM for a task graph in JSON, then
atomically creates the children, links them under the root, and flips the
root ``triage -> todo``. The root stays alive as parent of every leaf child so
it wakes back up when the graph completes and its assignee (the orchestrator
profile) can judge completion and add more work.

Mirrors ``kanban_specify`` (lazy aux import, lenient parse, never raises on
expected failures). ``fanout=false`` collapses to the ``specify`` behaviour
(tighten + promote, no children), making ``decompose`` a strict superset.
Unknown assignees are rewritten to ``default_assignee`` — a child NEVER ends
up with ``assignee=None``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_capability as capability
from hermes_cli import kanban_design as design
from hermes_cli import kanban_discovery as discovery
from hermes_cli import kanban_m2 as m2
from hermes_cli import kanban_streams as streams
from hermes_cli import kanban_wiki_landing as wiki_landing
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import profiles as profiles_mod
from hermes_cli.kanban_phases import CHAIN_ARROW, cascade_violations, normalize_phase, parse_phase
from hermes_cli.kanban_specify import (
    _call_aux, _extract_json_blob, _load_triage_task, _task_prompt_fields, _title_body,
    _with_m2_block, _with_discovery_report,
)
from hermes_cli.kanban_specify import _profile_author as _specify_author
from utils import env_int

logger = logging.getLogger(__name__)

# Decomposer replies are bigger than the specify floor: reasoning models burn
# part of the completion budget on reasoning_tokens before any content, and a
# truncated reply fails JSON parsing (observed on qwen3.7-flash: 2923
# reasoning tokens against a 4000 cap -> 'malformed JSON'). 12000 gives the
# 4-bucket checklist bodies room to complete.
HERMES_KANBAN_DECOMPOSE_MAX_TOKENS = max(4000, env_int("HERMES_KANBAN_DECOMPOSE_MAX_TOKENS", 12000))


_SYSTEM_PROMPT = """You are the Kanban decomposer for the Hermes Agent board.

A user dropped a rough idea into the Triage column. Your job is to break it
into a small graph of concrete child tasks and route each one to the best-
matching profile from the available roster.

You will be given:
  - The original task title and body
  - The list of available profiles (each with name + description)
  - The fallback "default_assignee" used when no profile fits

Output a single JSON object with this exact shape:

  {
    "fanout": true,
    "rationale": "<one sentence on why this decomposition>",
    "tasks": [
      {
        "title": "<concrete task title, imperative voice, <= 80 chars>",
        "body":  "<detailed spec for the worker on this child task>",
        "assignee": "<profile name from the roster, or null for default>",
        "phase": "<one of the 7 phase names, or null when no phase applies>",
        "covers": ["<stream tag this card OWNS>", ...],
        "fold_in": {"<stream tag this card ABSORBS>": "<why>"},
        "deliverable": true,
        "card_type": "<optional: 'design' for a DESIGN child, else omit>",
        "design": {"<optional: the structured D1-D6 payload, see DESIGN CARDS>"},
        "parents": [<int>, ...]
      },
      ...
    ]
  }

PHASE CASCADE — the DEFAULT shape for a multi-step quest
  A quest that spans behavior + design + implementation decomposes along this
  fixed phase chain, in this order (it is a dependency chain, not a menu):

    grounding -> review -> research -> system-design -> engineering-planning -> implementation -> testing

  - Use the phase NAMES above verbatim (lowercase, hyphenated). Any other
    value is discarded.
  - SKIP phases that do not apply. Emit only the phases the quest genuinely
    needs, keeping the surviving phases in the canonical order above. A
    pure-implementation quest is legitimately `implementation -> testing`.
    A docs-only quest may be a single `implementation` card. Never pad the
    graph to all 7 phases, and never reorder them.
  - LINK the chain: each phase card lists the PREVIOUS emitted phase card in
    its "parents" (data dependency), so the cascade runs one phase at a time
    instead of all at once. Independent work inside one phase runs in
    parallel with no parents between siblings.
  - `"phase": null` is correct for support cards that are not a step in the
    chain (a fan-out audit, a release note, an adoption check).
  - PHASE IS NOT STATUS: it is advisory routing data. Never use it to mean
    "done" or "blocked".

Rules:
  - "parents" is a list of INDICES (0-based) into this same "tasks" list,
    expressing actual data dependencies. Tasks with no parents run in
    PARALLEL. Tasks with parents wait until every parent completes.
  - Prefer parallelism WITHIN a phase. If two tasks can be done
    independently, give them no parents so the dispatcher fans them out at
    once. Across phases, always link predecessor -> successor.
  - Use 2-6 tasks for normal work. Don't create 20 tiny tasks. Don't
    cram everything into 1 task. The phase cascade is guidance, NOT a quota —
    never inflate the graph to hit all 7 phases.
  - ROUTING ORDER (mandatory): (1) PHASE — which stage of work this is;
    (2) CAPABILITY — classify each child as ARTIFACT (produces code,
    files, infra) or BEHAVIOR (research, review, adoption, docs, wiki
    landing, user-gate work) BEFORE choosing a profile; (3) PROFILE —
    only then pick the roster profile whose capability matches. A
    research/design/review card goes to a research-profile; a docs/wiki
    card goes to a docs-profile; a code card goes to a code-profile.
    Never route a BEHAVIOR child to an ARTIFACT-only profile.
  - Pick assignees from the roster by matching the task to the profile's
    CAPABILITY tag and DESCRIPTION (not just the name). A roster entry
    rendered as ``name [ARTIFACT]`` or ``name [BEHAVIOR]`` declares that
    tag; entries without a bracket are undeclared. When nothing matches
    well, use null and the system will route to the default_assignee.
  - WIKI-OWNING CARDS (a card whose title/body names a wiki page, slug, or
    ~/knowledge/wiki; any research/design/audit/survey card; any intake card;
    any card whose acceptance criteria mention doctrine, provenance, or
    "ground"): the child's body MUST open with this block, verbatim, ABOVE all
    other content:

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

    Use this exact wording — do not paraphrase or shorten it. This block is the
    single strongest measured lever on retrieval behaviour (cards carrying it
    ran 55-68% verb share vs 8-38% without).
  - NON-CODE SCOPE CHECKLIST (mandatory on every decomposition — do not
    skip even when the root body does not mention them): for each of
    ADOPTION (behavioral/rollout change), DOCS (wiki/documentation
    landing), RESEARCH-FEEDBACK (loop back into research), GATE (needs a
    user verdict) — decide whether the quest has such scope and, when it
    does, emit a dedicated child card for it with the matching capability.
    Never swallow non-code scope silently into a code child.
  - Each child task body MUST end with these four bucket headings, each
    either the concrete commitment for that bucket or the word "none":

        Adoption: <commitment or "none">
        Docs: <commitment or "none">
        Research-feedback: <commitment or "none">
        Gate: <needs-user-verdict detail or "none">

    (so a reviewer can see they were considered, not just omitted).
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.
  - Start each child body with the frontmatter line `phase: <name>` when the
    child has a phase, so the phase survives even if the field is dropped.
  - RESEARCH CHILDREN (mandatory): when a child's deliverable is research/
    reconnaissance/survey (not building something), its body MUST start with
    this exact five-field block — placed after any `phase:` line, before the
    body sections. Never rename or drop a field:

        web_searches: <integer N of DISTINCT discovery queries; floor 3>
        lead_ledger: attached
        anchor_note: <anchors the child starts from + the out-of-list angles to search>
        vendor_caveat: <the evaluator to name whenever a vendor-reported number is cited>
        wiki_handoff_disposition: <wiki-worthy -> emit a wiki-keeper landing child | no wiki value (internal-only)>

    Non-research children MUST NOT carry this block. The review lane hard-fails
    a research card whose block is incomplete.

When the task is genuinely a single unit of work (no useful decomposition),
return:

  {
    "fanout": false,
    "rationale": "<one sentence>",
    "title": "<tightened title>",
    "body":  "<concrete spec for a single worker>",
    "assignee": "<profile name from the roster, or null for default>",
    "phase": "<the single phase this card belongs to, or null>",
    "covers": ["<stream tag this card OWNS>", ...],
    "fold_in": {"<stream tag this card ABSORBS>": "<why>"},
    "deliverable": true
  }

In that case the task stays as one work item, just with a tightened spec and
a concrete assignee. If no profile fits, use null and the system will route to
the default_assignee.

No preamble, no closing remarks, no code fences. Output only the JSON object.
"""

# T5 discovery-report convention (single source of truth in kanban_discovery):
# a child whose work produces a flag-worthy mid-flight finding MAY declare
# itself a discovery-report source. Rendered from the module so the prompt, the
# renderer and the review gate key off the same field names.
_SYSTEM_PROMPT += discovery.discovery_prompt_block()

# W-8 declared stream coverage (single source of truth in kanban_streams, quest
# t_bec82259 / card t_e9fb49b3). This is the load-bearing half of the fix: the
# user turn lists EVERY stream the FULL root body declares — including the ones
# the prompt clamp cuts — so completeness stops depending on where the cut lands
# and stops relying on a keyword diff (measured 4/4 false-pass; see the module
# docstring). Binding only for a root that declares streams.
_SYSTEM_PROMPT += streams.coverage_prompt_block()

# W-10 declared deliverable (single source of truth in kanban_streams, card
# t_82eafe14). The user gate reaches exactly one child per root; the legacy
# title/phase heuristic that picked it is structurally biased (it landed on a
# verification card on both incident roots). This block makes the decomposer
# MARK the deliverable explicitly so the recipient is chosen on purpose.
_SYSTEM_PROMPT += streams.deliverable_prompt_block()

# T3 D1–D6 design-card contract (t_c5f32661): the decomposer assigns the
# `system-design` phase itself, so it must instantiate the same contract the
# specifier does — otherwise a design child is dispatched with zero D-sections
# and is guaranteed rework. Rendered from kanban_design, the single source of
# truth shared with the specifier prompt.
_SYSTEM_PROMPT += design.design_prompt_block()


_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}

Root phase (tagged by the triage specifier, when known): {root_phase}

Available profiles (assignees you may pick from):
{roster}

Default assignee (used when no profile fits a task): {default_assignee}

Phase chain for reference: {phase_chain}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None
    #: True when the ``fanout=false`` promotion was written under the D1–D6
    #: design contract (T3, t_c5f32661).
    design_card: bool = False


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return _specify_author("decomposer")


def _resolve_profile_from_cfg(cfg: dict, key: str, *, fallback: Optional[str] = None) -> str:
    """``kanban.<key>`` if it names an existing profile, else ``fallback``
    (the root task's own assignee) if that does, else the active default
    profile — so a task is never stranded for lack of an owner.
    ``orchestrator_profile`` owns the root after fan-out; ``default_assignee``
    catches children the decomposer can't route.

    The root's assignee sits before the active profile because the decomposer
    runs inside whatever profile hosts the dispatcher — an operator's
    credential-less incognito profile, say — and that profile must never
    silently become the owner of work the card was assigned away from (#114294).
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    explicit = (kanban_cfg.get(key) or "").strip()
    for candidate in (explicit, (fallback or "").strip()):
        if candidate:
            try:
                if profiles_mod.profile_exists(candidate):
                    return candidate
            except Exception:
                pass
    try:
        return profiles_mod.get_active_profile_name() or "default"
    except Exception:
        return "default"


def _build_roster() -> tuple[list[dict], set[str]]:
    """``(roster_for_prompt, valid_assignee_names)``; entries are
    ``{name, description, has_description, capability, capability_declared}`` —
    ``capability`` is the parsed ARTIFACT/BEHAVIOR tag, ``capability_declared``
    False when the profile carries no ``CAPABILITY:`` line (C2.4)."""
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return [], set()
    roster = []
    for p in all_profiles:
        desc = (p.description or "").strip()
        cap = capability.parse_capability(desc)
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
            "capability": cap.tag,
            "capability_declared": cap.declared,
        })
    return roster, {p.name for p in all_profiles}


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    lines = []
    for entry in roster:
        tag = f" [{entry['capability']}]" if entry.get("capability_declared") else ""
        warn = "" if entry["has_description"] else " ⚠ undescribed"
        lines.append(f"  - {entry['name']}{tag}{warn}: {entry['description']}")
    return "\n".join(lines)


def _normalize_assignee_choice(assignee: object, *, default_assignee: str, valid_names: set[str]) -> str:
    """A valid assignee, else ``default_assignee`` — promoted work is never
    left unassigned."""
    if not isinstance(assignee, str) or not assignee.strip():
        return default_assignee
    chosen = assignee.strip()
    return chosen if chosen in valid_names else default_assignee


@dataclass
class _Routing:
    """Config-derived routing context for one decomposition."""

    orchestrator: str
    default_assignee: str
    auto_promote: bool
    roster: list[dict]
    valid_names: set[str]


def _load_routing(*, root_assignee: Optional[str] = None) -> _Routing:
    from hermes_cli.config import load_config_readonly
    try:
        cfg = load_config_readonly()
    except Exception:  # decompose_task promises ok=False, never a raise, on config trouble
        cfg = {}
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    roster, valid_names = _build_roster()
    return _Routing(
        orchestrator=_resolve_profile_from_cfg(cfg, "orchestrator_profile", fallback=root_assignee),
        default_assignee=_resolve_profile_from_cfg(cfg, "default_assignee", fallback=root_assignee),
        auto_promote=bool(kanban_cfg.get("auto_promote_children", True)),
        roster=roster,
        valid_names=valid_names,
    )


def _apply_single(task: kb.Task, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    """``fanout=false``: single-task spec promotion (same effect as specify)."""
    title_val, body_val = _title_body(parsed)
    assignee_val = None
    if not task.assignee:
        assignee_val = _normalize_assignee_choice(
            parsed.get("assignee"), default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
    if title_val is None and body_val is None:
        return DecomposeOutcome(task.id, False, "decomposer returned fanout=false with no title/body")
    # T5: the single-task path reuses the specifier body shape, so a reply that
    # declared `discovery:` gets the section by the same rule as specify_task.
    body_val = _with_discovery_report(title_val, body_val)
    # W-8: a single card carries the declarations for every stream it covers
    # (the prompt contract says so explicitly). Declaration-driven — a reply
    # that declared nothing keeps its body byte-identically.
    body_val = streams.ensure_declaration_lines(
        body_val, covers=parsed.get("covers"), fold_in=parsed.get("fold_in"),
    )
    # W-10: the single card IS the deliverable when it says so; fold the
    # declaration line in for symmetry with the fan-out path. Inert on a
    # no-children card but keeps the reply schema honest (a reply that declared
    # nothing keeps body_val byte-identically).
    body_val = streams.ensure_deliverable_line(
        body_val,
        deliverable=parsed.get("deliverable"),
        delivers=(task.id if _declares_deliverable(parsed) else None),
    )
    # Structured ``phase`` field first, then the body frontmatter line the
    # prompt asks for, then whatever the card already carried (never wiped).
    phase = (
        normalize_phase(parsed.get("phase"))
        or parse_phase(body_val)
        or normalize_phase(task.current_step_key)
    )
    # T4 C4.1: the single-task path reuses the specify body shape, so it gets
    # the research M2 block by the same rule as specify_task.
    body_val = _with_m2_block(title_val, body_val)
    # T3 D1–D6: this path is the decomposer's own spec promotion, and it can
    # classify a single card as `system-design` just as the fan-out path can
    # (t_c5f32661). Same contract, same fold order (after M2), same module.
    if design.is_design_card(
        card_type=parsed.get("card_type"), phase=phase,
        title=title_val or parsed.get("title"), body=body_val,
        title_heuristic=False,
    ):
        design_card = True
        body_val = design.ensure_design_contract(
            body_val, parsed.get("design"), phase=phase,
        )
    else:
        design_card = False
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn, task.id, title=title_val, body=body_val, assignee=assignee_val,
            current_step_key=phase, author=author,
        )
    if not ok:
        return DecomposeOutcome(task.id, False, "task moved out of triage before promotion")
    return DecomposeOutcome(
        task.id, True, "single task (no fanout)", fanout=False, new_title=title_val,
        design_card=design_card,
    )


def _declares_deliverable(entry: dict) -> bool:
    """True when a reply entry marks its child as the root's deliverable (W-10).

    Accepts either declared shape, so a model that answers with the structured
    toggle (``"deliverable": true``) or the explicit root reference
    (``"delivers": "<root id>"``) is honoured. Anything else (absent, false,
    a junk string) is NOT a declaration — the gate then falls to the heuristic
    and the fallback is recorded.
    """
    value = entry.get("deliverable")
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() in streams.DELIVERABLE_TRUE_VALUES:
        return True
    delivers = entry.get("delivers")
    return bool(isinstance(delivers, str) and delivers.strip())


def _clean_children(task_id: str, raw_tasks: list, routing: _Routing) -> tuple[list[dict], str]:
    """Validate/normalise the LLM's ``tasks`` list; ``(children, "")`` or ``([], reason)``.
    Unknown assignees route to the default; never assignee=None. Capability-first
    routing (UG-2 C2.4): the LLM's pick is honored only when the chosen profile's
    capability tag matches the child's; else it is rerouted by capability; the
    default-assignee fallback stays the last resort. ``phase`` (T1) is
    canonicalized against the closed enum (body frontmatter line as fallback)
    and passed through for the DB layer to persist; an unknown phase is dropped
    rather than coerced."""
    caps = capability._build_capability_table()
    children: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return [], f"tasks[{idx}] is not an object"
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return [], f"tasks[{idx}].title is missing or empty"
        body = entry.get("body")
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee, default_assignee=routing.default_assignee, valid_names=routing.valid_names,
        )
        if isinstance(assignee, str) and assignee.strip() and assignee.strip() not in routing.valid_names:
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, routing.default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        raw_phase = entry.get("phase")
        phase = normalize_phase(raw_phase) or parse_phase(body)
        if phase is None and raw_phase not in (None, "", "null"):
            logger.info(
                "decompose: task %s child %d phase %r is not in the phase chain — leaving unset",
                task_id, idx, raw_phase,
            )
        children.append({
            "title": title.strip()[:200],
            "body": body.strip() if isinstance(body, str) else "",
            "assignee": chosen,
            "phase": phase,
            # T3: the LLM's explicit design signals for this child. Kept on the
            # child dict (like ``capability_tag``) so the fold below classifies
            # on exactly what the model declared, not on a re-parse of prose.
            "card_type": entry.get("card_type"),
            "design": entry.get("design"),
            # Drop non-int, out-of-range and self parent indices.
            "parents": [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx],
            # W-8 declared stream coverage, carried straight from the reply.
            # `ensure_declaration_lines` below folds these into the child body
            # as canonical `covers:` / `Fold-in:` lines — a declaration is never
            # invented for a child that did not make one.
            "covers": entry.get("covers"),
            "fold_in": entry.get("fold_in"),
            # W-10 declared deliverable (card t_82eafe14): the child that IS the
            # root's deliverable says so, so the user gate is routed on purpose
            # instead of by the title/phase heuristic. `ensure_deliverable_line`
            # below folds it into the body; a reply that declared nothing keeps
            # its body byte-identically.
            "deliverable": entry.get("deliverable"),
            "delivers": task_id if _declares_deliverable(entry) else None,
        })
    # UG-2: capability-first routing (C2.3/C2.4).
    capability._apply_capability_routing(task_id, children, routing, caps)
    # T5: a child that DECLARED `discovery:` gets its section completed to the
    # five exact field names. Declaration-driven — a child that did not ask for
    # one is left byte-identical, so no spurious discovery block can appear.
    # Applied BEFORE the bucket headings: those are routing boilerplate and must
    # not become part of the report section's trailing fields.
    for child in children:
        child["body"] = _with_discovery_report(child.get("title"), child.get("body")) or ""
    # W-8: fold each child's declared stream coverage into its body as canonical
    # `covers: <tag>` / `Fold-in: <tag> — <why>` lines. Declaration-driven and
    # idempotent: a child that declared nothing keeps its body byte-identically,
    # so the ordinary (streamless) fan-out is untouched.
    for child in children:
        child["body"] = streams.ensure_declaration_lines(
            child.get("body"), covers=child.pop("covers", None),
            fold_in=child.pop("fold_in", None),
        )
    # W-10: fold the declared-deliverable marker into the marking child's body.
    # Declaration-driven and idempotent — an undeclared child keeps its body
    # byte-identically, so the ordinary fan-out is untouched.
    for child in children:
        child["body"] = streams.ensure_deliverable_line(
            child.get("body"), deliverable=child.pop("deliverable", None),
            delivers=child.pop("delivers", None),
        )
    # T4 C4.1/C4.3: a research child carries the mandatory M2 acceptance block
    # so the review lane has the five exact field names to gate on. Classify
    # BEFORE the bucket headings are appended — those headings are routing
    # boilerplate, and letting "Research-feedback: none" reach the classifier
    # would mislabel every child that lacks a build verb in its title.
    for child in children:
        child["body"] = m2.ensure_m2_block(child["body"]) if m2.is_research_card(
            child["title"], child["body"],
        ) else child["body"]
    # T4 C4.2: a wiki-worthy research child gets its wiki-keeper landing child
    # gated on it (design O-A). Runs after capability routing so the research
    # children are already tagged, and before the bucket headings so the
    # appended landing card gets them too.
    emitted = wiki_landing.emit_wiki_landing_children(task_id, children, routing, caps)
    if emitted:
        logger.info(
            "decompose: task %s emitted %d wiki-landing child(ren): %s",
            task_id, len(emitted), emitted,
        )
    capability.append_bucket_headings(children)
    # T3 D1–D6 contract (t_c5f32661): the decomposer assigns the `system-design`
    # phase itself, so a design child must leave here WITH the contract — the
    # specifier is not the only path that produces design cards. This is the
    # LAST fold, deliberately:
    #   * after the M2 classification, so the D-sections' own vocabulary
    #     ("evaluate", "compare", "scout") cannot flip a design child into
    #     research and staple a spurious M2 block onto it;
    #   * after the bucket headings, because ``ensure_design_contract`` writes
    #     the six blocks at the END of the body — the routing boilerplate stays
    #     ABOVE D1, so the ordered D6 work list is never contaminated by a
    #     trailing "Adoption: none" line folded into its last item.
    # Ordinary (non-design) children pass through byte-identically.
    for child in children:
        if not design.is_design_card(
            card_type=child.get("card_type"), phase=child.get("phase"),
            title=child.get("title"), body=child.get("body"),
            title_heuristic=False,
        ):
            continue
        child["design_card"] = True
        child["body"] = design.ensure_design_contract(
            child.get("body"), child.get("design"), phase=child.get("phase"),
        )
        payload = child.get("design")
        child["design_metadata"] = design.validate_design_payload(payload).as_metadata()
    return children, ""


def _apply_fanout(task_id: str, parsed: dict, routing: _Routing, author: str) -> DecomposeOutcome:
    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(task_id, False, "decomposer returned fanout=true with empty tasks list")
    children, reason = _clean_children(task_id, raw_tasks, routing)
    if reason:
        return DecomposeOutcome(task_id, False, reason)
    # M3-style cascade audit: advisory only (logged), never blocks the fan-out.
    for violation in cascade_violations(raw_tasks):
        logger.warning("decompose: task %s phase cascade: %s", task_id, violation)
    # M3 bucket-heading coverage (SPEC §2.8): each child body should carry the
    # four headings. Logged advisory-only; a missing heading never fails the
    # already-committed fan-out.
    try:
        from hermes_cli import m3_audit
        with kbc.connect_closing() as conn:
            m3_audit.audit_and_log(logger, conn, task_id)
    except Exception:
        logger.exception("decompose: M3 bucket-heading audit failed on task %s", task_id)
    try:
        with kbc.connect_closing() as conn:
            child_ids = decompose_triage_task(
                conn,
                task_id,
                root_assignee=routing.orchestrator,
                children=children,
                author=author,
                auto_promote=routing.auto_promote,
            )
    except ValueError as exc:
        return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}")
    except Exception as exc:
        logger.exception("decompose: DB error on task %s", task_id)
        return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")
    if child_ids is None:
        return DecomposeOutcome(task_id, False, "task already decomposed or moved out of triage")
    # B2 / C1: post-commit, post-recompute advisories. Gate extraction runs
    # only after the graph committed (a gate failure must never fail a
    # committed fan-out, same pattern as the M3 audits below); an
    # orchestration-only root is scheduled graph-deferred so it stops
    # re-dispatching (SPEC §3.3/§3.4 C1 + D-C4).
    try:
        from hermes_cli import kanban_gate as gate

        with kbc.connect_closing() as conn:
            _root = kb.get_task(conn, task_id)
            _orchestration_only = gate.parse_orchestration_only(
                _root.body if _root else None
            )
        if _orchestration_only:
            with kbc.connect_closing() as conn:
                kb.schedule_task(
                    conn, task_id, reason="orchestration-only", reactivate_on="children_done",
                )
            try:
                with kbc.connect_closing() as conn:
                    gate.propagate_deferral(conn, task_id, child_ids, author=author)
            except Exception:
                logger.exception(
                    "decompose: deferral child stamping failed on task %s", task_id,
                )
        gate.propagate_root_gate(task_id, child_ids, children, author=author)
    except Exception:
        logger.exception(
            "decompose: gate propagation failed (advisory only) on task %s", task_id,
        )
    # W-9 (plan §2): the post-decompose coverage receipt (W-7). Advisory —
    # post-commit, inside its own try/except so a receipt failure can never
    # abort the already-committed graph, and it writes only a comment + one
    # `decomposition_receipt` event (no status/body/link touch). `child_ids` is
    # passed explicitly: the receipt must judge exactly the set this fan-out
    # emitted, not a later live-links union.
    try:
        from hermes_cli import kanban_coverage as coverage

        outcome = coverage.emit_receipt(task_id, child_ids=child_ids)
        if not outcome.ok:
            logger.warning(
                "decompose: coverage receipt skipped on task %s: %s",
                task_id, outcome.reason,
            )
        elif outcome.gaps:
            logger.warning(
                "decompose: task %s coverage receipt reports %d GAP(S) of %d "
                "declared stream(s) (%d child(ren) emitted)",
                task_id, outcome.gaps, outcome.declared_total, outcome.children_total,
            )
    except Exception:
        logger.exception(
            "decompose: coverage receipt failed (advisory only) on task %s", task_id,
        )
    # M3 audit: record one capability_routed event per child (task_events,
    # no schema change) so the before/after audit can count mis-routes.
    try:
        with kbc.connect_closing() as conn:
            capability.audit_log_routed_children(conn, task_id, child_ids, children)
    except Exception:
        logger.exception("decompose: M3 capability audit logging failed on task %s", task_id)
    # M3 audit (T4): the wiki-landing emissions, so the BA-P2.1 landing-rate
    # query reads events instead of re-deriving "was this research wiki-worthy".
    try:
        with kbc.connect_closing() as conn:
            wiki_landing.audit_log_landing_children(conn, task_id, child_ids, children)
    except Exception:
        logger.exception("decompose: wiki-landing audit logging failed on task %s", task_id)
    return DecomposeOutcome(
        task_id, True, f"decomposed into {len(child_ids)} children", fanout=True, child_ids=child_ids,
    )


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks. Expected failures
    (not in triage, no aux client, API error, malformed/empty reply) surface
    as ``ok=False``."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, reason)

    # Hand-merge at the pin: upstream added ``root_assignee=task.assignee`` (#114294);
    # our commit only INSERTED the phase derivation below and left the routing call alone.
    # Keep both — they are independent.
    routing = _load_routing(root_assignee=task.assignee)
    # Root phase: explicit tag first, then the body frontmatter the specifier
    # wrote. Passed to the prompt so a cascade continues the chain the root is
    # already on instead of restarting at `grounding`.
    root_phase = normalize_phase(getattr(task, "current_step_key", None)) or parse_phase(task.body)
    raw, reason = _call_aux(
        "decompose", task_id, aux_task="kanban_decomposer", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(
            **_task_prompt_fields(task),
            root_phase=root_phase or "(untagged — start the chain at grounding)",
            phase_chain=CHAIN_ARROW,
            roster=_format_roster(routing.roster),
            default_assignee=routing.default_assignee,
        ),
        max_tokens=HERMES_KANBAN_DECOMPOSE_MAX_TOKENS, timeout=timeout or 180, log=logger,
    )
    if raw is None:
        return DecomposeOutcome(task_id, False, reason)

    parsed = _extract_json_blob(raw, _FENCE_RE)
    if parsed is None:
        return DecomposeOutcome(task_id, False, "LLM returned malformed JSON")

    audit_author = author or _profile_author()
    if not parsed.get("fanout"):
        return _apply_single(task, parsed, routing, audit_author)
    return _apply_fanout(task_id, parsed, routing, audit_author)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kbc.connect_closing() as conn:
        rows = kb.list_tasks(conn, status="triage", tenant=tenant, limit=1000)
    return [row.id for row in rows]
