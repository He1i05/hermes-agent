"""Wiki-landing child emission for the kanban decomposer (T4 C4.2, design §6 O-A).

Closes the research->wiki gap: a research card that declares a ``wiki-worthy``
wiki-handoff disposition gets a ``wiki-keeper`` child emitted AUTOMATICALLY by
the decomposer, gated on the research card through the parent->child links the
board already has. The design's chosen option (O-A) over the two rejects:

* O-B (scout writes the wiki in its own card) conflates the research and wiki
  writer roles and erodes the review boundary — wiki-keeper owns wiki writes.
* O-C (a post-hoc wiki-sweep cron) reintroduces exactly the "separate step"
  gap the audit flagged (research landing nowhere).

Emitting the child here — at decomposition, before any worker runs — makes wiki
landing a card action rather than a step somebody has to remember, and the
existing parent gating holds the landing until the research card completes.

Two inputs drive the decision, both of which the card itself already carries:

1. the card is a RESEARCH card — it carries the five-field M2 acceptance block
   (:mod:`hermes_cli.kanban_m2`). The M2 block is only ever rendered onto a
   research card, so its presence is the research signature; no second
   classifier is invented here.
2. its ``wiki_handoff_disposition`` value names ``wiki-worthy``. The
   ``no wiki value (internal-only)`` form is an explicit, acceptable disposition
   and emits nothing (design I4.2 — a wiki-worthy card has a reachable landing
   path, an internal-only card has an explicit "no wiki value" disposition).

An UNFILLED placeholder (``<wiki-worthy -> ...>``) is a template instruction,
never a declaration, and emits nothing — so a card whose block was rendered
without a real disposition is not silently turned into a wiki page.

The emitted child is deterministic: a reviewer can re-derive the expected child
set from the research children alone, with no LLM in the loop.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# The profile that owns wiki writes. Every docs/wiki card routes here (see
# kanban_capability._BEHAVIOR_PROFILE); the decomposer resolves it by capability
# when the profile is missing from the roster.
WIKI_KEEPER = "wiki-keeper"

# Research children carry the five-field M2 block; the disposition field is the
# wiki-handoff disposition within it.
DISPOSITION_FIELD = "wiki_handoff_disposition"

# A decomposition may not explode the board: a quest with several wiki-worthy
# research cards gets at most this many landing children per pass (the extras
# are named in the fan-out log so the orchestrator can emit them deliberately).
MAX_WIKI_LANDING_CHILDREN = 2

# The declaration that asks for a landing. Matched case-insensitively against
# the disposition value; ``wiki worthy`` / ``wiki_worthy`` are the same claim.
_WIKI_WORTHY_RE = re.compile(r"(?i)\bwiki[-_ ]?worthy\b")

# An explicit "no landing" claim. Checked FIRST so "no wiki value" /
# "internal-only" can never be read as worthy through the token above.
_NEGATED_RE = re.compile(
    r"(?i)(\bno wiki value\b|\binternal[-_ ]only\b|\bnot\s+wiki[-_ ]?worthy\b|"
    r"\bwiki[-_ ]handoff\s*[:=]\s*none\b|^none$|^n/?a$)"
)

# A rendered-but-unfilled field value: ``<anything>`` is a template instruction,
# never a claim (mirrors kanban_m2's placeholder contract).
_PLACEHOLDER_RE = re.compile(r"^<.*>$", re.DOTALL)

# The wiki-keeper's own card shapes — a landing child for one of these is
# redundant (the docs card already lands the page).
_DOCS_INTENT_RE = re.compile(r"(?i)\b(wiki|docs?|documentation|knowledge)\b")

# The verb-first lookup block a wiki-owning card must open with. Defined ONCE
# here for emitted cards; the specifier/decomposer prompts carry their own
# inline copies of the same wording (hoisting all three into one constant is a
# follow-up refactor, not this card's business).
STEP0_BLOCK = """\
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
Then record `wiki_lookup: {verbs:[...], fallback_reason: null|<text>}` in completion metadata."""


def is_wiki_worthy(value: object) -> bool:
    """True when a ``wiki_handoff_disposition`` value DECLARES a wiki landing.

    Conservative: only an explicit ``wiki-worthy`` claim counts. An unfilled
    placeholder, a blank value, ``none``, or the explicit
    ``no wiki value (internal-only)`` disposition all return False — emitting a
    wiki page the card never asked for is the expensive direction to be wrong.
    """
    if not isinstance(value, str):
        return False
    text = value.strip().lstrip("-*>").strip()
    if not text or _PLACEHOLDER_RE.match(text):
        return False
    if _NEGATED_RE.search(text):
        return False
    return bool(_WIKI_WORTHY_RE.search(text))


def disposition_of(body: object) -> Optional[str]:
    """The card's declared wiki-handoff disposition, else None.

    Read through :mod:`hermes_cli.kanban_m2` so the field name has exactly one
    source of truth (a rename there cannot silently stop emitting landings).
    """
    from hermes_cli import kanban_m2 as m2

    try:
        found = m2.parse_m2_block(body)
    except Exception:  # pragma: no cover — parse is total, this is belt+braces
        logger.exception("wiki-landing: M2 parse failed; treating card as non-research")
        return None
    return found.get(DISPOSITION_FIELD)


def _is_research_child(child: dict) -> bool:
    """True when this cleaned child is a research card.

    The five-field M2 block IS the research signature: the block is only ever
    rendered onto a card ``kanban_m2.is_research_card`` classified as research,
    so its presence is decisive. The child's CAPABILITY tag is deliberately NOT
    consulted — capability classification keys off its own keyword list (which
    does not know every research-shaped title), and a second classifier that can
    disagree with the M2 renderer would silently drop landings. One source of
    truth, and it is the one the review lane already gates on.
    """
    from hermes_cli import kanban_m2 as m2

    return m2.has_m2_block(child.get("body"))


# Keywords that mark a declared BEHAVIOR profile as wiki/docs-owned.
_DOCS_KEYWORDS = ("wiki", "docs", "documentation", "knowledge")


def _is_docs_child(child: dict, docs_owned: set[str] | None = None) -> bool:
    """True when the child is already a docs/wiki card (landing it is its job).

    The TITLE heuristic is deliberately subordinate to the project's existing
    doctrine that an imperative build verb in a title wins: "Implement wiki
    export for the survey" is a BUILD card that happens to mention the wiki, and
    counting it as pre-existing landing coverage would suppress the landing for
    its research parent — the same silent-suppression failure mode as the
    assignee collision.
    """
    if child.get("assignee") in (docs_owned or {WIKI_KEEPER}):
        return True
    from hermes_cli.kanban_m2 import has_build_verb

    title = child.get("title") or ""
    if has_build_verb(title):
        return False
    return bool(_DOCS_INTENT_RE.search(title))


def docs_owned_assignees(routing, caps: dict) -> set[str]:
    """Names that genuinely OWN wiki/docs work — the only ones that count as coverage.

    Deliberately does NOT include the decomposition's default assignee: a plain
    build sibling that merely shares the *resolved fallback* name is not a
    landing, and treating it as one would silently suppress the wiki-landing for
    a wiki-worthy research card (the gap this engine exists to close).
    """
    names = {WIKI_KEEPER}
    for name, cap in (caps or {}).items():
        if (
            cap.declared
            and cap.tag == "BEHAVIOR"
            and any(k in _DOCS_KEYWORDS for k in cap.keywords)
        ):
            names.add(name)
    return names


def _already_landed(children: list[dict], idx: int, docs_owned: set[str]) -> bool:
    """True when an existing child is ALREADY a landing child for research child ``idx``.

    The prompt asks the LLM to emit the wiki-keeper child itself, and this
    engine emits it when the model did not — without this guard a decomposer
    that complied with the prompt would get a DUPLICATE landing card. A card
    counts as coverage when it is a docs/wiki-OWNED card (``docs_owned``, or a
    docs-intent title) whose parent list names ``idx`` (the standard gating edge
    for a landing). Sharing the resolved assignee is NOT sufficient on its own.
    """
    for other in children:
        if other is None or other is children[idx]:
            continue
        if other.get("assignee") not in docs_owned and not _is_docs_child(other):
            continue
        if idx in (other.get("parents") or []):
            return True
    return False


def resolve_wiki_assignee(routing, caps: dict) -> str:
    """The profile that must own the landing card.

    Preference order: the canonical ``wiki-keeper`` when it is on the roster,
    else any declared BEHAVIOR profile carrying a wiki/docs/knowledge keyword,
    else the decomposition's default assignee — so a landing child NEVER lands
    unassigned (assignee=None strands work on the board).
    """
    valid = getattr(routing, "valid_names", None) or set()
    if WIKI_KEEPER in valid:
        return WIKI_KEEPER
    candidates = sorted(
        name
        for name, cap in (caps or {}).items()
        if cap.declared
        and cap.tag == "BEHAVIOR"
        and any(k in _DOCS_KEYWORDS for k in cap.keywords)
    )
    if candidates:
        return candidates[0]
    return routing.default_assignee


def _landing_title(parent_title: str) -> str:
    """A <=80-char imperative title naming the research card it lands."""
    base = "Land wiki page for: "
    room = 80 - len(base)
    title = (parent_title or "research card").strip()
    return base + (title[: room - 1] + "…" if len(title) > room else title)


def _landing_body(parent: dict, disposition: str) -> str:
    """The emitted card's body: Step 0 block, goal, approach, acceptance.

    Opens with the verb-first lookup block (a wiki-owning card) and names the
    gated parent in prose as well as in the graph, so a worker reading only its
    own body still knows where the findings are. It declares its own ``Docs:``
    bucket commitment; the decomposer's bucket appender fills the rest.
    """
    parent_title = (parent.get("title") or "").strip() or "the research card"
    return (
        f"{STEP0_BLOCK}\n\n"
        f"**Goal**\n"
        f"Land the wiki page(s) for this card's gated parent research card "
        f"(\"{parent_title}\") and wire them into the knowledge graph.\n\n"
        f"**Approach**\n"
        f"- Read the parent card's handoff (`kanban_show` on the parent: summary + "
        f"artifacts) instead of re-running the research.\n"
        f"- Query the index verb-first (Step 0 above); seed the search with the "
        f"parent's `anchor_note` vocabulary.\n"
        f"- Ingest into `~/knowledge/wiki` with typed frontmatter and wikilinks to "
        f"the related pages; run `python3 ../wiki-validate.py` before committing.\n"
        f"- Commit + push `He1i05/knowledge-wiki` per the vault convention (never "
        f"edit `raw/`).\n\n"
        f"**Acceptance criteria**\n"
        f"- [ ] at least one wiki page created/updated under `~/knowledge/wiki` for "
        f"this research output\n"
        f"- [ ] page passes `wiki-validate.py`\n"
        f"- [ ] wikilinks connect it to the existing graph (not an orphan page)\n"
        f"- [ ] wiki commit sha recorded in completion metadata\n\n"
        f"**Out of scope**\n"
        f"- Re-doing the research (the parent card owns it); editing the parent "
        f"card's own artifacts.\n\n"
        f"Docs: wiki landing for the gated parent research card (this card)\n"
        f"Research-feedback: parent declared wiki-handoff disposition: {disposition}"
    )


def emit_wiki_landing_children(
    task_id: str, children: list[dict], routing, caps: dict,
    *, limit: int = MAX_WIKI_LANDING_CHILDREN,
) -> list[str]:
    """Append a gated ``wiki-keeper`` child for every wiki-worthy research child.

    Mutates ``children`` in place (the caller's sibling-graph/parent-index
    contract is index-based, so an appended entry whose ``parents`` names an
    earlier index is exactly the standard parent gating). Returns the list of
    emitted titles (empty when nothing was emitted) — the caller logs/aits them.

    Idempotent by construction: it only ever reads the children it was handed,
    so a second call would emit duplicates. The decomposer calls it once per
    decomposition, before the DB write.
    """
    emitted: list[str] = []
    if not children:
        return emitted
    assignee = resolve_wiki_assignee(routing, caps)
    # Coverage is checked against profiles that genuinely OWN wiki/docs work —
    # never against the resolved fallback name (see docs_owned_assignees).
    docs_owned = docs_owned_assignees(routing, caps)
    # Snapshot: landings are appended to ``children`` as we go, and iterating a
    # list being appended to would also walk the landing cards themselves.
    for idx, child in enumerate(list(children)):
        if len(emitted) >= max(0, limit):
            remaining = [
                c.get("title") for c in children[idx:]
                if _is_research_child(c) and is_wiki_worthy(disposition_of(c.get("body")))
            ]
            if remaining:
                logger.info(
                    "decompose: task %s wiki-landing cap (%d) reached; "
                    "wiki-worthy research children left unlanded: %s",
                    task_id, limit, remaining,
                )
            break
        if not _is_research_child(child):
            continue
        disposition = disposition_of(child.get("body")) or ""
        if not is_wiki_worthy(disposition):
            continue
        if _already_landed(children, idx, docs_owned):
            logger.info(
                "decompose: task %s child %d already has a gated docs/wiki child — "
                "no duplicate landing emitted", task_id, idx,
            )
            continue
        landing = {
            "title": _landing_title(child.get("title") or ""),
            "body": _landing_body(child, disposition.strip()),
            "assignee": assignee,
            "parents": [idx],
            "wiki_landing": True,
        }
        children.append(landing)
        emitted.append(landing["title"])
        logger.info(
            "decompose: task %s emitted wiki-landing child for research child %d "
            "(%r) -> assignee %r", task_id, idx, child.get("title"), assignee,
        )
    return emitted


def audit_log_landing_children(
    conn, task_id: str, child_ids: list[str], children: list[dict],
    run_id: Optional[int] = None,
) -> int:
    """M3-style audit: one ``wiki_landing_emitted`` event per emitted landing.

    Payload carries the gated research parent (title + assignee) and the
    disposition that triggered the emission, so the BA-P2.1 landing-rate query
    can be answered from events without re-parsing prose. Never raises past the
    caller's guard — like ``capability.audit_log_routed_children``, logging must
    not fail a fan-out that already committed.
    """
    from hermes_cli.kanban_db import _append_event

    written = 0
    for cid, child in zip(child_ids, children):
        if not child.get("wiki_landing"):
            continue
        parent_idx = (child.get("parents") or [None])[0]
        parent = children[parent_idx] if isinstance(parent_idx, int) and 0 <= parent_idx < len(children) else {}
        _append_event(
            conn, cid, "wiki_landing_emitted",
            {
                "root": task_id,
                "research_parent_title": parent.get("title"),
                "research_parent_assignee": parent.get("assignee"),
                "capability_reason": parent.get("capability_reason"),
                "disposition": disposition_of(parent.get("body")),
                "assignee": child.get("assignee"),
            },
            run_id=run_id,
        )
        written += 1
    return written
