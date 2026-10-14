"""M2 research-card acceptance block (T4, design §6 C4.1/C4.3).

A ``phase=research`` card carries a mandatory five-field acceptance block so the
review lane has a concrete, checkable contract before approving it:

    web_searches: <integer N of DISTINCT discovery queries; floor 3>
    lead_ledger: attached
    anchor_note: <anchors the card starts from + the out-of-list angles to search>
    vendor_caveat: <the evaluator to name whenever a vendor-reported number is cited>
    wiki_handoff_disposition: <wiki-worthy -> wiki-keeper landing child | no wiki value (internal-only)>

``M2_FIELDS`` is the single source of truth for the field names and their
render order — the triage specifier prompt, the renderer here, and the review
gate all key off it, so a rename cannot drift between them.

Scope: this module renders/parses/validates the block only. Emitting the
wiki-landing child card for a ``wiki-worthy`` disposition is the decomposer's
job (T4 C4.2), not this template's.

Backward compatibility: nothing here runs unless a card is classified as
research (``is_research_card``); non-research cards render byte-identically to
before.
"""

from __future__ import annotations

import re
from typing import Optional

# Exact field names, in render order. Do NOT rename/reorder — a review gate and
# the specifier prompt both rely on this tuple.
M2_FIELDS: tuple[str, ...] = (
    "web_searches",
    "lead_ledger",
    "anchor_note",
    "vendor_caveat",
    "wiki_handoff_disposition",
)

# The reviewer's floor: fewer than this many DISTINCT discovery queries is a
# hard fail (scout SOUL step 2a).
M2_MIN_WEB_SEARCHES: int = 3

# Unfilled-field guidance. A rendered-but-unfilled field is visibly a template
# instruction, never a claim the card has already satisfied — so a reviewer can
# tell a filled disposition from a placeholder at a glance.
M2_PLACEHOLDERS: dict[str, str] = {
    "web_searches": f"<integer N >= {M2_MIN_WEB_SEARCHES}: distinct discovery queries>",
    "lead_ledger": "attached",
    "anchor_note": "<anchors named in the card body + the out-of-list angles to search>",
    "vendor_caveat": "<evaluator (vendor-reported vs third-party) for any vendor number cited>",
    "wiki_handoff_disposition": "<wiki-worthy -> wiki-keeper landing child | no wiki value (internal-only)>",
}

# ``web_searches: 5`` / ``- web_searches : 5`` / ``**lead_ledger: attached**``
# (leading markers and a trailing markdown emphasis/backtick wrapper tolerated —
# an LLM may bold the whole line).
_FIELD_RE = re.compile(
    r"^[ \t]*(?:[-*>][ \t]*)*(?P<name>" + "|".join(M2_FIELDS) + r")\s*:\s*(?P<value>.*?)"
    r"[ \t]*[*`]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# A ``phase: research`` body-frontmatter line. Read locally (tolerant regex)
# rather than importing kanban_phases, so this template never hard-depends on a
# module another in-flight branch may not have landed yet.
_PHASE_LINE_RE = re.compile(r"^\s*[-*>]?\s*phase\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# Behaviour keywords that make a card's deliverable research — the same shape
# the scout SOUL's research protocol keys off.
# The T3 design-card marker (kanban_design.DESIGN_CARD_TYPE_LINE); ensure_m2_block
# keeps it as the first body line of a design card (no import — literal copy is
# intentional: kanban_m2 stays import-light and the T3 tests pin the literal).
_DESIGN_CARD_TYPE_LINE = "card_type: design"


_RESEARCH_RE = re.compile(
    r"(?i)\b(research|reconnaissance|survey|landscape|scout(?:ing)?|"
    r"investigat\w*|evaluate|evaluation|assessment|compare|comparison|"
    r"benchmark(?:ing)?|state[- ]of[- ]the[- ]art|prior art|literature)\b",
)

# An imperative build verb in the TITLE wins: "implement the research index" is
# an ARTIFACT card, not a research card.
_CODE_VERB_RE = re.compile(
    r"(?i)\b(implement|refactor|code|write tests?|add tests?|ship|deploy|"
    r"patch|hotfix|configure|install|migrate|optimize|build|fix|debug)\b",
)

# The decomposer appends the 4 non-code bucket headings (C2.3) to every child
# body. Their EMPTY form is pure routing boilerplate — the literal heading
# "Research-feedback: none" would otherwise classify every titled-without-a-
# code-verb child as research and staple a spurious M2 block onto it. Scrub
# only the placeholder form; a bucket line carrying real content still counts.
# The tail tolerates a sentence-final period and any mix of trailing
# whitespace/backtick/bold wrappers (`None.`, `` `none.` ``, `**none.**`): the
# writers that emit these bodies answer the heading in prose style, and a
# placeholder is a placeholder whether or not it is punctuated. Only
# punctuation may follow the placeholder token — the first real word keeps the
# line (and its research signal) intact.
_BUCKET_BOILERPLATE_RE = re.compile(
    r"^[ \t]*(?:[-*>][ \t]*)*(?:#{1,4}[ \t]*|\*\*)?[ \t]*"
    r"(?:adoption|docs|research-feedback|gate)[ \t]*:[ \t`*]*(?:none|n/?a|-)"
    r"[ \t`*.…]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _coerce_web_searches(value: object) -> Optional[int]:
    """``value`` as a positive int, else ``None`` (never raises)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        m = re.match(r"^\s*(\d+)\s*$", value)
        if m:
            return int(m.group(1))
    return None


def render_m2_block(
    *,
    web_searches: object = None,
    lead_ledger: object = None,
    anchor_note: object = None,
    vendor_caveat: object = None,
    wiki_handoff_disposition: object = None,
) -> str:
    """Render the five-line M2 block in canonical field order.

    A missing/blank/invalid argument falls back to its placeholder, so the
    block ALWAYS renders all five fields with their exact names. ``web_searches``
    is normalized to a bare integer (the floor when the caller's value is
    absent or below it — the reviewer enforces the real count against the
    worker's trace).
    """
    supplied = {
        "web_searches": web_searches,
        "lead_ledger": lead_ledger,
        "anchor_note": anchor_note,
        "vendor_caveat": vendor_caveat,
        "wiki_handoff_disposition": wiki_handoff_disposition,
    }
    lines: list[str] = []
    for name in M2_FIELDS:
        raw = supplied.get(name)
        if name == "web_searches":
            n = _coerce_web_searches(raw)
            lines.append(f"{name}: {M2_MIN_WEB_SEARCHES if n is None or n < M2_MIN_WEB_SEARCHES else n}")
            continue
        text = raw.strip() if isinstance(raw, str) else ""
        lines.append(f"{name}: {text or M2_PLACEHOLDERS[name]}")
    return "\n".join(lines)


def parse_m2_block(text: object) -> dict[str, str]:
    """``{field: value}`` for every M2 field present in ``text`` (raw values)."""
    if not isinstance(text, str) or not text.strip():
        return {}
    found: dict[str, str] = {}
    for m in _FIELD_RE.finditer(text):
        name = m.group("name").lower()
        # First occurrence wins — a card must not shadow its own declaration.
        found.setdefault(name, m.group("value").strip())
    return found


def has_m2_block(text: object) -> bool:
    """True when all five fields are present (values may still be placeholders)."""
    return set(parse_m2_block(text)) >= set(M2_FIELDS)


def missing_m2_fields(text: object) -> list[str]:
    """Present-in-canonical-order list of the M2 fields absent from ``text``."""
    found = parse_m2_block(text)
    return [name for name in M2_FIELDS if name not in found]


def validate_m2_block(text: object) -> list[str]:
    """Human-readable problems with a research card's M2 block; ``[]`` = valid.

    Advisory and never raises: the review lane consumes these strings, and a
    malformed block must not break card generation.
    """
    problems: list[str] = []
    found = parse_m2_block(text)
    for name in M2_FIELDS:
        if name not in found:
            problems.append(f"missing field: {name}")
    if "web_searches" in found:
        n = _coerce_web_searches(found["web_searches"])
        if n is None:
            problems.append(
                f"web_searches must be an integer, got {found['web_searches']!r}"
            )
        elif n < M2_MIN_WEB_SEARCHES:
            problems.append(
                f"web_searches {n} below the floor of {M2_MIN_WEB_SEARCHES} distinct queries"
            )
    return problems


def parse_phase_line(text: object) -> Optional[str]:
    """The canonical phase named by a ``phase:`` frontmatter line, else None."""
    if not isinstance(text, str) or not text.strip():
        return None
    for m in _PHASE_LINE_RE.finditer(text):
        token = re.sub(r"[\s_]+", "-", m.group("value").strip().lower()).strip("-")
        if token:
            return token
    return None


def has_build_verb(title: object) -> bool:
    """True when a card TITLE carries an imperative build verb.

    Public form of the same rule ``is_research_card`` applies: a build verb in
    the title wins over any incidental mention of a research/docs topic, so
    "Implement wiki export for the survey" is an ARTIFACT card, not a docs card.
    Other modules share this rather than re-deriving the verb list.
    """
    return bool(_CODE_VERB_RE.search(title if isinstance(title, str) else ""))


def is_research_card(title: object, body: object = "") -> bool:
    """True when a card's deliverable is research, not a build.

    An explicit ``phase: research`` line wins outright; otherwise a build verb
    in the title makes it ARTIFACT even when the body mentions research in
    passing, and a research keyword anywhere else classifies it research.
    Conservative by design — a card only gains the M2 block when it clearly
    reads as research, so non-research generation stays byte-identical.

    The decomposer's empty 4-bucket headings (``Research-feedback: none``) are
    routing boilerplate, not intent — they are scrubbed before matching, or
    every child without a build verb in its title would be misclassified.
    """
    if parse_phase_line(body) == "research" or parse_phase_line(title) == "research":
        return True
    title_l = (title or "").lower() if isinstance(title, str) else ""
    body_s = body if isinstance(body, str) else ""
    text = f"{title_l}\n{_BUCKET_BOILERPLATE_RE.sub('', body_s)}"
    if _CODE_VERB_RE.search(title_l):
        return False
    return bool(_RESEARCH_RE.search(text))


def ensure_m2_block(body: object, **values: object) -> str:
    """``body`` guaranteed to carry the M2 block, idempotently.

    Any value already present in the body is preserved (a caller's filled
    ``anchor_note`` is never overwritten); the remaining fields render from
    ``values`` or their placeholder. Existing M2 lines are removed before the
    block is re-inserted, so calling this twice yields the same body — a
    partial LLM block can never leave a duplicated or stale field behind. The
    block lands immediately after a ``phase:`` frontmatter line when one
    exists, else at the top.
    """
    text = body if isinstance(body, str) else ""
    found = parse_m2_block(text)
    merged = {name: found.get(name, values.get(name)) for name in M2_FIELDS}
    block = render_m2_block(**merged)
    # Drop existing M2 lines (wherever they sit) so re-rendering is idempotent,
    # then collapse the blank-line seam the removal can leave behind.
    stripped = re.sub(r"\n{3,}", "\n\n", _FIELD_RE.sub("", text)).strip("\n")

    phase_line = ""
    remainder = stripped
    m = _PHASE_LINE_RE.search(stripped)
    if m:
        phase_line = m.group(0).strip()
        remainder = stripped[m.end():].lstrip("\n")
    head = f"{phase_line}\n{block}" if phase_line else block
    # T3 seam: a design card's `card_type: design` marker is its FIRST body
    # line (kanban_design.DESIGN_CARD_TYPE_LINE, pinned by the T3 contract
    # tests). The M2 block must never be reordered above it — insert beneath
    # the marker (and beneath the optional phase line) instead.
    first = stripped.split("\n", 1)[0].strip() if stripped else ""
    if first == _DESIGN_CARD_TYPE_LINE:
        head = "\n".join(part for part in (first, phase_line, block) if part)
        body_rest = stripped[len(first):].lstrip("\n") if remainder == stripped else remainder
        return f"{head}\n\n{body_rest}" if body_rest else head
    return f"{head}\n\n{remainder}" if remainder else head
