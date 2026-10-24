"""D1–D6 design-card contract for the Kanban triage specifier (T3, design §5.1).

A card sitting on the ``system-design`` phase is a reviewable deliverable, not a
prose dump. It MUST carry six named sections:

  D1. Named semantics  — the design's named concepts, one line of definition each
  D2. Invariants       — properties that must hold before/after (review checklist)
  D3. Failure modes    — enumerated risks, each with a mitigation
  D4. Integration surface — the exact files/config/columns the design touches
  D5. Options          — >=2 genuine alternatives with trade-offs + a named
                         recommendation
  D6. Planning readiness — an ordered work list the engineering-planning card
                         consumes directly

This module is the single source of truth for that contract: the specifier
prompt renders it (``kanban_specify._SYSTEM_PROMPT``), the specifier detects
design cards with ``is_design_card`` and folds a structured ``design`` payload
back into the body with ``render_design_sections``/``merge_design_sections``,
and the review lane validates a design card ITEM BY ITEM with
``validate_design_card`` + ``DesignValidation.review_metadata`` (BA-P3.1: the
D1–D6 checklist travels in review metadata, one entry per item).

Dependency-light by design: the T1 phase cascade may be parked for user
verdict, so ``hermes_cli.kanban_phases`` is used when importable and a local
fallback covers ``system-design`` while it is not. Nothing here writes to the
board or raises on malformed input — validation reports, it never gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

#: Canonical phase name whose cards carry the D1–D6 contract.
DESIGN_PHASE = "system-design"

#: Marker used for a section that is genuinely not applicable.
DESIGN_PLACEHOLDER = "none"

#: Emitted/validated in this order — the order the review checklist follows.
DESIGN_SECTION_ORDER: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5", "D6")


@dataclass(frozen=True)
class DesignSection:
    """One section of the design-card contract."""

    code: str
    heading: str
    purpose: str
    requirement: str


DESIGN_SECTIONS: dict[str, DesignSection] = {
    "D1": DesignSection(
        code="D1",
        heading="D1. Named semantics",
        purpose="the design's NAMED concepts, one line of definition each",
        requirement=">=1 item; every item names a concept AND defines it",
    ),
    "D2": DesignSection(
        code="D2",
        heading="D2. Invariants",
        purpose="properties that must hold before/after (these become the review checklist)",
        requirement=">=1 item; each states a property that must hold",
    ),
    "D3": DesignSection(
        code="D3",
        heading="D3. Failure modes with mitigations",
        purpose="enumerated risks of the design, with a mitigation per risk",
        requirement=">=1 item; every item carries 'Mitigation: <...>'",
    ),
    "D4": DesignSection(
        code="D4",
        heading="D4. Integration surface",
        purpose="the exact files/config/columns the design touches",
        requirement=">=1 item; each item names a concrete file, column, config key or env var",
    ),
    "D5": DesignSection(
        code="D5",
        heading="D5. Options with trade-offs and recommendation",
        purpose=">=2 genuine alternatives with trade-offs, plus a named recommendation",
        requirement=">=2 option items (each with + pros / - cons) and a RECOMMENDATION line naming one",
    ),
    "D6": DesignSection(
        code="D6",
        heading="D6. Planning readiness",
        purpose="ordered work list the engineering-planning phase consumes directly",
        requirement=">=1 item, as an ORDERED (numbered) list",
    ),
}


# ---------------------------------------------------------------------------
# Phase handling (T1-aware, T1-optional)
# ---------------------------------------------------------------------------

# Local fallback so the design branch works while the phase cascade branch is
# parked for user verdict (design §9: P2/P3 do not depend on P1).
_LOCAL_PHASE_ALIASES: dict[str, str] = {
    "system-design": DESIGN_PHASE,
    "system-designs": DESIGN_PHASE,
    "design": DESIGN_PHASE,
    "architecture": DESIGN_PHASE,
}


def normalize_phase(value: Any) -> Optional[str]:
    """Canonical phase name for ``value`` (delegates to T1's closed enum when
    ``hermes_cli.kanban_phases`` is importable, else a local fallback)."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:  # T1 landed: keep one source of truth for the enum.
        from hermes_cli.kanban_phases import normalize_phase as _norm

        canonical = _norm(value)
        if canonical:
            return canonical
    except Exception:
        pass
    token = re.sub(r"[\s_]+", "-", value.strip().lower()).strip("-")
    return _LOCAL_PHASE_ALIASES.get(token)


# ``phase: system-design`` frontmatter line (same shape T1 writes).
_PHASE_LINE_RE = re.compile(r"(?im)^\s*[-*>#\s]*phase\s*:\s*(.+?)\s*$")


def body_phase(body: Any) -> Optional[str]:
    """First canonical ``phase: <name>`` frontmatter line in ``body``."""
    if not isinstance(body, str) or not body.strip():
        return None
    for match in _PHASE_LINE_RE.finditer(body):
        canonical = normalize_phase(match.group(1))
        if canonical:
            return canonical
    return None


# ``card_type: design`` frontmatter line — this card's own marker. Deliberately
# a DIFFERENT key from T1's ``phase:`` so the two changes never collide.
_CARD_TYPE_LINE_RE = re.compile(r"(?im)^\s*[-*>#\s]*card[ _-]?type\s*:\s*(.+?)\s*$")

#: Frontmatter line written as the FIRST body line of a design card.
DESIGN_CARD_TYPE_LINE = "card_type: design"


def body_card_type(body: Any) -> Optional[str]:
    """First ``card_type: <value>`` frontmatter line in ``body`` (lowercased)."""
    if not isinstance(body, str) or not body.strip():
        return None
    for match in _CARD_TYPE_LINE_RE.finditer(body):
        token = match.group(1).strip().strip("\"'").lower()
        if token:
            return token
    return None


# Design cards are detected on explicit signals only — never on a stray word.
# Deliberately conservative: a title like "design a migration plan" (an
# engineering-planning card) must NOT match, and neither must a card ABOUT the
# template ("implement the design-card template").
#
# The bare ``system-design`` alternative carries a negative lookahead: a title
# that merely names the PHASE — "… (system-design children get no design
# contract)", "fix the system-design phase cascade" — is an implementation or
# ops card, not a design. Only when the phrase stands as the design subject
# ("System design for the kanban decomposer") does it count. Without the
# lookahead the phase vocabulary leaks into the heuristic, which is the same
# false-positive class as "implement the design-card template".
_DESIGN_NOUNS = r"(?:pipeline|contract|schema|api|system|architecture|engine|protocol|interface|component|service)"
_PHASE_MENTION_TAIL = (
    r"(?!\s*(?:children|child|card|cards|phase|phases|cascade|step|steps|"
    r"tag|tags|key|keys|column|columns|field|fields)\b)"
)
_DESIGN_INTENT_RE = re.compile(
    rf"(?i)\b(?:"
    rf"system[- ]design{_PHASE_MENTION_TAIL}|"
    rf"design[- ]doc(?:ument)?|architecture\s+(?:design|doc(?:ument)?)|"
    rf"design\s+card\b(?![\s-]*template)|"
    rf"decide\s+how\s+to\s+build|"
    rf"design\s+(?:the|a|an|this)\s+(?:\w+\s+){{0,3}}?{_DESIGN_NOUNS}\b"
    rf")\b"
)


def is_design_card(
    *, card_type: Any = None, phase: Any = None, title: Any = "", body: Any = "",
    title_heuristic: bool = True,
) -> bool:
    """True when a triage idea is a DESIGN card (gets the D1–D6 contract).

    Explicit signals win and are checked in order: an emitted ``card_type``
    ("design"/"task"), a phase of ``system-design`` (argument or body
    frontmatter), a body already carrying >=2 D-sections, and finally a
    conservative design-intent phrase in the title. A plain card never
    matches by accident.

    ``title_heuristic=False`` drops that last signal, leaving only the EXPLICIT
    ones. The specifier needs the title fallback (the model may classify by
    prose alone); the decomposer does not — it assigns ``phase`` itself, so it
    must never pull an ordinary child into the contract off a title that merely
    mentions design (e.g. "Fix the system-design phase cascade"). Callers on
    the decomposer path pass ``title_heuristic=False`` so non-design children
    stay byte-identical.
    """
    if isinstance(card_type, str):
        token = card_type.strip().lower()
        if token in {"design", "design-card", "design_card", "system-design"}:
            return True
        if token in {"task", "plain", "work", "implementation"}:
            return False
    if body_card_type(body) in {"design", "design-card", "design_card", "system-design"}:
        return True
    if normalize_phase(phase) == DESIGN_PHASE:
        return True
    if body_phase(body) == DESIGN_PHASE:
        return True
    present = [code for code, lines in split_sections(body).items() if lines]
    if len(present) >= 2:
        return True
    if not title_heuristic:
        return False
    return bool(_DESIGN_INTENT_RE.search(str(title or "")))


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

# A section heading is ``**D1. ...**`` / ``## D1. ...`` / ``D1. ...``. The ``.``
# or ``:`` separator is required so prose like "D1–D6 contract" is not a heading.
_SECTION_RE = re.compile(
    r"(?im)^[ \t]*(?:\*\*|__|#{1,6}[ \t]*)?\(?D[ \t]*([1-6])\)?[ \t]*[.:]"
)
# Any other markdown heading ends the current section region.
_OTHER_HEADING_RE = re.compile(r"(?im)^[ \t]*(?:\*\*[^*]+\*\*|#{1,6}[ \t]*\S)")
_MARKER_RE = re.compile(r"^[ \t]*(?P<marker>[-*+]|\d+[.)])[ \t]+")
_NUMBERED_RE = re.compile(r"^[ \t]*\d+[.)][ \t]+")
# A recommendation line is structural: it terminates the previous item instead
# of being folded into it as a wrapped continuation.
_STRUCTURAL_LINE_RE = re.compile(r"(?i)^\s*(?:[-*]\s*)?\*{0,2}(?:recommendation|recommended|recommend)\b")


@dataclass
class _Item:
    text: str
    marker: str = ""

    @property
    def numbered(self) -> bool:
        return bool(_NUMBERED_RE.match(f"{self.marker} "))

    @property
    def bulleted(self) -> bool:
        return bool(self.marker) and not self.numbered


def _heading_code(line: str) -> Optional[str]:
    match = _SECTION_RE.match(line)
    return f"D{match.group(1)}" if match else None


def _items_in(region: Sequence[str]) -> list[_Item]:
    """Item lines of one section region: bullets/numbered lines start a new
    item, wrapped continuation lines are folded into the previous item, and an
    un-bulleted prose line is its own item.

    The explicit placeholder (``- none``, the marker the renderer writes for a
    section that was not supplied) is NOT content and is dropped here, so every
    consumer — parsing, validation, review metadata — agrees that the section
    is empty rather than counting a placeholder as an item.
    """
    items: list[_Item] = []
    for line in region:
        if not line.strip():
            continue
        match = _MARKER_RE.match(line)
        if match:
            text = line[match.end():].strip()
            if text.lower() == DESIGN_PLACEHOLDER:
                continue
            items.append(_Item(text=text, marker=match.group("marker")))
            continue
        text = line.strip()
        if text.lower() == DESIGN_PLACEHOLDER:
            continue
        if _STRUCTURAL_LINE_RE.match(line):
            items.append(_Item(text=text))
            continue
        if items and not _heading_code(line) and not _OTHER_HEADING_RE.match(line):
            items[-1].text = f"{items[-1].text} {text}".strip()
        else:
            items.append(_Item(text=text))
    return [i for i in items if i.text]


def _strip_markup(text: str) -> str:
    """Drop bold/backtick markup and collapse whitespace (for prose messages)."""
    return re.sub(r"\s+", " ", text.replace("**", "").replace("__", "").replace("`", "")).strip()


def _collapse(text: str) -> str:
    """Collapse whitespace, keeping inline markup (``backticks`` mark exact
    identifiers — a reviewer needs them)."""
    return re.sub(r"\s+", " ", text).strip()


def _item_text(item: _Item) -> str:
    """Display text of one item: the numbered marker is KEPT (D6's whole point
    is the ORDER), a bullet marker is dropped (decoration)."""
    return _collapse(f"{item.marker} {item.text}" if item.numbered else item.text)


def split_sections(body: Any) -> dict[str, list[_Item]]:
    """Split ``body`` into ``{D1..D6: [items]}`` (empty lists when absent)."""
    sections: dict[str, list[_Item]] = {code: [] for code in DESIGN_SECTION_ORDER}
    if not isinstance(body, str) or not body.strip():
        return sections
    current: Optional[str] = None
    region: list[str] = []
    for line in body.splitlines():
        code = _heading_code(line)
        if code:
            if current:
                sections[current] = _items_in(region)
            current, region = code, []
            continue
        if current and _OTHER_HEADING_RE.match(line):
            sections[current] = _items_in(region)
            current, region = None, []
            continue
        if current:
            region.append(line)
    if current:
        sections[current] = _items_in(region)
    return sections


def parse_design_sections(body: Any) -> dict[str, list[str]]:
    """Item-by-item text of each D-section (placeholder ``none`` dropped).

    Inline markup is preserved: a backticked identifier in D1/D4 is the exact
    concept/file the reviewer must check, not decoration.
    """
    parsed: dict[str, list[str]] = {}
    for code, items in split_sections(body).items():
        parsed[code] = [_item_text(item) for item in items]
    return parsed


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------

def _as_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Iterable):
        return [v for v in value if v is not None]
    return [value]


def _pick(item: Any, *keys: str) -> str:
    """First non-empty string value among ``keys`` (dict items) or the item
    itself when it is already a scalar. ``None`` / containers yield ``""``."""
    if item is None:
        return ""
    if isinstance(item, Mapping):
        for key in keys:
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    if isinstance(item, (list, tuple, set, dict, bytes)):
        text = item.decode("utf-8", "replace") if isinstance(item, bytes) else ""
        return text.strip()
    return str(item).strip()


def _render_d1(value: Any) -> list[str]:
    out = []
    for item in _as_items(value):
        name, definition = _pick(item, "name", "concept", "term"), _pick(item, "definition", "defines", "meaning")
        if name and definition:
            out.append(f"- `{name}` — {definition}")
        elif name or definition:
            out.append(f"- {name or definition}")
    return out


def _render_d2(value: Any) -> list[str]:
    return [f"- {text}" for text in (_pick(i, "invariant", "property", "statement") for i in _as_items(value)) if text]


def _render_d3(value: Any) -> list[str]:
    out = []
    for item in _as_items(value):
        failure = _pick(item, "failure", "failure_mode", "risk", "mode")
        mitigation = _pick(item, "mitigation", "mitigate", "guard")
        if failure and mitigation:
            out.append(f"- {failure} — Mitigation: {mitigation}")
        elif failure or mitigation:
            out.append(f"- {failure or mitigation}")
    return out


def _render_d4(value: Any) -> list[str]:
    out = []
    for item in _as_items(value):
        surface, change = _pick(item, "surface", "file", "path", "location"), _pick(item, "change", "touches", "what")
        if surface and change:
            out.append(f"- `{surface}` — {change}")
        elif surface or change:
            out.append(f"- {surface or change}")
    return out


def _signless(text: str) -> str:
    """Drop a leading ``+``/``-`` trade-off sign the model may have included.

    ``_render_d5`` adds its own ``+``/``-`` markers, so a payload written as
    ``"+cheap"`` would otherwise render the doubled ``+ +cheap`` a reviewer has
    to re-read. Only the marker is normalised — the wording is untouched.
    """
    return text.lstrip().lstrip("+-").strip()


def _render_d5(value: Any, recommendation: Any = None) -> list[str]:
    """Render D5 options; accepts a list of options, or a mapping carrying
    ``options`` + ``recommendation``."""
    options: Any = value
    rec = recommendation
    if isinstance(value, Mapping) and ("options" in value or "alternatives" in value):
        options = value.get("options") or value.get("alternatives")
        rec = rec or value.get("recommendation") or value.get("recommended")
    out: list[str] = []
    for idx, item in enumerate(_as_items(options)):
        label = _pick(item, "label", "id")
        if not label:
            label = chr(ord("A") + idx) if idx < 26 else str(idx + 1)
        name = _pick(item, "name", "option", "title")
        pros = _signless(_pick(item, "pros", "pro", "upside", "tradeoffs"))
        cons = _signless(_pick(item, "cons", "con", "downside"))
        line = f"- Option {label}"
        if name:
            line += f": {name}"
        details = []
        if pros:
            details.append(f"+ {pros}")
        if cons:
            details.append(f"- {cons}")
        if details:
            line += " — " + "; ".join(details)
        out.append(line)
    rec_text = "" if rec is None else (
        rec.strip() if isinstance(rec, str) else _pick(rec, "label", "option", "name", "text", "reason")
    )
    if rec_text:
        # Blank line so the recommendation is never folded into the last option
        # as a wrapped continuation line.
        out.append(f"\nRECOMMENDATION: {rec_text}")
    return out


def _render_d6(value: Any) -> list[str]:
    out = []
    for idx, item in enumerate(_as_items(value)):
        step = _pick(item, "step", "task", "work", "text")
        if step:
            out.append(f"{idx + 1}. {step}")
    return out


_RENDERERS = {
    "D1": _render_d1,
    "D2": _render_d2,
    "D3": _render_d3,
    "D4": _render_d4,
    "D6": _render_d6,
}

#: Structured-payload keys accepted for each section (tolerant of the LLM's
#: casing / wording: ``d1``, ``D1:``, ``named_semantics``, ...).
_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "D1": ("D1", "d1", "named_semantics", "semantics"),
    "D2": ("D2", "d2", "invariants"),
    "D3": ("D3", "d3", "failure_modes", "failures"),
    "D4": ("D4", "d4", "integration_surface", "integration"),
    "D5": ("D5", "d5", "options", "options_and_tradeoffs"),
    "D6": ("D6", "d6", "planning_readiness", "planning", "work_list"),
}


def payload_section(items: Mapping[str, Any], code: str) -> Any:
    """Pull one section's value out of a structured design payload.

    Accepts the short keys (``D1``/``d1``), the semantic names
    (``named_semantics``, ``invariants``, ...) and the rendered heading text
    (``"D1. Named semantics"``) — whichever the model chose.
    """
    if not isinstance(items, Mapping):
        return None
    for key in _PAYLOAD_KEYS[code]:
        if key in items:
            return items[key]
    heading = DESIGN_SECTIONS[code].heading
    for key, value in items.items():
        if isinstance(key, str) and (key.strip() == heading or _heading_code(key.strip()) == code):
            return value
    return None


def render_section(code: str, payload: Any, recommendation: Any = None) -> list[str]:
    """Rendered lines for ONE section: structured payload items, or the
    explicit ``none`` placeholder when nothing usable was supplied."""
    value = payload_section(payload or {}, code) if isinstance(payload, Mapping) else None
    if value is None and isinstance(payload, Mapping) and code in payload:
        value = payload[code]
    if code == "D5":
        lines = _render_d5(value, recommendation)
    elif code in _RENDERERS:
        lines = _RENDERERS[code](value)
    else:  # pragma: no cover — DESIGN_SECTION_ORDER is closed
        lines = []
    return lines or [f"- {DESIGN_PLACEHOLDER}"]


def render_design_sections(items: Mapping[str, Any]) -> str:
    """Instantiate the design-card template: ALL six sections, in order, from a
    structured ``{D1..D6: ...}`` payload (keys may also be the heading text).

    A section with no usable items renders the explicit ``none`` placeholder so
    the heading is always present — a reviewer can then see the section was
    considered, and ``validate_design_card`` reports it as a violation.
    """
    blocks: list[str] = []
    for code in DESIGN_SECTION_ORDER:
        section = DESIGN_SECTIONS[code]
        lines = render_section(
            code, items, items.get("recommendation") if isinstance(items, Mapping) else None,
        )
        blocks.append(f"**{section.heading}**\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def render_design_skeleton() -> str:
    """The empty design-card template (all six headings, placeholders)."""
    return render_design_sections({})


def _strip_design_regions(body: Any) -> str:
    """``body`` with every D-section region (heading + its lines) removed, the
    non-contract text preserved. Shared by ``merge_design_sections`` and
    ``ensure_design_contract`` so the two folds agree on what a region is."""
    text = body if isinstance(body, str) else ""
    lines = text.splitlines()
    kept: list[str] = []
    idx = 0
    while idx < len(lines):
        if _heading_code(lines[idx]):
            idx += 1
            while idx < len(lines) and not _heading_code(lines[idx]) and not _OTHER_HEADING_RE.match(lines[idx]):
                idx += 1
            continue
        kept.append(lines[idx])
        idx += 1
    return "\n".join(kept).strip()


def merge_design_sections(body: Any, block: str) -> str:
    """Return ``body`` with any existing D-section regions replaced by
    ``block`` (idempotent: merging the same block twice yields one copy)."""
    head = _strip_design_regions(body)
    if not head:
        return block.strip()
    return f"{head}\n\n{block.strip()}"


def _existing_section_lines(items: Sequence[_Item]) -> list[str]:
    """Body items re-emitted faithfully: an ordered item keeps its numbering
    (D6's whole point is the order), a bulleted one is re-bulleted, and a
    marker-less one stays marker-less — so ``ensure_design_contract`` is
    idempotent on a body it already folded.

    A structural line (the D5 ``RECOMMENDATION:`` line) is re-emitted with the
    blank line ``_render_d5`` puts before it, which is what keeps it from being
    folded into the last option as a wrapped continuation on the next parse.
    """
    out: list[str] = []
    for item in items:
        text = _item_text(item)
        if not text:
            continue
        if item.numbered:
            out.append(text)
        elif item.bulleted:
            out.append(f"- {text}")
        else:
            if out and _STRUCTURAL_LINE_RE.match(text):
                out.append("")
            out.append(text)
    return out


def _with_design_marker(body: str, phase: Any = None) -> str:
    """``body`` whose first line is the ``card_type: design`` marker (written
    once, never duplicated) and which carries a ``phase: <name>`` frontmatter
    line when ``phase`` is given and the body has none."""
    text = body or ""
    lines = text.splitlines()
    header: list[str] = []
    if not any(body_card_type(line) for line in lines[:3]):
        header.append(DESIGN_CARD_TYPE_LINE)
    canonical = normalize_phase(phase)
    if canonical and body_phase(text) is None:
        header.append(f"phase: {canonical}")
    if not header:
        return text.strip()
    return "\n".join([*header, "", text.strip()]) if text.strip() else "\n".join(header)


def ensure_design_contract(body: Any, payload: Any = None, *, phase: Any = None) -> str:
    """``body`` guaranteed to carry ALL SIX D-sections, in D1..D6 order.

    The idempotent, LOSSLESS fold: used wherever no structured ``design``
    payload is guaranteed (decomposer children, backfill of a card produced
    before the contract existed). Distinct from the specifier's
    ``merge_design_sections`` path, which renders purely from the payload:

      * a section the body already carries keeps its item text VERBATIM (only
        re-emitted, so the numbering of an ordered list survives);
      * a section the body lacks is rendered from ``payload`` when the caller
        has one, else the explicit ``none`` placeholder — the heading is always
        present, so ``validate_design_card`` reports the gap by section instead
        of a silent prose dump;
      * the ``card_type: design`` marker is written as the FIRST body line (the
        invariant the T3 tests and the M2 seam both pin);
      * ``phase`` is written as a frontmatter line when the body carries none.

    Running it twice yields the same body: the D-regions are re-derived from
    what is already there.
    """
    text = body if isinstance(body, str) else ""
    raw = split_sections(text)
    items: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}
    recommendation = items.get("recommendation") if isinstance(items, Mapping) else None
    blocks: list[str] = []
    for code in DESIGN_SECTION_ORDER:
        lines = _existing_section_lines(raw.get(code) or [])
        if not lines:
            lines = render_section(code, items, recommendation)
        blocks.append(f"**{DESIGN_SECTIONS[code].heading}**\n" + "\n".join(lines))
    head = _strip_design_regions(text)
    merged = "\n\n".join([head, *blocks]) if head else "\n\n".join(blocks)
    return _with_design_marker(merged, phase=phase)


# ---------------------------------------------------------------------------
# Validation (BA-P3.1 item-by-item)
# ---------------------------------------------------------------------------

# A concrete integration surface: a backticked identifier/path (no spaces — a
# backticked PROSE phrase is not a surface), a path-ish filename, a dotted
# module path, a snake_case identifier, or a named schema/config surface.
# The last two alternations exist because a design card legitimately names
# surfaces like ``envelope field idempotency_key`` — a concrete field of a
# schema the design introduces — which carries no dot-path and no keyword. The
# token rules require a separator BETWEEN word characters (``\w[._/]\w``,
# ``_`` between word chars), so prose such as "something vague" stays rejected.
_SURFACE_RE = re.compile(
    r"`[^`\s]+`"
    r"|`[^`]*\w[._/]\w[^`]*`"
    r"|[\w./-]+\.(?:py|md|ya?ml|json|sql|toml|sh|ini|cfg|txt)"
    r"|\b[\w-]+(?:\.[\w-]+){2,}\b"
    r"|\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"
    r"|(?i:\b(?:column|table|schema|config key|env var|frontmatter field|"
    r"tasks\.[a-z_]+|profile\.yaml|config\.yaml)\b)"
)
_MITIGATION_RE = re.compile(r"(?i)\bmitigat\w*")
_OPTION_LABEL_RE = re.compile(r"(?i)\boption\s+([A-Za-z0-9][\w.-]*)")
_TRADEOFF_RE = re.compile(r"(?i)(?:\+\s*\S|[-—]\s*\S|\btrade-?offs?\b|\bpros?\b|\bcons?\b)")
_RECOMMENDATION_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?\*{0,2}(?:recommendation|recommended|recommend)\*{0,2}\s*[:\-—]?\s*(?P<text>.+?)\s*\*{0,2}$"
)
_D1_SEPARATOR_RE = re.compile(r"\s(?:—|–|-|:)\s")


@dataclass
class SectionReport:
    """Item-by-item report for one D-section."""

    code: str
    heading: str
    present: bool = False
    items: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.present and not self.issues

    def as_metadata(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "present": self.present,
            "ok": self.ok,
            "items": list(self.items),
            "issues": list(self.issues),
        }


@dataclass
class DesignPayloadValidation:
    """Verdict on a STRUCTURED ``design`` payload emitted by the model.

    Mirrors ``DesignValidation`` but runs on the JSON the specifier receives,
    before it is rendered — so a malformed payload is reported (and the
    rendered body still carries every heading) instead of silently producing a
    half-empty card or a parse failure.
    """

    ok: bool
    violations: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    item_counts: dict[str, int] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "design_payload_ok": self.ok,
            "missing_sections": list(self.missing),
            "violations": list(self.violations),
            "item_counts": dict(self.item_counts),
        }


def validate_design_payload(payload: Any) -> DesignPayloadValidation:
    """Validate a structured ``design`` payload against the D1–D6 contract.

    Reports, never raises: an unusable payload yields ``ok=False`` with a
    violation per section, and the caller still renders all six headings (with
    ``none`` placeholders) so the card is reviewable rather than broken.
    """
    if not isinstance(payload, Mapping):
        return DesignPayloadValidation(
            ok=False,
            violations=["design payload is not an object"],
            missing=list(DESIGN_SECTION_ORDER),
            item_counts={code: 0 for code in DESIGN_SECTION_ORDER},
        )
    violations: list[str] = []
    missing: list[str] = []
    counts: dict[str, int] = {}
    for code in DESIGN_SECTION_ORDER:
        lines = render_section(code, payload, payload.get("recommendation"))
        items = [line for line in lines if line.strip().lstrip("- ").strip().lower() != DESIGN_PLACEHOLDER]
        counts[code] = len(items)
        if not items:
            missing.append(code)
            violations.append(f"{code}: missing (rendered as '{DESIGN_PLACEHOLDER}')")
    if counts["D5"] and counts["D5"] < 2:
        violations.append("D5: fewer than 2 options")
    if not str(payload.get("recommendation") or "").strip():
        violations.append("D5: no named recommendation in the payload")
    return DesignPayloadValidation(
        ok=not violations, violations=violations, missing=missing, item_counts=counts,
    )


@dataclass
class DesignValidation:
    """Item-by-item verdict for a design card body (never raises)."""

    is_design: bool
    sections: dict[str, SectionReport]
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.is_design and not self.violations and all(r.ok for r in self.sections.values())

    def review_metadata(self) -> dict[str, Any]:
        """BA-P3.1 payload: one entry per section, one ``items`` list each."""
        payload: dict[str, Any] = {
            "design_contract": "D1-D6",
            "is_design_card": self.is_design,
            "design_ok": self.ok,
            "violations": list(self.violations),
        }
        for code in DESIGN_SECTION_ORDER:
            payload[code] = self.sections[code].as_metadata()
        return payload


def _validate_d1(report: SectionReport, items: list[_Item]) -> None:
    if not items:
        report.issues.append("no named concepts (>=1 required)")
    for pos, item in enumerate(items, 1):
        text = _strip_markup(item.text)
        if not _D1_SEPARATOR_RE.search(text):
            report.issues.append(f"item {pos} lacks a definition separated by '—' (named concept only): {text!r}")


def _validate_d2(report: SectionReport, items: list[_Item]) -> None:
    if not items:
        report.issues.append("no invariants (>=1 required)")


def _validate_d3(report: SectionReport, items: list[_Item]) -> None:
    if not items:
        report.issues.append("no failure modes (>=1 required)")
    for pos, item in enumerate(items, 1):
        if not _MITIGATION_RE.search(_strip_markup(item.text)):
            report.issues.append(f"item {pos} has no mitigation: {_strip_markup(item.text)!r}")


def _validate_d4(report: SectionReport, items: list[_Item]) -> None:
    if not items:
        report.issues.append("no integration surface items (>=1 required)")
        return
    concrete = 0
    for pos, item in enumerate(items, 1):
        text = _strip_markup(item.text)
        if _SURFACE_RE.search(item.text) or _SURFACE_RE.search(text):
            concrete += 1
        else:
            report.issues.append(f"item {pos} names no concrete file/column/config key: {text!r}")
    if concrete == 0:
        report.issues.append("no exact file, column, config key or env var referenced anywhere")


def _validate_d5(report: SectionReport, items: list[_Item]) -> None:
    labelled: list[tuple[str, _Item]] = []
    fallback: list[_Item] = []
    rec_text = ""
    for item in items:
        if _RECOMMENDATION_RE.match(item.text or ""):
            if not rec_text:
                rec_text = _RECOMMENDATION_RE.match(item.text or "").group("text") or ""
            continue
        text = _strip_markup(item.text)
        match = _OPTION_LABEL_RE.search(text)
        if match:
            labelled.append((match.group(1), item))
        else:
            fallback.append(item)
    options = labelled or [(_OPTION_LABEL_RE.search(_strip_markup(i.text)) or None, i) for i in fallback]
    counts = len(labelled) if labelled else len(fallback)
    if counts < 2:
        report.issues.append(f"only {counts} option(s); >=2 genuine options required")
    for item in [i for _, i in options]:
        text = _strip_markup(item.text)
        if not _TRADEOFF_RE.search(text):
            report.issues.append(f"option states no trade-offs (+ pros / - cons): {text!r}")
    if not rec_text:
        report.issues.append("no named recommendation (RECOMMENDATION: Option <X>)")
    elif labelled and not any(
        re.search(rf"(?i)\boption\s+{re.escape(label)}\b", rec_text) for label, _ in labelled
    ):
        report.issues.append(
            "recommendation does not name one of the option labels: " + ", ".join(f"Option {l}" for l, _ in labelled)
        )


def _validate_d6(report: SectionReport, items: list[_Item]) -> None:
    if not items:
        report.issues.append("no planning-readiness work list (>=1 ordered item required)")
        return
    if not any(item.numbered for item in items):
        report.issues.append("work list is not ordered (a numbered list is required)")


def validate_design_card(body: Any, *, is_design: Optional[bool] = None) -> DesignValidation:
    """Validate a card body against the D1–D6 contract, item by item.

    ``is_design`` defaults to ``is_design_card(body=body)``. Returns a report;
    it never raises and never rewrites the body — enforcement lives in the
    review lane (design §5.3), this is its checklist.
    """
    raw = split_sections(body)
    if is_design is None:
        is_design = is_design_card(body=body)
    sections: dict[str, SectionReport] = {}
    for code in DESIGN_SECTION_ORDER:
        items = raw.get(code) or []
        section = DESIGN_SECTIONS[code]
        report = SectionReport(
            code=code, heading=section.heading, present=bool(items),
            items=[_item_text(i) for i in items],
        )
        if code == "D1":
            _validate_d1(report, items)
        elif code == "D2":
            _validate_d2(report, items)
        elif code == "D3":
            _validate_d3(report, items)
        elif code == "D4":
            _validate_d4(report, items)
        elif code == "D5":
            _validate_d5(report, items)
        else:
            _validate_d6(report, items)
        sections[code] = report
    violations = [f"{code}: {'; '.join(r.issues)}" for code, r in sections.items() if r.issues]
    return DesignValidation(is_design=bool(is_design), sections=sections, violations=violations)


def design_review_metadata(body: Any, *, is_design: Optional[bool] = None) -> dict[str, Any]:
    """Convenience wrapper for the review lane's ``kanban_request_review``
    metadata (BA-P3.1: D1–D6 verified item-by-item)."""
    return validate_design_card(body, is_design=is_design).review_metadata()


# ---------------------------------------------------------------------------
# Prompt block (single source of truth for BOTH card templates)
# ---------------------------------------------------------------------------


def design_prompt_block() -> str:
    """The D1–D6 contract as the card templates' prompt block.

    Rendered from ``DESIGN_SECTIONS`` so the specifier prompt, the decomposer
    prompt and the review checklist cannot drift apart. Both templates append
    this verbatim — the specifier's copy is byte-identical to the block that
    preceded this module owning it (pinned by the T3 contract tests).
    """
    lines = [
        "",
        "DESIGN CARDS (card_type: design) — the D1–D6 content contract",
        "  When the idea is a system design / architecture decision (\"decide how",
        "  to build X\", \"design the Y pipeline\"), it is a DESIGN CARD. Emit",
        "  \"card_type\": \"design\" and put the design in a structured \"design\"",
        "  object. The body MUST then carry ALL SIX of these sections, with their",
        "  exact headings, AFTER the four standard sections above:",
        "",
    ]
    for code in DESIGN_SECTION_ORDER:
        section = DESIGN_SECTIONS[code]
        lines.append(f"  **{section.heading}** — {section.purpose}")
        lines.append(f"      ({section.requirement})")
    lines += [
        "",
        "  Rules for a design card:",
        "    - Always emit all six headings, in D1..D6 order, even when a section",
        "      genuinely does not apply — write its placeholder \"none\" then.",
        "    - Give the same content TWICE: once structured (the \"design\" object)",
        "      and once inside \"body\" as the six sections. The body wins; the",
        "      structured object is the machine-readable copy.",
        "    - D5 must offer >=2 GENUINE alternatives (not a straw man), each with",
        "      trade-offs, and end with a line naming the recommendation, e.g.",
        "      \"RECOMMENDATION: Option A ...\". The recommendation MUST name one of",
        "      the option labels.",
        "    - D6 is an ORDERED work list (numbered) the engineering-planning card",
        "      consumes directly — no prose.",
        "    - A design card that does not apply to this idea (an ordinary task)",
        "      must NOT be labelled design and must NOT carry D1–D6.",
        "",
        "  Structured shape for the \"design\" object (all six keys):",
        "",
        '    "design": {',
        '      "D1": [{"name": "<concept>", "definition": "<one line>"}],',
        '      "D2": [{"invariant": "<property that must hold>"}],',
        '      "D3": [{"failure": "<risk>", "mitigation": "<how it is handled>"}],',
        '      "D4": [{"surface": "<exact file/column/config key>", "change": "<what changes>"}],',
        '      "D5": {"options": [{"label": "A", "name": "<alt>", "pros": "<pro>", "cons": "<con>"}]},',
        '      "recommendation": "Option A — <why>",',
        '      "D6": [{"step": "<ordered step>"}]',
        '    }',
    ]
    return "\n".join(lines)
