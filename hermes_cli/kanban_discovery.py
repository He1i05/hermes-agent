"""Discovery-report card convention (T5, design §8.2 / phase P4).

A card that produced a flag-worthy mid-flight finding MAY declare itself a
DISCOVERY-REPORT source. The declaration is a card-body frontmatter line and a
standardized summary section:

    discovery: flag-med | flag-high

    **DISCOVERY SUMMARY**
    - finding: <one line: what was discovered>
    - confidence: <high|medium|low + why>
    - impact-lane: <phase the finding lands on, or none>
    - suggested-action: <continue|refine|pivot>
    - affected-card-ids: <t_..., t_... or none>

    flag-med   an observation worth noting; no immediate action.
    flag-high  a finding that, if true, changes the goal or invalidates
               in-flight sibling work -> the orchestrator evaluates it (design
               §8.3) and may act on `affected-card-ids`.

SCOPE — this module is the single source of truth for the CONVENTION only:
the frontmatter key + vocabulary, the section heading and its five exact field
names, render/parse/validate, and the prompt block the two card templates
(``kanban_specify`` / ``kanban_decompose``) render from. It is deliberately
the same shape as the sibling ``hermes_cli/kanban_m2.py`` (T4 research
acceptance block) and ``hermes_cli/kanban_design.py`` (T3 design contract) so
the phases cannot drift apart.

What this module does NOT do (owned elsewhere, by design):
  * STOP mechanics / graceful park / pivot drill — UG-3 (``t_b6044898``,
    runbook ``~/.hermes/docs/runbooks/t5-graceful-stop-runbook.md``). P4
    excludes them explicitly; nothing here blocks, parks or kills a card.
  * The orchestrator's evaluation step + PIVOT DECISION record — the sibling
    runbook card (``t_1d0325bc``). This module only guarantees the declaration
    it is evaluated from is well-formed.

Backward compatibility: the convention is DECLARATION-DRIVEN. Nothing is
injected into a card that did not declare ``discovery:``, so ordinary and
plain research cards render byte-identically to before. (Injecting a section
onto cards that merely *look* report-shaped is the spurious-injection defect
class fixed on ``t_71b6eca9`` — designed out here rather than patched later.)
"""

from __future__ import annotations

import re
from typing import Any, Optional

#: The exact frontmatter key. Do NOT rename — templates + review lane key off it.
DISCOVERY_FIELD = "discovery"

#: The exact level vocabulary, low -> high. Only these two strings are valid.
DISCOVERY_LEVELS: tuple[str, ...] = ("flag-med", "flag-high")

#: The standardized section heading (rendered bold, per the specifier's house
#: section style: the board UI renderer turns ``**X**`` into ``<strong>`` and
#: dash bullets into a ``<ul>`` while headings are clamped to ``#``-``####``).
DISCOVERY_SECTION_HEADING = "DISCOVERY SUMMARY"

#: The five exact field names of the section, in render order.
DISCOVERY_SECTION_FIELDS: tuple[str, ...] = (
    "finding",
    "confidence",
    "impact-lane",
    "suggested-action",
    "affected-card-ids",
)

#: ``suggested-action`` vocabulary (design §8.3 decision space).
DISCOVERY_ACTIONS: tuple[str, ...] = ("continue", "refine", "pivot")

#: Unfilled-field guidance. A rendered-but-unfilled field is visibly a template
#: instruction, never a claim the card has already reported — so a reviewer can
#: tell a filled report from a placeholder at a glance.
DISCOVERY_PLACEHOLDERS: dict[str, str] = {
    "finding": "<one line: what was discovered>",
    "confidence": "<high | medium | low + why>",
    "impact-lane": "<the phase/area the finding lands on, or none>",
    "suggested-action": "<continue | refine | pivot>",
    "affected-card-ids": "<t_..., t_... — or none when no in-flight card is impacted>",
}

#: Tolerated spellings of the two levels (models write ``flag_high``, ``high``).
_LEVEL_ALIASES: dict[str, str] = {
    "flag-med": "flag-med",
    "flag_med": "flag-med",
    "med": "flag-med",
    "medium": "flag-med",
    "flag-high": "flag-high",
    "flag_high": "flag-high",
    "high": "flag-high",
}

#: The frontmatter line, wherever it sits: ``discovery: flag-high`` /
#: ``- discovery : flag_high`` / ``**discovery: flag-med**``.
_LEVEL_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*>][ \t]*)*(?:\*\*|__)?[ \t]*" + DISCOVERY_FIELD + r"[ \t]*(?:\*\*|__)?[ \t]*:"
    r"[ \t]*(?P<value>.*?)[ \t]*[*`]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: A body-frontmatter ``phase:`` line — the discovery line is placed after it.
_PHASE_LINE_RE = re.compile(r"^[ \t]*[-*>]?[ \t]*phase[ \t]*:[ \t]*(?P<value>.+?)[ \t]*$",
                            re.IGNORECASE | re.MULTILINE)

#: The section heading alone on a line, bold or ``#``-prefixed.
_HEADING_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*" + DISCOVERY_SECTION_HEADING.replace(" ", r"[ \t]+")
    + r"[ \t]*(?:\*\*|__)?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: ``- finding: ...`` / ``**impact-lane**: ...`` / ``suggested-action : ...``.
_FIELD_RE = re.compile(
    r"^[ \t]*(?:[-*+][ \t]*)*(?:\*\*|__)?[ \t]*(?P<name>"
    + "|".join(re.escape(name) for name in DISCOVERY_SECTION_FIELDS)
    + r")[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(?P<value>.*?)[ \t]*[*`]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: A placeholder is an angle-bracket instruction, never a real value.
_PLACEHOLDER_RE = re.compile(r"^<.*>$")


def normalize_level(value: Any) -> Optional[str]:
    """The canonical level for ``value``, else ``None``.

    Separators fold to ``-`` so ``flag_high`` / ``Flag-High`` normalize; the
    enum is CLOSED — an unknown value is rejected rather than coerced, so a
    typo can never be read downstream as a real severity.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    token = re.sub(r"[\s_]+", "-", value.strip().lower()).strip("-")
    return _LEVEL_ALIASES.get(token)


def parse_discovery_level(text: Any) -> Optional[str]:
    """The canonical level declared by a ``discovery:`` line in ``text``, else None."""
    if not isinstance(text, str) or not text.strip():
        return None
    for m in _LEVEL_LINE_RE.finditer(text):
        level = normalize_level(m.group("value"))
        if level:
            return level
    return None


def render_frontmatter(level: Any) -> str:
    """The canonical frontmatter line for ``level`` (``discovery: flag-high``).

    An unknown/absent level renders the ``flag-med|flag-high`` choice marker so
    an unfilled declaration is visibly unfilled rather than silently defaulted.
    """
    canonical = normalize_level(level)
    return f"{DISCOVERY_FIELD}: {canonical or 'flag-med | flag-high'}"


def render_section(
    *,
    finding: Any = None,
    confidence: Any = None,
    impact_lane: Any = None,
    suggested_action: Any = None,
    affected_card_ids: Any = None,
) -> str:
    """Render the ``DISCOVERY SUMMARY`` section in canonical field order.

    A missing/blank argument falls back to its placeholder, so the section
    ALWAYS carries all five exact field names.
    """
    supplied = {
        "finding": finding,
        "confidence": confidence,
        "impact-lane": impact_lane,
        "suggested-action": suggested_action,
        "affected-card-ids": affected_card_ids,
    }
    lines = [f"**{DISCOVERY_SECTION_HEADING}**"]
    for name in DISCOVERY_SECTION_FIELDS:
        raw = supplied.get(name)
        text = raw.strip() if isinstance(raw, str) else ""
        lines.append(f"- {name}: {text or DISCOVERY_PLACEHOLDERS[name]}")
    return "\n".join(lines)


def parse_section(text: Any) -> dict[str, str]:
    """``{field: value}`` for every section field present in ``text``."""
    if not isinstance(text, str) or not text.strip():
        return {}
    found: dict[str, str] = {}
    for m in _FIELD_RE.finditer(text):
        found.setdefault(m.group("name").lower(), m.group("value").strip())
    return found


def has_section(text: Any) -> bool:
    """True when the section heading AND all five fields are present."""
    if not isinstance(text, str) or not _HEADING_RE.search(text):
        return False
    return set(parse_section(text)) >= set(DISCOVERY_SECTION_FIELDS)


def missing_section_fields(text: Any) -> list[str]:
    """Present-in-canonical-order list of the section fields absent from ``text``."""
    found = parse_section(text)
    return [name for name in DISCOVERY_SECTION_FIELDS if name not in found]


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(value.strip()))


def is_discovery_card(title: Any = "", body: Any = "") -> bool:
    """True when a card DECLARES itself a discovery-report source.

    Declaration-driven on purpose: a card qualifies iff its body carries a
    *valid* ``discovery: flag-med|flag-high`` line. Merely reading like a report
    is not a declaration, so no section is ever stapled onto a card that did not
    ask for one. (``title`` is accepted for call-site symmetry with the sibling
    classifiers, but the declaration is a BODY frontmatter line by design — a
    title is prose and is never read as the declaration.)
    """
    return parse_discovery_level(body) is not None


def validate_discovery_card(text: Any, *, level: Any = None) -> list[str]:
    """Human-readable problems with a declared discovery-report card; ``[]`` = valid.

    Advisory and never raises — the review lane consumes these strings, and a
    malformed declaration must not break card generation.
    """
    problems: list[str] = []
    body = text if isinstance(text, str) else ""
    declared = normalize_level(level) or parse_discovery_level(body)
    if declared is None:
        problems.append(f"missing or invalid `{DISCOVERY_FIELD}:` frontmatter line "
                        f"(expected one of: {' | '.join(DISCOVERY_LEVELS)})")
    if not _HEADING_RE.search(body):
        problems.append(f"missing section heading: {DISCOVERY_SECTION_HEADING}")
    found = parse_section(body)
    for name in DISCOVERY_SECTION_FIELDS:
        if name not in found:
            problems.append(f"missing field: {name}")
            continue
        if _is_placeholder(found[name]) or not found[name]:
            problems.append(f"field {name} still carries its placeholder — the report is incomplete")
    action = found.get("suggested-action")
    if action and not _is_placeholder(action):
        token = re.sub(r"[\s_]+", "-", action.strip().lower()).strip("-")
        if token not in DISCOVERY_ACTIONS:
            problems.append(f"suggested-action must be one of "
                            f"{' | '.join(DISCOVERY_ACTIONS)}, got {action!r}")
    return problems


def ensure_discovery_section(body: Any, *, level: Any = None, **values: Any) -> str:
    """``body`` guaranteed to carry the declaration + section, idempotently.

    Only for a card that DECLARES discovery (or is handed an explicit
    ``level``) — any other body is returned unchanged. The frontmatter line is
    placed immediately after a leading ``phase:`` line when one exists, else at
    the top; the section is appended (a report summary reads after the spec's
    own sections). Existing values are preserved, so a second call is a no-op:
    every declaration line and section field is stripped before re-rendering,
    which makes a partially-filled report converge instead of duplicating.
    """
    text = body if isinstance(body, str) else ""
    declared = normalize_level(level) or parse_discovery_level(text)
    if declared is None:
        return text

    # --- frontmatter line: strip any existing declaration, then re-place it --
    stripped = _LEVEL_LINE_RE.sub("", text)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip("\n")
    line = render_frontmatter(declared)
    head = ""
    m = _PHASE_LINE_RE.match(stripped)
    if m:
        # Match the no-phase branch's blank-line seam, so the declaration line
        # is its own paragraph and the body reads the same either way.
        head = f"{m.group(0).strip()}\n{line}\n\n"
        stripped = stripped[m.end():].lstrip("\n")
    else:
        head = f"{line}\n\n"

    # --- summary section: strip every existing field line, re-render canonically
    existing = parse_section(stripped)
    # Field names in the section are hyphenated (`impact-lane`) but the render
    # kwargs are underscored (`impact_lane`), so a caller's value must be looked
    # up under BOTH spellings — otherwise every hyphenated caller value is
    # silently dropped and replaced by its placeholder.
    merged = {}
    for name in DISCOVERY_SECTION_FIELDS:
        supplied = values.get(name)
        if supplied is None:
            supplied = values.get(name.replace("-", "_"))
        merged[name.replace("-", "_")] = existing.get(name) or supplied
    remainder = _FIELD_RE.sub("", stripped)
    remainder = _HEADING_RE.sub("", remainder)
    remainder = re.sub(r"\n{3,}", "\n\n", remainder).strip("\n")
    section = render_section(**merged)
    remainder = f"{remainder}\n\n{section}" if remainder else section
    return f"{head}{remainder}"


def discovery_prompt_block() -> str:
    """The convention block the card templates render from (single source of truth).

    Rendered from the constants above so the specifier prompt, the decomposer
    prompt, the renderer and the review gate cannot drift apart.
    """
    fields = "\n".join(f"      - {name}: {DISCOVERY_PLACEHOLDERS[name]}"
                       for name in DISCOVERY_SECTION_FIELDS)
    return f"""
DISCOVERY-REPORT CARDS (opt-in — the `{DISCOVERY_FIELD}:` frontmatter key)
  A card whose work produced a FLAG-WORTHY mid-flight finding MAY declare
  itself a discovery-report source by writing this frontmatter line as the
  card's first line (after any `phase:` line):

      {DISCOVERY_FIELD}: {' | '.join(DISCOVERY_LEVELS)}

  - `flag-med`  — an observation worth noting; no immediate action.
  - `flag-high` — a finding that, if true, CHANGES THE GOAL or invalidates
    in-flight sibling work. The orchestrator evaluates a flag-high report
    after the card completes, so `affected-card-ids` must be filled.

  Such a card's body MUST also carry this section, with the exact heading and
  all five exact field names, at the END of the body:

      **{DISCOVERY_SECTION_HEADING}**
{fields}

  Rules for the declaration:
    - Only declare it when there IS a flag-worthy finding. An ordinary card,
      and a research card with nothing to flag, MUST NOT carry the line or
      the section — no placeholder discovery blocks on clean cards.
    - `suggested-action` is exactly one of: {' | '.join(DISCOVERY_ACTIONS)}.
    - `affected-card-ids` lists the board ids (`t_...`) the finding impacts,
      comma-separated, or the word "none" — a card you cannot name is a card
      the orchestrator cannot evaluate, so never leave it blank.
    - This section only REPORTS. Do not park, block, kill or re-scope
      anything from it; the orchestrator owns the decision.
"""
