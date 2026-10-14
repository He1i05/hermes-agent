"""Canonical SDLC phase chain for Kanban cards (the "phase cascade", T1).

One closed enum shared by the triage specifier (tags the ROOT card's phase) and
the decomposer (emits the chain as its DEFAULT shape for multi-step quests,
skipping phases that do not apply).

The value travels in two places:
  * the card body, as a frontmatter line ``phase: <name>`` (human + LLM
    readable, travels with the spec when a card is re-decomposed);
  * ``tasks.current_step_key``, written by
    ``kanban_db_graph._insert_decomposed_child`` for decomposed children.

Phase is DERIVED, advisory routing data. It never overrides the dispatcher's
status machine (``status`` stays the single authority for spawning).
"""

from __future__ import annotations

import re
from typing import Any, Optional

# The closed enum, in canonical (dependency) order. Anything not in this tuple
# is not a phase: callers must refuse to persist an unknown value.
PHASE_CHAIN: tuple[str, ...] = (
    "grounding",
    "review",
    "research",
    "system-design",
    "engineering-planning",
    "implementation",
    "testing",
)

PHASE_SET: frozenset[str] = frozenset(PHASE_CHAIN)
PHASE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(PHASE_CHAIN)}

# Rendered once for the prompts so the enum has a single source of truth.
CHAIN_ARROW: str = " -> ".join(PHASE_CHAIN)

# Prose spellings the design docs / models use for the same phase.
_ALIASES: dict[str, str] = {
    "system-design": "system-design",
    "engineering-plan": "engineering-planning",
    "engineering-plans": "engineering-planning",
    "eng-planning": "engineering-planning",
    "design": "system-design",
    "planning": "engineering-planning",
    "plan": "engineering-planning",
    "implement": "implementation",
    "tests": "testing",
    "test": "testing",
}

# ``phase: research`` / ``- phase: research`` / ``Phase : system design``.
_PHASE_LINE_RE = re.compile(r"^\s*[-*>\s]*phase\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def normalize_phase(value: Any) -> Optional[str]:
    """Return the canonical phase name for ``value``, else ``None``.

    Separators (space / underscore) fold to ``-`` so ``system design`` and
    ``system_design`` are the same phase; a small alias table covers the
    prose spellings ("engineering plan", "design", "plan", "test"). Unknown
    values are rejected rather than coerced — the enum is closed.
    """
    if not isinstance(value, str):
        return None
    token = re.sub(r"[\s_]+", "-", value.strip().lower()).strip("-")
    if token in PHASE_SET:
        return token
    return _ALIASES.get(token)


def parse_phase(text: Any) -> Optional[str]:
    """First ``phase: <name>`` frontmatter line in ``text``, canonicalized.

    ``None`` when the text is absent/blank, has no phase line, or names
    something outside the enum (an unlabelled card, not an error).
    """
    if not isinstance(text, str) or not text.strip():
        return None
    for match in _PHASE_LINE_RE.finditer(text):
        phase = normalize_phase(match.group(1))
        if phase:
            return phase
    return None


def phase_line(phase: str) -> str:
    """The canonical body-frontmatter line for ``phase`` (validated)."""
    canonical = normalize_phase(phase)
    if canonical is None:
        raise ValueError(f"not a canonical phase: {phase!r} (expected one of {', '.join(PHASE_CHAIN)})")
    return f"phase: {canonical}"


def phase_index(phase: Any) -> Optional[int]:
    """Position of ``phase`` in the canonical chain; ``None`` when unknown."""
    canonical = normalize_phase(phase)
    return None if canonical is None else PHASE_INDEX[canonical]


def _reaches(entry_idx: int, entries: list, upstream: set[int]) -> bool:
    """True when card ``entry_idx`` depends (directly or transitively, via
    ``parents`` indices) on any index in ``upstream``. Cycle-safe."""
    seen: set[int] = {entry_idx}
    stack: list[int] = [entry_idx]
    while stack:
        idx = stack.pop()
        entry = entries[idx] if 0 <= idx < len(entries) else None
        if not isinstance(entry, dict):
            continue
        for parent in entry.get("parents") or []:
            if not isinstance(parent, int):
                continue
            if parent in upstream:
                return True
            if parent in seen or not (0 <= parent < len(entries)):
                continue
            seen.add(parent)
            stack.append(parent)
    return False


def cascade_violations(entries: Any) -> list[str]:
    """M3-style audit of an emitted phase cascade — advisory, never raises.

    ``entries`` is the decomposer's raw child list in emitted order; each entry
    may carry ``phase`` and ``parents`` (indices into this same list). Returns
    human-readable violations; an empty list means the cascade is well-formed.

    Skip-non-applicable semantics are encoded as: the classified phases appear
    in canonical order (a genuine subsequence, so gaps are fine), and every
    classified card after the first has a dependency path to an earlier
    classified card (so the chain runs incrementally instead of all at once).
    Zero or one classified card is trivially well-formed.
    """
    if not isinstance(entries, list):
        return []
    classified: list[tuple[int, str]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        phase = normalize_phase(entry.get("phase"))
        if phase:
            classified.append((idx, phase))
    if len(classified) <= 1:
        return []

    violations: list[str] = []
    order = [PHASE_INDEX[phase] for _, phase in classified]
    if order != sorted(order):
        violations.append(
            "phase order broken: emitted "
            + " -> ".join(phase for _, phase in classified)
            + f" — canonical chain is {CHAIN_ARROW}"
        )

    for pos, (idx, phase) in enumerate(classified):
        if pos == 0:
            continue
        upstream = {i for i, _ in classified[:pos]}
        if not _reaches(idx, entries, upstream):
            violations.append(
                f"tasks[{idx}] (phase {phase}) has no dependency path to an earlier "
                "phase card — chain linkage missing"
            )
    return violations
