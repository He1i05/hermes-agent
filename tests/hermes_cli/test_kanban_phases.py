"""Tests for the canonical SDLC phase chain (T1 phase cascade).

Pure unit tests — no DB, no LLM. They pin the closed enum, the tolerant
normalisation, the body-frontmatter fallback, and the advisory cascade audit.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_phases as ph


def test_chain_is_the_canonical_seven_phase_order():
    assert ph.PHASE_CHAIN == (
        "grounding",
        "review",
        "research",
        "system-design",
        "engineering-planning",
        "implementation",
        "testing",
    )
    assert ph.PHASE_SET == frozenset(ph.PHASE_CHAIN)
    assert len(ph.PHASE_SET) == 7


def test_normalize_phase_accepts_canonical_names():
    for name in ph.PHASE_CHAIN:
        assert ph.normalize_phase(name) == name


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("System Design", "system-design"),
        ("system_design", "system-design"),
        ("  RESEARCH  ", "research"),
        ("engineering plan", "engineering-planning"),
        ("engineering-plan", "engineering-planning"),
        ("eng-planning", "engineering-planning"),
        ("design", "system-design"),
        ("plan", "engineering-planning"),
        ("planning", "engineering-planning"),
        ("implement", "implementation"),
        ("test", "testing"),
        ("tests", "testing"),
    ],
)
def test_normalize_phase_maps_prose_spellings(raw, expected):
    assert ph.normalize_phase(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", 7, ["research"], "deployment", "grounding-x"])
def test_normalize_phase_rejects_unknown_values(raw):
    """The enum is closed — anything else is None, never a coerced guess."""
    assert ph.normalize_phase(raw) is None


def test_parse_phase_reads_frontmatter_line():
    body = "phase: research\n\n**Goal**\nFind out."
    assert ph.parse_phase(body) == "research"


def test_parse_phase_tolerates_leading_markers_and_case():
    assert ph.parse_phase("- Phase: System Design\nbody") == "system-design"
    assert ph.parse_phase("> PHASE : implementation\n") == "implementation"


def test_parse_phase_ignores_unknown_and_missing_lines():
    assert ph.parse_phase("phase: deployment\n") is None
    assert ph.parse_phase("**Goal**\nno phase here") is None
    assert ph.parse_phase(None) is None


def test_parse_phase_only_accepts_phase_keyed_lines():
    """A body sentence mentioning a phase is not a tag."""
    assert ph.parse_phase("This card is in the research phase of the work") is None


def test_phase_line_rejects_non_canonical_input():
    assert ph.phase_line("research") == "phase: research"
    assert ph.phase_line("System Design") == "phase: system-design"
    with pytest.raises(ValueError):
        ph.phase_line("deployment")


def test_phase_index_follows_chain_order():
    assert ph.phase_index("grounding") == 0
    assert ph.phase_index("implementation") == 5
    assert ph.phase_index("testing") == 6
    assert ph.phase_index("nope") is None


def test_cascade_violations_accepts_a_linked_subsequence():
    """Skip-non-applicable: gaps are fine, order + linkage are not."""
    entries = [
        {"phase": "research", "parents": []},
        {"phase": "implementation", "parents": [0]},
        {"phase": "testing", "parents": [1]},
    ]
    assert ph.cascade_violations(entries) == []


def test_cascade_violations_accepts_single_and_unphased_graphs():
    assert ph.cascade_violations([{"phase": "implementation", "parents": []}]) == []
    assert ph.cascade_violations([{"parents": []}, {"title": "a", "parents": [0]}]) == []
    assert ph.cascade_violations([]) == []
    assert ph.cascade_violations("not a list") == []


def test_cascade_violations_accepts_transitive_linkage():
    """Linkage may run through an unphased support card."""
    entries = [
        {"phase": "grounding", "parents": []},
        {"phase": None, "parents": [0]},
        {"phase": "implementation", "parents": [1]},
    ]
    assert ph.cascade_violations(entries) == []


def test_cascade_violations_flags_out_of_order_phases():
    entries = [
        {"phase": "implementation", "parents": []},
        {"phase": "research", "parents": [0]},
    ]
    violations = ph.cascade_violations(entries)
    assert len(violations) == 1
    assert "phase order broken" in violations[0]


def test_cascade_violations_flags_missing_chain_link():
    entries = [
        {"phase": "research", "parents": []},
        {"phase": "implementation", "parents": []},
    ]
    violations = ph.cascade_violations(entries)
    assert len(violations) == 1
    assert "no dependency path" in violations[0]


def test_cascade_violations_ignores_unknown_phase_values():
    """Unclassified support cards neither break order nor count as chain steps."""
    entries = [
        {"phase": "research", "parents": []},
        {"phase": "deployment", "parents": []},
        {"phase": "implementation", "parents": [0]},
    ]
    assert ph.cascade_violations(entries) == []


def test_cascade_violations_is_cycle_safe():
    entries = [
        {"phase": "research", "parents": [1]},
        {"phase": "implementation", "parents": [0]},
    ]
    # Order is fine; linkage is satisfied transitively and must not hang.
    assert ph.cascade_violations(entries) == []


def test_cascade_violations_ignores_non_int_parents():
    entries = [
        {"phase": "research", "parents": []},
        {"phase": "implementation", "parents": ["0", None]},
    ]
    violations = ph.cascade_violations(entries)
    assert len(violations) == 1
    assert "no dependency path" in violations[0]
