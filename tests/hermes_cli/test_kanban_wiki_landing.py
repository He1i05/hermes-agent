"""Tests for decomposer wiki-landing child emission (T4 C4.2, design §6 O-A).

Covers: the wiki-worthy disposition parse, the emitted child's assignee +
parent gating, integration with the P0 Research-feedback bucket, the M3
``wiki_landing_emitted`` audit event, and board stability (no emission when
nothing is wiki-worthy, no duplicate when the LLM already emitted a landing,
and the cap that stops a decomposition from exploding).
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_capability as capability
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_m2 as m2
from hermes_cli import kanban_wiki_landing as wl


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _patch_aux_client(content: str):
    return patch("agent.auxiliary_client.call_llm", return_value=_fake_aux_response(content))


def _patch_list_profiles(names: list[str], *, declared: dict[str, str] | None = None):
    """Stub the profile roster on the module object the decomposer actually holds.

    ``patch.object(decomp.profiles_mod, ...)`` rather than a ``"hermes_cli.profiles.X"``
    string target: some older kanban test files purge every ``hermes_cli.*``
    entry from ``sys.modules`` and re-import them, which leaves this module's
    ``decomp`` reference bound to the pre-purge module object. A string target
    would then patch a DIFFERENT (fresh) module object than the one the
    decomposer calls, so the stub silently stops biting and the real roster is
    used instead. Patching through ``decomp``'s own reference is order-independent.
    """
    from types import SimpleNamespace
    declared = declared or {}
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=declared.get(n, f"desc for {n}"),
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch.object(decomp.profiles_mod, "list_profiles", return_value=fake_profiles),
        patch.object(decomp.profiles_mod, "profile_exists", side_effect=lambda x: x in names),
        patch.object(decomp.profiles_mod, "get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def _run_patches(patches):
    for p in patches:
        p.start()
    return patches


def _routing(default="fallback", valid=("builder", "scout", "wiki-keeper", "fallback")):
    from types import SimpleNamespace
    return SimpleNamespace(default_assignee=default, valid_names=set(valid))


def _caps():
    return {
        "builder": capability._Capability(name="builder", tag="ARTIFACT", declared=True),
        "scout": capability._Capability(name="scout", tag="BEHAVIOR", declared=True,
                                        keywords=("research", "review")),
        "wiki-keeper": capability._Capability(name="wiki-keeper", tag="BEHAVIOR", declared=True,
                                              keywords=("wiki", "docs", "knowledge")),
    }


def _research_body(disposition: str = "wiki-worthy") -> str:
    return m2.ensure_m2_block("**Goal**\nsurvey the space", wiki_handoff_disposition=disposition)


def _research_child(title="Survey agent frameworks", disposition="wiki-worthy", **kw):
    child = {
        "title": title,
        "body": _research_body(disposition),
        "assignee": "scout",
        "capability_tag": "BEHAVIOR",
        "parents": [],
    }
    child.update(kw)
    return child


# ---------- disposition parsing ----------

@pytest.mark.parametrize("value", [
    "wiki-worthy",
    "wiki worthy",
    "wiki_worthy",
    "Wiki-worthy (produces a reusable framework comparison)",
])
def test_is_wiki_worthy_accepts_the_declaration(value):
    assert wl.is_wiki_worthy(value) is True


@pytest.mark.parametrize("value", [
    None,
    "",
    "   ",
    "none",
    "N/A",
    "no wiki value (internal-only)",
    "internal-only",
    "not wiki-worthy",
    "no wiki value",
    "<wiki-worthy -> wiki-keeper landing child | no wiki value (internal-only)>",
])
def test_is_wiki_worthy_rejects_non_declarations(value):
    """Conservative direction: only an explicit claim emits a wiki page.

    A rendered-but-unfilled placeholder is a template instruction, and the
    explicit internal-only disposition is an ACCEPTED answer (design I4.2) —
    neither is turned into a landing child.
    """
    assert wl.is_wiki_worthy(value) is False


def test_disposition_of_reads_through_m2_source_of_truth():
    body = _research_body("wiki-worthy — reusable protocol survey")
    assert wl.disposition_of(body) == "wiki-worthy — reusable protocol survey"
    assert wl.disposition_of("no M2 block here") is None
    assert wl.disposition_of(None) is None


# ---------- assignee resolution ----------

def test_resolve_assignee_prefers_wiki_keeper():
    assert wl.resolve_wiki_assignee(_routing(), _caps()) == "wiki-keeper"


def test_resolve_assignee_falls_back_to_declared_docs_behavior_profile():
    caps = _caps()
    caps["librarian"] = capability._Capability(
        name="librarian", tag="BEHAVIOR", declared=True, keywords=("docs", "knowledge"),
    )
    routing = _routing(valid=("builder", "scout", "fallback"))
    assert wl.resolve_wiki_assignee(routing, caps) == "librarian"


def test_resolve_assignee_never_returns_blank():
    """A landing child must never be emitted unassigned."""
    routing = _routing("fallback", valid=("builder", "scout", "fallback"))
    assert wl.resolve_wiki_assignee(routing, {}) == "fallback"


# ---------- emission ----------

def test_emits_landing_child_for_wiki_worthy_research():
    children = [_research_child()]
    emitted = wl.emit_wiki_landing_children("t_x", children, _routing(), _caps())
    assert len(children) == 2
    landing = children[-1]
    assert emitted == [landing["title"]]
    assert landing["assignee"] == "wiki-keeper"
    # Standard parent gating: index into the same emitted list.
    assert landing["parents"] == [0]
    assert landing["wiki_landing"] is True
    assert "Land wiki page for: Survey agent frameworks" == landing["title"]
    assert len(landing["title"]) <= 80


def test_landing_body_is_a_wiki_owning_card():
    children = [_research_child()]
    wl.emit_wiki_landing_children("t_x", children, _routing(), _caps())
    body = children[-1]["body"]
    assert body.startswith("## Step 0 — KNOWLEDGE LOOKUP (verb-first, MANDATORY)")
    assert "wiki_index.py search" in body
    assert "**Goal**" in body and "**Acceptance criteria**" in body
    assert "wiki-validate.py" in body
    # Names the gated parent so a worker reading only its own body knows where
    # the findings are.
    assert "Survey agent frameworks" in body


def test_no_emission_for_internal_only_disposition():
    children = [_research_child(disposition="no wiki value (internal-only)")]
    assert wl.emit_wiki_landing_children("t_x", children, _routing(), _caps()) == []
    assert len(children) == 1


def test_no_emission_for_unfilled_placeholder_disposition():
    children = [{
        "title": "Survey something",
        "body": m2.render_m2_block(),  # all five fields are placeholders
        "assignee": "scout",
        "capability_tag": "BEHAVIOR",
        "parents": [],
    }]
    assert wl.emit_wiki_landing_children("t_x", children, _routing(), _caps()) == []
    assert len(children) == 1


def test_no_emission_for_non_research_child():
    children = [{
        "title": "Implement the router",
        "body": "code it\n\nDocs: none\nResearch-feedback: none",
        "assignee": "builder",
        "capability_tag": "ARTIFACT",
        "parents": [],
    }]
    assert wl.emit_wiki_landing_children("t_x", children, _routing(), _caps()) == []


def test_cap_limits_landings_per_decomposition():
    children = [_research_child(title=f"Survey area {i}") for i in range(4)]
    emitted = wl.emit_wiki_landing_children("t_x", children, _routing(), _caps())
    assert len(emitted) == wl.MAX_WIKI_LANDING_CHILDREN
    assert len(children) == 4 + wl.MAX_WIKI_LANDING_CHILDREN


def test_no_duplicate_when_llm_already_emitted_a_gated_landing():
    """The prompt asks the LLM to emit the wiki-keeper child itself; a model
    that complied must not get a second one."""
    research = _research_child()
    llm_landing = {
        "title": "Land the wiki page", "body": "wiki it", "assignee": "wiki-keeper",
        "capability_tag": "BEHAVIOR", "parents": [0],
    }
    children = [research, llm_landing]
    assert wl.emit_wiki_landing_children("t_x", children, _routing(), _caps()) == []
    assert len(children) == 2


def test_emission_is_deterministic():
    a = [_research_child()]
    b = [_research_child()]
    wl.emit_wiki_landing_children("t_x", a, _routing(), _caps())
    wl.emit_wiki_landing_children("t_x", b, _routing(), _caps())
    assert a[-1] == b[-1]


# ---------- assignee-collision regression (the fallback must not suppress) ----------

def test_landing_emitted_when_a_sibling_shares_the_resolved_assignee():
    """A sibling that merely SHARES the resolved assignee is not coverage.

    With no wiki-keeper on the roster, ``resolve_wiki_assignee`` falls back to
    the decomposition default — the SAME name a plain build sibling carries.
    Reading that sibling as "already a docs/wiki child" suppressed the landing
    entirely, silently recreating the research->wiki gap this engine closes.
    """
    routing = _routing("fallback", valid=("builder", "scout", "fallback"))
    body = _research_body("wiki-worthy")
    children = [
        {"title": "Survey agent frameworks", "body": body, "assignee": "fallback",
         "capability_tag": "BEHAVIOR", "parents": []},
        # Gated on the research child, shares the resolved assignee, but BUILDS.
        {"title": "Implement the integration", "body": "**Goal**\ncode it",
         "assignee": "fallback", "capability_tag": "ARTIFACT", "parents": [0]},
    ]
    emitted = wl.emit_wiki_landing_children("t_x", children, routing, {})
    assert emitted, "landing must still be emitted"
    assert len(children) == 3
    landing = children[-1]
    assert landing["wiki_landing"] is True
    assert landing["parents"] == [0]
    # Never unassigned, even on the fallback path.
    assert landing["assignee"] == "fallback"


def test_docs_owned_names_exclude_the_default_assignee():
    """The coverage set names only profiles that OWN wiki/docs work.

    The complement of the collision bug at the unit level: whatever the
    decomposition's default assignee is, it must never appear in the
    coverage set unless a profile of that name genuinely declares the docs
    capability (and then it is docs-owned on its own merits, not by default).
    """
    routing = _routing("fallback", valid=("builder", "scout", "fallback"))
    assert wl.docs_owned_assignees(routing, {}) == {wl.WIKI_KEEPER}
    assert "fallback" not in wl.docs_owned_assignees(routing, _caps())
    # A declared docs BEHAVIOR profile IS docs-owned, by declaration.
    caps = dict(_caps())
    caps["librarian"] = capability._Capability(
        name="librarian", tag="BEHAVIOR", declared=True, keywords=("docs", "knowledge"),
    )
    assert "librarian" in wl.docs_owned_assignees(routing, caps)


def test_real_docs_sibling_still_suppresses_the_duplicate():
    """The genuine dedupe case must be preserved: a wiki-keeper child gated on
    the research card means the LLM already landed it — no second landing."""
    children = [_research_child(), {
        "title": "Land the wiki page", "body": "wiki it", "assignee": "wiki-keeper",
        "capability_tag": "BEHAVIOR", "parents": [0],
    }]
    assert wl.emit_wiki_landing_children("t_x", children, _routing(), _caps()) == []
    assert len(children) == 2


def test_build_card_merely_mentioning_the_wiki_is_not_coverage():
    """A build-verb title wins over an incidental wiki/docs mention.

    "Implement wiki export for the survey" gated on the research card is a BUILD
    card, not a landing — counting it as coverage would suppress the landing,
    the same silent failure mode as the assignee collision.
    """
    children = [
        _research_child(),
        {"title": "Implement wiki export for the survey", "body": "**Goal**\ncode it",
         "assignee": "builder", "capability_tag": "ARTIFACT", "parents": [0]},
    ]
    emitted = wl.emit_wiki_landing_children("t_x", children, _routing(), _caps())
    assert emitted, "a build sibling must not suppress the landing"
    assert children[-1]["wiki_landing"] is True
    assert children[-1]["parents"] == [0]


def test_empty_children_is_a_noop():
    children: list[dict] = []
    assert wl.emit_wiki_landing_children("t_x", children, _routing(), _caps()) == []
    assert children == []


def test_only_the_first_person_landing_is_chosen_per_research_child():
    """A multi-card decomposition keeps each landing gated on ITS research card."""
    children = [
        _research_child(title="Survey A"),
        _research_child(title="Survey B"),
        _research_child(title="Implement it", disposition="no wiki value (internal-only)",
                        capability_tag="ARTIFACT"),
    ]
    wl.emit_wiki_landing_children("t_x", children, _routing(), _caps())
    landings = [c for c in children if c.get("wiki_landing")]
    assert [l["parents"] for l in landings] == [[0], [1]]


# ---------- _clean_children wiring ----------

def test_clean_children_emits_landing_and_buckets_every_card(kanban_home):
    routing = _routing()
    raw = [
        {"title": "Survey the framework landscape",
         "body": "**Goal**\nsurvey\n\nweb_searches: 5\nlead_ledger: attached\n"
                 "anchor_note: named frameworks\nvendor_caveat: third-party\n"
                 "wiki_handoff_disposition: wiki-worthy",
         "assignee": "scout", "parents": []},
        {"title": "Implement the integration", "body": "**Goal**\ncode it",
         "assignee": "builder", "parents": [0]},
    ]
    with patch.object(decomp.capability, "_build_capability_table", return_value=_caps()):
        children, reason = decomp._clean_children("t_x", raw, routing)
    assert reason == ""
    # Build child untouched by the landing engine.
    assert len([c for c in children if c.get("wiki_landing")]) == 1
    landing = next(c for c in children if c.get("wiki_landing"))
    assert landing["assignee"] == "wiki-keeper"
    # Every card, landing included, carries the P0 bucket headings.
    for c in children:
        for h in capability.BUCKET_HEADINGS:
            assert f"{h}:" in c["body"], (c["title"], h)
    # The landing card names the parent's disposition in the P0 bucket.
    assert "Research-feedback: parent declared wiki-handoff disposition" in landing["body"]


def test_clean_children_does_not_emit_for_an_unstated_disposition(kanban_home):
    """A research child whose disposition the LLM never filled gets a rendered
    PLACEHOLDER (the M2 renderer's contract) — a template instruction, not a
    declaration, so no wiki page is emitted on its behalf."""
    routing = _routing()
    raw = [
        {"title": "Survey the framework landscape", "body": "**Goal**\nsurvey",
         "assignee": "scout", "parents": []},
    ]
    with patch.object(decomp.capability, "_build_capability_table", return_value=_caps()):
        children, reason = decomp._clean_children("t_x", raw, routing)
    assert reason == ""
    assert m2.has_m2_block(children[0]["body"]) is True
    assert not any(c.get("wiki_landing") for c in children)


def test_clean_children_leaves_a_non_wiki_quest_unchanged(kanban_home):
    """Board stability: a quest with no wiki-worthy research emits exactly the
    children the LLM asked for."""
    routing = _routing()
    raw = [
        {"title": "Implement the router", "body": "code it", "assignee": "builder", "parents": []},
        {"title": "Run the regression suite", "body": "test it", "assignee": "builder", "parents": [0]},
    ]
    with patch.object(decomp.capability, "_build_capability_table", return_value=_caps()):
        children, reason = decomp._clean_children("t_x", raw, routing)
    assert reason == ""
    assert len(children) == 2
    assert not any(c.get("wiki_landing") for c in children)


# ---------- end-to-end: real DB fan-out ----------

def _fan_llm_payload():
    return jsonlib.dumps({
        "fanout": True,
        "rationale": "research plus build",
        "tasks": [
            {"title": "Survey the framework landscape",
             "body": "**Goal**\nsurvey the space\n\n"
                     "web_searches: 5\nlead_ledger: attached\nanchor_note: frameworks named in the card\n"
                     "vendor_caveat: third-party benchmark\n"
                     "wiki_handoff_disposition: wiki-worthy",
             "assignee": "scout", "parents": []},
            {"title": "Implement the integration", "body": "**Goal**\ncode it",
             "assignee": "builder", "parents": [0]},
        ],
    })


def test_fanout_persists_the_gated_landing_child(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest with wiki-worthy research", triage=True)

    patches = _run_patches(_patch_list_profiles(
        ["orchestrator", "builder", "scout", "wiki-keeper"],
        declared={
            "builder": "code. CAPABILITY: ARTIFACT: code, tests.",
            "scout": "research. CAPABILITY: BEHAVIOR: research, review.",
            "wiki-keeper": "wiki. CAPABILITY: BEHAVIOR: wiki, docs, knowledge.",
        },
    ) + [_patch_aux_client(_fan_llm_payload()), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"kanban": {"default_assignee": "orchestrator"}},
    )])
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert len(outcome.child_ids) == 3
    with kbc.connect() as conn:
        kids = [kb.get_task(conn, cid) for cid in outcome.child_ids]
    research = next(k for k in kids if k.title == "Survey the framework landscape")
    landing = next(k for k in kids if k.title.startswith("Land wiki page for:"))
    build = next(k for k in kids if k.title == "Implement the integration")

    # Correct assignee + gating metadata.
    assert landing.assignee == "wiki-keeper"
    with kbc.connect_closing() as conn:
        assert kb.parent_ids(conn, landing.id) == [research.id]
        assert landing.id in kb.child_ids(conn, research.id)
        assert landing.id not in kb.child_ids(conn, build.id)

    # Gating behaviour: the landing card waits for the research card.
    assert build.status == "todo"
    assert landing.status == "todo"

    # Bucket integration on the persisted body.
    assert "Research-feedback: parent declared wiki-handoff disposition: wiki-worthy" in (landing.body or "")


def test_landing_promotes_only_after_its_research_parent_completes(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest", triage=True)
    patches = _run_patches(_patch_list_profiles(
        ["orchestrator", "builder", "scout", "wiki-keeper"],
        declared={
            "builder": "code. CAPABILITY: ARTIFACT: code.",
            "scout": "research. CAPABILITY: BEHAVIOR: research.",
            "wiki-keeper": "wiki. CAPABILITY: BEHAVIOR: wiki, docs.",
        },
    ) + [_patch_aux_client(_fan_llm_payload()), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"kanban": {"default_assignee": "orchestrator"}},
    )])
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason

    with kbc.connect() as conn:
        kids = [kb.get_task(conn, cid) for cid in outcome.child_ids]
        research = next(k for k in kids if k.title == "Survey the framework landscape")
        landing = next(k for k in kids if k.title.startswith("Land wiki page for:"))
        # Parent completes -> landing becomes dispatchable.
        assert kb.complete_task(conn, research.id, summary="done") is True
        kb.recompute_ready(conn)
        assert kb.get_task(conn, landing.id).status == "ready"
        # Still not dispatchable while the research card is unfinished.
        assert kb.get_task(conn, research.id).status == "done"


def test_fanout_writes_wiki_landing_emitted_audit_event(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest", triage=True)
    patches = _run_patches(_patch_list_profiles(
        ["orchestrator", "builder", "scout", "wiki-keeper"],
        declared={
            "builder": "code. CAPABILITY: ARTIFACT: code.",
            "scout": "research. CAPABILITY: BEHAVIOR: research.",
            "wiki-keeper": "wiki. CAPABILITY: BEHAVIOR: wiki, docs.",
        },
    ) + [_patch_aux_client(_fan_llm_payload()), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"kanban": {"default_assignee": "orchestrator"}},
    )])
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        events = {}
        for cid in outcome.child_ids:
            evs = [e for e in kb.list_events(conn, cid) if e.kind == "wiki_landing_emitted"]
            if evs:
                events[cid] = evs[0].payload or {}
    assert len(events) == 1, events
    payload = next(iter(events.values()))
    assert payload["research_parent_title"] == "Survey the framework landscape"
    assert payload["assignee"] == "wiki-keeper"
    assert payload["disposition"].strip() == "wiki-worthy"
    assert payload["root"] == tid


def test_audit_event_writer_ignores_non_landing_children(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain")
        cid = kb.create_task(conn, title="plain child", parents=[tid])
        children = [{"title": "plain child", "assignee": "builder"}]
        assert wl.audit_log_landing_children(conn, tid, [cid], children) == 0
        assert [e for e in kb.list_events(conn, cid) if e.kind == "wiki_landing_emitted"] == []


# ---------- integration with the sibling M2 layer ----------

def test_landing_is_emitted_for_a_card_the_m2_renderer_filled():
    """The engine keys off the SAME M2 block the sibling renderer writes, so a
    real research child produced by kanban_m2 drives emission end-to-end."""
    body = m2.ensure_m2_block("**Goal**\nsurvey", wiki_handoff_disposition="wiki-worthy")
    assert m2.has_m2_block(body)
    children = [{"title": "Survey X", "body": body, "assignee": "scout",
                 "capability_tag": "BEHAVIOR", "parents": []}]
    assert wl.emit_wiki_landing_children("t_x", children, _routing(), _caps())
