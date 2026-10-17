"""Tests for UG-2 capability-manifest routing (T2, design §4).

Covers: capability parse, routing order (phase -> capability -> profile),
fallback intactness, 4-bucket headings on child bodies, and M3
``capability_routed`` audit events written on fan-out.
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
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_list_profiles(names: list[str], *, declared: dict[str, str] | None = None):
    """Pretend the named profiles exist; ``declared`` maps name -> description
    carrying a CAPABILITY: ARTIFACT/BEHAVIOR tag."""
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
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def _run_patches(patches):
    for p in patches:
        p.start()
    return patches


def _stop(patches):
    for p in patches:
        p.stop()


# ---------- capability parsing ----------

def test_parse_capability_declared_tags():
    c = capability.parse_capability(
        "Owns code. CAPABILITY: ARTIFACT: code, tests, infra."
    )
    assert c.tag == "ARTIFACT" and c.declared
    assert "code" in c.keywords and "tests" in c.keywords
    c = capability.parse_capability(
        "Read-only worker. CAPABILITY: BEHAVIOR: research, verification, review, design."
    )
    assert c.tag == "BEHAVIOR" and c.declared
    assert "research" in c.keywords


def test_parse_capability_undeclared_defaults_artifact():
    c = capability.parse_capability("just a plain description, no tag")
    assert c.tag == "ARTIFACT" and not c.declared
    c = capability.parse_capability("")
    assert c.tag == "ARTIFACT" and not c.declared


# ---------- routing order ----------

def _routing(default="fallback"):
    from types import SimpleNamespace
    return SimpleNamespace(default_assignee=default)


def test_route_mismatch_reroutes_to_declared_capability_profile():
    caps = {
        "builder": capability._Capability(name="builder", tag="ARTIFACT", declared=True),
        "scout": capability._Capability(name="scout", tag="BEHAVIOR", declared=True),
    }
    routing = _routing()
    # A research-shaped child the LLM mis-routed to builder.
    assignee, reason = capability.route_assignee(
        "BEHAVIOR", "builder", routing, caps,
    )
    assert assignee == "scout"
    assert reason == "capability:scout"


def test_route_llm_pick_honored_when_capability_matches():
    caps = {
        "builder": capability._Capability(name="builder", tag="ARTIFACT", declared=True),
        "scout": capability._Capability(name="scout", tag="BEHAVIOR", declared=True),
    }
    assignee, reason = capability.route_assignee("ARTIFACT", "builder", _routing(), caps)
    assert (assignee, reason) == ("builder", "llm")


def test_route_undeclared_profile_keeps_llm_pick():
    """Existing fallback routing intact: an undeclared profile never triggers a
    reroute (conservative default = ARTIFACT/undeclared)."""
    caps = {
        "engineer": capability._Capability(name="engineer", tag="ARTIFACT", declared=False),
    }
    assignee, reason = capability.route_assignee("BEHAVIOR", "engineer", _routing(), caps)
    assert (assignee, reason) == ("engineer", "llm")


def test_route_no_valid_pick_and_no_declared_match_uses_default():
    caps = {"engineer": capability._Capability(name="engineer", tag="ARTIFACT", declared=False)}
    assignee, reason = capability.route_assignee("BEHAVIOR", None, _routing("fallback"), caps)
    assert (assignee, reason) == ("fallback", "default")


def test_route_ambiguous_declared_roster_keeps_llm_pick():
    caps = {
        "scout": capability._Capability(name="scout", tag="BEHAVIOR", declared=True),
        "docs": capability._Capability(name="docs", tag="BEHAVIOR", declared=True),
        "builder": capability._Capability(name="builder", tag="ARTIFACT", declared=True),
    }
    assignee, reason = capability.route_assignee("BEHAVIOR", "builder", _routing(), caps)
    assert assignee == "builder"  # kept, flagged as mismatch-kept
    assert reason == "llm_mismatch_kept"


def test_classify_child_shape():
    assert capability.classify_child("Update wiki pages", "refresh stale docs") == "BEHAVIOR"
    assert capability.classify_child("Implement router", "write code and tests") == "ARTIFACT"


# ---------- clean_children wiring ----------

def test_clean_children_applies_capability_routing_and_buckets(kanban_home):
    routing = MagicMock()
    routing.default_assignee = "fallback"
    routing.valid_names = {"builder", "scout", "fallback"}
    raw = [
        {"title": "research the options", "body": "grounded survey", "assignee": "builder", "parents": []},
        {"title": "implement router", "body": "code it", "assignee": "builder", "parents": []},
    ]
    with patch.object(decomp.capability, "_build_capability_table", return_value={
        "builder": capability._Capability(name="builder", tag="ARTIFACT", declared=True),
        "scout": capability._Capability(name="scout", tag="BEHAVIOR", declared=True),
    }):
        children, reason = decomp._clean_children("t_x", raw, routing)
    assert reason == ""
    # BEHAVIOR child rerouted off the ARTIFACT-only profile.
    assert children[0]["assignee"] == "scout"
    assert children[0]["capability_reason"] == "capability:scout"
    assert children[1]["assignee"] == "builder"
    # 4-bucket headings appended to every child body (C2.3).
    for c in children:
        for h in ("Adoption", "Docs", "Research-feedback", "Gate"):
            assert f"{h}:" in c["body"]


def test_clean_children_survives_capability_table_failure(kanban_home):
    """Routing is best-effort: a capability-table failure must not abort the
    fan-out (falls through to default routing)."""
    routing = MagicMock()
    routing.default_assignee = "fallback"
    routing.valid_names = {"engineer"}
    raw = [{"title": "ship it", "body": "code", "assignee": "engineer", "parents": []}]
    with patch.object(decomp.capability, "_build_capability_table", return_value={}):
        children, reason = decomp._clean_children("t_x", raw, routing)
    # _build_capability_table swallows internally, so routing still ran with
    # an empty table; the LLM pick on an undeclared profile is kept.
    assert reason == ""
    assert children[0]["assignee"] == "engineer"


# ---------- M3 audit events ----------

def test_fanout_writes_capability_routed_events(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest with research", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research the options", "body": "grounded survey", "assignee": "builder", "parents": []},
            {"title": "implement router", "body": "code it", "assignee": "builder", "parents": []},
        ],
    })

    patches = _run_patches(_patch_list_profiles(
        ["orchestrator", "builder", "scout"],
        declared={
            "builder": "code. CAPABILITY: ARTIFACT: code, tests.",
            "scout": "research. CAPABILITY: BEHAVIOR: research, review.",
        },
    ) + [_patch_aux_client(llm_payload), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"kanban": {"default_assignee": "orchestrator"}},
    )])
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        events = {}
        for cid in outcome.child_ids:
            evs = kb.list_events(conn, cid)
            cap_events = [e for e in evs if e.kind == "capability_routed"]
            assert len(cap_events) == 1, (cid, [e.kind for e in evs])
            events[cid] = cap_events[0].payload or {}
    payloads = list(events.values())
    assert payloads[0]["capability_tag"] == "BEHAVIOR"
    assert payloads[0]["assignee"] == "scout"
    assert payloads[0]["reason"] == "capability:scout"
    assert payloads[1]["capability_tag"] == "ARTIFACT"
    assert payloads[1]["assignee"] == "builder"
    assert payloads[1]["reason"] == "llm"


def test_fanout_false_single_path_untouched(kanban_home):
    """fanout=false (specify path) keeps its existing semantics."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="single", triage=True)
    llm_payload = jsonlib.dumps({
        "fanout": False, "rationale": "one", "title": "T", "body": "B",
        "assignee": "engineer",
    })
    patches = _run_patches(_patch_list_profiles(["orchestrator", "engineer"]) + [
        _patch_aux_client(llm_payload),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
    ])
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.assignee == "engineer"


# ---------- M7 metrics ----------

def test_m7_metrics_reads_review_events(kanban_home):
    import importlib
    m7 = importlib.import_module("hermes_cli.m7_metrics")
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="card under review",
            body="Adoption: none\nDocs: land the wiki page\nResearch-feedback: none\nGate: none",
            triage=False,
        )
        from hermes_cli.kanban_db import _append_event
        _append_event(conn, tid, "review_requested", {})
        _append_event(conn, tid, "changes_requested", {"note": "Docs bucket not delivered"})
        conn.commit()
    verdict = m7.collect(kbc.connect(), None, 50) if False else m7.collect(
        kbc.connect().__enter__(), None, 50,
    )
    assert verdict["review_events_total"] >= 1
    card = next(c for c in verdict["cards"] if c["task_id"] == tid)
    assert card["declared_buckets"] == {"Docs": "land the wiki page"}
    assert card["empty_buckets"] == ["Adoption", "Research-feedback", "Gate"]


# ---------- rollback ----------

def test_prompt_revert_is_single_commit_shape():
    """Rollback (prompt-revert) contract: the UG-2 delta touches exactly two
    in-repo files + one new module; reverting the delta commit removes all of
    it. Assert the wiring points exist so a revert leaves no dangling refs."""
    import inspect
    src = inspect.getsource(decomp)
    assert "capability" in src
    assert decomp._SYSTEM_PROMPT.count("NON-CODE SCOPE CHECKLIST") == 1
    assert decomp._SYSTEM_PROMPT.count("ROUTING ORDER") == 1


# ---------- roster rendering (T2/C2.4) ----------

def _roster_profiles(declared: dict[str, str], undeclared: list[str]):
    """Fake ``list_profiles`` result: declared profiles carry a CAPABILITY line."""
    from types import SimpleNamespace
    out = []
    for name, desc in declared.items():
        out.append(SimpleNamespace(
            name=name, is_default=False, description=desc,
            description_auto=False, model="m", provider="p", skill_count=1,
        ))
    for name in undeclared:
        out.append(SimpleNamespace(
            name=name, is_default=True, description="",
            description_auto=False, model="m", provider="p", skill_count=1,
        ))
    return out


def test_build_roster_emits_capability_key():
    """The roster entry must actually carry the parsed tag the docstring
    promises — a doc-only 'capability' key left the decomposer routing on
    free prose while the prompt claimed a tag was available."""
    profiles = _roster_profiles(
        {
            "builder": "Owns code. CAPABILITY: ARTIFACT: code, tests, infra.",
            "scout": "Read-only. CAPABILITY: BEHAVIOR: research, review.",
        },
        undeclared=["default"],
    )
    with patch("hermes_cli.profiles.list_profiles", return_value=profiles):
        roster, valid = decomp._build_roster()

    by_name = {e["name"]: e for e in roster}
    assert by_name["builder"]["capability"] == "ARTIFACT"
    assert by_name["builder"]["capability_declared"] is True
    assert by_name["scout"]["capability"] == "BEHAVIOR"
    assert by_name["scout"]["capability_declared"] is True
    # Undeclared profile: conservative default, flagged as not declared.
    assert by_name["default"]["capability_declared"] is False
    assert valid == {"builder", "scout", "default"}


def test_format_roster_shows_declared_tags_only():
    """Declared profiles render a [TAG] bracket; undeclared ones keep the
    pre-existing rendering (no bracket, ⚠ undescribed marker preserved)."""
    profiles = _roster_profiles(
        {"builder": "Owns code. CAPABILITY: ARTIFACT: code.", },
        undeclared=["default"],
    )
    with patch("hermes_cli.profiles.list_profiles", return_value=profiles):
        rendered = decomp._format_roster(decomp._build_roster()[0])

    assert "- builder [ARTIFACT]:" in rendered
    assert "[BEHAVIOR]" not in rendered
    # Undeclared entry unchanged from the historical format.
    assert "⚠ undescribed" in rendered
    assert "[ARTIFACT]" not in rendered.split("- default")[1]
