"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
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


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim retained for call-site compatibility: extra_body plumbing
    # now lives inside call_llm, which _patch_aux_client already mocks.
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_decompose_fanout_persists_phase_to_current_step_key(kanban_home):
    """The emitted phase chain survives into the DB on every child (T1)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="multi-step quest", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "phase cascade",
        "tasks": [
            {"title": "research", "body": "phase: research", "assignee": "researcher",
             "phase": "research", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer",
             "phase": "implementation", "parents": [0]},
            {"title": "verify", "body": "test it", "assignee": "engineer",
             "phase": "testing", "parents": [1]},
            {"title": "audit", "body": "support card, no phase", "assignee": "engineer",
             "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        phases = [kb.get_task(conn, cid).current_step_key for cid in outcome.child_ids]
    assert phases == ["research", "implementation", "testing", None]


def test_decompose_phase_falls_back_to_body_frontmatter(kanban_home):
    """A child that tags only `phase:` in its body still persists the phase."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "frontmatter only",
        "tasks": [
            {"title": "design it", "body": "phase: System Design\n\n**Goal**\nx",
             "assignee": "engineer", "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child.current_step_key == "system-design"


def test_decompose_drops_unknown_phase_without_failing_the_graph(kanban_home):
    """An out-of-enum phase is advisory-only: dropped, never a fan-out failure."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "bad phase",
        "tasks": [
            {"title": "deploy it", "body": "ship", "assignee": "engineer",
             "phase": "deployment", "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child.current_step_key is None


def test_decompose_user_prompt_carries_root_phase_and_chain(kanban_home):
    """Prompt plumbing: root phase + canonical chain are rendered into the user turn."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="quest", body="phase: system-design\n\n**Goal**\ndecide", triage=True,
        )

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_aux_response(jsonlib.dumps({
            "fanout": True, "rationale": "r",
            "tasks": [{"title": "t", "body": "b", "assignee": "engineer", "parents": []}],
        }))

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=_capture):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    user = captured["messages"][1]["content"]
    assert "Root phase" in user and "system-design" in user
    assert "grounding -> review -> research" in user


def test_decompose_user_prompt_reports_untagged_root(kanban_home):
    """An untagged root tells the model to start the chain at grounding."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain quest", body="no phase here", triage=True)

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_aux_response(jsonlib.dumps({
            "fanout": True, "rationale": "r",
            "tasks": [{"title": "t", "body": "b", "assignee": "engineer", "parents": []}],
        }))

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=_capture):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    user = captured["messages"][1]["content"]
    assert "untagged" in user and "grounding" in user


def test_decompose_root_phase_prefers_persisted_step_key(kanban_home):
    """current_step_key (specifier-tagged) outranks the body frontmatter line."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="quest", body="phase: research\n\nx", triage=True,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET current_step_key = 'system-design' WHERE id = ?", (tid,),
            )

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_aux_response(jsonlib.dumps({
            "fanout": True, "rationale": "r",
            "tasks": [{"title": "t", "body": "b", "assignee": "engineer", "parents": []}],
        }))

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=_capture):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    user = captured["messages"][1]["content"]
    assert "Root phase (tagged by the triage specifier, when known): system-design" in user


def test_decompose_unknown_root_step_key_falls_back_to_body(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quest", body="phase: testing\n\nx", triage=True)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET current_step_key = 'legacy-step-7' WHERE id = ?", (tid,))

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_aux_response(jsonlib.dumps({
            "fanout": True, "rationale": "r",
            "tasks": [{"title": "t", "body": "b", "assignee": "engineer", "parents": []}],
        }))

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=_capture):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    user = captured["messages"][1]["content"]
    assert "when known): testing" in user


def test_decompose_single_task_persists_root_phase(kanban_home):
    """fanout=false: the root card keeps the phase the decomposer classified."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="one thing", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened",
        "body": "**Goal**\nDo it.",
        "phase": "implementation",
    })

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.current_step_key == "implementation"


def test_decompose_system_prompt_documents_the_cascade():
    """The prompt itself is the contract: chain, skip rule, phase field."""
    prompt = decomp._SYSTEM_PROMPT
    assert "grounding -> review -> research -> system-design -> engineering-planning -> implementation -> testing" in prompt
    assert "SKIP phases that do not apply" in prompt
    assert '"phase"' in prompt
    assert "phase: <name>" in prompt


def test_decompose_fanout_children_inherit_root_assignee_when_unrouted(kanban_home):
    """Unrouted children fall back to the ROOT task's assignee, not
    the decomposer's active profile (#114294). The active profile here is ``private``
    (an incognito profile with no credentials), so the old fallback spawned
    workers that deadlocked on capability blockers."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="zdr", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "made_up", "parents": []},
            {"title": "build", "body": "code it", "assignee": None, "parents": [0]},
        ],
    })

    # get_active_profile_name() is mocked to names[0] = "private" — the
    # global default chain would resolve there without kanban.default_assignee.
    patches = _patch_list_profiles(["private", "zdr"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0.assignee == "zdr"
    assert c1.assignee == "zdr"
    # Same class for the root: no ``orchestrator_profile`` must not hand the
    # orchestration card to the dispatcher's own (here: incognito) profile.
    assert root.assignee == "zdr"


def test_decompose_explicit_default_assignee_wins_over_root_assignee(kanban_home):
    """An explicitly configured ``kanban.default_assignee`` stays
    authoritative for unroutable children; the root task's assignee only
    fills in when no explicit default is set (explicit config → card
    assignee → active profile)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="engineer", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "made_up", "parents": []},
            {"title": "build", "body": "code it", "assignee": None, "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["engineer", "docs", "private"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"kanban": {"default_assignee": "docs"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0.assignee == "docs"
    assert c1.assignee == "docs"

def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_load_routing_falls_back_to_defaults_when_config_unreadable(kanban_home, monkeypatch):
    """decompose_task promises ok=False on expected failures; a config read that raises (missing
    profile home, HomeInitializationError) must not escape _load_routing as an exception."""
    from hermes_cli import config as config_mod

    def _boom():
        raise FileNotFoundError("profile home is gone")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    routing = decomp._load_routing()
    assert routing.default_assignee == "default" and routing.auto_promote is True


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not in triage" in outcome.reason


