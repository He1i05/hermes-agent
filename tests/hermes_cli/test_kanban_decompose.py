"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
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
        with _patch_aux_client(llm_payload):
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
        with _patch_aux_client(llm_payload), patch(
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
        with _patch_aux_client(llm_payload), patch(
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
        with _patch_aux_client(llm_payload), patch(
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


def _make_repo(path: Path) -> Path:
    """A throwaway git repo with one commit (no global git config needed)."""
    path.mkdir(parents=True, exist_ok=True)
    git = ["git", "-C", str(path), "-c", "user.name=Test User",
           "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([*git, *args], check=True, capture_output=True, text=True)

    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True, text=True)
    (path / "README.md").write_text("base\n", encoding="utf-8")
    run("add", "README.md")
    run("commit", "-m", "init")
    return path


def test_decompose_worktree_child_resolves_to_its_own_tree(kanban_home):
    """A repo-touching child must be created DISPATCHABLE: kind=worktree plus a
    repo-root ``workspace_path``. With a bare kind and no path, dispatch asked
    the board for a ``default_workdir`` that is unset here, failed the spawn,
    and the child burned its retry budget without ever starting.

    Proven with the dispatcher's own resolver, on the row the decomposer wrote —
    not by asserting the column."""
    repo = _make_repo(kanban_home / "hermes-agent")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="patch the repo", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "implement then verify",
        "tasks": [
            {
                "title": "implement", "body": "do it", "assignee": "engineer", "parents": [],
                "workspace_kind": "worktree",
            },
            {
                "title": "verify", "body": "check it", "assignee": "scout", "parents": [0],
                "workspace_kind": "not-a-kind",
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "engineer", "scout"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        impl = kb.get_task(conn, outcome.child_ids[0])
        verify = kb.get_task(conn, outcome.child_ids[1])
    assert impl.workspace_kind == "worktree"
    # The anchor (not the deploy checkout itself) — dispatch turns this into
    # <repo>/.worktrees/<child-id>, one tree per child.
    assert Path(impl.workspace_path).resolve() == repo.resolve()
    workspace, branch = kbw._resolve_worktree_workspace(impl, board="default")
    assert workspace == (repo / ".worktrees" / impl.id).resolve()
    assert workspace.is_dir()
    assert (workspace / ".git").is_file()  # a LINKED worktree, not the main checkout
    assert branch == f"wt/{impl.id}"
    # The bogus kind is dropped and the child inherits the root's kind.
    assert verify.workspace_kind == "scratch"
    # ...and the verify child still waits for the implement child.
    assert verify.status == "todo"


def test_decompose_worktree_child_falls_back_when_no_anchor_is_reachable(kanban_home):
    """No repo to anchor on means no worktree child: emitting kind=worktree with
    an unresolvable path is the spawn-failure this card exists to prevent. The
    child inherits the root's kind and its body still carries the worktree
    command, so the work is not lost — only the per-child provisioning."""
    assert decomp._worktree_anchor(None) is None
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="patch a repo that is not here", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "single child",
        "tasks": [
            {"title": "implement", "body": "do it", "assignee": "engineer", "parents": [],
             "workspace_kind": "worktree", "workspace_path": "/home/someone/tmp-worktrees/x"},
        ],
    })
    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child.workspace_kind == "scratch"
    # The LLM's un-resolvable path is dropped, never persisted to fail dispatch.
    assert child.workspace_path is None


def test_deploy_tree_state_reads_live_checkout_and_orders_lanes_by_commit(kanban_home):
    """The prompt's deploy-tree line is READ, not asserted: it reports the
    checkout's actual branch, and lists carry lanes newest-commit-first so
    "the first lane" is the lane a card should branch from."""
    repo = _make_repo(kanban_home / "hermes-agent")
    subprocess.run(["git", "-C", str(repo), "branch", "carry/v0.22.7-release"],
                   check=True, capture_output=True, text=True)
    (repo / "CARRY.md").write_text("bump\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test User",
                    "-c", "user.email=test@example.com", "add", "CARRY.md"],
                   check=True, capture_output=True, text=True)
    # Explicit future dates: two commits inside the same second share a
    # committerdate, and the sort then falls back to the name order this test
    # distinguishes itself from.
    future = {"GIT_AUTHOR_DATE": "2030-01-01T00:00:00 +0000",
              "GIT_COMMITTER_DATE": "2030-01-01T00:00:00 +0000"}
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test User",
                    "-c", "user.email=test@example.com", "commit", "-m", "carry bump"],
                   check=True, capture_output=True, text=True, env={**os.environ, **future})
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "carry/v0.23.0-main"],
                   check=True, capture_output=True, text=True)
    state = decomp._deploy_tree_state()
    assert state.startswith("Deploy checkout:")
    assert "branch=carry/v0.23.0-main" in state
    lanes = state.split("carry lanes: ", 1)[1].splitlines()[0]
    # Newest commit first; alphabetical order would put 0.22.7 ahead of 0.23.0.
    assert lanes.split(", ")[0] == "carry/v0.23.0-main"


def test_deploy_tree_state_reports_unknown_without_a_checkout(kanban_home):
    """An unreadable checkout degrades to a STATED unknown (the fixture home has
    no ``hermes-agent``), never a raise that would fail the fan-out."""
    state = decomp._deploy_tree_state()
    assert state.startswith("Deploy checkout:")
    assert "state unknown" in state
