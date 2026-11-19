"""Pre-dispatch deploy-tree assertion (``hermes_cli.kanban_db_deploy_tree``).

The deploy checkout — ``<hermes-home>/hermes-agent`` — is the tree systemd runs:
every newly spawned worker, every ``hermes`` CLI call and every no-agent cron
script imports it from disk. On 2026-11-10 a card body said "create a feature
branch from the deploy tree … commit there" and the worker ran ``git checkout
-b`` in that tree, serving unreviewed WIP to the live install for ~30 minutes.

The guard under test refuses to spawn a *repo-touching* card while that checkout
is off-lane or dirty, names the reason in
``deploy_tree_guard.py``'s vocabulary, books the run ``spawn_failed`` /
``infrastructure`` (no retry budget spent) and parks the card ``needs_input``.
Scope is not global: only worktree cards anchored inside the checkout and cards
whose body *names* it are asserted, so a dirty update-day cannot wedge the board.

Every git probe runs against a tmp repo; the live checkout is never touched.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_deploy_tree as dt
from hermes_cli import kanban_db_dispatch as kbd


# ---------------------------------------------------------------------------
# Fixtures: isolated board home + a throwaway deploy checkout
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    ).stdout


@pytest.fixture
def deploy_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp git repo standing in for the deploy checkout, on the carry lane."""
    repo = tmp_path / "deploy" / "hermes-agent"
    repo.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-b", "carry/v0.23.0-main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    monkeypatch.setenv(dt.DEPLOY_ROOT_ENV, str(repo))
    monkeypatch.delenv(dt.ALLOW_OFFLANE_ENV, raising=False)
    return repo


class _Task:
    """Minimal stand-in for ``kanban_db.Task`` on the pure-helper tests."""

    def __init__(self, *, body: str | None = None, workspace_kind: str = "scratch",
                 workspace_path: str | None = None):
        self.body = body
        self.workspace_kind = workspace_kind
        self.workspace_path = workspace_path


def _off_lane(repo: Path) -> None:
    _git(repo, "checkout", "-b", "feature/off-lane")


def _head8(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()[:8]


# ---------------------------------------------------------------------------
# AC1 — the pure precondition: None on a clean lane, two named reasons
# ---------------------------------------------------------------------------


def test_clean_lane_returns_none(deploy_repo: Path) -> None:
    task = _Task(body=f"work in `{deploy_repo}` please")
    assert dt.deploy_tree_precondition(task, "/tmp/elsewhere") is None


def test_off_lane_branch_returns_named_reason(deploy_repo: Path) -> None:
    _off_lane(deploy_repo)
    task = _Task(body=f"commit into `{deploy_repo}`")
    reason = dt.deploy_tree_precondition(task, "/tmp/elsewhere")
    assert reason == f"deploy-tree-off-lane branch=feature/off-lane head={_head8(deploy_repo)}"


def test_dirty_tree_returns_named_reason(deploy_repo: Path) -> None:
    (deploy_repo / "README.md").write_text("modified\n", encoding="utf-8")
    (deploy_repo / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    task = _Task(body=f"commit into `{deploy_repo}`")
    assert dt.deploy_tree_precondition(task, "/tmp/elsewhere") == "deploy-tree-dirty files=2"


def test_main_lane_is_accepted(deploy_repo: Path) -> None:
    _git(deploy_repo, "checkout", "-b", "main")
    task = _Task(body=f"commit into `{deploy_repo}`")
    assert dt.deploy_tree_precondition(task, "/tmp/elsewhere") is None


def test_unreadable_root_is_inert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dt.DEPLOY_ROOT_ENV, str(tmp_path / "not-a-repo"))
    task = _Task(body=f"commit into `{tmp_path / 'not-a-repo'}`")
    assert dt.deploy_tree_precondition(task, "/tmp/elsewhere") is None


# ---------------------------------------------------------------------------
# Scope — which cards are asserted (not global, and normalised body matching)
# ---------------------------------------------------------------------------


def test_card_not_naming_the_tree_is_out_of_scope(deploy_repo: Path) -> None:
    _off_lane(deploy_repo)
    task = _Task(body="fix the parser cache in an unrelated repo")
    assert dt.deploy_tree_precondition(task, "/tmp/elsewhere") is None


def test_dirty_tree_does_not_touch_other_worktrees(deploy_repo: Path) -> None:
    (deploy_repo / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    task = _Task(workspace_kind="worktree", workspace_path="/tmp/other-repo")
    assert dt.deploy_tree_precondition(task, "/tmp/other-repo/.worktrees/x") is None


def test_worktree_workspace_inside_the_checkout_is_in_scope(deploy_repo: Path) -> None:
    _off_lane(deploy_repo)
    workspace = deploy_repo / ".worktrees" / "t_1"
    task = _Task(workspace_kind="worktree", workspace_path=str(deploy_repo))
    task.body = None
    assert dt.deploy_tree_precondition(task, str(workspace)) is not None


def test_tilde_form_of_the_path_is_in_scope(deploy_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _off_lane(deploy_repo)
    # ``deploy_repo`` is <tmp>/deploy/hermes-agent; make "~/hermes-agent" its tilde form.
    monkeypatch.setattr(Path, "home", lambda: deploy_repo.parent)
    task = _Task(body="branch from `~/hermes-agent` and commit there")
    reason = dt.deploy_tree_precondition(task, "/tmp/elsewhere")
    assert reason is not None and reason.startswith("deploy-tree-off-lane")


def test_home_variable_form_of_the_path_is_in_scope(
    deploy_repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _off_lane(deploy_repo)
    monkeypatch.setenv("HOME", str(deploy_repo.parent))
    task = _Task(body="branch from `$HOME/hermes-agent` and commit there")
    reason = dt.deploy_tree_precondition(task, "/tmp/elsewhere")
    assert reason is not None and reason.startswith("deploy-tree-off-lane")


# ---------------------------------------------------------------------------
# AC2/AC3/AC4 — the dispatcher seam
# ---------------------------------------------------------------------------


def _spawn_recorder():
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    return spawns, fake_spawn


def test_naming_card_is_refused_and_blocked_needs_input(
    kanban_home, deploy_repo: Path, all_assignees_spawnable,
) -> None:
    _off_lane(deploy_repo)
    spawns, fake_spawn = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="repo card", assignee="alice",
            body=f"create a branch in `{deploy_repo}` and commit there",
        )
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == [], "a repo-touching card must not spawn while the tree is off-lane"
    assert [row[0] for row in res.deploy_tree_refused] == [tid]
    assert res.deploy_tree_refused[0][1].startswith("deploy-tree-off-lane branch=feature/off-lane")

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert (task.status, task.block_kind) == ("blocked", "needs_input")

        blocked = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"]
        assert blocked, "the sticky needs_input block must be visible on the card"
        reason = blocked[-1].payload["reason"]
        assert reason.startswith("deploy-tree-off-lane branch=feature/off-lane")
        assert "restore the deploy checkout" in reason

        runs = conn.execute(
            "SELECT outcome, status, error, metadata FROM task_runs WHERE task_id = ? "
            "ORDER BY id", (tid,),
        ).fetchall()
        spawn_runs = [r for r in runs if r["outcome"] == "spawn_failed"]
        assert len(spawn_runs) == 1, "the spawn attempt is closed as spawn_failed"
        assert "deploy-tree-off-lane" in (spawn_runs[0]["error"] or "")
        assert '"infrastructure": true' in (spawn_runs[0]["metadata"] or "")

        # The cause is the host's tree, not the card: no retry budget is spent.
        assert task.consecutive_failures == 0
        # Sticky block is the terminal state, not a crash/auto-block.
        assert tid not in res.auto_blocked


def test_non_naming_card_still_spawns(deploy_repo: Path, kanban_home, all_assignees_spawnable) -> None:
    _off_lane(deploy_repo)
    spawns, fake_spawn = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unrelated", assignee="alice", body="fix a parser bug")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == [tid], "scope is not global: unrelated cards must keep dispatching"
    assert res.deploy_tree_refused == []


def test_worktree_card_anchored_on_the_checkout_is_refused(
    deploy_repo: Path, kanban_home, all_assignees_spawnable,
) -> None:
    _off_lane(deploy_repo)
    spawns, fake_spawn = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="repo worktree card", assignee="alice", body="implement the thing",
            workspace_kind="worktree", workspace_path=str(deploy_repo),
        )
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == []
    assert [row[0] for row in res.deploy_tree_refused] == [tid]


def test_escape_hatch_spawns_and_logs_the_bypass(
    deploy_repo: Path, kanban_home, all_assignees_spawnable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _off_lane(deploy_repo)
    monkeypatch.setenv(dt.ALLOW_OFFLANE_ENV, "1")
    spawns, fake_spawn = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="repair the tree", assignee="alice",
            body=f"restore `{deploy_repo}` to the carry lane",
        )
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == [tid], "the escape hatch must let a repair card spawn while off-lane"
    assert res.deploy_tree_refused == []

    with kbc.connect() as conn:
        events = [e for e in kb.list_events(conn, tid) if e.kind == "deploy_tree_bypassed"]
        assert len(events) == 1
        payload = events[0].payload
        assert payload["root"] == str(deploy_repo.resolve())
        assert payload["via"] == f"{dt.ALLOW_OFFLANE_ENV}=1"
        # The event names the refusal it overrode — a bypass is never a mystery.
        assert payload["reason"].startswith("deploy-tree-off-lane branch=feature/off-lane")


def test_escape_hatch_does_not_log_for_out_of_scope_cards(
    deploy_repo: Path, kanban_home, all_assignees_spawnable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(dt.ALLOW_OFFLANE_ENV, "1")
    spawns, fake_spawn = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unrelated", assignee="alice", body="parser bug")
        kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == [tid]
    with kbc.connect() as conn:
        assert not [e for e in kb.list_events(conn, tid) if e.kind == "deploy_tree_bypassed"]


def test_review_lane_card_is_refused_and_blocked_too(
    deploy_repo: Path, kanban_home, all_assignees_spawnable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reviewer run ends in the ``review`` phase, not ``ready``; the sticky
    block must land for that lane as well."""
    import hermes_cli.config as cfgmod

    _off_lane(deploy_repo)
    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    spawns, fake_spawn = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="review the branch", assignee="reviewer",
            body=f"review the change committed in `{deploy_repo}`",
        )
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(
            conn, tid, summary="PR ready", expected_run_id=claimed.current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == []
    assert [row[0] for row in res.deploy_tree_refused] == [tid]
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert (task.status, task.block_kind) == ("blocked", "needs_input")
