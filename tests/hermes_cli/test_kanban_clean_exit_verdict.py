"""Hermetic tests for the cause-first clean-exit verdict (design T7–T14).

A worker that exits ``rc=0`` while its card is still ``running`` used to be booked
``crashed`` / ``protocol_violation`` unconditionally. The verdict now reads the cause:
provider wall -> ``rate_limited``, artifact evidence inside the run window ->
``work_present``, otherwise the protocol violation. These tests pin each branch plus the
accounting invariants (``consecutive_failures`` +0, violation streak +1, no crash report)
and the two structural requirements: the evidence probe runs outside the board write
transaction, and the git probe never raises or hangs into a false positive.

Everything runs against a scratch ``HERMES_HOME`` — never the live board.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as kdg
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _claim_dead_worker(
    conn,
    tid: str,
    pid: int,
    *,
    rc: int = 0,
    workspace: Path | None = None,
    kind: str = "scratch",
    wall: str | None = None,
    session_marker: bool = True,
) -> None:
    """Claim ``tid`` for a worker that already exited ``rc`` and wrote its log."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, workspace_kind=?, "
        "workspace_path=? WHERE id=?",
        (pid, kind, str(workspace) if workspace is not None else None, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        if session_marker:
            # printed by the submitter at claim; the audit's own run-attribution key
            f.write("Session: test-session\n")
        if wall:
            f.write(f"{wall}\n")
        f.write(f"\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}\n")


def _latest_run(conn, tid: str):
    return conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? "
        "ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()


def _scratch(tmp_path: Path, name: str = "scratch") -> Path:
    ws = tmp_path / name
    ws.mkdir()
    return ws


# ── T7: provider wall before evidence ────────────────────────────────────────


def test_t7_provider_wall_books_rate_limited_before_evidence(kanban_home, tmp_path):
    """A 429/usage-limit abort exits rc=0; even with fresh artifacts the wall wins, so the
    card takes the cooldown instead of being requeued into a proven-exhausted provider."""
    ws = _scratch(tmp_path, "wall")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wall", assignee="a")
        _claim_dead_worker(
            conn, tid, 72001, workspace=ws,
            wall="Error: HTTP 429 — session usage limit exceeded, out of credits",
        )
        (ws / "done.md").write_text("work that landed anyway", encoding="utf-8")

        crashed = kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        task = kb.get_task(conn, tid)
        meta = kb._json_dict(run["metadata"])
        assert run["outcome"] == "rate_limited"
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert "protocol_violation" not in meta
        # evidence is preserved for the retry worker even though the wall decides the booking
        assert meta["evidence"] == ["done.md"]
        assert kbd._protocol_violation_streak(conn, tid) == 0
        assert tid not in crashed
        assert tid in kbd.detect_crashed_workers._last_rate_limited


# ── T8: evidence inside the run window -> work_present ───────────────────────


def test_t8_artifact_inside_run_window_books_work_present(kanban_home, tmp_path):
    """Landed work with no landed handoff is a run-level ``work_present``: not a crash, no
    failure counted, streak +1, task back at its source phase, evidence preserved."""
    ws = _scratch(tmp_path, "wp")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wp", assignee="a")
        _claim_dead_worker(conn, tid, 72002, workspace=ws)
        (ws / "report.md").write_text("the deliverable", encoding="utf-8")

        crashed = kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        task = kb.get_task(conn, tid)
        meta = kb._json_dict(run["metadata"])
        assert run["outcome"] == "work_present"
        assert meta["work_present"] is True
        assert meta["evidence"] == ["report.md"]
        assert "report.md" in (run["error"] or "")
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert "report.md" in (task.last_failure_error or "")
        assert kbd._protocol_violation_streak(conn, tid) == 1
        event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert event["kind"] == "work_present"
        assert kb._json_dict(event["payload"])["evidence"] == ["report.md"]
        # a work_present run is not a crash report
        assert tid not in crashed
        assert tid in kbd.detect_crashed_workers._last_work_present


def test_t8b_reviewer_run_returns_to_review_with_retry_status(kanban_home, tmp_path):
    """A reviewer run that landed work without a handoff must go back to ``review``, not
    ``ready`` — the same source-phase rule every other reclaim path follows."""
    ws = _scratch(tmp_path, "review-wp")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review-wp", assignee="a")
        host = kb._claimer_id().split(":", 1)[0]
        assert kb.claim_task(conn, tid, claimer=f"{host}:impl") is not None
        assert kb.request_review(conn, tid, summary="implemented") is True
        assert kb.claim_review_task(conn, tid, claimer=f"{host}:reviewer") is not None
        conn.execute(
            "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, workspace_kind='scratch', "
            "workspace_path=? WHERE id=?",
            (76001, str(ws), tid),
        )
        conn.commit()
        log = kb.worker_log_path(tid)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"Session: reviewer-session\nreview notes\n{KANBAN_WORKER_EXIT_TRAILER}0\n")
        (ws / "review-notes.md").write_text("the review work", encoding="utf-8")

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        task = kb.get_task(conn, tid)
        assert run["outcome"] == "work_present"
        assert kb._json_dict(run["metadata"])["retry_status"] == "review"
        assert task.status == "review"
        # unblocking a card parked by this streak restores the reviewer phase, not ready
        assert kb._resume_status_from_events(conn, tid) == "review"


# ── T9: pin against over-correction ──────────────────────────────────────────


def test_t9_empty_fresh_scratch_still_books_protocol_violation(kanban_home, tmp_path):
    """Nothing landed, nothing left behind: today's verdict, now with complete text."""
    ws = _scratch(tmp_path, "empty")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="empty", assignee="a")
        _claim_dead_worker(conn, tid, 72003, workspace=ws)

        crashed = kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        task = kb.get_task(conn, tid)
        meta = kb._json_dict(run["metadata"])
        assert run["outcome"] == "crashed"
        assert meta["protocol_violation"] is True
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert kbd._protocol_violation_streak(conn, tid) == 1
        assert tid in crashed
        # the retry worker's text names every terminal tool, derived from the guard's set
        from agent.kanban_stop import _TERMINAL_KANBAN_TOOLS

        for tool in _TERMINAL_KANBAN_TOOLS:
            assert tool in (run["error"] or "")


# ── T10: stale artifacts are not this run's work ─────────────────────────────


def test_t10_stale_file_outside_run_window_is_not_work_present(kanban_home, tmp_path):
    ws = _scratch(tmp_path, "stale")
    old = ws / "old.md"
    old.write_text("left over from an earlier attempt", encoding="utf-8")
    old_ts = time.time() - 3600
    os.utime(old, (old_ts, old_ts))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale", assignee="a")
        _claim_dead_worker(conn, tid, 72004, workspace=ws)

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        assert run["outcome"] == "crashed"
        assert kb._json_dict(run["metadata"])["protocol_violation"] is True


# ── T11: the violation streak still bounds work_present ──────────────────────


def test_t11_repeated_work_present_trips_gave_up_with_operator_message(kanban_home, tmp_path):
    """A card that keeps landing work and never files must still reach ``gave_up`` — with a
    message that sends the operator to the workspace, not just back to retry."""
    ws = _scratch(tmp_path, "loop")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="loop", assignee="a")
        for i in range(kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT):
            _claim_dead_worker(conn, tid, 73000 + i, workspace=ws)
            (ws / f"run_{i}.md").write_text(f"work from run {i}", encoding="utf-8")
            kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert "keeps producing work without a handoff" in (task.last_failure_error or "")
        gave_up = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up' "
            "ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert gave_up is not None
        payload = kb._json_dict(gave_up["payload"])
        assert payload["work_present"] is True
        assert payload["protocol_violations"] >= kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT


# ── T12: diagnostics keep their current meaning ──────────────────────────────


def test_t12_crash_threshold_neither_counts_nor_breaks_on_work_present():
    """``kanban_diagnostics._rule_repeated_crashes`` counts trailing ``crashed`` and breaks
    only on ``completed``/``reclaimed``. A ``work_present`` row must therefore neither count
    nor break — pinned so a refactor cannot silently start counting it as a crash."""
    task = {"id": "t_diag", "status": "ready", "consecutive_failures": 0, "assignee": "a"}
    now = int(time.time())
    cfg = {"crash_threshold": 2}
    runs = [
        {"id": 1, "outcome": "crashed", "error": "pid 1 not alive"},
        {"id": 2, "outcome": "work_present", "error": "work landed, no handoff"},
    ]
    assert kdg._rule_repeated_crashes(task, [], runs, now, cfg) == []

    runs.append({"id": 3, "outcome": "crashed", "error": "pid 3 not alive"})
    hits = kdg._rule_repeated_crashes(task, [], runs, now, cfg)
    assert len(hits) == 1
    assert "crashed 2x" in hits[0].title


# ── T13: worktrees use git, not a capped walk ────────────────────────────────


def _git(cwd: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True,
        env=env,
    )


def _init_repo(path: Path) -> Path:
    """A repo whose base commit is two hours old, i.e. outside any run window this test opens."""
    old = f"@{int(time.time()) - 7200} +0000"
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(
        path, "-c", "user.email=test@example.invalid", "-c", "user.name=test",
        "commit", "--allow-empty", "-q", "-m", "base",
        env={**os.environ, "GIT_AUTHOR_DATE": old, "GIT_COMMITTER_DATE": old},
    )
    return path


def test_t13_dirty_worktree_books_work_present(kanban_home, tmp_path):
    repo = _init_repo(tmp_path / "wt")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wt-dirty", assignee="a")
        _claim_dead_worker(conn, tid, 74001, workspace=repo, kind="worktree")
        (repo / "changed.txt").write_text("edited but never filed", encoding="utf-8")

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        assert run["outcome"] == "work_present"
        assert kb._json_dict(run["metadata"])["evidence"] == ["changed.txt"]


def test_t13b_commit_made_during_run_books_work_present(kanban_home, tmp_path):
    repo = _init_repo(tmp_path / "wt-clean")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wt-commit", assignee="a")
        _claim_dead_worker(conn, tid, 74002, workspace=repo, kind="worktree")
        _git(
            repo, "-c", "user.email=test@example.invalid", "-c", "user.name=test",
            "commit", "--allow-empty", "-q", "-m", "work landed during the run",
        )

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        assert run["outcome"] == "work_present"
        assert kb._json_dict(run["metadata"])["evidence"] == ["HEAD committed during this run"]


def test_t13c_clean_worktree_without_new_commit_is_not_evidence(kanban_home, tmp_path):
    repo = _init_repo(tmp_path / "wt-pristine")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wt-clean2", assignee="a")
        _claim_dead_worker(conn, tid, 74003, workspace=repo, kind="worktree")

        # clean status + a HEAD commit older than the run window: no evidence
        assert kbd._workspace_evidence(str(repo), "worktree", time.time()) == []

        kbd.detect_crashed_workers(conn)
        run = _latest_run(conn, tid)
        assert run["outcome"] == "crashed"


# ── T14: the probe never runs under the board write lock ─────────────────────


def test_t14_evidence_probe_runs_outside_the_board_write_transaction(
    kanban_home, tmp_path, monkeypatch,
):
    """Phase 2 (probe) must not hold the board lock; phase 3 still books through the
    conditional UPDATE, so the run row exists exactly once."""
    ws = _scratch(tmp_path, "phase2")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="phase2", assignee="a")
        _claim_dead_worker(conn, tid, 75001, workspace=ws)
        (ws / "x.txt").write_text("x", encoding="utf-8")

        observed: dict[str, bool] = {}
        real = kbd._workspace_evidence

        def spy(path, kind, since):
            observed["in_use"] = bool(conn.in_transaction)
            return real(path, kind, since)

        monkeypatch.setattr(kbd, "_workspace_evidence", spy)

        kbd.detect_crashed_workers(conn)

        assert observed == {"in_use": False}
        runs = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id=?", (tid,)
        ).fetchall()
        assert [r["outcome"] for r in runs] == ["work_present"]


def test_t14b_unreadable_workspace_still_books_protocol_violation(kanban_home, tmp_path):
    """A doubt-raising probe returns no evidence — the verdict falls back, it never raises
    out of the sweep and never invents ``work_present``."""
    missing = tmp_path / "never-created"
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="missing", assignee="a")
        _claim_dead_worker(conn, tid, 75002, workspace=missing)

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        assert run["outcome"] == "crashed"
        assert kb._json_dict(run["metadata"])["protocol_violation"] is True


def test_t14c_git_probe_timeout_is_not_evidence(monkeypatch, tmp_path):
    """A hung git must yield ``[]`` (bounded at 3 s), not block the sweep or book work."""
    ws = tmp_path / "hung"
    ws.mkdir()

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=kbd._GIT_PROBE_TIMEOUT_SECONDS)

    monkeypatch.setattr(kbd.subprocess, "run", _timeout)
    assert kbd._workspace_evidence(str(ws), "worktree", time.time()) == []


def test_t14d_run_scope_requires_a_session_marker(kanban_home, tmp_path):
    """No ``Session:`` marker means the log cannot be attributed to this run: the verdict
    falls back to the protocol violation instead of reading a previous attempt's output."""
    ws = _scratch(tmp_path, "no-marker")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no-marker", assignee="a")
        _claim_dead_worker(conn, tid, 75003, workspace=ws, session_marker=False)
        (ws / "orphan.md").write_text("cannot be attributed", encoding="utf-8")

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        assert run["outcome"] == "crashed"
        assert kb._json_dict(run["metadata"])["protocol_violation"] is True


# ── T14e–g: run attribution for BOTH log shapes in the audit population ──────


def test_t14e_one_shot_exit_summary_log_is_attributable():
    """A one-shot worker's log ends with the CLI exit summary (``Session:        <id>``).
    That line is a separator, not a boundary to read forward from: the run body sits before
    it, and the wall signature in the body must still be found."""
    text = (
        "Query: work kanban task\n"
        "Error: HTTP 429 session usage limit exceeded\n"
        "\nResume this session with:\n  hermes --resume 20260915_000000_abc\n\n"
        "Session:        20260915_000000_abc\n"
        "Messages:       3 (1 user, 2 tool calls)\n"
    )
    segment = kbd._run_scoped_log_segment(text)
    assert segment is not None
    assert "HTTP 429" in segment
    assert "Messages:" not in segment


def test_t14f_consecutive_one_shot_runs_are_separated_by_their_summaries():
    first = "run one body\n\nSession:        s1\nMessages:       1\n"
    second = "run two body\n\nSession:        s2\nMessages:       2\n"
    segment = kbd._run_scoped_log_segment(first + second)
    assert segment is not None
    assert "run two body" in segment
    assert "run one body" not in segment


def test_t14g_claim_marker_wins_over_the_exit_summary():
    """Managed submitter: the claim marker is the boundary and the aligned exit summary that
    follows it is part of the same run block (so the body is not truncated away)."""
    text = "Session: sid-1\nbody of the run\n\nSession:        20260915_000000_abc\nMessages: 3\n"
    segment = kbd._run_scoped_log_segment(text)
    assert segment is not None
    assert "body of the run" in segment
