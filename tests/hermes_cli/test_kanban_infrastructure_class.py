"""Hermetic tests for the host-loss infrastructure classifier (design t_82e97446, Appendix H).

A worker whose run dies because the gateway owner missed its RPC window exits nonzero
carrying one of a small, closed set of strings. The crash sweep used to book that as an
ordinary ``_record_task_failure``: ``consecutive_failures`` climbed to the card's limit and
a healthy card parked — and when the same fingerprint appeared 3+ times in one sweep the
``is_systemic`` branch tripped it at 1 with ``sticky=True``, so a host outage parked cards
HARDER. These tests pin:

  * the predicate (anchor in the last 200 characters, cause-first label after the first
    wrapper) and its closed vocabulary;
  * the booking: ``infrastructure: true`` in run metadata and event payload, the card's
    retry budget untouched, no ``gave_up``, no ``sticky``, never ``blocked``;
  * the streak arithmetic (it counts the run being booked, never N-1) and the burst
    escalation (3+ in one sweep -> wider spacing, one warning, still no block);
  * the cooldown reader the guard uses to space the retry;
  * the positive controls: a CONTENT crash still charges the counter and still trips.

Everything runs against a scratch ``HERMES_HOME`` — never the live board.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

WALL_CONNFAIL = "Kanban owner refused: Gateway connection failed; no local fallback"
WALL_EU = "Kanban owner refused: Kanban execution unknown; retained by owner"
WALL_DISC = "Kanban owner refused: Gateway disconnected; turn outcome is unknown."
WALL_TIMEOUT = "Kanban owner refused: Gateway connection timed out after 30s"
WALL_PROTO = "Kanban owner refused: Gateway protocol mismatch (owner=3 worker=2)"
WALL_BARE_TIMEOUT = "Kanban owner refused: Gateway bootstrap timed out"
WALL_OTHER = "Kanban owner refused: something the classifier has never seen"
WALL_OWNER_UNAVAILABLE = (
    "Traceback (most recent call last):\n"
    "  File \"kanban_worker_client.py\", line 113, in submit\n"
    "agent.runtime_session_store.WorkerPersistenceError: owner_unavailable"
)
CONTENT_WALL = "AssertionError: 2 + 2 == 5"


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


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8


def _claim_dead_worker(
    conn, tid: str, pid: int, *, rc: int = 1, wall: str | None = None,
    workspace: Path | None = None,
) -> None:
    """Claim ``tid`` for a worker that already exited ``rc`` and wrote its log."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, workspace_kind='scratch', "
        "workspace_path=? WHERE id=?",
        (pid, str(workspace) if workspace is not None else None, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write("Session: test-session\n")
        if wall:
            f.write(f"{wall}\n")
        f.write(f"\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}\n")


def _latest_run(conn, tid: str):
    return conn.execute(
        "SELECT outcome, error, metadata, ended_at FROM task_runs WHERE task_id=? "
        "ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()


def _meta(conn, tid: str) -> dict:
    return kb._json_dict(_latest_run(conn, tid)["metadata"])


def _events(conn, tid: str, kind: str) -> list[dict]:
    return [
        kb._json_dict(r["payload"])
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id", (tid, kind)
        )
    ]


def _add_closed_run(conn, tid: str, *, outcome: str, metadata: dict | None = None) -> None:
    """Append an already-closed run row (history the streak walk reads)."""
    now = int(time.time())
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (tid, outcome, now - 60, now - 30, outcome, json.dumps(metadata or {})),
    )
    conn.commit()


def _assert_booked_as_infrastructure(conn, tid: str, expected_class: str, *, streak: int = 1) -> None:
    run = _latest_run(conn, tid)
    task = kb.get_task(conn, tid)
    meta = kb._json_dict(run["metadata"])
    assert run["outcome"] == "crashed"
    assert meta["infrastructure"] is True
    assert meta["infrastructure_class"] == expected_class
    assert meta["infrastructure_streak"] == streak
    assert meta["retry_status"] == "ready"
    assert task.status == "ready", "a host-caused death must not park the card"
    assert task.consecutive_failures == 0, "the card's retry budget must be untouched"
    assert (task.last_failure_error or "").startswith("[host-caused] ")
    assert not _events(conn, tid, "gave_up"), "the breaker must never trip on a host loss"
    assert not kb._has_sticky_block(conn, tid)


# ── T1–T4: the vocabulary, booked ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("wall", "expected"),
    [
        (WALL_CONNFAIL, "gateway_connection_failed"),
        (WALL_EU, "execution_unknown"),
        (WALL_DISC, "gateway_disconnected"),
        (WALL_TIMEOUT, "gateway_timeout"),
        (WALL_PROTO, "gateway_protocol_mismatch"),
        (WALL_BARE_TIMEOUT, "owner_refused_timeout"),
        (WALL_OTHER, "owner_refused_other"),
        (WALL_OWNER_UNAVAILABLE, "owner_unavailable"),
    ],
    ids=[
        "t1-connfailed", "t2-execution-unknown", "t3-disconnected", "t4a-timeout",
        "t4b-protocol", "t4c-bootstrap-timeout", "t4d-other", "t4e-owner-unavailable",
    ],
)
def test_each_owner_loss_arm_is_booked_as_infrastructure(kanban_home, wall, expected):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="owner-loss", assignee="a")
        _claim_dead_worker(conn, tid, 71001, wall=wall)

        crashed = kbd.detect_crashed_workers(conn)

        _assert_booked_as_infrastructure(conn, tid, expected)
        assert tid in crashed, "the run is still reported as a crash"
        assert tid in kbd.detect_crashed_workers._last_infrastructure
        payload = _events(conn, tid, "crashed")[-1]
        assert payload["infrastructure"] is True
        assert payload["infrastructure_class"] == expected
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"


# ── T5: a burst is a host outage, not three bad cards ────────────────────────


def test_t5_burst_of_three_escalates_spacing_and_never_blocks(kanban_home):
    with kbc.connect() as conn:
        tids = [kb.create_task(conn, title=f"burst-{i}", assignee="a") for i in range(3)]
        for i, (tid, wall) in enumerate(zip(tids, (WALL_CONNFAIL, WALL_EU, WALL_DISC))):
            _claim_dead_worker(conn, tid, 72000 + i, wall=wall)

        result = kbd.DispatchResult()
        kbd._run_reclaim_phase(
            conn, result, stale_timeout_seconds=3600, failure_limit=2,
            reconcile_orphans=False,
        )

        for tid in tids:
            meta = _meta(conn, tid)
            assert meta["infrastructure"] is True
            assert meta["infrastructure_burst"] is True
            assert meta["infrastructure_burst_count"] == 3
            assert meta["infrastructure_cooldown_seconds"] == 900
            assert kb.get_task(conn, tid).consecutive_failures == 0
            assert not _events(conn, tid, "gave_up")
            assert not kb._has_sticky_block(conn, tid)
        assert sorted(result.infrastructure_burst.values()) == [3]
        assert sorted(result.infrastructure_crashed) == sorted(tids)


# ── T6: a host loss after a real failure preserves that failure's count ──────


def test_t6_host_loss_does_not_erase_an_earlier_content_failure(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="prior-failure", assignee="a")
        _claim_dead_worker(conn, tid, 73001, wall=CONTENT_WALL)
        kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, tid).consecutive_failures == 1

        _claim_dead_worker(conn, tid, 73002, wall=WALL_CONNFAIL)
        kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 1, "the host loss must neither add nor reset"
        assert task.status == "ready"
        assert _meta(conn, tid)["infrastructure"] is True
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"


# ── T7: the reviewer lane keeps its phase ────────────────────────────────────


def test_t7_reviewer_host_loss_returns_to_review(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reviewer", assignee="a")
        host = kb._claimer_id().split(":", 1)[0]
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:impl")
        assert kb.request_review(
            conn, tid, summary="implemented", expected_run_id=claimed.current_run_id
        ) is True
        assert kb.claim_review_task(conn, tid, claimer=f"{host}:reviewer") is not None
        pid = 74001
        conn.execute("UPDATE tasks SET worker_pid=?, worker_started_at=NULL WHERE id=?", (pid, tid))
        conn.commit()
        log = kb.worker_log_path(tid)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"Session: reviewer-session\n{WALL_CONNFAIL}\n{KANBAN_WORKER_EXIT_TRAILER}1\n",
            encoding="utf-8",
        )

        kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        meta = _meta(conn, tid)
        assert meta["infrastructure"] is True
        assert meta["retry_status"] == "review"
        assert task.status == "review", "a killed reviewer returns to review, never to ready"
        assert task.consecutive_failures == 0


# ── T8–T11: positive controls — a card's own failures still charge ──────────


def test_t8_content_crash_still_charges_the_counter(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="content", assignee="a")
        _claim_dead_worker(conn, tid, 75001, wall=CONTENT_WALL)

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 1
        assert task.status == "ready"
        assert "infrastructure" not in kb._json_dict(run["metadata"])
        assert kbd.check_respawn_guard(conn, tid) is None
        assert not (task.last_failure_error or "").startswith("[host-caused]")


def test_t9_two_content_crashes_still_trip_without_sticky(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="content-twice", assignee="a")
        _claim_dead_worker(conn, tid, 75002, wall=CONTENT_WALL)
        kbd.detect_crashed_workers(conn)
        _claim_dead_worker(conn, tid, 75003, wall=CONTENT_WALL)
        kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 2
        gave_up = _events(conn, tid, "gave_up")
        assert gave_up and gave_up[-1]["failures"] == 2
        assert "sticky" not in gave_up[-1]


def test_t10_clean_exit_without_handoff_is_not_infrastructure(kanban_home, tmp_path):
    ws = tmp_path / "clean"
    ws.mkdir()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="clean", assignee="a")
        _claim_dead_worker(conn, tid, 75004, rc=0, workspace=ws)

        kbd.detect_crashed_workers(conn)

        meta = _meta(conn, tid)
        assert meta["protocol_violation"] is True
        assert "infrastructure" not in meta
        assert kb.get_task(conn, tid).consecutive_failures == 0


def test_t11_terminal_provider_still_trips_at_once(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="terminal", assignee="a", max_retries=5)
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, tid, claimer=f"{host}:w0")
        pid = 75005
        conn.execute("UPDATE tasks SET worker_pid=?, worker_started_at=NULL WHERE id=?", (pid, tid))
        conn.commit()
        kbd._record_worker_exit(pid, _exited_status(kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE))

        kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert kb._has_sticky_block(conn, tid)
        assert "infrastructure" not in _meta(conn, tid)


# ── T12: card content outranks host loss, even with gateway text in the tail ─


def test_t12_work_present_outranks_an_owner_loss_tail(kanban_home, tmp_path):
    """rc=0 with an owner-loss line in the log AND landed work: the card-content verdict
    wins, so no infrastructure stamp is written for a run the card is answerable for."""
    ws = tmp_path / "wp"
    ws.mkdir()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="precedence", assignee="a")
        _claim_dead_worker(conn, tid, 75006, rc=0, wall=WALL_CONNFAIL, workspace=ws)
        (ws / "report.md").write_text("landed", encoding="utf-8")

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        meta = kb._json_dict(run["metadata"])
        assert run["outcome"] == "work_present"
        assert "infrastructure" not in meta


# ── T13: the predicate itself ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("worker_output", "expected"),
    [
        (WALL_CONNFAIL, "gateway_connection_failed"),
        (WALL_EU, "execution_unknown"),
        (WALL_DISC, "gateway_disconnected"),
        (WALL_TIMEOUT, "gateway_timeout"),
        (WALL_PROTO, "gateway_protocol_mismatch"),
        (WALL_BARE_TIMEOUT, "owner_refused_timeout"),
        (WALL_OTHER, "owner_refused_other"),
        (WALL_OWNER_UNAVAILABLE, "owner_unavailable"),
        # a report that merely quotes the wrapper earlier in the output: the label is
        # read after the FIRST wrapper (documented consequence of the stability rule),
        # so the quoted cause is what the token reports — both classes are host-caused
        (
            "I read the log: 'Kanban owner refused: Gateway connection failed' is what we saw\n"
            + WALL_EU,
            "gateway_connection_failed",
        ),
        # mixed tail, outcome printed first, cause second -> the cause decides
        (f"{WALL_EU}\n{WALL_CONNFAIL}", "gateway_connection_failed"),
        # mixed tail, cause printed first -> same answer
        (f"{WALL_CONNFAIL}\n{WALL_EU}", "gateway_connection_failed"),
        # no evidence at all
        ("", None),
        (None, None),
        # a freestanding timeout with no wrapper is not a host refusal
        ("gateway connection timed out, retrying\nstill not back", None),
    ],
)
def test_t13_host_loss_class_vocabulary(worker_output, expected):
    assert kbd._host_loss_class(worker_output) == expected


# ── T14: the cooldown reader ──────────────────────────────────────────────────


def test_t14_cooldown_reader_clamps_and_falls_back(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cooldown", assignee="a")
        now = int(time.time())
        # 900s stamp, 400s elapsed -> still cooling
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, metadata) "
            "VALUES (?, 'crashed', ?, ?, 'crashed', ?)",
            (tid, now - 800, now - 400,
             json.dumps({"infrastructure": True, "infrastructure_cooldown_seconds": 900})),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"

        # garbage stamp -> falls back to the 300s default, so 400s elapsed releases it
        conn.execute("UPDATE task_runs SET metadata=? WHERE task_id=?",
                     (json.dumps({"infrastructure": True,
                                  "infrastructure_cooldown_seconds": "not-a-number"}), tid))
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) is None

        # absent stamp -> same fallback
        conn.execute("UPDATE task_runs SET metadata=? WHERE task_id=?",
                     (json.dumps({"infrastructure": True}), tid))
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) is None

    assert kbd._coerce_infrastructure_cooldown(900, default=300) == 900
    assert kbd._coerce_infrastructure_cooldown("junk", default=300) == 300
    assert kbd._coerce_infrastructure_cooldown(None, default=300) == 300
    assert kbd._coerce_infrastructure_cooldown(5, default=300) == 60
    assert kbd._coerce_infrastructure_cooldown(99999, default=300) == 3600


# ── T15 + T16: nothing sticky, and a second sweep changes nothing ───────────


def test_t15_burst_cards_stay_movable_and_unblocked(kanban_home):
    with kbc.connect() as conn:
        tids = [kb.create_task(conn, title=f"movable-{i}", assignee="a") for i in range(3)]
        for i, (tid, wall) in enumerate(zip(tids, (WALL_CONNFAIL, WALL_EU, WALL_DISC))):
            _claim_dead_worker(conn, tid, 76000 + i, wall=wall)
        kbd.detect_crashed_workers(conn)

        kb.recompute_ready(conn, failure_limit=2)

        for tid in tids:
            task = kb.get_task(conn, tid)
            assert task.status in ("ready", "running")
            assert not kb._has_sticky_block(conn, tid)


def test_t16_second_sweep_is_a_no_op(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="idempotent", assignee="a")
        _claim_dead_worker(conn, tid, 77001, wall=WALL_CONNFAIL)
        first = kbd.detect_crashed_workers(conn)
        assert first == [tid]
        before = _meta(conn, tid)

        second = kbd.detect_crashed_workers(conn)

        assert second == []
        assert _meta(conn, tid) == before
        assert kb.get_task(conn, tid).consecutive_failures == 0
        assert len(_events(conn, tid, "crashed")) == 1
        assert not _events(conn, tid, "gave_up")


# ── T17: the streak counts the run being booked (never N-1) ──────────────────


def test_t17_streak_counts_the_run_being_booked(kanban_home):
    with kbc.connect() as conn:
        # (a) no prior infrastructure run -> 1
        a = kb.create_task(conn, title="streak-a", assignee="a")
        _add_closed_run(conn, a, outcome="completed")
        _claim_dead_worker(conn, a, 78001, wall=WALL_CONNFAIL)
        kbd.detect_crashed_workers(conn)
        assert _meta(conn, a)["infrastructure_streak"] == 1

        # (b) one prior infrastructure run -> 2
        b = kb.create_task(conn, title="streak-b", assignee="a")
        _add_closed_run(conn, b, outcome="crashed", metadata={"infrastructure": True})
        _claim_dead_worker(conn, b, 78002, wall=WALL_CONNFAIL)
        kbd.detect_crashed_workers(conn)
        assert _meta(conn, b)["infrastructure_streak"] == 2

        # (c) a content crash in between -> 1
        c = kb.create_task(conn, title="streak-c", assignee="a")
        _add_closed_run(conn, c, outcome="crashed", metadata={})
        _claim_dead_worker(conn, c, 78003, wall=WALL_CONNFAIL)
        kbd.detect_crashed_workers(conn)
        assert _meta(conn, c)["infrastructure_streak"] == 1

        # (d) two host losses separated by a neutral rate_limited run -> 2
        d = kb.create_task(conn, title="streak-d", assignee="a")
        _add_closed_run(conn, d, outcome="crashed", metadata={"infrastructure": True})
        _add_closed_run(conn, d, outcome="rate_limited")
        _claim_dead_worker(conn, d, 78004, wall=WALL_CONNFAIL)
        kbd.detect_crashed_workers(conn)
        assert _meta(conn, d)["infrastructure_streak"] == 2


# ── T18: padding never changes the label, and past the anchor it is a miss ──


def test_t18_window_padding_never_changes_the_label():
    dying = "Kanban owner refused: Gateway connection failed; no local fallback"
    pad_before = "log line about the run that says nothing about the handshake\n" * 1

    assert kbd._host_loss_class(dying) == "gateway_connection_failed"
    # enough preceding text to move the 200-char boundary, wrapper still inside it
    assert kbd._host_loss_class(pad_before * 2 + dying) == "gateway_connection_failed"
    # the wrapper pushed out of the window entirely -> no evidence, ordinary card path
    assert kbd._host_loss_class(dying + "\n" + ("noise\n" * 60)) is None
