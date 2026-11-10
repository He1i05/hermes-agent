"""Interrupted-run retry policy (``cron/retry_interrupted.py``).

A cron attempt killed through no fault of the job used to be lost forever: the ledger
terminalized it ``unknown`` and nothing requeued it (``cron/executions.py``: "it is not a retry
queue"). The retry path keeps that invariant — one row per attempt, terminal rows immutable —
and adds one bounded re-run:

* the terminal write that *knows* the outcome class also stamps a retry intent;
* a planning pass turns the intent into a ``next_run_at`` amendment + a ``retry_interrupted``
  marker on the job;
* the ordinary due scan fires it, so at-most-once claim, dedupe and fencing apply unchanged.

Every test here points ``HERMES_HOME``, the cron store, the ledger and the kanban env at a
tmpdir: a test must never touch the live board or cron store.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pytest

import cron.executions as executions_mod
import cron.retry_interrupted as retry_mod
from cron import jobs


@pytest.fixture(autouse=True)
def _isolated_store(monkeypatch, tmp_path):
    """Own cron store + ledger + kanban env; never the live ones."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setattr(
        executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    # _kill() backdates each claim past this process's own start: these rows model a run killed
    # in a PREVIOUS process lifetime. S1 itself is covered by tests/cron/test_sweep_scoping.py.
    with jobs.use_cron_store(tmp_path):
        yield tmp_path


def _dead_pid() -> int:
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True, text=True, check=True,
    )
    return int(proc.stdout.strip())


def _job(**kwargs) -> dict:
    kwargs.setdefault("schedule", "every 5m")
    return jobs.create_job(prompt="retry fixture", model="fixture", deliver="local", **kwargs)


def _new_attempt(job_id: str, *, attempt: int = 1, retry_of: str | None = None) -> str:
    record = executions_mod.create_execution(
        job_id, source="builtin", attempt=attempt, retry_of=retry_of)
    return record["id"]


def _claim_boottime(age_seconds: float) -> float:
    """Boot-clock stamp for a claim ``age_seconds`` old (S1 compares on the boot clock)."""
    return (executions_mod._now_boottime() or 0.0) - age_seconds


def _kill(execution_id: str, *, age_seconds: float = 600.0) -> None:
    """Make the attempt look like a run killed mid-flight: a dead owner, a pre-restart claim.

    The killed run's fire claim is released with its owner: ``_claim_is_live`` drops a claim
    whose owner pid has exited, so the retry's claim is not blocked by a dead firer. Both clock
    readings are backdated: a row that says ``claimed_at`` 10 minutes ago but carries a boot
    stamp of "now" would be a claim that postdates this very process, which is the opposite of
    what a killed predecessor looks like.
    """
    record = executions_mod.get_execution(execution_id)
    with executions_mod._transaction() as conn:
        conn.execute(
            """UPDATE executions
                  SET status='running', started_at=?, process_id='dead-worker', pid=?,
                      process_started_at=NULL, claimed_at=?, claimed_boottime=?
                WHERE id=?""",
            (executions_mod._hermes_now().isoformat(), _dead_pid(),
             (executions_mod._hermes_now() - timedelta(seconds=age_seconds)).isoformat(),
             _claim_boottime(age_seconds),
             execution_id),
        )
    if record is not None:
        jobs.update_job(str(record["job_id"]), {"fire_claim": None})


def _set_now(monkeypatch, when: datetime) -> None:
    monkeypatch.setattr(jobs, "_hermes_now", lambda: when)
    monkeypatch.setattr(executions_mod, "_hermes_now", lambda: when)
    monkeypatch.setattr(retry_mod, "_hermes_now", lambda: when)


def _dispatch_what_is_due(job_id: str) -> dict | None:
    """The claim + receipt half of a dispatch (``_process_due_job``), without running a job."""
    due = {job["id"]: job for job in jobs.get_due_jobs()}
    if job_id not in due:
        return None
    claimed = jobs.claim_job_for_fire(job_id, return_job=True)
    if not isinstance(claimed, dict):
        return None  # lost the fire claim: no new attempt was started
    attempt, retry_of = retry_mod.dispatch_identity(claimed)
    return executions_mod.create_execution(
        job_id, source="builtin", scheduled_instant=claimed.get("_scheduled_instant"),
        attempt=attempt, retry_of=retry_of)


def _rows(job_id: str) -> list[dict]:
    """Attempts only: a ``missed`` row is the due scan's note about skipped slots, not a run."""
    rows = executions_mod.list_executions(job_id=job_id, limit=100)
    return sorted((row for row in rows if row["status"] != "missed"),
                  key=lambda row: (row["claimed_at"], row["id"]))


# --- (a) a killed-mid-run execution is retried exactly once ----------------------------------

def test_killed_mid_run_is_retried_exactly_once(monkeypatch):
    job = _job()
    parent_id = _new_attempt(job["id"])
    _kill(parent_id)

    assert executions_mod.recover_interrupted_executions() == 1

    parent = executions_mod.get_execution(parent_id)
    assert parent["status"] == "unknown"
    assert parent["outcome_class"] == "owner_gone"
    assert parent["retry_pending"] == 1
    finished = datetime.fromisoformat(parent["finished_at"])
    retry_at = datetime.fromisoformat(parent["retry_at"])
    # backoff 60 s, one-sided jitter (<=20%) that may only pull it EARLIER
    assert 48 <= (retry_at - finished).total_seconds() <= 60

    # The planner converts the intent into a schedule amendment and clears it.
    assert retry_mod.plan_pending_retries() == 1
    stored = jobs.get_job(job["id"])
    # Compare instants, not their ISO spellings: the store canonicalises to UTC while
    # ``retry_at`` keeps the profile zone's offset.
    assert datetime.fromisoformat(stored["next_run_at"]) == retry_at
    assert stored["retry_interrupted"] == {
        "at": parent["retry_at"], "attempt": 2, "row_id": parent_id, "expr": None}
    assert executions_mod.get_execution(parent_id)["retry_pending"] == 0

    # The ordinary due scan fires the parked instant, classified inside grace (never catch_up).
    _set_now(monkeypatch, retry_at + timedelta(seconds=1))
    row = _dispatch_what_is_due(job["id"])
    assert row is not None
    assert row["attempt"] == 2
    assert row["retry_of"] == parent_id
    assert datetime.fromisoformat(row["scheduled_instant"]) == retry_at
    assert jobs.get_job(job["id"])["last_dispatch"]["kind"] == "on_time"

    # Exactly one retry: the occurrence is consumed, and no third attempt exists.
    assert _dispatch_what_is_due(job["id"]) is None
    assert [(r["attempt"], r["retry_of"]) for r in _rows(job["id"])] == [
        (1, None), (2, parent_id)]

    # The terminal row is unchanged apart from the intent it carried once (immutability).
    still = executions_mod.get_execution(parent_id)
    assert still["status"] == "unknown"
    assert still["finished_at"] == parent["finished_at"]
    assert still["error"] == parent["error"]


def test_the_retry_does_not_consume_another_repeat_slot():
    """A retry re-runs an occurrence that already counted: ``repeat.completed`` must not move."""
    job = _job(repeat=3)
    first = _new_attempt(job["id"])
    _kill(first)
    executions_mod.recover_interrupted_executions()
    retry_mod.plan_pending_retries()
    stored = jobs.get_job(job["id"])
    completed_before = stored["repeat"]["completed"]
    # A gated run snapshot of the retry instant, exactly as the tick hands it to the job.
    from cron.occurrences import scheduled_instant

    snapshot = dict(stored,
                    _scheduled_instant=scheduled_instant(stored["next_run_at"]))
    assert retry_mod.is_off_lattice_rerun(snapshot) is True
    jobs.mark_job_run(job["id"], True, ladder_rung=retry_mod.is_off_lattice_rerun(snapshot))
    assert jobs.get_job(job["id"])["repeat"]["completed"] == completed_before


# --- (c) the retry cap is honoured (no runaway firing) ----------------------------------------

@pytest.mark.parametrize("max_attempts", [0, 1, 2])
def test_retry_cap_is_honoured(max_attempts):
    job = _job(retry={"max_attempts": max_attempts, "backoff_seconds": 0})
    _kill(_new_attempt(job["id"]))

    for _ in range(6):  # more cycles than any cap allows
        executions_mod.recover_interrupted_executions()
        retry_mod.plan_pending_retries()
        nxt = _dispatch_what_is_due(job["id"])
        if nxt is None:
            break
        _kill(nxt["id"])

    rows = _rows(job["id"])
    assert [row["attempt"] for row in rows] == list(range(1, 2 + max_attempts))
    assert not any(row["attempt"] > max_attempts + 1 for row in rows)
    last = rows[-1]
    exhausted = executions_mod.get_execution(last["id"])
    if max_attempts == 0:
        # Never retried at all: the killed attempt is a visible terminal row and no intent.
        assert exhausted["status"] == "unknown"
    assert exhausted["retry_pending"] == 0
    assert exhausted["retry_at"] is None


def test_a_five_minute_job_cannot_be_refired_hundreds_of_times():
    """Backoff is capped by the job's own grace window (half the period, 150 s for ``every 5m``),
    so the worst-case chain for one occurrence is 60 + 120 + 150 + 150 + 150 = 630 s — not the
    200 fires a five-minute script would collect from an un-bounded retry."""
    job = _job()
    delays = [retry_mod.delay_seconds(job, attempt) for attempt in range(1, 6)]
    assert delays == [60.0, 120.0, 150.0, 150.0, 150.0]
    assert sum(delays) == pytest.approx(630.0)
    assert sum(delays) < (retry_mod.MAX_ATTEMPTS_CEILING + 1) * 60 * 5


# --- (d) catch-up and retry never double-fire ------------------------------------------------

def test_an_overdue_natural_occurrence_wins_and_no_retry_is_planned():
    job = _job()
    parent_id = _new_attempt(job["id"])
    _kill(parent_id)
    assert executions_mod.recover_interrupted_executions() == 1
    assert executions_mod.get_execution(parent_id)["retry_pending"] == 1

    stored = jobs.load_jobs()
    stored[0]["next_run_at"] = (jobs._hermes_now() - timedelta(seconds=3000)).isoformat()
    jobs.save_jobs(stored)

    # R1: the stored instant is already past grace (catch-up owns it), so the retry is refused
    # and its intent is dropped durably.
    assert retry_mod.plan_pending_retries() == 0
    assert executions_mod.get_execution(parent_id)["retry_pending"] == 0
    assert jobs.get_job(job["id"])["next_run_at"] != executions_mod.get_execution(
        parent_id)["retry_at"]

    due = jobs.get_due_jobs()
    assert [row["id"] for row in due] == [job["id"]]
    assert due[0]["last_dispatch"]["kind"] == "catch_up"
    # Exactly one run for the overdue slot — the catch-up fire, not a retry of the killed
    # attempt — and no second dispatch of any kind.
    caught_up = _dispatch_what_is_due(job["id"])
    assert caught_up is not None
    assert caught_up["attempt"] == 1
    assert caught_up["retry_of"] is None
    assert jobs.get_due_jobs() == []
    assert [row["attempt"] for row in _rows(job["id"])] == [1, 1]


def test_a_planned_retry_is_never_classified_catch_up(monkeypatch):
    job = _job()
    parent_id = _new_attempt(job["id"])
    _kill(parent_id)
    executions_mod.recover_interrupted_executions()
    assert retry_mod.plan_pending_retries() == 1
    retry_at = datetime.fromisoformat(executions_mod.get_execution(parent_id)["retry_at"])

    # A prompt fire on its own instant classifies on_time; only a >300 s slip could be "late",
    # and the delay is capped at the job's grace window, so catch_up is unreachable.
    _set_now(monkeypatch, retry_at + timedelta(seconds=1))
    due = jobs.get_due_jobs()
    assert [row["id"] for row in due] == [job["id"]]
    assert due[0]["last_dispatch"]["kind"] in {"on_time", "late"}
    row = _dispatch_what_is_due(job["id"])
    assert row["attempt"] == 2
    assert jobs.get_due_jobs() == []
    assert len(_rows(job["id"])) == 2


def test_a_retry_that_outlived_a_second_outage_is_handled_by_catch_up_not_twice(monkeypatch):
    """Host down at the retry instant: the instant is then past grace, so the catch-up policy
    owns it (one fire or a logged skip) — the ledger still holds exactly one retry row."""
    job = _job()
    parent_id = _new_attempt(job["id"])
    _kill(parent_id)
    executions_mod.recover_interrupted_executions()
    assert retry_mod.plan_pending_retries() == 1
    retry_at = datetime.fromisoformat(executions_mod.get_execution(parent_id)["retry_at"])

    _set_now(monkeypatch, retry_at + timedelta(seconds=3600))
    due = jobs.get_due_jobs()
    assert [row["id"] for row in due] == [job["id"]]
    assert due[0]["last_dispatch"]["kind"] == "catch_up"
    # The late retry fires ONCE as the catch-up run of its own instant, and no third attempt
    # exists. (An outage-delayed retry is owned by the catch-up policy, which books the run
    # against the instant it fires — the documented residual: the parked identity is dropped
    # when the claimed instant is more than FIRE_CLAIM_SKEW_SECONDS from now.)
    late = _dispatch_what_is_due(job["id"])
    assert late is not None
    assert jobs.get_due_jobs() == []
    assert len(_rows(job["id"])) == 2


# --- ledger plumbing ------------------------------------------------------------------------

def test_retry_pending_rows_are_exempt_from_pruning(monkeypatch):
    monkeypatch.setattr(executions_mod, "MAX_TERMINAL_EXECUTIONS", 3)
    job = _job()
    parent_id = _new_attempt(job["id"])
    _kill(parent_id)
    assert executions_mod.recover_interrupted_executions() == 1

    for index in range(10):  # newer terminal rows would otherwise evict it
        filler = executions_mod.create_execution(f"filler-{index}", source="builtin")
        executions_mod.finish_execution(filler["id"], success=True)

    parent = executions_mod.get_execution(parent_id)
    assert parent is not None
    assert parent["retry_pending"] == 1
    assert parent["retry_at"] is not None


def test_only_one_planner_can_claim_an_intent():
    job = _job()
    parent_id = _new_attempt(job["id"])
    _kill(parent_id)
    executions_mod.recover_interrupted_executions()

    assert executions_mod.claim_retry_intent(parent_id) is True
    assert executions_mod.claim_retry_intent(parent_id) is False
    assert retry_mod.plan_pending_retries() == 0


def test_a_retry_for_a_paused_job_is_dropped_durably():
    job = _job()
    parent_id = _new_attempt(job["id"])
    _kill(parent_id)
    executions_mod.recover_interrupted_executions()
    jobs.pause_job(job["id"])

    assert retry_mod.plan_pending_retries() == 0
    assert executions_mod.get_execution(parent_id)["retry_pending"] == 0
    assert jobs.get_job(job["id"])["state"] == "paused"


def test_a_kill_switch_off_voids_the_owed_intent(monkeypatch):
    """With retries switched off the intent is not planned — and not left pending either: an
    owed intent that will never be converted must stop keeping its row prune-exempt."""
    job = _job()
    parent_id = _new_attempt(job["id"])
    _kill(parent_id)
    executions_mod.recover_interrupted_executions()
    next_run_before = jobs.get_job(job["id"])["next_run_at"]
    monkeypatch.setattr(retry_mod, "retry_enabled", lambda cfg=None: False)

    assert retry_mod.plan_pending_retries() == 0
    parent = executions_mod.get_execution(parent_id)
    assert parent["retry_pending"] == 0      # voided, so the row can be pruned again
    assert parent["retry_at"] is not None    # the record of when it was owed survives
    assert jobs.get_job(job["id"])["next_run_at"] == next_run_before


def test_a_verified_receipt_cancels_the_owed_retry():
    """A canonical receipt proves the attempt finished after all: the sweep's intent must be
    cancelled in that same write, so the planner cannot re-fire a completed run."""
    job = _job()
    parent_id = _new_attempt(job["id"])
    _kill(parent_id)
    assert executions_mod.recover_interrupted_executions() == 1
    assert executions_mod.get_execution(parent_id)["retry_pending"] == 1

    settled = executions_mod.recover_receipted_execution(
        parent_id, job["id"], success=True)

    assert settled is not None and settled["status"] == "completed"
    assert settled["retry_pending"] == 0
    assert settled["retry_at"] is None
    assert retry_mod.plan_pending_retries() == 0


def test_owner_wedged_is_never_retried(monkeypatch):
    """A live owner whose claim went silent is released as wedged — but it was never killed, so
    re-firing it would be a concurrent second execution."""
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    job = _job()
    record = executions_mod.create_execution(job["id"], source="builtin")
    with executions_mod._transaction() as conn:
        conn.execute(
            """UPDATE executions SET status='running', process_id='wedged-worker',
                   pid=?, process_started_at=?, claimed_at=?, claimed_boottime=?
                WHERE id=?""",
            (os.getpid(), executions_mod._process_start_time(os.getpid()),
             (executions_mod._hermes_now() - timedelta(seconds=9000)).isoformat(),
             _claim_boottime(9000.0),
             record["id"]),
        )

    assert executions_mod.recover_interrupted_executions() == 1
    row = executions_mod.get_execution(record["id"])
    assert row["outcome_class"] == "owner_wedged"
    assert row["retry_pending"] == 0
    assert row["retry_at"] is None
