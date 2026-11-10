"""Dead-owner sweep scoping: the 2026-11-09 sweep-scoping defect.

On 2026-11-09 the post-boot scheduler claimed ``kanban-release-watchdog`` at 09:13:40, started
it at 09:13:46, and swept that same execution to ``unknown`` 32 s later at 09:14:18.939 — a
fresh run declared unaccountable by the process that launched it.

The sweep now demands positive proof of death and never judges a claim that postdates its own
start:

* S1 — precedence, monotonic: a claim newer than this process's own start is out of scope; the
  one exemption is the self-sufficient proof (owner pid absent) after a full reap interval.
* S2 — machine scope: a row from another machine's pid namespace is undecidable, never dead.
* S3 — proof: pid absence, a RECORDED fingerprint mismatch, or a boot change on this machine.
  A fingerprint that could not be read at claim time is not proof.
* S4 — the unchanged wedged bound for a LIVE owner (see test_stale_running_recovery.py).

Every test here uses a tmp ledger and never touches the live store.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import timedelta

import pytest

import cron.executions as executions_mod


@pytest.fixture(autouse=True)
def _ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(
        executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    # The sweeper's own start is deterministic in these tests: it began an hour ago, so a row
    # stamped "now" genuinely postdates it and a backdated row genuinely predates it.
    monkeypatch.setattr(
        executions_mod, "_process_start_monotonic", lambda: time.monotonic() - 3600.0)
    return executions_mod


def _dead_pid() -> int:
    """PID of a process that has already exited."""
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True, text=True, check=True,
    )
    return int(proc.stdout.strip())


def _seed(
    executions,
    job_id: str,
    *,
    status: str = "running",
    pid: int | None = None,
    process_started_at: int | None = None,
    claimed_at: str | None = None,
    boot_id: str | None = None,
    machine_id: str | None = None,
    attempt: int = 1,
) -> str:
    """A foreign owner's row, shaped like a real claim (owner image, pid, fingerprint)."""
    record = executions.create_execution(job_id, source="builtin", attempt=attempt)
    with executions._transaction() as conn:
        conn.execute(
            """UPDATE executions
                  SET status=?, started_at=?, process_id='foreign-image', pid=?,
                      process_started_at=?, claimed_at=COALESCE(?, claimed_at),
                      boot_id=?, machine_id=?
                WHERE id=?""",
            (status, executions._hermes_now().isoformat() if status == "running" else None,
             os.getpid() if pid is None else pid,
             process_started_at, claimed_at,
             boot_id if boot_id is not None else executions._read_boot_id(),
             machine_id if machine_id is not None else executions._machine_scope_id(),
             record["id"]),
        )
    return record["id"]


def _status(execution_id: str) -> str:
    return executions_mod.get_execution(execution_id)["status"]


def _backdated(seconds: float = 7200.0) -> str:
    return (executions_mod._hermes_now() - timedelta(seconds=seconds)).isoformat()


# --- (b) the incident shape: a post-restart execution is NOT swept ---------------------------

def test_the_2026_11_09_incident_shape_cannot_be_swept(caplog):
    """claimed 09:13:40 / started 09:13:46 / would-be sweep 09:14:18.939 (39 s later).

    The owner pid is LIVE but its start-time reading does not match the claim (a boot-scoped
    fingerprint across a reboot, or a recycled pid — the ledger cannot tell which). Today's
    code called that "provably dead" and swept an ordinary on-cadence run 33 s in. S1 makes it
    impossible: the claim postdates the sweeper's own start.
    """
    live_but_mismatched = executions_mod._process_start_time(os.getpid()) + 60_000_000
    execution_id = _seed(
        executions_mod, "kanban-release-watchdog",
        process_started_at=live_but_mismatched,
        claimed_at=executions_mod._hermes_now().isoformat(),  # 39 s into this process's life
    )

    with caplog.at_level(logging.DEBUG, logger="cron.executions"):
        assert executions_mod.recover_interrupted_executions() == 0

    assert _status(execution_id) == "running"
    assert "left to a later sweep" in caplog.text
    assert "claimed after this process started" in caplog.text


def test_pre_restart_claim_with_a_mismatched_fingerprint_is_swept():
    """The same evidence is conclusive for a claim that predates the sweeper (the restart
    batch): the owner instance cannot be the live process, so the row is terminalized."""
    unrelated = executions_mod._process_start_time(os.getpid()) + 60_000_000
    execution_id = _seed(
        executions_mod, "weekly-job",
        process_started_at=unrelated, claimed_at=_backdated(),
    )

    assert executions_mod.recover_interrupted_executions() == 1

    record = executions_mod.get_execution(execution_id)
    assert record["status"] == "unknown"
    assert record["outcome_class"] == "owner_gone"


def test_postdating_claim_is_swept_once_its_owner_pid_is_gone_and_it_is_old_enough():
    """The self-sufficient proof still reaches a postdating claim — but only after one reap
    interval, so a crashed `hermes cron run` row clears without a gateway restart while no
    fresh run is judged in its first seconds."""
    fresh_id = _seed(executions_mod, "crashed-cli", pid=_dead_pid(),
                     claimed_at=executions_mod._hermes_now().isoformat())

    assert executions_mod.recover_interrupted_executions() == 0
    assert _status(fresh_id) == "running"

    aged_id = _seed(executions_mod, "crashed-cli-old", pid=_dead_pid(),
                    claimed_at=_backdated(executions_mod.SWEEP_POSTDATING_MIN_AGE_SECONDS + 60))

    assert executions_mod.recover_interrupted_executions() == 1
    assert _status(aged_id) == "unknown"


def test_clock_step_backwards_does_not_make_a_fresh_claim_eligible():
    """The successor boot on 2026-11-09 ran -92 min until NTP corrected at 10:46:35. A
    wall-clock comparison would read the fresh claim as OLDER than the process that made it;
    the monotonic comparison reads it as postdating the sweeper, i.e. out of scope."""
    stepped = (executions_mod._hermes_now() + timedelta(minutes=92)).isoformat()
    execution_id = _seed(executions_mod, "clock-step", pid=_dead_pid(), claimed_at=stepped)

    assert executions_mod.recover_interrupted_executions() == 0
    assert _status(execution_id) == "running"

    older = _seed(executions_mod, "pre-boot", pid=_dead_pid(), claimed_at=_backdated())
    assert executions_mod.recover_interrupted_executions() == 1
    assert _status(older) == "unknown"


# --- S2 / S3: machine scope, boot change, and what is NOT proof ------------------------------

def test_a_row_from_another_machine_is_never_swept_however_dead_it_looks():
    execution_id = _seed(
        executions_mod, "other-host-job", pid=_dead_pid(), claimed_at=_backdated(),
        machine_id="some-other-host.example",
    )

    assert executions_mod.recover_interrupted_executions() == 0
    assert _status(execution_id) == "running"


def test_a_legacy_row_without_machine_id_is_treated_as_same_machine():
    execution_id = _seed(
        executions_mod, "legacy-job", pid=_dead_pid(), claimed_at=_backdated(),
        machine_id="",
    )

    assert executions_mod.recover_interrupted_executions() == 1
    assert _status(execution_id) == "unknown"


def test_a_boot_change_terminalises_a_same_machine_row_even_with_a_colliding_pid():
    """S3(c): the fingerprint cannot settle a cross-boot claim in either direction (a
    coincidence inside the drift tolerance would otherwise read as "alive" and wedge the row
    until S4). The recorded boot id settles it."""
    execution_id = _seed(
        executions_mod, "previous-boot-job",
        process_started_at=executions_mod._process_start_time(os.getpid()),  # coincidental match
        claimed_at=_backdated(),
        boot_id="00000000-0000-0000-0000-000000000000",
    )

    assert executions_mod.recover_interrupted_executions() == 1
    record = executions_mod.get_execution(execution_id)
    assert record["status"] == "unknown"
    assert record["outcome_class"] == "owner_gone"


def test_an_unreadable_claim_fingerprint_is_not_proof_of_death(monkeypatch):
    """A fingerprint that could not be read at claim time (NULL) says nothing. With no boot
    change to corroborate it, the row must be left alone — the module header's original
    promise, which the old ``pid == os.getpid()`` branch contradicted."""
    monkeypatch.setattr(executions_mod, "_read_boot_id", lambda: "")
    execution_id = _seed(
        executions_mod, "unreadable-fingerprint",
        process_started_at=None, claimed_at=_backdated(6000), boot_id="",
    )

    assert executions_mod.recover_interrupted_executions() == 0
    assert _status(execution_id) == "running"


def test_owner_is_live_with_no_fingerprint_is_fail_safe():
    """``_owner_is_live(foreign live pid, None)`` used to return False for every foreign pid,
    which handed every such row to the sweep. Unreadable is not dead."""
    assert executions_mod._owner_is_live(os.getpid(), None) is True
    assert executions_mod._owner_is_live(_dead_pid(), None) is False
