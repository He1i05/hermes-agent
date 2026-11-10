"""A running ``no_agent`` script must stamp ``progress_at``, so the live-owner stale bound
measures SILENCE rather than run length.

``recover_interrupted_executions`` reclaims a live-owned row only once its owner has been silent
past ``_live_owner_stale_after_seconds()`` (``max(3 × HERMES_CRON_TIMEOUT, script timeout,
7200)``), measured from the owner's last ``progress_at`` stamp. The agent path stamps it from its
run-monitor loop (``ExecutionProgressStamper``); the script path used to stamp nothing, so a
script row fell back to ``claimed_at`` and a healthy long ``no_agent`` run was reclaimed to
``unknown`` — ``owner_wedged`` — while it was still running (2026-11-09 incident rows:
``progress_at`` NULL except the agent run).

These tests drive the real supervisor loop. The sweep is run from a SECOND process because the
sweeper deliberately skips rows whose ``process_id`` is its own, so a same-process sweep could
never judge the row under test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

import cron.executions as executions_mod
from cron import scheduler, scheduler_script
from cron.executions import _hermes_now

REPO_ROOT = Path(__file__).resolve().parents[2]

# Run ``recover_interrupted_executions`` in a child process (its own ``_PROCESS_ID``).
_SWEEPER = """\
import json
import sys
import time

from cron.executions import recover_interrupted_executions

count = int(sys.argv[1])
gap = float(sys.argv[2])
for index in range(count):
    if index:
        time.sleep(gap)
    print(json.dumps({"reclaimed": recover_interrupted_executions()}), flush=True)
"""


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    """Hermetic HERMES_HOME + executions ledger: the live store is never opened."""
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "3600")
    monkeypatch.setattr(executions_mod, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    # The stamp cadence is the agent path's own knob; shrink it so a test-sized script stamps.
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.2)
    bound = executions_mod._live_owner_stale_after_seconds()
    assert bound == pytest.approx(7200.0), bound  # max(3×600, 3600 script, 7200 floor)
    return home


def _progress_at(execution_id: str):
    with executions_mod._transaction() as conn:
        row = conn.execute(
            "SELECT progress_at FROM executions WHERE id=?", (execution_id,)).fetchone()
    return None if row is None else row["progress_at"]


def _row(execution_id: str) -> dict:
    with executions_mod._transaction() as conn:
        row = conn.execute(
            "SELECT status, error, claimed_at, progress_at FROM executions WHERE id=?",
            (execution_id,)).fetchone()
    return dict(row)


def _sweep_from_another_process(tmp_path, *, count: int = 1, gap: float = 0.0) -> list[int]:
    """Sweep from a child process and return one reclaim count per sweep."""
    driver = tmp_path / "sweeper.py"
    driver.write_text(_SWEEPER, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(driver), str(count), str(gap)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180, check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line)["reclaimed"] for line in proc.stdout.splitlines() if line.strip()]


def _wait_until(predicate, *, timeout: float = 15.0, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def _running_job_with_ancient_claim(job_id: str, age_seconds: float) -> str:
    """A live-owned ``running`` row whose claim is already older than the stale bound (the
    incident shape: a long script run, no progress stamp yet)."""
    row = executions_mod.create_execution(job_id, source="cron")
    assert executions_mod.mark_execution_running(row["id"]) is not None
    with executions_mod._transaction() as conn:
        conn.execute(
            "UPDATE executions SET claimed_at=? WHERE id=?",
            ((_hermes_now() - timedelta(seconds=age_seconds)).isoformat(), row["id"]),
        )
    return row["id"]


def test_live_no_agent_script_run_past_the_bound_is_not_reclaimed(ledger, tmp_path):
    """The script keeps running well past ``_live_owner_stale_after_seconds()``: because the
    supervisor loop keeps stamping, an out-of-process sweep leaves the row ``running``."""
    home = ledger
    scripts = home / "scripts"
    scripts.mkdir()
    script = scripts / "long_job.py"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    # Holds the run open until the test says so, so every sweep below lands mid-run.
    script.write_text(
        "import os, time\n"
        f"open({str(ready) + '.tmp'!r}, 'w').write('1')\n"
        f"os.replace({str(ready) + '.tmp'!r}, {str(ready)!r})\n"
        f"deadline = time.monotonic() + 120\n"
        f"while not os.path.exists({str(release)!r}) and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )

    bound = executions_mod._live_owner_stale_after_seconds()
    execution_id = _running_job_with_ancient_claim("long-script-job", bound + 600)
    assert _progress_at(execution_id) is None  # nothing has stamped yet: the defect's shape

    job = {"id": "long-script-job", "name": "long script", "no_agent": True,
           "execution_id": execution_id}
    results, errors = [], []

    def drive() -> None:
        try:
            results.append(
                scheduler_script._run_job_script_with_claim_heartbeat(job, str(script)))
        except BaseException as exc:  # surfaced through ``errors`` below
            errors.append(exc)

    worker = threading.Thread(target=drive, name="cron-script-progress-test")
    worker.start()
    try:
        _wait_until(lambda: ready.exists(), what="the script to start")
        first_stamp = _wait_until(lambda: _progress_at(execution_id), what="a progress stamp")

        # A claim older than the bound, a live owner, and three sweeps from another process.
        sweeps = _sweep_from_another_process(tmp_path, count=3, gap=0.3)
        assert sweeps == [0, 0, 0], sweeps
        assert _row(execution_id)["status"] == "running"

        # Still stamping, not a single stamp at startup.
        later_stamp = _progress_at(execution_id)
        assert later_stamp is not None and later_stamp > first_stamp
    finally:
        release.write_text("1", encoding="utf-8")
        worker.join(timeout=60)

    assert not errors, errors
    assert results == [(True, "")]

    # Control: the SAME row, same live owner, once the stamps stop for longer than the bound,
    # is still reclaimed — the fix makes the bound measure silence, it does not disable it.
    with executions_mod._transaction() as conn:
        conn.execute(
            "UPDATE executions SET progress_at=? WHERE id=?",
            ((_hermes_now() - timedelta(seconds=bound + 1)).isoformat(), execution_id),
        )
    assert _sweep_from_another_process(tmp_path) == [1]
    assert _row(execution_id)["status"] == "unknown"
    assert "wedged" in (_row(execution_id)["error"] or "").lower()


def test_silent_no_agent_script_run_is_still_reclaimed_at_the_bound(ledger):
    """A script row whose owner went silent past the bound is reclaimed — the wedge backstop
    still fires, it just no longer fires on a run that is visibly working."""
    bound = executions_mod._live_owner_stale_after_seconds()
    execution_id = executions_mod.create_execution("silent-script-job", source="cron")["id"]
    # Same live pid, but a foreign process_id — exactly how a sweeper sees another worker's row.
    with executions_mod._transaction() as conn:
        conn.execute(
            """UPDATE executions
                  SET status='running', process_id='wedged-worker', pid=?,
                      process_started_at=?, claimed_at=?
                WHERE id=?""",
            (os.getpid(), executions_mod._process_start_time(os.getpid()),
             (_hermes_now() - timedelta(seconds=bound + 600)).isoformat(), execution_id),
        )
    assert _progress_at(execution_id) is None

    assert executions_mod.recover_interrupted_executions() == 1
    row = _row(execution_id)
    assert row["status"] == "unknown"
    assert "wedged" in (row["error"] or "").lower()


def test_no_agent_run_job_stamps_progress_for_the_script(ledger):
    """End to end: ``run_job`` (the no_agent path, given only an ``execution_id``) stamps the
    row while its script runs. Before the fix the row's ``progress_at`` stayed NULL."""
    home = ledger
    scripts = home / "scripts"
    scripts.mkdir()
    script = scripts / "short_job.py"
    script.write_text("import time\ntime.sleep(1.0)\n", encoding="utf-8")
    execution_id = executions_mod.create_execution("no-agent-job", source="cron")["id"]
    assert executions_mod.mark_execution_running(execution_id) is not None

    success, _doc, _response, error = scheduler.run_job(
        {"id": "no-agent-job", "name": "no agent", "no_agent": True, "script": str(script)},
        execution_id=execution_id,
    )

    assert success is True and error is None
    stamp = _progress_at(execution_id)
    assert stamp is not None, "a running no_agent script must stamp progress_at"
