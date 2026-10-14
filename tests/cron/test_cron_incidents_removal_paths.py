"""Every path that drops a job from ``cron/jobs.json`` must also clear its incident rows.

t_4e6cff2a taught ``remove_job`` (the ``hermes cron delete`` path) to delete the removed job's
``cron_incidents`` rows. ``remove_job`` is NOT the only removal path: the due scan retires a stale
one-shot (``_DueScan.retire``), the retention sweep prunes completed one-shots, and
``claim_dispatch`` pops a wedged one-shot — each writes the store via
``save_jobs(removed_ids=...)`` / ``scan.retire(...)`` without touching the incident ledger. The
retire/wedged paths then left permanently-open (``detected``/``alerted``) orphans the operator
could never clear (``close_incidents_for_recovered_job`` only fires after a SUCCESSFUL run, and a
job that no longer exists never runs); the sweep left stale ``resolved`` rows. Same orphan class
t_4e6cff2a closed for ``remove_job``.

Fix: the set-based removal paths route their removed ids through the shared
``cron.incidents.delete_incidents_for_job_ids`` cleanup (``_clear_incidents_for_removed_jobs`` in
``cron.jobs``). Deleting — not resolving — is deliberate: a resolved row re-opens on the same
signature and keeps occupying ``hermes cron incidents`` for a job that can never run again.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import cron.incidents as incidents

FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def cron_env(tmp_path, monkeypatch):
    """Isolated job store + incident ledger sharing one throwaway HERMES_HOME, with a pinned clock."""
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    monkeypatch.setattr(jobs_mod, "_hermes_now", lambda: FIXED_NOW)

    # The incident table lives in the same executions.db the scheduler uses.
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", hermes_home / "cron" / "executions.db")
    return hermes_home


def _oneshot(jid, run_at, *, state="scheduled", last_run_at=None, completed=0):
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "once", "run_at": run_at.isoformat()},
        "next_run_at": run_at.isoformat(),
        "last_run_at": last_run_at,
        "enabled": True,
        "state": state,
        "repeat": {"times": 1, "completed": completed},
        "deliver": "local",
    }


# ── due-scan retire path (_DueScan.retire) ─────────────────────────────────


def test_due_scan_retire_clears_incident(cron_env):
    """A stale one-shot retired by the due scan must not leave its incident behind."""
    from cron.jobs import get_due_jobs, load_jobs, save_jobs

    stale = _oneshot("stale1", FIXED_NOW - timedelta(hours=3))
    save_jobs([stale])
    incident_id, _ = incidents.upsert_incident("stale1", "provider timeout boom")

    get_due_jobs()

    # Retired from the store (past the grace window)...
    assert [j["id"] for j in load_jobs()] == []
    # ...and its incident went with it — no permanently-open orphan.
    assert incidents.get_incident(incident_id) is None


# ── retention sweep path (_sweep_completed_oneshots) ───────────────────────


def test_retention_sweep_clears_incident(cron_env):
    """A completed one-shot aged out by the retention sweep must not leave its incident behind."""
    from cron.jobs import get_due_jobs, load_jobs, save_jobs

    finished = FIXED_NOW - timedelta(days=30)
    old = _oneshot("done1", finished, state="completed", last_run_at=finished.isoformat())
    save_jobs([old])
    incident_id, _ = incidents.upsert_incident("done1", "old failure")
    incidents.set_incident_state(incident_id, "resolved")

    get_due_jobs()

    assert [j["id"] for j in load_jobs()] == []
    assert incidents.get_incident(incident_id) is None


# ── claim_dispatch wedged-removal path ─────────────────────────────────────


def test_claim_dispatch_wedged_removal_clears_incident(cron_env):
    """A wedged one-shot popped by claim_dispatch must not leave its incident behind."""
    from cron.jobs import claim_dispatch, load_jobs, save_jobs

    wedged = _oneshot("wedged1", FIXED_NOW - timedelta(minutes=5), completed=1)
    save_jobs([wedged])
    incident_id, _ = incidents.upsert_incident("wedged1", "tick died mid-run")

    # repeat.times=1, completed=1, last_run_at=None -> the wedged-claim branch removes it.
    assert claim_dispatch("wedged1") is False

    assert [j["id"] for j in load_jobs()] == []
    assert incidents.get_incident(incident_id) is None


# ── guard: live jobs' incidents are never collateral ───────────────────────


def test_removal_paths_spare_live_job_incidents(cron_env):
    """Only the removed job's rows are cleared — a surviving job's incident is untouched."""
    from cron.jobs import get_due_jobs, load_jobs, save_jobs

    live = _oneshot("live1", FIXED_NOW + timedelta(hours=1))  # future -> stays scheduled
    stale = _oneshot("stale2", FIXED_NOW - timedelta(hours=3))
    save_jobs([live, stale])
    live_inc, _ = incidents.upsert_incident("live1", "still alive")
    stale_inc, _ = incidents.upsert_incident("stale2", "gone")

    get_due_jobs()

    assert [j["id"] for j in load_jobs()] == ["live1"]
    assert incidents.get_incident(live_inc) is not None
    assert incidents.get_incident(stale_inc) is None
