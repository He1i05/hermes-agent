"""Deleting a cron job must not orphan its failure incidents in ``cron_incidents``.

``remove_job`` used to drop the job from ``cron/jobs.json`` (and clear its output dir + notepad)
but never touched the incident ledger. Because ``close_incidents_for_recovered_job`` only fires
after a SUCCESSFUL run, a deleted job's incident could never auto-resolve: it stayed
``detected``/``alerted`` forever and showed up in ``hermes cron incidents`` as permanently open
(5 such orphans on the live store, t_4e6cff2a).

Fix: ``remove_job`` now deletes the job's incident rows (``cron.incidents.delete_incidents_for_job``),
and ``prune_orphan_incidents`` sweeps any residual orphans whose ``job_id`` is absent from
``cron/jobs.json``. Deleting — not resolving — is deliberate: a resolved row re-opens on the same
signature and would keep occupying the CLI view for a job that can never run again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.incidents as incidents


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated job store + incident ledger sharing one throwaway HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    # The incident table lives in the same executions.db the scheduler uses.
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", hermes_home / "cron" / "executions.db")
    return hermes_home


# ── remove_job clears the ledger ───────────────────────────────────────────


def test_remove_job_clears_open_incident_and_spares_siblings(cron_env):
    from cron.jobs import create_job, remove_job

    job = create_job(prompt="Check the feed", schedule="every 1h")
    other = create_job(prompt="Other job", schedule="every 2h")
    open_id, _ = incidents.upsert_incident(job["id"], "provider timeout boom")
    closed_id, _ = incidents.upsert_incident(job["id"], "a distinct acked failure")
    incidents.ack_incident(closed_id)
    sibling_id, _ = incidents.upsert_incident(other["id"], "keep me")

    assert remove_job(job["id"]) is True

    # Every row for the removed job is gone — open AND closed (the job no longer exists).
    assert incidents.get_incident(open_id) is None
    assert incidents.get_incident(closed_id) is None
    assert incidents.list_incidents() == [incidents.get_incident(sibling_id)]
    # The sibling job's incident is untouched.
    assert incidents.get_incident(sibling_id)["state"] == "detected"


def test_remove_job_leaves_no_orphan_in_incidents_view(cron_env):
    """Acceptance shape: after remove_job, no incident's job_id is absent from jobs.json."""
    from cron.jobs import create_job, load_jobs, remove_job

    job = create_job(prompt="ephemeral", schedule="every 5m")
    incidents.upsert_incident(job["id"], "boom")
    incidents.upsert_incident("ghost-job-id", "pre-existing orphan")

    assert remove_job(job["id"]) is True

    live_ids = {j["id"] for j in load_jobs()}
    orphans = [inc for inc in incidents.list_incidents() if inc["job_id"] not in live_ids]
    # The removed job left nothing behind; the pre-existing orphan (a job that never existed
    # here) is a separate residual handled by prune, not by remove_job.
    assert [o["job_id"] for o in orphans] == ["ghost-job-id"]


def test_remove_job_survives_incident_store_failure(cron_env, monkeypatch):
    """Incident cleanup is best effort — a ledger error must never block job removal."""
    from cron.jobs import create_job, get_job, remove_job

    job = create_job(prompt="Check the feed", schedule="every 1h")

    def _boom(job_id):
        raise RuntimeError("db locked")

    monkeypatch.setattr(incidents, "delete_incidents_for_job", _boom)
    assert remove_job(job["id"]) is True
    assert get_job(job["id"]) is None


# ── prune sweeps residual orphans ──────────────────────────────────────────


def test_prune_orphan_incidents_removes_only_absent_jobs(cron_env):
    from cron.jobs import create_job

    live = create_job(prompt="live job", schedule="every 1h")
    live_id, _ = incidents.upsert_incident(live["id"], "keep me")
    orphan_a, _ = incidents.upsert_incident("ae41a539959b", "orphan one")
    orphan_b, _ = incidents.upsert_incident("e2ea65344147", "orphan two")

    removed = incidents.prune_orphan_incidents()

    assert {row["id"] for row in removed} == {orphan_a, orphan_b}
    assert incidents.get_incident(orphan_a) is None
    assert incidents.get_incident(orphan_b) is None
    # The live job's incident is guarded — never touched.
    assert incidents.get_incident(live_id)["state"] == "detected"

    # Idempotent: a second sweep has nothing left to do.
    assert incidents.prune_orphan_incidents() == []


def test_cli_prune_reports_and_clears(cron_env, capsys):
    from cron.jobs import create_job
    from hermes_cli.cron import cron_incidents

    live = create_job(prompt="live job", schedule="every 1h")
    incidents.upsert_incident(live["id"], "keep me")
    orphan_id, _ = incidents.upsert_incident("deadbeef0001", "orphan")

    args = argparse.Namespace(incident_action="prune", state=None, incident_id=None)
    assert cron_incidents(args) == 0
    out = capsys.readouterr().out
    assert "Pruned 1" in out and "deadbeef0001" in out
    assert incidents.get_incident(orphan_id) is None

    # Clean second run reports no orphans instead of an empty/blank line.
    assert cron_incidents(args) == 0
    assert "No orphaned incidents" in capsys.readouterr().out
