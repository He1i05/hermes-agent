"""The local missed-run policy preserves grace and manual triggers."""
from datetime import timedelta

import pytest

from cron import jobs


@pytest.mark.parametrize("catch_up", [True, False])
def test_missed_policy_preserves_grace_and_manual(tmp_path, monkeypatch, catch_up):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"cron:\n  catch_up_missed: {str(catch_up).lower()}\n", encoding="utf-8")
    with jobs.use_cron_store(tmp_path / "cron"):
        now = jobs._hermes_now()
        for name, lag in [("stale", 14400), ("grace", 30), ("manual", 14400)]:
            job = jobs.create_job(prompt=name, schedule="every 1h", model="fixture", deliver="local")
            stored = jobs.load_jobs()
            row = next(row for row in stored if row["id"] == job["id"])
            row["next_run_at"] = (now - timedelta(seconds=lag)).isoformat()
            if name == "manual":
                row["manual_run_at"] = row["next_run_at"]
            jobs.save_jobs(stored)
        due = {job["prompt"] for job in jobs.get_due_jobs()}
        assert due == ({"stale", "grace", "manual"} if catch_up else {"grace", "manual"})
        stale = next(row for row in jobs.load_jobs() if row["prompt"] == "stale")
        assert jobs._ensure_aware(jobs.datetime.fromisoformat(stale["next_run_at"])) > now


@pytest.mark.parametrize("uncomputable", [False, True])
def test_default_and_uncomputable_still_catch_up(tmp_path, monkeypatch, uncomputable):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if uncomputable:
        (tmp_path / "config.yaml").write_text("cron:\n  catch_up_missed: false\n", encoding="utf-8")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="default", schedule="every 1h", model="fixture", deliver="local")
        stored = jobs.load_jobs()
        stored[0]["next_run_at"] = (jobs._hermes_now() - timedelta(hours=4)).isoformat()
        jobs.save_jobs(stored)
        if uncomputable:
            monkeypatch.setattr(jobs, "compute_next_run", lambda *args, **kwargs: None)
        assert [row["id"] for row in jobs.get_due_jobs()] == [job["id"]]


@pytest.mark.parametrize("schedule", ["every 1m", "every 5m", "every 1h", "0 9 * * 1"])
def test_a_planned_retry_can_never_be_classified_catch_up(schedule):
    """The interrupted-run retry rides the ONE due scan, so "retry vs catch-up" is not a race
    between two dispatchers but a classification invariant: the retry delay is capped at the
    job's own grace window, so its lateness can only classify ``on_time``/``late``. Catch-up
    only ever acts on an instant already past grace, and the same instant cannot be both.
    """
    from cron.retry_interrupted import delay_seconds

    parsed = jobs.parse_schedule(schedule)
    grace = jobs._compute_grace_seconds(parsed, None)
    assert grace > 0
    for attempt in range(1, 7):  # past the 5-attempt ceiling on purpose: the cap must hold
        delay = delay_seconds({"schedule": parsed}, attempt)
        assert delay <= grace
        for lateness in (0.0, 1.0, grace - 1.0, float(grace)):
            assert jobs._classify_dispatch_lateness(lateness, grace) != "catch_up"
    # Control: one second past grace IS the catch-up branch, which only the natural schedule
    # can reach (an outage, not a retry).
    assert jobs._classify_dispatch_lateness(grace + 1, grace) == "catch_up"
