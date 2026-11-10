"""The per-job ``retry`` field in the cron job schema (validation, defaults, compatibility).

``jobs.json`` says WHETHER and HOW MUCH an interrupted run is retried; the ledger
(``cron/executions.py``) says which attempt and holds the evidence. The field is additive: an
absent key means the defaults (``1/60/1800``), ``max_attempts: 0`` means never retry this job,
and an older build that strips unknown keys cannot break a record written by a newer one.
"""

from __future__ import annotations

import pytest

from cron import jobs
from cron.retry_interrupted import DEFAULT_RETRY_POLICY, job_retry_policy


@pytest.fixture(autouse=True)
def _store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    with jobs.use_cron_store(tmp_path):
        yield tmp_path


def _job(**kwargs) -> dict:
    kwargs.setdefault("schedule", "every 5m")
    return jobs.create_job(prompt="retry schema fixture", model="fixture", deliver="local",
                           **kwargs)


def test_retry_policy_persists_and_resolves():
    job = _job(retry={"max_attempts": 2, "backoff_seconds": 30, "max_backoff_seconds": 120})

    stored = jobs.get_job(job["id"])
    assert stored["retry"] == {
        "max_attempts": 2, "backoff_seconds": 30, "max_backoff_seconds": 120}
    assert job_retry_policy(stored).max_attempts == 2


def test_partial_policy_takes_the_documented_defaults():
    job = _job(retry={"max_attempts": 0})

    stored = jobs.get_job(job["id"])
    assert stored["retry"] == {
        "max_attempts": 0, "backoff_seconds": 60, "max_backoff_seconds": 1800}
    assert job_retry_policy(stored).max_attempts == 0


def test_absent_retry_reads_as_defaults_and_is_not_persisted():
    job = _job()

    stored = jobs.load_jobs()[0]
    assert "retry" not in stored  # optional keys are persisted only when set
    assert job_retry_policy(jobs.get_job(job["id"])) == DEFAULT_RETRY_POLICY


@pytest.mark.parametrize(
    "value",
    [
        {"max_attempts": True},          # bool is an int in Python
        {"max_attempts": "2"},
        {"max_attempts": -1},
        {"max_attempts": 6},
        {"backoff_seconds": -1},
        {"backoff_seconds": 3601},
        {"max_backoff_seconds": 10, "backoff_seconds": 60},  # cap below the first delay
        {"max_backoff_seconds": 21601},
        {"attempts": 2},                 # unknown key
        ["max_attempts", 2],             # not an object
        "2",
    ],
)
def test_invalid_retry_is_rejected_before_anything_is_stored(value):
    with pytest.raises(ValueError):
        _job(retry=value)
    assert jobs.load_jobs() == []


def test_update_can_set_and_clear_the_policy():
    job = _job()
    jobs.update_job(job["id"], {"retry": {"max_attempts": 3, "backoff_seconds": 90}})
    assert jobs.get_job(job["id"])["retry"]["max_attempts"] == 3

    jobs.update_job(job["id"], {"retry": None})
    assert job_retry_policy(jobs.get_job(job["id"])) == DEFAULT_RETRY_POLICY

    with pytest.raises(ValueError):
        jobs.update_job(job["id"], {"retry": {"max_attempts": 99}})
    assert jobs.get_job(job["id"])["retry"] is None


def test_legacy_record_without_the_key_loads_and_reads_as_defaults():
    job = _job()
    stored = jobs.load_jobs()
    stored[0].pop("retry", None)
    jobs.save_jobs(stored)

    assert job_retry_policy(jobs.get_job(job["id"])) == DEFAULT_RETRY_POLICY


def test_the_due_scan_normalizer_leaves_retry_alone():
    job = _job(retry={"max_attempts": 4})
    raw = jobs.load_jobs()
    assert jobs._normalize_due_scan_records(raw) is False
    assert jobs.load_jobs()[0]["retry"] == {
        "max_attempts": 4, "backoff_seconds": 60, "max_backoff_seconds": 1800}
    assert jobs.get_job(job["id"])["retry"]["max_attempts"] == 4


def test_malformed_stored_policy_cannot_raise_on_the_tick_path():
    """A hand-edited jobs.json must not crash the scheduler: the reader clamps and falls back."""
    assert job_retry_policy({"retry": "nonsense"}) == DEFAULT_RETRY_POLICY
    assert job_retry_policy({"retry": {"max_attempts": True, "backoff_seconds": "x"}}) == (
        DEFAULT_RETRY_POLICY)
    assert job_retry_policy({"retry": {"max_attempts": 99}}).max_attempts == 5
