"""Bounded retry for cron attempts killed through no fault of the job.

The audit ledger (``cron/executions.py``) keeps one immutable row per attempt and is not a
retry queue. This module is its one bounded exception: an attempt that ended ``unknown`` with
``outcome_class='owner_gone'`` (its owner was proved gone before any terminal state) or
``failed`` with ``outcome_class='interrupted'`` (SIGTERM, a killed run thread, a transport
cancel) was killed mid-flight, so its occurrence never completed. Re-running it cannot
resurrect a terminal state: the retry is a NEW ledger row (``retry_of`` / ``attempt``) fired
through the ordinary due scan.

Two stores, two jobs:

* ``jobs.json`` says WHETHER and HOW MUCH — the per-job ``retry`` object
  (``max_attempts`` / ``backoff_seconds`` / ``max_backoff_seconds``; absent = defaults).
* ``executions.db`` says WHICH ATTEMPT and holds the evidence — the same fenced UPDATE that
  terminalizes the attempt is the only writer of the retry *intent* (``retry_pending``,
  ``retry_at``), so no rule "a terminal row may be updated later" is introduced.

``plan_pending_retries`` turns an intent into a schedule amendment: ``next_run_at`` is pulled
to the retry instant and a ``retry_interrupted`` marker (with the schedule ``expr`` it was
planned under) is stamped on the job. The existing due scan then fires it, so every guarantee
that already exists applies unchanged — at-most-once claim, occurrence dedupe, ownership
fencing, delivery.

Delays are capped at the job's own grace window (``cron.jobs._compute_grace_seconds``), so a
planned retry is always *inside* grace and therefore classifies ``on_time``/``late`` — never
``catch_up``. That is what makes "retry" and ``cron.catch_up_missed`` mutually exclusive by
construction instead of a race between two dispatchers: catch-up only ever acts on an instant
already past grace, and the same instant cannot be both.

Kill switches: ``cron.retry_interrupted: false`` in config.yaml, and per-job
``retry.max_attempts: 0``. Recurring jobs only: finite one-shots are pre-claimed by
``claim_dispatch`` (at-most-times, #38758) and must not regain a consumed dispatch; a killed
one-shot stays a visible row and ``hermes cron resume <job> --run-now`` remains the operator
path.

Side-effect note: a killed attempt may already have written partial side effects, so a retry
makes that occurrence at-least-once instead of at-most-once. That is why only *proved death* /
*interruption* outcomes are retried, never job-caused failures, and why a job with
non-idempotent external side effects should be pinned to ``retry.max_attempts: 0``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from cron.constants import is_recurring
from hermes_time import now as _hermes_now

logger = logging.getLogger("cron.scheduler")

# Persisted on the job while a retry is parked on the schedule:
# {"at": <retry instant>, "attempt": <the attempt the retry will be>, "row_id": <parent row>,
#  "expr": <the cron expression it was planned under>}. A direct jobs.json schedule edit changes
# ``expr`` and thereby invalidates the parked retry, exactly like ``cron.unreachable_retry``.
STATE_KEY = "retry_interrupted"

# Config kill switch, read at most once per planning pass (never on an idle tick with no
# candidates — see tests/cron/test_idle_tick_config_skip.py).
CONFIG_KEY = "cron.retry_interrupted"

# Only these two classes are "no fault of the job". ``owner_wedged`` is deliberately NOT here:
# that owner was never killed and may still be writing (a live no_agent script run is reclaimed
# this way today), so a retry would be a genuine concurrent second execution.
RETRYABLE_OUTCOME_CLASSES = frozenset({"owner_gone", "interrupted"})

# Hard ceilings, applied by the schema normalizer AND re-clamped when planning (defence in depth
# against a hand-edited jobs.json).
MAX_ATTEMPTS_CEILING = 5
MAX_BACKOFF_SECONDS_CEILING = 3600
MAX_BACKOFF_TOTAL_CEILING_SECONDS = 21600

# One-sided de-synchronisation of a boot where a dozen jobs are all swept in the same second:
# jitter only ever pulls a retry EARLIER, so rule R1 (below) can be decided on the base instant.
JITTER_FRACTION = 0.2

# Bound the work one planning pass may do (and the job-store writes it can cause locks for).
MAX_INTENTS_PER_PASS = 8


@dataclass(frozen=True)
class RetryPolicy:
    """The per-job ``retry`` object, resolved to defaults when absent or malformed."""

    max_attempts: int = 1
    backoff_seconds: int = 60
    max_backoff_seconds: int = 1800


DEFAULT_RETRY_POLICY = RetryPolicy()


def retry_enabled(cfg: Optional[dict] = None) -> bool:
    """``cron.retry_interrupted`` — default ON (the ``retry_unreachable`` precedent).

    Default ON is an intentional decision, not an accident: it is what turns the 2026-11-09
    class (a killed run lost forever) into one bounded re-run. A job with non-idempotent
    external side effects opts out with ``retry.max_attempts: 0``.
    """
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
        except Exception:  # config unreadable — keep the reliability default
            return True
    cron_cfg = (cfg or {}).get("cron")
    if not isinstance(cron_cfg, dict):
        return True
    return cron_cfg.get("retry_interrupted") is not False


def _clamped_int(raw: Any, default: int, low: int, high: int) -> int:
    if raw is None or isinstance(raw, bool):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(low, min(value, high))


def job_retry_policy(job: Optional[dict[str, Any]]) -> RetryPolicy:
    """Resolve a job record's ``retry`` object. Never raises: this runs on the tick path."""

    raw = (job or {}).get("retry")
    if not isinstance(raw, dict):
        return DEFAULT_RETRY_POLICY
    return RetryPolicy(
        max_attempts=_clamped_int(raw.get("max_attempts"), 1, 0, MAX_ATTEMPTS_CEILING),
        backoff_seconds=_clamped_int(
            raw.get("backoff_seconds"), 60, 0, MAX_BACKOFF_SECONDS_CEILING),
        max_backoff_seconds=_clamped_int(
            raw.get("max_backoff_seconds"), 1800, 0, MAX_BACKOFF_TOTAL_CEILING_SECONDS),
    )


def clear_state(job: dict[str, Any]) -> None:
    """The parked retry is spent (or invalidated): the marker goes away."""
    job.pop(STATE_KEY, None)


def _grace_seconds(job: dict[str, Any]) -> float:
    from cron.jobs import _compute_grace_seconds

    return float(_compute_grace_seconds(job.get("schedule") or {}, job.get("grace_minutes")))


def delay_seconds(job: dict[str, Any], attempt: int) -> float:
    """Exponential backoff for the Nth retry, capped by the job's own grace window.

    ``delay_n = min(backoff * 2**(n-1), min(max_backoff, grace_seconds(job)))``. Reusing the
    grace window means a planned retry is always inside the job's late window, so the due scan
    can never classify it ``catch_up`` — the no-double-fire invariant of the catch-up policy.
    """
    policy = job_retry_policy(job)
    cap = max(0.0, min(float(policy.max_backoff_seconds), _grace_seconds(job)))
    exponent = max(0, int(attempt) - 1)
    try:
        base = float(policy.backoff_seconds) * (2 ** exponent)
    except (OverflowError, ValueError):
        base = cap
    return max(0.0, min(base, cap))


def _jitter_seconds(execution_id: str, delay: float) -> float:
    """Deterministic, one-sided (earlier-only) jitter derived from the parent attempt id.

    Deterministic so a crashed planning pass recomputes the same instant (and tests can assert
    an exact window); one-sided so a retry can never land LATER than its base delay, which is
    what lets R1 be decided on the base instant regardless of jitter.
    """
    if delay <= 0:
        return 0.0
    digest = hashlib.sha256(str(execution_id).encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return delay * JITTER_FRACTION * fraction


def retry_instant(
    job: dict[str, Any], *, execution_id: str, attempt: int, finished_at: datetime,
) -> Optional[datetime]:
    """The instant the Nth retry of this occurrence should fire, or None when unplannable."""
    from cron.jobs import _seconds_after

    delay = delay_seconds(job, attempt)
    return _seconds_after(finished_at, delay - _jitter_seconds(execution_id, delay))


def _retry_refusal(
    job: dict[str, Any], *, attempt: int, retry_at: datetime,
) -> Optional[str]:
    """Why this attempt must NOT be retried, or None. Ordered gates; each alone prevents a
    double fire or an unintended run."""
    from cron.jobs import _instant_at_or_before, _parse_aware, is_job_runnable, is_terminal_job

    policy = job_retry_policy(job)
    if policy.max_attempts <= 0:
        return "retry disabled for this job (retry.max_attempts=0)"
    if not isinstance(job.get("schedule"), dict):
        # A raw/unparsed schedule (legacy or hand-edited record, test double): nothing on this
        # path can reason about its cadence, so leave the occurrence to the natural schedule.
        return "schedule is not parsed (no cadence to bound a retry by)"
    if not is_recurring(job):
        return "not a recurring job (one-shots keep the operator resume path)"
    if attempt > min(policy.max_attempts, MAX_ATTEMPTS_CEILING):
        return f"retry cap reached ({attempt - 1}/{policy.max_attempts})"
    if is_terminal_job(job) or not is_job_runnable(job):
        return "job is paused, disabled or terminal"
    # R1 — the natural occurrence always wins. A stored instant at-or-before the retry instant
    # either already owns this slot (catch-up/due scan) or is the sooner fire; never both.
    natural = _parse_aware(job.get("next_run_at"))
    if natural is not None and _instant_at_or_before(natural, retry_at):
        return f"natural occurrence at {job.get('next_run_at')} fires first"
    # R2 — pending-slot reverse gate: a slot the store will restore and fire itself must not
    # also be retried (should be impossible: the fire claim clears pending_slot).
    if isinstance(job.get("pending_slot"), dict):
        return "pending_slot still names the killed instant"
    # R3 — completion dedupe: a crash after a successful retry must not fire a third run.
    from cron.occurrences import completed_occurrence

    if completed_occurrence(job, retry_at.isoformat()):
        return "retry occurrence already completed"
    return None


def retry_intent_for(
    job: Optional[dict[str, Any]],
    *,
    execution_id: str,
    outcome_class: Optional[str],
    attempt: Optional[int] = None,
    finished_at: Optional[str] = None,
) -> Optional[tuple[str, int]]:
    """``(retry_at_iso, attempt)`` to stamp on a terminal row, or None when nothing is owed.

    Called by the writers that *know* the outcome class (the dead-owner sweep for
    ``owner_gone``, the three interruption paths for ``interrupted``) so no prose classifier
    ever sits on a side-effect decision. ``finish_execution`` is handed the decided intent and
    never a policy, which keeps the ledger writer job-free.
    """
    if outcome_class not in RETRYABLE_OUTCOME_CLASSES or not isinstance(job, dict):
        return None
    from cron.jobs import _parse_aware

    if attempt is None:
        from cron.executions import get_execution

        record = get_execution(execution_id) or {}
        attempt = int(record.get("attempt") or 1)
    finished = _parse_aware(finished_at) if finished_at else None
    if finished is None:
        finished = _hermes_now()
    retry_at = retry_instant(job, execution_id=execution_id, attempt=int(attempt),
                             finished_at=finished)
    if retry_at is None:
        return None
    refusal = _retry_refusal(job, attempt=int(attempt), retry_at=retry_at)
    if refusal:
        logger.info(
            "Job '%s' (%s): attempt %s ended %s but is not retried — %s",
            job.get("name", job.get("id", "?")), job.get("id"), attempt, outcome_class, refusal)
        return None
    return (retry_at.isoformat(), int(attempt) + 1)


def is_retry_interrupted_fire(job: dict[str, Any], next_run: Optional[str]) -> bool:
    """True for the exact instant parked by the planner (off the cron lattice).

    The expression fingerprint keeps a direct jobs.json schedule edit from inheriting the
    exception, as in ``cron.unreachable_retry.is_retry_fire``.
    """
    state = job.get(STATE_KEY) or {}
    return state.get("at") == next_run and state.get("expr") == (job.get("schedule") or {}).get("expr")


def is_retry_interrupted_run(job: dict[str, Any]) -> bool:
    """True when this dispatch fired the planner's instant: the attempt is a RETRY of an
    occurrence that already counted, so it must not spend another ``repeat`` slot."""
    from cron.occurrences import scheduled_instant

    state = job.get(STATE_KEY) or {}
    instant = scheduled_instant(job.get("_scheduled_instant"))
    return instant is not None and scheduled_instant(state.get("at")) == instant


def is_off_lattice_rerun(job: dict[str, Any]) -> bool:
    """True for every re-run of an occurrence that already counted (this retry path and the
    unreachable-model ladder): ``mark_job_run(..., ladder_rung=True)`` must not bump
    ``repeat.completed`` a second time."""
    from cron.unreachable_retry import is_retry_run

    return is_retry_run(job) or is_retry_interrupted_run(job)


def dispatch_identity(job: dict[str, Any]) -> tuple[int, Optional[str]]:
    """``(attempt, retry_of)`` for the execution row this dispatch is about to create.

    The attempt number and parent row come from the planner's marker, and only when the
    dispatch really is firing that marker's instant — a manual run or a later natural
    occurrence is attempt 1 of its own occurrence.
    """
    from cron.occurrences import scheduled_instant

    state = job.get(STATE_KEY) or {}
    instant = scheduled_instant(job.get("_scheduled_instant"))
    if instant is None or scheduled_instant(state.get("at")) != instant:
        return 1, None
    try:
        attempt = max(2, int(state.get("attempt") or 2))
    except (TypeError, ValueError):
        attempt = 2
    return attempt, state.get("row_id")


def plan_pending_retries(limit: int = MAX_INTENTS_PER_PASS) -> int:
    """Turn owed retry intents into schedule amendments; return how many were planned.

    Cheap and safe on an idle tick: one indexed query, and the config kill switch is only read
    once a candidate exists (preserves #33612 / tests/cron/test_idle_tick_config_skip.py).

    The intent is claimed first (R4 — a fenced ``retry_pending 1 -> 0``), so two schedulers
    sharing a store cannot both amend the schedule; the loser sees ``rowcount == 0`` and drops
    out. A gate that refuses after the claim drops the intent durably and logs why (F6).
    """
    from cron.executions import claim_retry_intent, pending_retry_intents
    from cron.jobs import _parse_aware, get_job, _with_job, is_job_runnable, is_terminal_job

    intents = pending_retry_intents(limit)
    if not intents:
        return 0
    if not retry_enabled():
        logger.info(
            "cron.retry_interrupted is off — %d interrupted-run retr%s left unplanned",
            len(intents), "y" if len(intents) == 1 else "ies")
        return 0

    planned = 0
    for row in intents:
        execution_id = str(row.get("id") or "")
        job_id = str(row.get("job_id") or "")
        retry_at_text = row.get("retry_at")
        attempt = int(row.get("attempt") or 1)
        retry_at = _parse_aware(retry_at_text)
        if not execution_id or not job_id or retry_at is None:
            continue
        if not claim_retry_intent(execution_id):
            continue  # another planner won this intent
        job = get_job(job_id)
        if job is None:
            logger.warning(
                "cron.retry_interrupted.dropped execution=%s job=%s — the job record is gone",
                execution_id, job_id)
            continue
        refusal = _retry_refusal(job, attempt=attempt, retry_at=retry_at)
        if refusal:
            logger.info(
                "Job '%s' (%s): interrupted-run retry dropped — %s",
                job.get("name", job_id), job_id, refusal)
            continue

        marker = {
            "at": retry_at.isoformat(),
            "attempt": attempt + 1,
            "row_id": execution_id,
            "expr": (job.get("schedule") or {}).get("expr"),
        }

        def apply(jobs, _i, stored):
            # Re-check under the jobs lock: the store may have been paused/edited since the read.
            from cron.jobs import _instant_at_or_before, save_jobs

            if is_terminal_job(stored) or not is_job_runnable(stored):
                logger.info(
                    "Job '%s' (%s): interrupted-run retry dropped — job became unrunnable",
                    stored.get("name", job_id), job_id)
                return False
            natural = _parse_aware(stored.get("next_run_at"))
            if natural is not None and _instant_at_or_before(natural, retry_at):
                logger.info(
                    "Job '%s' (%s): interrupted-run retry dropped — natural occurrence %s fires first",
                    stored.get("name", job_id), job_id, stored.get("next_run_at"))
                return False
            stored["next_run_at"] = retry_at.isoformat()
            stored[STATE_KEY] = marker
            if stored.get("state") != "paused":
                stored["state"] = "scheduled"
            # R5 — a normal CAS on the job store; no fire claim is stamped (that would advance
            # next_run_at a second time).
            save_jobs(jobs)
            return True

        if _with_job(job_id, apply, missing=False):
            planned += 1
            logger.info(
                "Job '%s' (%s): retry %d/%d of interrupted attempt %s scheduled at %s",
                job.get("name", job_id), job_id, attempt, job_retry_policy(job).max_attempts,
                execution_id, retry_at.isoformat())
    return planned
