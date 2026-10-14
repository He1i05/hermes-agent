"""Durable cron failure incidents with signature dedup and ack.

The executions ledger records every attempt; this module groups the *failures* into incidents keyed
by ``(job_id, error signature)`` so the same job failing with the same error does not re-ping the
operator every run once acknowledged. Lifecycle: ``detected`` → ``alerted`` → ``closed``. The same
job + same normalized error resolves to the SAME incident id, so a closed incident stays closed
until the error text changes and mints a new one. ``alerted`` means a failure ping actually reached
the operator (``alerted_at`` = when the latest one did; the scheduler withholds repeats until
``cron.failure_repeat_alert_hours`` have passed). Incidents share ``cron/executions.db`` with
``cron.executions`` (one ledger file).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from cron import executions as _executions
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override (mirrors ``cron.executions.EXECUTIONS_FILE``).
EXECUTIONS_FILE: Optional[Path] = None

# ``resolved``: the job ran OK after the failure (auto); ``closed``: the operator acked the signature
# and wants it silent. Only ``closed`` is terminal; a repeat of a resolved error re-opens it.
INCIDENT_STATES = ("detected", "alerted", "resolved", "closed")
_FAILURE_TYPE_ORDER = (
    ("rate_limit", (r"\b429\b", "rate limit", "usage limit", "quota")),
    ("timeout", ("timeout", "timed out")),
    ("auth", (r"\b401\b", "unauthorized", "authentication", "auth")),
    ("delivery", ("delivery", "deliver", "delivering")),
    ("config", ("config", "configuration", "validation")),
    ("script", ("script", "no_agent")),
    ("agent", ("agent", "model", "provider", "inference")),
)
MAX_ERROR_CHARS = 500
_MAX_SIGNATURE_ERROR_CHARS = 200
# Measured durations ("idle for 603s", "retry in 12.5 seconds") differ run to run for the same
# failure, so they are masked out of the signature. Status codes and other numbers still count.
_DURATION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?)\b"
)

_lock = threading.RLock()


def _db_path() -> Path:
    """Shared cron DB path. The ``cron.executions`` override wins when installed so redirecting the
    executions ledger also redirects the incident table (they must stay in the SAME database); then
    this module's own override, then the canonical profile home."""
    for override in (_executions.EXECUTIONS_FILE, EXECUTIONS_FILE):
        if override is not None:
            return Path(override)
    return get_hermes_home().resolve() / "cron" / "executions.db"


def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from cron.jobs import _ensure_cron_dir
    from hermes_cli.sqlite_util import open_db

    path = _db_path()
    _ensure_cron_dir(path.parent)
    return open_db(path, db_label="cron/executions.db", synchronous_full=True, initialize=_initialize_schema)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    conn.execute(
        """CREATE TABLE IF NOT EXISTS cron_incidents (
             id            TEXT PRIMARY KEY,
             job_id        TEXT NOT NULL,
             error_sig     TEXT NOT NULL,
             state         TEXT NOT NULL,
             failure_type  TEXT NOT NULL DEFAULT 'unknown',
             first_seen_at TEXT NOT NULL,
             last_seen_at  TEXT NOT NULL,
             acked_at      TEXT,
             alerted_at    TEXT,
             closed_at     TEXT,
             error         TEXT NOT NULL,
             output_file   TEXT
           )"""
    )
    # Ledgers created before the alert-once gate lack ``alerted_at``; add it in place.
    add_column_if_missing(conn, "cron_incidents", "alerted_at", "alerted_at TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cron_incidents_job "
        "ON cron_incidents(job_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cron_incidents_state "
        "ON cron_incidents(state)"
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def _normalize_error(error: str) -> str:
    """Strip whitespace and lowercase before signing (dedup normalization)."""
    return re.sub(r"\s+", " ", str(error or "")).strip().lower()


def _redact_error(error: str) -> str:
    """Redact secrets (best-effort; the scheduler path never fails on it) then bound the length."""
    text = str(error or "")
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text, force=True)  # persisted to disk: always scrub
    except Exception:
        pass
    return text[:MAX_ERROR_CHARS]


# Volatile tokens that must NOT participate in the dedup signature. An error that merely
# EMBEDS a run-varying token — the dated output dir in rsync's stderr (``.../hermes-backup/
# 2026-09-28/...`` vs ``2026-10-08/...`` minted one incident per night and defeated dedup,
# t_49fc646c), a wall-clock time, a hex request id, an rsync ``mkstemp`` random stem — is the
# SAME failure to the operator. Applied only to the signature, never to the stored error text.
_VOLATILE_TOKEN_PATTERNS = (
    # ISO-8601 datetimes (normalized text is lowercased first, hence the lowercase 't').
    (re.compile(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?"), "<ts>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
    (re.compile(r"\d{2}:\d{2}:\d{2}"), "<time>"),
    (re.compile(r"\b[0-9a-f]{8,}\b"), "<hex>"),
    # rsync mkstemp random suffix: "...mkstemp \"/path/.manifest.json.a4imyc\" failed" — a
    # 5-8 char alphanumeric token right after a dot and immediately before a closing quote.
    (re.compile(r"(?<=\.)[a-z0-9]{5,8}(?=\")"), "<tmp>"),
)


def _dedup_normalize_error(error: str) -> str:
    """Signature-key normalization: whitespace/lowercase, then scrub volatile tokens (run dates,
    times, hex ids, temp-file stems) so the same failure keeps ONE signature across runs."""
    text = _normalize_error(error)
    for pattern, replacement in _VOLATILE_TOKEN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _error_signature(job_id: str, error: str) -> str:
    """Dedup key: stable for same job + same normalized error prefix. Volatile tokens are
    scrubbed first (``_dedup_normalize_error``) so an error that embeds the run date keeps the
    same signature across runs instead of minting one incident per night."""
    normalized = _dedup_normalize_error(error)[:_MAX_SIGNATURE_ERROR_CHARS]
    return hashlib.sha256(job_id.encode() + normalized.encode()).hexdigest()[:12]


def _incident_id(job_id: str, error_sig: str) -> str:
    return f"{job_id[:6]}_{error_sig}"


def _classify_failure_type(error: str) -> str:
    """Classify a failure from error-text keywords; ``unknown`` is the default."""
    text = _normalize_error(error)
    if not text:
        return "unknown"
    for kind, patterns in _FAILURE_TYPE_ORDER:
        for pattern in patterns:
            if pattern.startswith("\\b") and pattern.endswith("\\b"):
                if re.search(pattern, text):
                    return kind
            elif pattern in text:
                return kind
    return "unknown"


def upsert_incident(
    job_id: str, error: str, *, job_name: Optional[str] = None, failure_type: Optional[str] = None,
    output_file: Optional[str] = None,
) -> tuple[str, bool]:
    """Record (or refresh) the incident for ``job_id`` + ``error``; returns ``(incident_id,
    is_new)``. An existing row for the signature refreshes
    ``last_seen_at``/``error``/``output_file`` and keeps its state — a ``closed`` incident stays
    closed, while a ``resolved`` one (job recovered, then broke the same way again) re-opens as
    ``detected`` so the operator is alerted once more. A changed error text (after the
    volatile-token scrub the signature applies) mints a new incident."""
    job_id = str(job_id or "")
    sig = _error_signature(job_id, error)
    stored_error = _redact_error(error)
    incident_id = _incident_id(job_id, sig)
    now = _hermes_now().isoformat()
    failure_type = failure_type or _classify_failure_type(error)
    output_file = str(output_file) if output_file is not None else None

    with _transaction() as conn:
        row = conn.execute(
            "SELECT id, state FROM cron_incidents WHERE id=?", (incident_id,)
        ).fetchone()
        if row is not None:
            reopen = row["state"] == "resolved"
            conn.execute(
                """UPDATE cron_incidents
                   SET last_seen_at=?, error=?, output_file=?,
                       state=CASE WHEN state='resolved' THEN 'detected' ELSE state END,
                       alerted_at=CASE WHEN state='resolved' THEN NULL ELSE alerted_at END,
                       closed_at=CASE WHEN state='resolved' THEN NULL ELSE closed_at END
                   WHERE id=?""",
                (now, stored_error, output_file, incident_id),
            )
            return incident_id, reopen
        conn.execute(
            """INSERT INTO cron_incidents
               (id, job_id, error_sig, state, failure_type,
                first_seen_at, last_seen_at, error, output_file)
               VALUES (?, ?, ?, 'detected', ?, ?, ?, ?, ?)""",
            (incident_id, job_id, sig, failure_type, now, now,
             stored_error, output_file),
        )
        return incident_id, True


def set_incident_state(incident_id: str, state: str) -> bool:
    """Transition an incident's lifecycle state; return whether it changed. ``closed`` is terminal
    for that signature (re-open happens by a changed error minting a NEW incident). Unknown states
    are rejected (no-op, ``False``). ``alerted`` also stamps ``alerted_at`` — every time, so the
    cooldown reminder (see ``cron.scheduler._upsert_incident_for_failure``) restarts its window."""
    if state not in INCIDENT_STATES:
        return False
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        row = conn.execute(
            "SELECT state FROM cron_incidents WHERE id=?", (incident_id,)
        ).fetchone()
        if row is None or row["state"] == "closed":
            return False
        if state == "alerted":
            conn.execute(
                "UPDATE cron_incidents SET state='alerted', alerted_at=? WHERE id=?",
                (now, incident_id),
            )
            return True
        if row["state"] == state:
            return False
        if state == "closed":
            conn.execute(
                """UPDATE cron_incidents
                   SET state='closed', closed_at=?, acked_at=?
                   WHERE id=? AND state != 'closed'""",
                (now, now, incident_id),
            )
        else:
            conn.execute(
                "UPDATE cron_incidents SET state=? WHERE id=?",
                (state, incident_id),
            )
        return True


def ack_incident(incident_id: str) -> bool:
    """Acknowledge (close) an incident; ``False`` when missing or already closed."""
    return set_incident_state(incident_id, "closed")


def close_incidents_for_recovered_job(job_id: str) -> int:
    """Mark every open incident for ``job_id`` ``resolved`` after a successful run; returns how many.
    Without this the ledger only ever grows: a one-off failure (a config drift skip, a provider
    outage) stayed ``detected``/``alerted`` forever after the job recovered, so ``hermes cron
    incidents`` showed dozens of "open" incidents for jobs that had been green for weeks (32 of 32 on
    one install). ``resolved`` is distinct from the operator's ``closed`` on purpose: a repeat of the
    same error re-opens a resolved incident and alerts again (see ``upsert_incident``), whereas
    ``closed`` keeps that signature silent."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cursor = conn.execute(
            """UPDATE cron_incidents SET state='resolved', closed_at=?
               WHERE job_id=? AND state IN ('detected', 'alerted')""",
            (now, str(job_id or "")),
        )
        return int(cursor.rowcount or 0)


def delete_incidents_for_job(job_id: str) -> int:
    """Delete every incident row for ``job_id``; returns how many rows were removed.

    Called from ``cron.jobs.remove_job`` so a deleted job leaves no incident behind. **Delete, not
    resolve, is the deliberate choice:** ``resolved`` is only meaningful for a job that still
    exists — a resolved row re-opens if the same signature recurs (see ``upsert_incident``), it
    keeps occupying ``hermes cron incidents``, and ``close_incidents_for_recovered_job`` can never
    fire for a job whose next run does not exist. A job that is gone can never recover, so its
    incident rows are dead weight; removing them keeps the ledger from accumulating permanently
    open incidents for jobs that no longer exist (the t_4e6cff2a orphan class)."""
    with _transaction() as conn:
        cursor = conn.execute(
            "DELETE FROM cron_incidents WHERE job_id=?", (str(job_id or ""),)
        )
        return int(cursor.rowcount or 0)


def delete_incidents_for_job_ids(job_ids: Iterable[str]) -> int:
    """Delete every incident row for any of ``job_ids`` in one transaction; returns rows removed.

    Batch form of :func:`delete_incidents_for_job` for the removal paths that drop SEVERAL records
    in one store write — the due scan's retention sweep and the retired/wedged one-shot removals
    (``_clear_incidents_for_removed_jobs`` in ``cron.jobs``). One transaction instead of one per id
    keeps a sweep that prunes many aged records from opening N connections. Empty/blank ids are
    ignored; an empty set is a no-op returning 0.
    """
    ids = sorted({str(j) for j in job_ids if j})
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    with _transaction() as conn:
        cursor = conn.execute(
            f"DELETE FROM cron_incidents WHERE job_id IN ({placeholders})", tuple(ids)
        )
        return int(cursor.rowcount or 0)


def prune_orphan_incidents() -> List[Dict[str, Any]]:
    """Delete incidents whose ``job_id`` is absent from the live cron store; returns the rows
    removed (idempotent — a second call removes nothing).

    Guard: the set of valid ids is read from ``cron.jobs.load_jobs()`` (``cron/jobs.json``), so an
    incident for a job that still exists is never touched. Sweeps the residual orphans left behind
    by a ``remove_job`` from before the ledger-clearing fix landed."""
    from cron.jobs import load_jobs  # late import: cron.jobs imports this module's caller lazily

    live_ids = {str(job.get("id")) for job in load_jobs() if isinstance(job, dict)}
    with _transaction() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM cron_incidents").fetchall()]
        orphans = [row for row in rows if str(row.get("job_id")) not in live_ids]
        for orphan in orphans:
            conn.execute("DELETE FROM cron_incidents WHERE id=?", (orphan["id"],))
    return orphans


def _state_filter(state: Optional[str]) -> tuple[str, tuple]:
    return ("", ()) if state is None else (" WHERE state=?", (state,))


def list_incidents(state: Optional[str] = None) -> list[dict[str, Any]]:
    """Return incidents, newest-activity first, optionally filtered by state."""
    if state is not None and state not in INCIDENT_STATES:
        return []
    where, params = _state_filter(state)
    with _transaction() as conn:
        # last_seen_at carries a DST-varying offset: order by instant, not text.
        rows = conn.execute(
            "SELECT * FROM cron_incidents" + where
            + " ORDER BY julianday(last_seen_at) DESC, last_seen_at DESC, id DESC",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_incident(incident_id: str) -> Optional[dict[str, Any]]:
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM cron_incidents WHERE id=?", (incident_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def count_incidents(state: Optional[str] = None) -> int:
    if state is not None and state not in INCIDENT_STATES:
        return 0
    where, params = _state_filter(state)
    with _transaction() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM cron_incidents" + where, params).fetchone()
    return int(row["n"]) if row is not None else 0
