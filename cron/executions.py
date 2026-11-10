"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue. Interrupted attempts
become ``unknown`` only after their owner is proved gone — a start-time reading that fails to
match the claim-time fingerprint is not proof of death. Terminal states are immutable.

One bounded amendment to that rule (``cron/retry_interrupted.py``): an attempt killed through no
fault of the job (owner proved gone, or an interruption written by the failure paths) records a
retry *intent* (``retry_pending``/``retry_at``) in the SAME fenced UPDATE that terminalizes it,
and a later planning pass converts the intent into a schedule amendment. The retry itself is a
new row (``retry_of``/``attempt``); the terminal row is never rewritten.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from cron.constants import CLAIM_TTL_INACTIVITY_HEADROOM
from hermes_cli.observability.shared_metrics_gateway import record_cron_finish

logger = logging.getLogger(__name__)

# Optional test override. Production resolves the path at transaction time so dashboard operations
# that temporarily enter another profile cannot leak that profile's records into the import-time
# home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
HANDOFF_ADOPTION_GRACE_SECONDS = 30.0
# Floor for the live-owner stale-claim bound (#115692); see _live_owner_stale_after_seconds.
LIVE_OWNER_STALE_CLAIM_FLOOR_SECONDS = 7200.0
# Dead-owner sweep scoping (S1). A claim that postdates this process's own start is out of scope
# for an inferential verdict; the self-sufficient proof (owner pid absent) still applies, but only
# once the claim has had one full reap interval to terminalize itself. Both numbers exist to stop a
# *fresh* run being declared unaccountable by the process that launched it (2026-11-09: claimed
# 09:13:40, started 09:13:46, swept 09:14:18.939).
SWEEP_START_SLACK_SECONDS = 5.0
SWEEP_POSTDATING_MIN_AGE_SECONDS = 300.0

# ``cancelled``: an operator stopped the run (``hermes cron cancel``, removing the job mid-run).
# ``missed``: never started — one row books every occurrence the scheduler skipped past.
_TERMINAL_STATES = ("completed", "failed", "unknown", "cancelled", "missed")
_STATUS_CHECK = "('claimed','running','completed','failed','unknown','cancelled','missed')"
# Outcomes the attempt's own firer booked: receipt recovery, the external-worker waiter and the
# dead-owner sweep must treat these as final, never replay or re-book them.
FIRER_FINISHED_STATES = frozenset({"completed", "failed", "cancelled"})
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


# --- executions ledger --------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from cron.jobs import _ensure_cron_dir
    from hermes_cli.sqlite_util import open_db

    path = EXECUTIONS_FILE or (get_hermes_home().resolve() / "cron" / "executions.db")
    _ensure_cron_dir(path.parent)
    return open_db(path, db_label="cron/executions.db", synchronous_full=True, initialize=_initialize_schema)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               {_STATUS_CHECK}),
             handoff_pending INTEGER NOT NULL DEFAULT 0,
             handoff_started_at REAL,
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT
           )"""
    )
    _widen_status_check(conn)
    add_column_if_missing(
        conn, "executions", "handoff_pending",
        "handoff_pending INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn, "executions", "handoff_started_at", "handoff_started_at REAL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    add_column_if_missing(conn, "executions", "delivery_outcome", "delivery_outcome TEXT")
    add_column_if_missing(conn, "executions", "scheduled_instant", "scheduled_instant TEXT")
    add_column_if_missing(conn, "executions", "progress_at", "progress_at TEXT")
    # --- interrupted-run retry evidence (cron/retry_interrupted.py) --------------------------
    # ``attempt``: 1 = the original fire, N+1 = the Nth retry of that occurrence.
    # ``retry_of``: the parent attempt's id (NULL for an original fire).
    # ``outcome_class``: machine-readable verdict written once, at the terminal write, by the
    #   writer that knows it (``owner_gone`` / ``owner_wedged`` / ``interrupted``) — never
    #   re-derived from the prose in ``error``.
    # ``retry_pending``/``retry_at``: the retry intent, stamped in the same fenced UPDATE that
    #   terminalizes the row (there is no post-terminal rewrite rule).
    # ``boot_id``/``machine_id``: what the dead-owner sweep needs to tell "another boot" and
    #   "another machine" from "a dead owner" without trusting a boot-scoped fingerprint.
    add_column_if_missing(conn, "executions", "attempt",
                          "attempt INTEGER NOT NULL DEFAULT 1")
    add_column_if_missing(conn, "executions", "retry_of", "retry_of TEXT")
    add_column_if_missing(conn, "executions", "outcome_class", "outcome_class TEXT")
    add_column_if_missing(conn, "executions", "retry_pending",
                          "retry_pending INTEGER NOT NULL DEFAULT 0")
    add_column_if_missing(conn, "executions", "retry_at", "retry_at TEXT")
    add_column_if_missing(conn, "executions", "boot_id", "boot_id TEXT")
    add_column_if_missing(conn, "executions", "machine_id", "machine_id TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_retry_pending "
        "ON executions(retry_at) WHERE retry_pending=1"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_occurrence "
        "ON executions(job_id, scheduled_instant) WHERE status='completed'"
    )


def _widen_status_check(conn: sqlite3.Connection) -> None:
    """Rebuild a ledger created before ``cancelled``/``missed`` existed: SQLite cannot alter a
    CHECK constraint in place. Copies every column the old table has; runs once per file, and
    re-reads the schema under the write lock so two processes opening it together rebuild once."""
    def current_sql() -> Optional[str]:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='executions'").fetchone()
        return row[0] if row is not None else None

    if "'missed'" in (current_sql() or "'missed'"):
        return
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        old_sql = current_sql() or "'missed'"
        if "'missed'" not in old_sql:
            columns = ", ".join(r[1] for r in conn.execute("PRAGMA table_info(executions)"))
            conn.execute("ALTER TABLE executions RENAME TO executions_pre_widen")
            conn.execute(old_sql.replace("'unknown')", "'unknown','cancelled','missed')", 1))
            conn.execute(
                f"INSERT INTO executions ({columns}) SELECT {columns} FROM executions_pre_widen")
            conn.execute("DROP TABLE executions_pre_widen")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def _fetch(conn: sqlite3.Connection, execution_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()
    return dict(row) if row is not None else None


def _emit_execution_state(
    record: Optional[dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Project durable state to monitoring without affecting ledger behavior."""
    try:
        from agent.monitoring.cron_health import emit_execution_state

        emit_execution_state(record, delivery_outcome=delivery_outcome)
    except Exception:
        pass


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        # The claim-time fingerprint could not be read (#115692): that is *not* proof of death,
        # so a live foreign pid must read as live. The old ``pid == os.getpid()`` answer
        # contradicted this module's own header and handed every foreign row to the sweep.
        return True
    current = _process_start_time(pid)
    if current is None:
        return True  # cannot compare -> cannot prove death; a misread must not rewrite state
    # Drifted same-host readings (#117505) are not proof of death; a live misread is still
    # bounded by the stale-claim sweep below.
    from gateway.status import start_time_fingerprints_match
    return start_time_fingerprints_match(started_at, current)


_BOOT_ID_CACHE: Optional[str] = None
_MACHINE_SCOPE_CACHE: Optional[str] = None


def _read_boot_id() -> str:
    """This boot's id (``""`` when unreadable): the only portable "same boot" witness.

    Cached: a boot id cannot change within one process lifetime (a reboot ends the process), and
    the claim path reads it for every attempt.
    """
    global _BOOT_ID_CACHE
    if _BOOT_ID_CACHE is None:
        try:
            _BOOT_ID_CACHE = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii", errors="replace").strip()
        except OSError:
            _BOOT_ID_CACHE = ""
    return _BOOT_ID_CACHE


def _machine_scope_id() -> str:
    """A machine identity that is stable ACROSS processes on the same host.

    ``cron.jobs._machine_id()`` is ``hostname:pid`` (per-process attribution), so it can never
    scope a row to a machine. ``HERMES_MACHINE_ID`` when set, else the hostname. Cached: the
    claim path reads it for every attempt.
    """
    global _MACHINE_SCOPE_CACHE
    if _MACHINE_SCOPE_CACHE is None:
        explicit = os.getenv("HERMES_MACHINE_ID", "").strip()
        if explicit:
            _MACHINE_SCOPE_CACHE = explicit
        else:
            try:
                import socket

                _MACHINE_SCOPE_CACHE = socket.gethostname()
            except Exception:
                _MACHINE_SCOPE_CACHE = ""
    return _MACHINE_SCOPE_CACHE


_MY_START_MONOTONIC: Optional[float] = None


def _process_start_monotonic() -> Optional[float]:
    """The monotonic instant this process started (None when /proc cannot settle it).

    Both sides of the claim-vs-sweeper comparison must come from the monotonic clock:
    ``process_started_at`` is ticks-since-boot and cannot be compared with ``claimed_at``, and
    the wall clock steps (this host ran -92 min after the 2026-11-09 boot until NTP corrected).
    """
    global _MY_START_MONOTONIC
    if _MY_START_MONOTONIC is not None:
        return _MY_START_MONOTONIC
    ticks = _process_start_time(os.getpid())
    if ticks is None:
        return None
    try:
        uptime = float(Path("/proc/uptime").read_text(encoding="ascii").split()[0])
    except (OSError, ValueError, IndexError):
        return None
    try:
        hz = float(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, ValueError, OSError):
        hz = 100.0
    age = max(0.0, uptime - (float(ticks) / hz))
    _MY_START_MONOTONIC = time.monotonic() - age
    return _MY_START_MONOTONIC


def _claim_precedence_refusal(row: dict[str, Any], *, pid_absent: bool) -> Optional[str]:
    """S1 — why this row is out of scope for THIS sweeper, or None when it is in scope.

    A claim that postdates the sweeper's own start measured on the monotonic clock cannot be
    judged by it: its owner is at least as young as the sweeper, so a boot-change or
    fingerprint verdict about it describes a *different* process (2026-11-09: a run claimed at
    09:13:40 was declared unaccountable at 09:14:18.939 by the process that launched it).

    The one exemption is the self-sufficient proof — the owner pid is *absent* — and only after
    the claim has had a full reap interval to terminalize itself. That keeps the #86721
    guarantee (a crashed `hermes cron run` clears without a gateway restart) while no fresh run
    can be swept out from under itself in its first few seconds.

    A wall clock step changes nothing here: elapsed seconds are measured on the monotonic clock,
    and a backwards step makes the claim look newer than it is, i.e. it fails safe (skipped).
    """
    my_start = _process_start_monotonic()
    if my_start is None:
        return "this process's own start is unreadable (/proc unavailable)"
    try:
        age = _claim_age_seconds(row["claimed_at"])
    except (TypeError, ValueError):
        return "unreadable claim stamp"
    claimed_mono = time.monotonic() - age
    if claimed_mono <= my_start + SWEEP_START_SLACK_SECONDS:
        return None  # claimed before this process started: the pre-restart batch
    if not pid_absent:
        return "claimed after this process started (owner not provably gone by pid absence)"
    if age < SWEEP_POSTDATING_MIN_AGE_SECONDS:
        return (
            "claimed after this process started and younger than one reap interval "
            f"({max(0.0, age):.0f}s < {SWEEP_POSTDATING_MIN_AGE_SECONDS:.0f}s)"
        )
    return None


def _same_machine(recorded: Optional[str]) -> bool:
    """S2 — machine scope. A row from another machine's pid namespace is undecidable, never dead.

    An empty ``machine_id`` is a pre-upgrade row: it falls through to today's behaviour.
    """
    stamp = (recorded or "").strip()
    if not stamp:
        return True
    return stamp == _machine_scope_id()


def _boot_changed(recorded: Optional[str]) -> bool:
    """S3(c) — a boot change, usable only when BOTH sides recorded a non-empty id."""
    stamp = (recorded or "").strip()
    current = _read_boot_id()
    return bool(stamp and current and stamp != current)


def _owner_death_proof(
    row: dict[str, Any], *, pid_absent: bool, owner_live: bool,
) -> bool:
    """S3 — positive proof of death, or False when the row must be left alone (S4 still bounds it).

    Exactly three disjuncts: (a) pid absent, self-sufficient; (b) a fingerprint that was
    RECORDED at claim time and now mismatches beyond the drift tolerance — ``owner_live`` is
    the already-evaluated comparison, and it is False only when that comparison actually ran and
    failed (an unreadable current reading reads as live there); a recorded
    ``process_started_at IS NULL`` is not this case; (c) a boot change, which also catches the
    coincidence S3(b) cannot (a fingerprint that happens to land within tolerance across a boot
    would otherwise read as "alive" and wedge the row).
    """
    if pid_absent:
        return True  # (a)
    if _boot_changed(row["boot_id"]):
        return True  # (c)
    if row["process_started_at"] is None:
        return False  # fingerprint unreadable at claim time: not proof (see the header)
    return not owner_live  # (b)



def _live_owner_stale_after_seconds() -> Optional[float]:
    """Age past which a claimed/running row with a LIVE owner is treated as wedged.

    Derived from the existing knobs, never a bare wall-clock constant:
    ``max(3 × HERMES_CRON_TIMEOUT, cron script timeout, 7200)``. Returns ``None`` (never reclaim
    live owners — today's behaviour) when the inactivity timeout is 0/unlimited or not a finite
    positive number: with no bound to derive from, fail closed.
    """
    from cron.scheduler import _cron_inactivity_seconds
    from cron.scheduler_script import _get_script_timeout

    inactivity = float(_cron_inactivity_seconds())
    if not math.isfinite(inactivity) or inactivity <= 0:
        return None
    return max(
        inactivity * CLAIM_TTL_INACTIVITY_HEADROOM,
        float(_get_script_timeout()),
        LIVE_OWNER_STALE_CLAIM_FLOOR_SECONDS,
    )


def _claim_age_seconds(claimed_at: str) -> float:
    """Seconds since ``claimed_at`` (NOT NULL, always the aware ISO string from hermes_time.now)."""
    return (_hermes_now() - datetime.fromisoformat(claimed_at)).total_seconds()


def _stale_age_seconds(claimed_at: str, progress_at: Optional[str]) -> float:
    """Seconds since the owner last proved it was making progress.

    A wedged worker (#115692) stops stamping ``progress_at``; a long but healthy run keeps
    stamping it, so the derived stale bound measures silence, not run length. Rows written
    before the column existed have no stamp and fall back to claim age.
    """
    return _claim_age_seconds(progress_at or claimed_at)


def touch_execution_progress(execution_id: str) -> bool:
    """Stamp ``progress_at`` on a running attempt this process owns. Returns False when the row
    is no longer ours or no longer running (the caller treats that as informational only)."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET progress_at=?
               WHERE id=? AND status IN ('claimed','running') AND process_id=? AND pid=?""",
            (now, execution_id, _PROCESS_ID, os.getpid()),
        )
        return cur.rowcount == 1


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    # ``retry_pending = 0``: an owed retry intent must never be pruned away (the retry planner
    # would otherwise lose the evidence that a killed run still owes a re-run).
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown','cancelled','missed')
               AND retry_pending = 0
             ORDER BY julianday(finished_at) DESC, finished_at DESC,
                      julianday(claimed_at) DESC, claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (max(0, int(MAX_TERMINAL_EXECUTIONS)),),
    )


def create_execution(
    job_id: str, *, source: str, scheduled_instant: Optional[str] = None,
    attempt: int = 1, retry_of: Optional[str] = None,
) -> dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch.

    ``attempt``/``retry_of``: the retry path's identity for a re-run of an occurrence that was
    killed mid-flight (see ``cron/retry_interrupted.py``); an ordinary fire is attempt 1 with no
    parent. The claim also records which machine and boot claimed it, so a later sweeper can
    tell "another machine/boot" from "a dead owner" without trusting a boot-scoped fingerprint.
    """
    from cron.occurrences import scheduled_instant as canonical_instant

    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at, scheduled_instant, attempt, retry_of, boot_id, machine_id)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?, ?, ?, ?, ?, ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
             _process_start_time(pid), now, canonical_instant(scheduled_instant),
             max(1, int(attempt)), retry_of, _read_boot_id(), _machine_scope_id()),
        )
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record  # type: ignore[return-value]



def set_execution_occurrence(execution_id: str, instant: Optional[str]) -> None:
    """Bind the store-claimed snapshot before a provider hands it to a worker."""
    from cron.occurrences import scheduled_instant

    with _transaction() as conn:
        cur = conn.execute(
            "UPDATE executions SET scheduled_instant=? WHERE id=? AND status='claimed' "
            "AND handoff_pending=0 AND process_id=? AND pid=?",
            (scheduled_instant(instant), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Cron occurrence could not be bound before dispatch")


def set_execution_retry_identity(
    execution_id: str, attempt: int, retry_of: Optional[str],
) -> None:
    """Stamp ``attempt``/``retry_of`` on a claim created BEFORE its fire claim was taken.

    An external provider's ``claim_fire`` persists the attempt and only then learns which
    instant it won, so the retry identity of that instant has to be stamped afterwards — fenced
    to this process's own still-claimed row, exactly like ``set_execution_occurrence``.
    """
    if int(attempt) <= 1 and not retry_of:
        return
    with _transaction() as conn:
        conn.execute(
            """UPDATE executions SET attempt=?, retry_of=?
               WHERE id=? AND status='claimed' AND handoff_pending=0
                 AND process_id=? AND pid=?""",
            (max(1, int(attempt)), retry_of, str(execution_id), _PROCESS_ID, os.getpid()),
        )


def mark_execution_handoff_pending(execution_id: str) -> Optional[dict[str, Any]]:
    """Fence restart recovery while an external worker is adopting a claim."""
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET handoff_pending=1, handoff_started_at=?
               WHERE id=? AND status='claimed'
                 AND process_id=? AND pid=?""",
            (time.time(), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def adopt_claimed_execution(execution_id: str) -> Optional[dict[str, Any]]:
    """Atomically transfer and start an attempt in its worker process.

    The dispatching gateway creates the row before spawning a restart-safe
    worker.  Adoption is the single ``claimed`` → ``running`` gate: only the
    winner may acknowledge ownership or run side effects.
    """
    pid = os.getpid()
    process_started_at = _process_start_time(pid)
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET process_id=?, pid=?, process_started_at=?,
                   status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL
               WHERE id=? AND status='claimed' AND handoff_pending=1""",
            (_PROCESS_ID, pid, process_started_at, now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def mark_execution_running(execution_id: str) -> Optional[dict[str, Any]]:
    """Transition one claimed attempt to running exactly once."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL
               WHERE id=? AND status='claimed' AND handoff_pending=0
                 AND process_id=? AND pid=?""",
            (now, execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None, cancelled: bool = False,
    outcome_class: Optional[str] = None,
    retry_intent: Optional[tuple[str, int]] = None,
) -> Optional[dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten.

    Only the owning process may finish its attempt; receipt recovery for a departed owner
    settles through ``recover_receipted_execution``.

    ``outcome_class``: the machine-readable verdict of the writer that knows it, written only
    when a failure is terminalized (``interrupted`` from the three interruption paths). It is
    never re-derived from the prose in ``error``.

    ``retry_intent``: ``(retry_at, attempt)`` decided by the caller through
    ``cron.retry_interrupted.retry_intent_for`` — written in the SAME UPDATE that terminalizes
    the row, so no terminal row is ever updated afterwards. The ledger writer stays job-free:
    it is handed a decided intent, never a policy.
    """
    now = _hermes_now().isoformat()
    status = "cancelled" if cancelled else ("completed" if success else "failed")
    detail = None if success else (str(error) if error else "unknown failure")
    retry_at: Optional[str] = None
    if retry_intent is not None and not success and not cancelled:
        retry_at = str(retry_intent[0])
    with _transaction() as conn:
        owner = (_PROCESS_ID, os.getpid())
        cur = conn.execute(
            """UPDATE executions
               SET status=?, finished_at=?, error=?, handoff_pending=0,
                   handoff_started_at=NULL, delivery_outcome=?,
                   outcome_class=COALESCE(?, outcome_class),
                   retry_pending=?, retry_at=?
               WHERE id=? AND status IN ('claimed','running')
                 AND process_id=? AND pid=?""",
            (status, now, detail, delivery_outcome,
             None if success else outcome_class,
             1 if retry_at else 0, retry_at,
             execution_id, *owner),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _fetch(conn, execution_id)
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    record_cron_finish(record, delivery_outcome)
    return record



def record_missed_occurrences(
    job_id: str, *, count: int, first_instant: Optional[str], detail: str,
) -> Optional[dict[str, Any]]:
    """Book occurrences the scheduler passed over without running as ONE terminal ``missed`` row
    (``missed ×N``), so run history shows the gap instead of nothing. Never started, so no
    owner fencing applies; ``first_instant`` is the earliest skipped slot."""
    from cron.occurrences import scheduled_instant as canonical_instant

    if count < 1:
        return None
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at, finished_at, error, scheduled_instant)
               VALUES (?, ?, 'scheduler', ?, ?, ?, 'missed', ?, ?, ?, ?)""",
            (execution_id, str(job_id), _PROCESS_ID, pid, _process_start_time(pid), now, now,
             f"missed \u00d7{count} ({detail})", canonical_instant(first_instant)),
        )
        _prune_unlocked(conn)
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def recover_receipted_execution(execution_id, job_id, *, success, error=None, delivery_outcome=None):
    """Settle a verified canonical receipt after owner loss, including its unknown audit row.

    Terminal verdicts stay immutable. A live firer still owns its bookkeeping
    (``execution_owner_live``).
    """
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        row = _recoverable_row(conn, execution_id, job_id)
        if row is None or _row_owner_live(row):
            return None
        # A verified canonical receipt proves the attempt FINISHED after all: any retry intent
        # the sweep stamped on it is cancelled here, in the same write, so a completed run can
        # never be re-fired by the retry planner.
        conn.execute(
            """UPDATE executions SET status=?, finished_at=?, error=?, handoff_pending=0,
               handoff_started_at=NULL, delivery_outcome=?, retry_pending=0, retry_at=NULL
               WHERE id=?""",
            ('completed' if success else 'failed', now, None if success else (str(error) if error else 'unknown failure'),
             delivery_outcome, execution_id))
        _prune_unlocked(conn)
        record = _fetch(conn, execution_id)
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    record_cron_finish(record, delivery_outcome)
    return record


def _recoverable_row(conn, execution_id, job_id):
    return conn.execute(
        """SELECT status, process_id, pid, process_started_at FROM executions
           WHERE id=? AND job_id=? AND status IN ('claimed','running','unknown')""",
        (str(execution_id), str(job_id))).fetchone()


def _row_owner_live(row) -> bool:
    if row['process_id'] == _PROCESS_ID:
        # The in-gateway ticker fires on pool threads: a claimed/running row of THIS process is
        # still in flight here. Only an unknown row was given up by its own firer.
        return row['status'] != 'unknown'
    return _owner_is_live(int(row['pid']), row['process_started_at'])


def execution_owner_live(execution_id: str, job_id: str) -> bool:
    """True while the attempt's own firer (this process included) can still finish it, so receipt
    recovery must not act for it: no output re-save, no job mark, no ledger write."""
    with _transaction() as conn:
        row = _recoverable_row(conn, execution_id, job_id)
    return row is not None and _row_owner_live(row)


def settle_delivery_outcome(execution_id: str, outcome: str) -> bool:
    """Replace a terminal attempt's ``queued`` delivery outcome with the drain's real one.

    The only post-terminal rewrite the ledger allows: the attempt's status is untouched and
    only a ``queued`` handoff can settle, so a replay or a late caller is a no-op.
    """
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET delivery_outcome=?
               WHERE id=? AND delivery_outcome='queued'
                 AND status IN ('completed','failed','unknown')""",
            (outcome, execution_id),
        )
    return cur.rowcount == 1


_OWNER_GONE_REASON = (
    "Scheduler restarted after this execution's owner exited before a durable "
    "terminal state; whether side effects ran is unknown."
)
_OWNER_WEDGED_REASON = (
    "Owner process is still alive but the claim outlived the derived stale bound; "
    "treated as wedged (#115692). The process was not terminated; whether side effects "
    "ran is unknown."
)


def settle_unstarted_execution(execution_id: str, job_id: str, error: str) -> None:
    """Close the receipt of a run that never started: a ``claimed`` row never resolves. Best-effort
    so a ledger write cannot mask the failure the caller logs or re-raises."""
    try:
        finish_execution(execution_id, success=False, error=error)
    except (sqlite3.Error, OSError) as record_err:
        logger.error("Job '%s': failed to close execution receipt %s (%s): %s",
                     job_id, execution_id, error, record_err)


def _pid_absent(pid: int) -> bool:
    """True only when the probe positively says the pid is gone (an unreadable probe is not)."""
    try:
        from gateway.status import _pid_exists

        return not _pid_exists(pid)
    except Exception:
        return False


def _sweep_deferral(row: dict[str, Any], *, pid_absent: bool) -> Optional[str]:
    """Why this row is out of scope for THIS sweep pass, or None when it is in scope.

    Composed of the two ordering gates that must both hold before any verdict:
    S2 (machine scope — another machine's pid namespace proves nothing) and S1 (the claim must
    not postdate this process's own start). S3's proof is evaluated by the caller.
    """
    if not _same_machine(row["machine_id"]):
        return "claimed by another machine's pid namespace (undecidable, never dead)"
    return _claim_precedence_refusal(row, pid_absent=pid_absent)


def recover_interrupted_executions() -> int:
    """Mark abandoned attempts unknown; record a bounded retry for the ones no fault of the job.

    Two classes of row are ever terminalized here:

    * **proved dead** (S2 + S3, then S1 precedence): the owner pid is absent, or a recorded
      fingerprint mismatches beyond the drift tolerance, or this is a different boot on the same
      machine. ``interrupted``-class evidence is never inferred from prose.
    * **wedged** (S4): a LIVE owner whose claim has been silent past the derived stale bound
      (#115692). Deliberately not retryable: that process was never killed and may still be
      writing, so re-firing is a concurrent second run, not a recovery.

    Everything else — a different machine, a claim younger than this process, an unreadable
    fingerprint with no boot change, an unparseable /proc — is left alone. A lost sweep is
    recoverable (a later process, or S4); a wrong sweep is not.

    A retry intent is decided here (``cron/retry_interrupted.retry_intent_for``) and written in
    the SAME fenced UPDATE that terminalizes the row, so the terminal row is never rewritten and
    a further planning pass converts the intent into a schedule amendment.
    """
    now = _hermes_now().isoformat()
    recovered: list[dict[str, Any]] = []
    # Derived on the first live-owned row only: the bound reads config, and the idle gateway
    # tick must stay config-free (tests/cron/test_idle_tick_config_skip.py).
    stale_after: Optional[float] = None
    stale_after_resolved = False
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT id, job_id, status, process_id, pid, process_started_at,
                      boot_id, machine_id, attempt,
                      handoff_pending, handoff_started_at, claimed_at, progress_at
               FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()

    # Decisions are made OUTSIDE the ledger write transaction: the retry gate reads the job
    # store and the config, and holding the ledger lock across those would invert the
    # jobs->ledger lock order that mark_job_run already establishes.
    decisions: list[tuple[dict[str, Any], str, str, Optional[tuple[str, int]]]] = []
    for row in rows:
        if row["process_id"] == _PROCESS_ID:
            continue
        pid = int(row["pid"])
        pid_absent = _pid_absent(pid)
        deferral = _sweep_deferral(row, pid_absent=pid_absent)
        if deferral:
            logger.debug(
                "Cron execution %s (job %s) left to a later sweep: %s",
                row["id"], row["job_id"], deferral)
            continue
        owner_live = _owner_is_live(pid, row["process_started_at"])
        if _owner_death_proof(row, pid_absent=pid_absent, owner_live=owner_live):
            retry_intent: Optional[tuple[str, int]] = None
            try:
                from cron.jobs import get_job
                from cron.retry_interrupted import retry_intent_for

                retry_intent = retry_intent_for(
                    get_job(str(row["job_id"])),
                    execution_id=str(row["id"]),
                    attempt=int(row["attempt"] or 1),
                    outcome_class="owner_gone",
                    finished_at=now,
                )
            except Exception as retry_exc:  # a planner bug must never block the sweep
                logger.debug(
                    "Cron execution %s: retry planning skipped (%s)", row["id"], retry_exc)
            decisions.append((dict(row), _OWNER_GONE_REASON, "owner_gone", retry_intent))
            continue
        if owner_live:
            # A live owner whose death is unproved is normally a legitimately running job. A
            # worker permanently deadlocked (e.g. futex_wait behind a route/proxy flip, #115692)
            # also passes this check, so a claim SILENT for longer than the derived bound is
            # treated as wedged and released — the external-worker wait loop polls this ledger
            # for a terminal status, so the job can fire again. Silence is measured from the
            # owner's last ``progress_at`` stamp (the run monitor refreshes it while the agent is
            # active), so a healthy multi-hour run is never reclaimed while it is still working.
            # The wedged worker PROCESS is NOT terminated here (leaked until host restart); rows
            # owned by this process (process_id == _PROCESS_ID) are skipped above.
            if not stale_after_resolved:
                stale_after = _live_owner_stale_after_seconds()
                stale_after_resolved = True
            if (
                stale_after is None
                or _stale_age_seconds(row["claimed_at"], row["progress_at"]) <= stale_after
            ):
                continue
            decisions.append((dict(row), _OWNER_WEDGED_REASON, "owner_wedged", None))
            continue
        logger.debug(
            "Cron execution %s (job %s): owner is not live but death is unproved "
            "(no pid-absence, boot change or recorded fingerprint mismatch)",
            row["id"], row["job_id"])

    if not decisions:
        return 0
    changed = 0
    with _transaction() as conn:
        for row, reason, outcome_class, retry_intent in decisions:
            handoff_started_at = row["handoff_started_at"]
            if (
                row["handoff_pending"]
                and handoff_started_at is not None
                and time.time() - float(handoff_started_at)
                < HANDOFF_ADOPTION_GRACE_SECONDS
            ):
                continue
            # The SELECT ran outside a write transaction and the liveness probe takes time: the
            # owner may have stamped progress since, so act only on the snapshot that was judged.
            cur = conn.execute(
                """UPDATE executions
                   SET status='unknown', finished_at=?, error=?, outcome_class=?,
                       retry_pending=?, retry_at=?,
                       handoff_pending=0, handoff_started_at=NULL
                   WHERE id=? AND status=? AND process_id=? AND pid=?
                     AND handoff_pending=?
                     AND handoff_started_at IS ?
                     AND claimed_at=? AND progress_at IS ?""",
                (now, reason, outcome_class,
                 1 if retry_intent else 0, retry_intent[0] if retry_intent else None,
                 row["id"], row["status"], row["process_id"], row["pid"],
                 row["handoff_pending"], row["handoff_started_at"],
                 row["claimed_at"], row["progress_at"]),
            )
            changed += cur.rowcount
            if cur.rowcount:
                record = _fetch(conn, row["id"])
                if record is not None:
                    recovered.append(record)
        if changed:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return changed


def pending_retry_intents(limit: int = 8) -> list[dict[str, Any]]:
    """Retry intents owed by terminal rows, oldest first. The retry planner's only query.

    Indexed by ``idx_executions_retry_pending`` and cheap enough for a throttled idle tick: an
    empty result means no config or job-store read at all (#33612).
    """
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT id, job_id, attempt, retry_at
               FROM executions
               WHERE retry_pending=1 AND retry_at IS NOT NULL
               ORDER BY julianday(retry_at) ASC, retry_at ASC, id ASC
               LIMIT ?""",
            (max(1, int(limit)),),
        ).fetchall()
    return [dict(row) for row in rows]


def claim_retry_intent(execution_id: str) -> bool:
    """Claim an owed retry intent (R4): the fenced ``retry_pending 1 -> 0``. Exactly one planner
    wins, so two schedulers sharing a store cannot both amend the schedule."""
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET retry_pending=0
               WHERE id=? AND retry_pending=1 AND status IN ('unknown','failed')""",
            (str(execution_id),),
        )
    return cur.rowcount == 1



def terminalize_dead_owner(execution_id: str, *, reason: str) -> bool:
    """Record one attempt as ``unknown`` with a cause this process actually observed.

    ``recover_interrupted_executions`` sweeps every attempt whose owner is provably
    dead, and knows nothing but that absence — so all it can write is
    ``_OWNER_GONE_REASON``, which asserts a scheduler restart. A waiter that held the
    worker's ``Popen`` knows more: the owner was that external worker, and it exited
    with a known status. Without this, a manual run whose worker dies is filed as
    "Scheduler restarted ..." (a restart that never happened) and, because the sweep
    leaves the row terminal, the waiter reports success and never records the run —
    the job's ``fire_claim`` then blocks the next manual fire for the whole lease
    (#128509).

    The attempt stays ``unknown``, not ``failed``: whether side effects ran is still
    unknown. Only the CAUSE becomes truthful. Returns False — leaving the caller to
    fall back to the generic sweep — when the row is absent, already terminal, owned by
    this process, inside the handoff adoption grace, or owned by a live process: a
    worker that is still running must never be terminalized out from under itself.
    """
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        row = conn.execute(
            """SELECT id, status, process_id, pid, process_started_at,
                      handoff_pending, handoff_started_at
               FROM executions WHERE id=?""",
            (execution_id,),
        ).fetchone()
        if row is None or row["status"] not in ("claimed", "running"):
            return False
        if row["process_id"] == _PROCESS_ID:
            return False
        if _owner_is_live(int(row["pid"]), row["process_started_at"]):
            return False
        handoff_started_at = row["handoff_started_at"]
        if (
            row["handoff_pending"]
            and handoff_started_at is not None
            and time.time() - float(handoff_started_at) < HANDOFF_ADOPTION_GRACE_SECONDS
        ):
            return False
        cur = conn.execute(
            """UPDATE executions
               SET status='unknown', finished_at=?, error=?,
                   handoff_pending=0, handoff_started_at=NULL
               WHERE id=? AND status=? AND process_id=? AND pid=?""",
            (now, reason, row["id"], row["status"], row["process_id"], row["pid"]),
        )
        if cur.rowcount != 1:
            return False
        record = _fetch(conn, execution_id)
        _prune_unlocked(conn)
    _emit_execution_state(record)
    return True


def give_up_execution(execution_id: str, *, reason: str) -> bool:
    """The firer itself gave up on its in-flight attempt (``CronExecutionUnknown``): record it
    ``unknown`` now, owner-fenced to this process, instead of leaving it ``running`` until a
    restart's dead-owner sweep. Receipt recovery never acts for a same-process running row, so
    the owner's later terminal receipt can settle it only once it is marked given up."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET status='unknown', finished_at=?, error=?,
                   handoff_pending=0, handoff_started_at=NULL
               WHERE id=? AND status IN ('claimed','running') AND process_id=? AND pid=?""",
            (now, reason, str(execution_id), _PROCESS_ID, os.getpid()))
        if cur.rowcount != 1:
            return False
        record = _fetch(conn, str(execution_id))
        _prune_unlocked(conn)
    _emit_execution_state(record)
    return True


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50, before_claimed_at: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: list[str] = []
    params: list[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        # Same (instant, text) key as the ORDER BY, so a page never skips or repeats a row.
        clauses.append("(julianday(claimed_at), claimed_at) < (julianday(?), ?)")
        params.extend([str(before_claimed_at)] * 2)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    # Stamps carry the local offset, which changes at DST and on a timezone change, so text order
    # is not time order. julianday() compares instants (ms); the text breaks same-ms ties.
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY julianday(claimed_at) DESC, claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_execution(execution_id: str) -> Optional[dict[str, Any]]:
    """Return one exact execution attempt, or ``None`` when it is absent."""
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?",
            (str(execution_id),),
        ).fetchone()
    return dict(row) if row is not None else None


def latest_execution(job_id: str) -> Optional[dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def live_inflight_execution(job_id: str) -> Optional[dict[str, Any]]:
    """The job's latest attempt while it is still claimed/running under a LIVE owner, else ``None``.

    This is scheduler OWNERSHIP, not recent activity: a run inside a long tool call writes no
    heartbeat yet stays owned, while a run whose process died (watchdog kill, crash) does not.
    Read-only — unlike ``recover_interrupted_executions`` it never rewrites a row.
    """
    # A ``missed`` row is the due scan's note about skipped slots, not an attempt: one booked while
    # a long run is still live (machine slept through its next slots) must not hide that run.
    with _transaction() as conn:
        row = conn.execute(
            """SELECT * FROM executions WHERE job_id=? AND status != 'missed'
               ORDER BY julianday(claimed_at) DESC, claimed_at DESC, id DESC LIMIT 1""",
            (job_id,),
        ).fetchone()
    record = dict(row) if row is not None else None
    if not record or record.get("status") not in ("claimed", "running"):
        return None
    if not _owner_is_live(int(record["pid"]), record.get("process_started_at")):
        return None
    return record


def cancellable_execution(job_id: str) -> Optional[dict[str, Any]]:
    """The job's newest claimed/running attempt under a LIVE owner, even when it is not the latest row.

    A Run-now refused while a run is live books its own newer ``failed`` row ("Fire claim was not
    acquired"), so ``live_inflight_execution`` (latest row only) would call the live run idle and a
    cancel would never reach it.
    """
    return live_executions([job_id]).get(str(job_id))


def live_executions(job_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Each job's newest claimed/running attempt under a live owner, in one query (see
    ``cancellable_execution``). The job list carries it as ``live_execution`` so a Stop button
    still finds a run that a refused Run-now's newer row hides from ``latest_execution``."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT * FROM executions WHERE job_id IN ({placeholders}) AND status IN ('claimed', 'running')
               ORDER BY julianday(claimed_at) DESC, claimed_at DESC, id DESC""",
            clean,
        ).fetchall()
    live: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["job_id"] not in live and _row_owner_live(row):
            live[row["job_id"]] = dict(row)
    return live


def latest_executions(job_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Load latest execution for many jobs in one query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    # One windowed sort: a per-row correlated ORDER BY julianday() cannot use the index and
    # grows quadratically with history (~90 ms at 1000 rows).
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e WHERE e.id IN (
                  SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                             PARTITION BY job_id
                             ORDER BY julianday(claimed_at) DESC, claimed_at DESC, id DESC
                           ) AS rn
                    FROM executions WHERE job_id IN ({placeholders}))
                  WHERE rn=1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
