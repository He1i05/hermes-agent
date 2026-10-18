"""In-code Kanban write guard: fail closed on a test-context write to a LIVE board.

A pytest process whose kanban DB path resolves to a production board must never
open or write it. The leak is real and reproduced (see #69283 and the 2026-09-12
design-card incident): `hermes_cli.kanban_db._board_path` gives the
``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_HOME`` env pins **precedence over
``HERMES_HOME``**, and a dispatched worker shell exports
``HERMES_KANBAN_DB=<real root>/kanban.db``. Redirecting ``HERMES_HOME`` to a
tempdir is therefore NOT isolation, and a stray ``pytest`` (or a test spawning a
child) commits probe rows into the operator's live board.

Since the pytest-only guard landed (wave 1/2), a THIRD leak vector surfaced
(wave 3, QUEST t_2fd6e188): a standalone debug/probe script launched from a
claimed worker's shell (``debug_a5.py`` / ``probe_c4*.py``) inherits the
worker's ``HERMES_KANBAN_*`` pins — ``HERMES_KANBAN_DB`` pinned to the live
board AND ``HERMES_KANBAN_TASK`` naming the worker's task — yet never ran
pytest, so the test-context guard failed OPEN and the probe wrote fixture rows
straight into the live board. The guard is therefore extended to ALSO fail
closed (outside test context) when a process carries worker claim pins but is
NOT a genuinely claimed worker for them.

A fourth defect surfaced in the wave-3 guard itself (t_e35a95a1): the guard
refused the worker's **own post-completion writes**. ``complete_task`` closes
the run and NULLs ``tasks.current_run_id`` / ``tasks.worker_pid`` INSIDE its
main transaction, so the live-claim predicate below turns false the moment the
card reaches ``done``. Everything the SAME worker process does afterwards hits
it: ``complete_task``'s own post-commit tail (``_clear_failure_counter`` /
``recompute_ready`` open their own ``write_txn``) and the worker's later
``kanban_comment`` / ``kanban_attach`` / heartbeat. The escaped exception made
the TOOL report failure even though the completion had committed, and skipped
``_cleanup_workspace`` + the ``kanban_task_completed`` hook — 30 of 31
dispatched-worker completions after the guard commit (93%, measured live). The
process-local :data:`_OWNED_WORKER_PINS` allowance below fixes that without
reopening the vector: a standalone probe is a SEPARATE process and never
observed a genuine claim, so it stays refused.

Why this lives in the library and not only in ``tests/conftest.py``: the
test-side deny-list (#69283) monkeypatches exactly one function and is
structurally blind to (a) writes that happen before the first fixture boundary
and (b) caller-supplied ``sqlite3.Connection`` objects, which never call
``connect()``. Both holes are reproduced in the design card (t_d46b036c).

**The checks are code INSIDE the function bodies of
``hermes_cli/kanban_db_connect.py`` (``_sqlite_connect``, ``write_txn``,
``init_db``, ``_backup_corrupt_db``) and ``kanban_transfer._snapshot_db`` — never
an externally-installed wrapper.** ``hermes_cli.kanban_db`` eagerly re-exports
``write_txn``/``init_db`` from ``kanban_db_connect``, which binds a SECOND module
attribute to the same function object; rebinding one module's name leaves
``kanban_db.create_task``'s own module-global lookup unguarded.

Failure mode: raise ``RuntimeError`` whose message starts with
``kanban_write_guard``. Not ``sys.exit`` — a library must not kill the
dispatcher's process, and the existing test corpus matches on the token.

Fail-OPEN + fail-safe conditions (deliberate, mirroring ``hermes_state_guard``):
an unresolvable path/root; an in-memory connection; the
``HERMES_KANBAN_GUARD_BYPASS`` escape hatch; a process with NO worker claim pin
(the gateway dispatcher / plain CLI — I3); and any DB-read/parse error inside
the claim lookup, which logs and degrades to the current behavior rather than
crash the dispatcher (I5). A guard that can wedge production gets deleted.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional

#: Literal marker in the raised message; the existing corpus matches on it and
#: operators grep for it in logs.
KANBAN_WRITE_GUARD_MARK = "kanban_write_guard"

#: Escape hatch (D8): a guard with no escape hatch eventually gets deleted. The
#: env twin crosses a process boundary; the module global is its in-process
#: sibling, mirroring ``hermes_state._STATE_DB_GUARD_BYPASS`` /
#: ``hermes_state_guard._STATE_DB_GUARD_BYPASS_ENV``.
GUARD_ENV_BYPASS = "HERMES_KANBAN_GUARD_BYPASS"
_KANBAN_GUARD_BYPASS = False

#: Extra deny-roots (D5): a custom-root deployment (``HERMES_HOME=/srv/hermes``)
#: injects its root here, and the hermetic conftest injects the REAL root it
#: captured before the session sandbox rewired the env — exactly the pattern of
#: ``hermes_state._STATE_DB_GUARD_EXTRA_DENY_ROOTS``.
_KANBAN_GUARD_EXTRA_DENY_ROOTS: tuple[Path, ...] = ()

#: The single worker claim pin this guard keys on: its PRESENCE means the
#: process claims to be a dispatched worker for a specific task, so a non-test
#: write to the live board must satisfy the live-claim gate. Absence means the
#: process is the gateway dispatcher / plain CLI (no worker sponsorship) and is
#: allowed (invariant I3). ``HERMES_KANBAN_DB`` / `_BOARD` / `_WORKSPACE` are
#: deliberately NOT keys: those pins are set in dispatcher and CLI shells too,
#: so keying on them would lock out every non-probe process.
WORKER_SPONSOR_PIN = "HERMES_KANBAN_TASK"

#: Process-local ledger of worker claims THIS process has itself validated:
#: ``{task_id: (pid, run_id)}``. Written ONLY inside :func:`is_claimed_worker`'s
#: fully-validated branch, so the single way to get an entry is to have been a
#: genuinely claimed worker for that task *in this process*.
#:
#: Why it must exist (t_e35a95a1): ``complete_task`` closes the run and NULLs
#: ``tasks.current_run_id`` / ``worker_pid`` inside its MAIN txn, so the
#: live-claim predicate is already false for everything the SAME worker does
#: afterwards — ``complete_task``'s own post-commit tail
#: (``_clear_failure_counter`` / ``recompute_ready`` open their own
#: ``write_txn``) and the worker's later ``kanban_comment`` /
#: ``kanban_attach`` / heartbeat. The escaped ``RuntimeError`` made the tool
#: report failure although the completion had committed, and skipped
#: ``_cleanup_workspace`` + the ``kanban_task_completed`` hook: 30 of 31
#: dispatched-worker completions after the wave-3 guard landed (93%, measured
#: on the live board).
#:
#: Why a process-local ledger is the RIGHT discriminator and not a hole: a
#: standalone probe is a SEPARATE process, so its own copy starts empty and it
#: can never populate it (its ``os.getpid()`` never matches
#: ``tasks.worker_pid``). ``os.getpid()`` is re-checked on every read, so a
#: forked descendant that inherits the dict still fails the match. The wave-3
#: vector stays closed without a DB token that a probe could replay.
_OWNED_WORKER_PINS: dict[str, tuple[int, int]] = {}

#: Bound on the ledger: a worker process is one-task-per-run, but an embedded
#: long-lived process (or a test session) may validate several; evict oldest.
_OWNED_WORKER_PINS_MAX = 8

#: Run outcomes that mean THIS process ended its own run by HANDING OFF — the
#: worker-lifecycle terminals where the worker legitimately retains its card
#: thread. Only these make ``_owns_ended_run`` true once
#: ``tasks.current_run_id`` is NULL.
#:
#: ``blocked`` is deliberately EXCLUDED even though a worker's own
#: ``kanban_block`` produces it: an orchestrator's STOP-pivot park produces the
#: SAME outcome (``_end_or_synthesize_run(outcome="blocked")`` in
#: ``block_task``) from a different process, and the two are indistinguishable
#: on the run row. Allowing it would hand a parked worker write access to a card
#: the orchestrator just took away — and to ``complete_task`` on it, since a
#: human-park ``blocked`` status is inside complete_task's accepted set. The
#: graceful-park contract (docs/runbooks/t5-graceful-stop-runbook.md §4) requires
#: that a parked worker stay refused, so the ambiguous outcome stays refused.
#: ``reclaimed`` / ``crashed`` / ``timed_out`` / ``released`` / ``gave_up`` /
#: ``spawn_failed`` are likewise absent: the worker stopped owning the card.
_TRAILING_RUN_OUTCOMES = frozenset({
    "completed", "review_requested", "changes_requested",
})

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def set_extra_deny_roots(*roots: Any) -> None:
    """Install extra production roots (test harness / custom-root deployments).

    Unresolvable entries are ignored rather than raised: this is a guard, and a
    guard must not break the caller that is trying to arm it.
    """
    resolved: list[Path] = []
    for root in roots:
        if not root:
            continue
        try:
            resolved.append(Path(root).expanduser().resolve())
        except Exception:
            continue
    global _KANBAN_GUARD_EXTRA_DENY_ROOTS
    _KANBAN_GUARD_EXTRA_DENY_ROOTS = tuple(resolved)


def _real_platform_kanban_root() -> Optional[Path]:
    """The REAL platform-default Hermes root — never the test sandbox.

    ``os.path.expanduser("~")`` reads ``HOME``/passwd, which ``tests/conftest.py``
    deliberately never rewrites; ``Path.home()`` IS monkeypatched by tests, and
    ``hermes_constants.get_default_hermes_root()`` honours the very
    ``HERMES_HOME`` redirect this guard exists to distrust. Same reasoning as
    ``hermes_state_guard._real_platform_state_root``.
    """
    try:
        home = Path(os.path.expanduser("~"))
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA", "").strip()
            root = Path(base) / "hermes" if base else home / "AppData" / "Local" / "hermes"
        else:
            root = home / ".hermes"
        return root.resolve()
    except Exception:
        return None


def _is_production_kanban_db(resolved: Path, root: Path) -> bool:
    """True iff *resolved* is exactly a **board** path under *root*.

    ``<root>/kanban.db`` (the default board, back-compat location) or
    ``<root>/kanban/boards/<slug>/kanban.db`` (a named board). **Nothing else**:
    a naive "any path under the real root" rule would also refuse the legitimate
    scratch surfaces ``<root>/kanban/workspaces/<task>/*.db`` and
    ``<root>/hermes-agent/.worktrees/<task>/*.db`` (12/12 validated in the design
    card), so the shape predicate is load-bearing, not a shortcut.
    """
    if resolved.name != "kanban.db":
        return False
    if resolved.parent == root:
        return True
    try:
        parts = resolved.relative_to(root).parts
    except ValueError:
        return False
    return (
        len(parts) == 4
        and parts[0] == "kanban"
        and parts[1] == "boards"
        and bool(parts[2])
    )


def _bypassed() -> bool:
    if _KANBAN_GUARD_BYPASS:
        return True
    return os.environ.get(GUARD_ENV_BYPASS, "").strip().lower() in _TRUTHY


def _in_test_context() -> bool:
    """Test run by environment OR ancestry.

    Reused verbatim from ``hermes_state_guard`` rather than re-implemented: the
    signal set (``PYTEST_CURRENT_TEST`` / ``PYTEST_VERSION`` / the inheritable
    ``HERMES_TEST_ISOLATION`` marker) plus the ancestry backstop is exactly what
    the state-DB guard needed for the same class of escape (#82770), and forking
    it would let the two guards drift apart.
    """
    try:
        from hermes_state_guard import _in_test_context as _hs_in_test_context

        return bool(_hs_in_test_context())
    except Exception:
        # Fail open: an import problem in the signal source must never stop the
        # production dispatcher.
        return False


def _deny_roots() -> list[Path]:
    roots: list[Path] = []
    real = _real_platform_kanban_root()
    if real is not None:
        roots.append(real)
    roots.extend(_KANBAN_GUARD_EXTRA_DENY_ROOTS)
    return roots


def _has_worker_sponsor_pin(env: Optional[dict] = None) -> bool:
    """True when a worker claim pin is present (this process claims to be a
    dispatched worker for a specific task).

    Invariant I3: the gateway dispatcher and plain-CLI admin shells carry no
    ``HERMES_KANBAN_TASK``, so their writes to the live board are never trapped.
    A process that DOES carry the pin must satisfy the live-claim gate (I1/I2).
    """
    source = os.environ if env is None else env
    return bool((source.get(WORKER_SPONSOR_PIN) or "").strip())


def _record_owned_pin(task_pin: str, pid: int, run_id: Optional[int]) -> None:
    """Remember a claim THIS process just validated (see :data:`_OWNED_WORKER_PINS`).

    Called only from the fully-validated branch of :func:`is_claimed_worker`, so
    an entry is proof of a real claim rather than a claim of one. ``run_id`` is
    required: without the run row we cannot later tell "my own run ends" from
    "the dispatcher reclaimed me", so no entry is recorded.
    """
    if run_id is None:
        return
    _OWNED_WORKER_PINS[task_pin] = (int(pid), int(run_id))
    while len(_OWNED_WORKER_PINS) > _OWNED_WORKER_PINS_MAX:
        oldest = next(iter(_OWNED_WORKER_PINS))
        _OWNED_WORKER_PINS.pop(oldest, None)


def _owns_ended_run(conn: sqlite3.Connection, task_pin: Optional[str] = None) -> bool:
    """True when THIS process ended the pinned task's own latest run (I1-tail).

    The post-completion allowance (t_e35a95a1). ``complete_task`` NULLs
    ``tasks.current_run_id`` inside its MAIN txn, so ``is_claimed_worker`` is
    already false for ``complete_task``'s own post-commit tail and for every
    later write the same worker makes. This predicate re-admits exactly that
    window and nothing else. Every clause is load-bearing:

    * the ledger entry exists — set ONLY by a fully-validated live claim in
      this process. A standalone probe is a separate process, so its copy is
      empty and it can never acquire one.
    * the recorded pid is ``os.getpid()`` — re-checked at read time, so a forked
      descendant that inherited the dict is still refused.
    * the recorded run is still the task's LATEST run — a respawn beside us
      (the graceful-park shape: the dispatcher reclaimed and re-spawned onto a
      new run) creates a higher id and correctly refuses us.
    * that run is CLOSED with a handoff outcome — ``completed`` /
      ``review_requested`` / ``changes_requested``. ``blocked`` is excluded on
      purpose (an orchestrator park produces the identical run row), and
      ``reclaimed`` / ``crashed`` / ``timed_out`` / ``released`` / ``gave_up`` /
      ``spawn_failed`` mean the worker stopped owning the card, so those stay
      refused.

    Fails CLOSED on any error (unlike the I5 claim-read fail-safe): this is an
    allowance, and an allowance that cannot be evaluated must not be granted.
    """
    if task_pin is None:
        return False
    try:
        recorded = _OWNED_WORKER_PINS.get(task_pin)
        if recorded is None:
            return False
        pid, run_id = recorded
        if int(pid) != os.getpid():
            return False
        row = conn.execute(
            "SELECT r.status AS run_status, r.outcome AS run_outcome, "
            "       r.ended_at AS run_ended, "
            "       (SELECT MAX(id) FROM task_runs WHERE task_id = t.id) AS latest_run "
            "FROM tasks t "
            "  LEFT JOIN task_runs r ON r.id = ? "
            "WHERE t.id = ?",
            (int(run_id), task_pin),
        ).fetchone()
        if row is None:
            return False
        if row["latest_run"] is not None and int(row["latest_run"]) != int(run_id):
            return False  # superseded: a newer run exists for this task.
        if row["run_ended"] is None:
            return False  # still open: the live-claim gate owns this case.
        outcome = str(row["run_outcome"] or row["run_status"] or "")
        return outcome in _TRAILING_RUN_OUTCOMES
    except Exception:
        return False


def _owns_handed_off_card(conn: sqlite3.Connection, task_pin: Optional[str] = None) -> bool:
    """True when THIS process handed the pinned card off and is only reading it.

    The post-handoff lockout (t_6c8342e0, second half). ``request_review``
    re-points ``tasks.assignee`` at the reviewer, and the dispatcher claims the
    review lane onto a NEW run seconds later (measured: 34s on the incident
    card). From that instant BOTH predicates above turn false for the worker
    that made the handoff — ``is_claimed_worker`` because ``current_run_id`` /
    ``worker_pid`` now name the reviewer, and ``_owns_ended_run`` because the
    recorded run is no longer the task's LATEST run. The worker was then refused
    every board verb for the rest of its life, including the read-only
    ``kanban_show`` and the ``kanban_comment`` handoff ack: 7 consecutive
    refusals at ``op=connect`` in wiki-keeper session ``20260915_105118_6d6362``,
    after which that session exported ``HERMES_KANBAN_GUARD_BYPASS=1`` and closed
    a LIVE reviewer run through the sanctioned CLI with the guard disarmed. The
    parent card's hand-rolled direct-SQLite writers were the same pressure on an
    earlier round. **A guard whose sanctioned path is worse than its bypass gets
    bypassed**, which is why this is a guard fix and not a loosening: no probe
    can reach it (the ledger entry below is unforgeable from another process),
    and it grants only the thread the worker already owns.

    Admitted iff every clause holds:

    * the ledger entry exists and is this pid — as in :func:`_owns_ended_run`.
    * the recorded run is an ANCESTOR of the task's latest run (the case
      ``_owns_ended_run`` deliberately refuses) and closed with a handoff
      outcome from :data:`_TRAILING_RUN_OUTCOMES`.
    * **every run after it was claimed from ``review``** — read from each run's
      own ``claimed`` event payload ``source_status``. This is the load-bearing
      clause and it is exactly what separates this window from the graceful-park
      shape: a dispatcher reclaim + ready-lane respawn creates a newer run whose
      ``claimed`` event carries no ``source_status`` (``_claim_and_open_run``
      only stamps it for the review lane), so a SUPERSEDED worker stays refused
      (pinned by ``test_t3_superseded_worker_stays_refused_after_respawn``).

    Lifecycle safety is NOT delegated to this predicate: it re-admits board
    *access*, and ``complete_task`` separately refuses to close a live run the
    caller does not own (``_completion_ownership_guard``). So the handed-off
    worker can read its card and leave notes, and still cannot close the
    reviewer's run.

    Fails CLOSED on any error (an allowance that cannot be evaluated must not be
    granted), like :func:`_owns_ended_run` and unlike the I5 claim fail-safe.
    """
    if task_pin is None:
        return False
    try:
        recorded = _OWNED_WORKER_PINS.get(task_pin)
        if recorded is None:
            return False
        pid, run_id = recorded
        if int(pid) != os.getpid():
            return False
        row = conn.execute(
            "SELECT r.status AS run_status, r.outcome AS run_outcome, "
            "       r.ended_at AS run_ended, "
            "       (SELECT MAX(id) FROM task_runs WHERE task_id = t.id) AS latest_run "
            "FROM tasks t "
            "  LEFT JOIN task_runs r ON r.id = ? "
            "WHERE t.id = ?",
            (int(run_id), task_pin),
        ).fetchone()
        if row is None or row["run_ended"] is None:
            return False
        outcome = str(row["run_outcome"] or row["run_status"] or "")
        if outcome not in _TRAILING_RUN_OUTCOMES:
            return False
        latest = row["latest_run"]
        if latest is None or int(latest) <= int(run_id):
            return False  # not superseded: _owns_ended_run owns this case.
        successors = conn.execute(
            "SELECT r.id AS run_id, "
            "       (SELECT e.payload FROM task_events e "
            "         WHERE e.task_id = r.task_id AND e.run_id = r.id "
            "           AND e.kind = 'claimed' "
            "         ORDER BY e.id DESC LIMIT 1) AS claimed_payload "
            "FROM task_runs r "
            "WHERE r.task_id = ? AND r.id > ? ORDER BY r.id",
            (task_pin, int(run_id)),
        ).fetchall()
        if not successors:
            return False
        for successor in successors:
            payload = _json_payload(successor["claimed_payload"])
            if payload is None or payload.get("source_status") != "review":
                return False  # a ready-lane respawn => superseded, stay refused.
        return True
    except Exception:
        return False


def _json_payload(raw: Any) -> Optional[dict]:
    """``json.loads(raw)`` when it yields a dict, else ``None`` (fail closed)."""
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def is_claimed_worker(env: Optional[dict] = None, conn: Optional[sqlite3.Connection] = None) -> bool:
    """True iff this process is a genuinely claimed dispatched worker.

    ``HERMES_KANBAN_TASK`` resolves to a task whose CURRENT run row (via
    ``tasks.current_run_id``) is ``status='running'``, has not expired, and is
    OWNED by this process — meaning ``tasks.worker_pid == os.getpid()``.

    Ownership is keyed on ``tasks.worker_pid`` (NOT on ``claim_lock ==
    _claimer_id()``): the dispatcher's claim path writes ``claim_lock`` with the
    **claimeR's** ``host:pid`` (the gateway pid, e.g. ``host:1278``) and only
    ``_set_worker_pid`` later records the **spawned worker's** pid. A real worker
    therefore does NOT match ``claim_lock == _claimer_id()`` and would be
    wrongly refused if that predicate was used (observed on live board
    t_84b75a96: claim_lock ``host:1278``, worker_pid 158773 == the worker's
    ``os.getpid()``). A standalone probe launched from a worker shell has a
    DIFFERENT pid, so ``tasks.worker_pid != probe os.getpid()`` even though it
    inherits every pin — this is the discriminator that closes the wave-3 vector.

    A validated call also RECORDS the claim in :data:`_OWNED_WORKER_PINS`, which
    is what lets :func:`_owns_ended_run` re-admit the same worker after its card
    reaches ``done``. This function itself still returns False then — the
    live-claim gate is unchanged, and the narrow post-completion window is a
    separate, explicit allowance at the call sites.

    Fail-safe (I5): any exception during the lookup returns True (allow) so a
    transient DB/parse error can never wedge a production worker — "a guard that
    can break the dispatcher gets deleted."
    """
    try:
        source = os.environ if env is None else env
        task_pin = (source.get(WORKER_SPONSOR_PIN) or "").strip()
        if not task_pin:
            # No worker sponsorship => dispatcher / plain CLI => not a worker
            # write; callers treat this as "no claim gate" (allowed, I3).
            return True
        if conn is None:
            return True  # Cannot read the claim store => fail-safe allow.
        row = conn.execute(
            "SELECT r.id AS run_id, r.status AS run_status, r.claim_expires AS run_exp, "
            "       t.worker_pid AS t_pid "
            "FROM tasks t "
            "  LEFT JOIN task_runs r ON r.id = t.current_run_id "
            "WHERE t.id = ?",
            (task_pin,),
        ).fetchone()
        if row is None:
            return False  # Pinned task does not exist => stale/forged pin => probe.
        if row["run_status"] != "running":
            return False  # Not a live running run => recycle/dead/forged => probe.
        if row["run_exp"] is not None and int(row["run_exp"]) < int(time.time()):
            return False  # Claim expired => stale => probe.
        if row["t_pid"] is not None and int(row["t_pid"]) != os.getpid():
            return False  # Run is owned by a different pid => standalone probe.
        # Validated worker claim: remember it so the post-completion window is
        # admissible. Recorded on BOTH allow paths (proven pid match, and the
        # loose `worker_pid IS NULL` case the dispatcher leaves behind when
        # spawn_fn reports no pid) — otherwise a pid-less local worker keeps the
        # exact lockout this allowance exists to fix. Recording is not a
        # widening: the loose path already allows that process DURING the run,
        # and `_owns_ended_run` still requires this pid, the LATEST run, and a
        # worker-lifecycle outcome.
        _record_owned_pin(task_pin, os.getpid(), _row_get(row, "run_id"))
        return True
    except Exception:
        return True  # I5: fail-safe on any claim-read error.


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """``row[key]`` tolerant of a row without the column (tuple/Raw/Dict)."""
    try:
        return row[key]
    except Exception:
        return default


def _pinned_task_id(env: Optional[dict] = None) -> Optional[str]:
    """The current ``HERMES_KANBAN_TASK`` pin, stripped, or ``None``."""
    source = os.environ if env is None else env
    value = (source.get(WORKER_SPONSOR_PIN) or "").strip()
    return value or None


def _open_claim_conn(resolved: Path) -> Optional[sqlite3.Connection]:
    """Open a short-lived connection to *resolved* for the claim lookup at a
    path-only guard site (connect / init_db / quarantine).

    A WAL-mode board reports its freshest commits via the ``-wal``/``-shm``
    side files, which a ``?mode=ro`` URI cannot access, so the lookup would see
    a stale (empty) snapshot and wrongly refuse a genuine worker. Opening
    normally (writable-capable) lets the lookup read the live claim; the
    connection is closed immediately and never mutated here. A failure returns
    None so the guard fails safe (I5).

    ``row_factory`` is set to :class:`sqlite3.Row` exactly as production's
    ``kanban_db_connect._open_configured`` does. Without it the column-name
    lookup inside :func:`is_claimed_worker` raises, the I5 bare-except
    fail-safe swallows it, and every path-only site returns "allow" — the
    guard reads healthy while the wave-3 vector is wide open (t_e35a95a1
    caveat; the write boundary still refuses, which is what made the hole
    invisible).
    """
    try:
        conn = sqlite3.connect(os.fspath(resolved))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _resolve_db_path(db_path: Any) -> Optional[Path]:
    try:
        return Path(db_path).expanduser().resolve()
    except Exception:
        return None


def _raise_guard(resolved: Path, op: str, *, test_context: bool) -> None:
    if test_context:
        detail = (
            "This process looks like a test run (PYTEST_CURRENT_TEST / "
            "PYTEST_VERSION / HERMES_TEST_ISOLATION) and its kanban DB path "
            "resolves to the REAL board"
        )
    else:
        detail = (
            "This process carries a worker claim pin (HERMES_KANBAN_TASK) but is "
            "NOT a genuinely claimed worker for it — the pinned task's current "
            "run is not a live 'running' run owned by this PID (tasks.worker_pid "
            "!= os.getpid(), or the run is not running, or the claim is "
            "expired). This is the standalone-probe pattern (debug_a5.py / "
            "probe_c4*.py)"
        )
    raise RuntimeError(
        f"{KANBAN_WRITE_GUARD_MARK}: refusing to open a production kanban "
        f"board. path={resolved}, op={op}. {detail}. The env pin "
        f"(HERMES_KANBAN_DB / HERMES_KANBAN_HOME) wins over HERMES_HOME, so "
        f"redirecting HERMES_HOME is NOT sufficient isolation. Fix: run via "
        f"scripts/run_tests.sh, ensure the write is made by the task's actual "
        f"dispatched worker process (matching tasks.worker_pid), or scrub ALL "
        f"HERMES_KANBAN_* pins before importing kanban_db. See #69283 / "
        f"QUEST t_2fd6e188."
    )


def conn_db_path(conn: sqlite3.Connection) -> Optional[Path]:
    """Authoritative on-disk path of an open connection, or ``None``.

    ``PRAGMA database_list`` reports the resolved ``main`` file — the only
    reliable way to see where a **caller-supplied** connection actually points
    (the caller chose the path; the library never saw it). ``None`` for
    in-memory / unavailable connections, which the guard treats as safe.
    """
    try:
        for row in conn.execute("PRAGMA database_list"):
            if row[1] == "main" and row[2]:
                return Path(row[2])
    except Exception:
        return None
    return None


def assert_conn_not_live_board(conn: sqlite3.Connection, *, op: str = "write_txn") -> None:
    """Cheap-then-precise guard for a ``write_txn``-style call site.

    Resolving the connection's path costs a ``PRAGMA``, so the bypass and worker
    pin checks run FIRST and production pays only an env lookup when no claim
    pin is present. Kept separate from :func:`assert_not_live_board` so the
    ordering is explicit rather than an evaluation-order accident of the
    caller's arguments, and so the caller's live ``conn`` can be handed to the
    claim validator without a second open.
    """
    if _bypassed():
        return
    in_test = _in_test_context()
    if not in_test and not _has_worker_sponsor_pin():
        return  # I3: no worker claim pin => dispatcher / plain CLI => allow.
    resolved = _resolve_db_path(conn_db_path(conn))
    if resolved is None:
        return  # fail open: in-memory / unresolvable => not this guard's call.
    if not any(_is_production_kanban_db(resolved, root) for root in _deny_roots()):
        return  # I4: hermetic / scratch path => allow.
    if in_test and not _has_worker_sponsor_pin():
        # Bare test-context write (no claim pin) to the live board => refuse.
        # This is the unchanged wave-1/2 guard: pytest must never touch it.
        _raise_guard(resolved, op, test_context=True)
        return
    # The live-claim gate applies in BOTH test and non-test contexts: a genuine
    # claimed worker (I1) is allowed; an off-claim probe / bare pytest-with-pin
    # (I2) is refused. Ownership is keyed on worker_pid == os.getpid().
    if is_claimed_worker(conn=conn):  # I1: genuinely claimed worker => allow.
        return
    # I1-tail: the SAME worker whose own card this process just moved to a
    # handoff terminal (``done`` / ``review``). ``complete_task`` NULLs
    # current_run_id inside its main txn, so the live-claim gate above is false
    # for the post-commit tail it runs itself and for every later write the
    # worker makes in this run (t_e35a95a1). A standalone probe is a different
    # process and holds no validated ledger entry, so it stays refused.
    if _owns_ended_run(conn, _pinned_task_id()):
        return
    # I1-post-handoff: the same worker whose card is now being reviewed. The
    # dispatcher claims the review lane onto a NEW run seconds after the
    # handoff, which supersedes the ledger's recorded run and turns the clause
    # above false — leaving the worker unable to read its own card or leave the
    # handoff ack. That lockout is what drove the incident's raw-SQL writers and
    # its `HERMES_KANBAN_GUARD_BYPASS` export (t_6c8342e0). Unreachable for a
    # probe: it needs a ledger entry only a validated live claim can create.
    if _owns_handed_off_card(conn, _pinned_task_id()):
        return
    _raise_guard(resolved, op, test_context=in_test)


def assert_not_live_board(
    db_path: Any, *, op: str = "connect", conn: Optional[sqlite3.Connection] = None
) -> None:
    """Fail closed when a test-context or off-claim process targets a production
    kanban board.

    ``db_path`` may be a ``Path``, ``str`` or ``None``; ``None`` (an in-memory
    connection) returns immediately so ``write_txn`` can call this
    unconditionally with :func:`conn_db_path`.

    ``op`` names the boundary in the message (``connect`` / ``write_txn`` /
    ``init_db`` / ``quarantine`` / ``snapshot``) so an operator can tell which
    entry point refused without a traceback.

    ``conn`` is optional: when the caller already holds a live connection to the
    board (``write_txn``), it is handed to the claim validator to avoid a second
    open; otherwise a short-lived connection is opened for the lookup
    (fail-safe if that open fails).
    """
    if db_path is None:
        return
    if _bypassed():
        return
    in_test = _in_test_context()
    if not in_test and not _has_worker_sponsor_pin():
        return  # I3: no worker claim pin => dispatcher / plain CLI => allow.
    resolved = _resolve_db_path(db_path)
    if resolved is None:
        return  # fail open: an unresolvable path is not this guard's call
    if not any(_is_production_kanban_db(resolved, root) for root in _deny_roots()):
        return  # I4: hermetic / scratch path => allow.
    if in_test and not _has_worker_sponsor_pin():
        # Bare test-context write to the live board => refuse (unchanged).
        _raise_guard(resolved, op, test_context=True)
        return
    claim_conn = conn
    if claim_conn is None and _has_worker_sponsor_pin():
        claim_conn = _open_claim_conn(resolved)
    try:
        if claim_conn is not None:
            if is_claimed_worker(conn=claim_conn):
                return  # I1: genuinely claimed worker => allow.
            # I1-tail: the same worker whose own card just reached a handoff
            # terminal (t_e35a95a1). This site is reached FIRST on every kanban
            # tool call (``connect``), so without it a worker could not even
            # open the board to comment/attach after completing. The ledger
            # entry was recorded while the claim was live — in practice by
            # ``complete_task``'s own guard pass, which is unavoidable.
            if _owns_ended_run(claim_conn, _pinned_task_id()):
                return
            # I1-post-handoff: the reviewer's run superseded ours after the
            # handoff, so the clause above is false while this worker still owns
            # the card's thread (t_6c8342e0 — the incident's lockout, which was
            # measured at exactly this ``op=connect`` site and is what drove the
            # bypass). Probe-unreachable: needs a validated-claim ledger entry.
            if _owns_handed_off_card(claim_conn, _pinned_task_id()):
                return
    finally:
        if conn is None and claim_conn is not None:
            try:
                claim_conn.close()
            except Exception:
                pass
    _raise_guard(resolved, op, test_context=in_test)
