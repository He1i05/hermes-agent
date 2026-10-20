"""Wave-3 invariant tests for the kanban write guard's claim discriminator.

These cover the invariant matrix from DESIGN t_42184086 / QUEST t_2fd6e188
(Appendix A discriminator, Appendix B test matrix), plus the negative sub-cases
each maps to:

    T1  probe-shaped process (claim pin but no live claim this PID owns) -> REFUSE
    T2  dispatcher-style write (no worker claim pin)                     -> ALLOW
    T3  claimed worker with a valid live claim                           -> ALLOW
    T4  hermetic temp path                                               -> ALLOW (always)

Negative sub-cases folded into T1/T3: (a) expired claim_expires, (b) run
status != 'running' but pins set, (c) run owned by a DIFFERENT pid, (d) pinned
task id not found.

The wave-3 attack is a STANDALONE process (debug_a5.py / probe_c4*.py) launched
from a claimed worker's shell that inherits the worker's pins but is NOT the
dispatcher and does NOT own the run (its pid != tasks.worker_pid). Those probes
run WITHOUT pytest, so the guard must refuse them in a NON-test context. The
hermetic tests therefore simulate the production process shape by controlling
the guard's test-context signal explicitly (`kg._in_test_context` -> False)
while arming the scratch board as the "live" one via extra-deny-roots — the
same injection conftest performs for the real root. Bare-pytest-to-live-board
refusal (wave 1/2) is preserved and asserted separately in test context.

Hermetic-only: every test uses the `kanban_home` fixture (temp board under a
per-test HERMES_HOME) and never points at ~/.hermes/kanban.db. Run via
`scripts/run_tests.sh` (never plain pytest against the live home).
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_guard as kg


def _hostname() -> str:
    try:
        return socket.gethostname() or "unknown"
    except Exception:
        return "unknown"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    # Pin the write-under-test to this board; strip bypass + worker pins so the
    # process inherits no stale claim (mirrors conftest's scrub list).
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv(kg.GUARD_ENV_BYPASS, raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    return db_path


def _arm_live_board(monkeypatch, db_path, *, in_test):
    """Arm the scratch board as a production board for the guard.

    Injects the board root into extra-deny-roots (so the shape predicate
    recognises it as "live") and, when `in_test` is False, simulates a
    non-test process (the realistic wave-3/dispatcher shape) by stubbing the
    guard's test-context signal.
    """
    monkeypatch.setattr(kg, "_KANBAN_GUARD_EXTRA_DENY_ROOTS", (db_path.parent,))
    monkeypatch.setattr(kg, "_in_test_context", lambda: in_test)


def _count_rows():
    # Plain (non-URI) open that bypasses the guarded `connect` path, so counting
    # never trips the guard regardless of the ambient claim pin / context.
    import sqlite3 as _sql

    db = os.environ.get("HERMES_KANBAN_DB")
    conn = _sql.connect(str(db))
    try:
        return conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    finally:
        conn.close()


def _claim_as_dispatcher(conn, tid, *, worker_pid=None):
    """Claim a task the way the gateway dispatcher does, then stamp worker_pid
    the way `_set_worker_pid` does. Runs with no worker pin (conftest / fixture
    strips it), so the claim write is always allowed."""
    kb.claim_task(conn, tid, claimer=f"{_hostname()}:{os.getpid()}")
    run_id = kb._current_run_id(conn, tid)
    if worker_pid is not None:
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (int(worker_pid), tid))
        conn.execute(
            "UPDATE task_runs SET worker_pid=? WHERE id=?", (int(worker_pid), run_id)
        )
    return run_id


# --- T1: probe-shaped process refused (non-test, like the real attack) ------

def test_t1_probe_task_missing_refused(kanban_home, monkeypatch):
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    before = _count_rows()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef0000")  # non-existent
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="probe", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert _count_rows() == before


def test_t1a_stale_pin_no_live_claim_refused(kanban_home, monkeypatch):
    """Task EXISTS but its current run is not a live running run (never
    claimed, still todo) => refuse (probe reusing a known task id)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unclaimed", assignee="w1")
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    before = _count_rows()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="probe-a", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert _count_rows() == before


def test_t1b_expired_claim_refused(kanban_home, monkeypatch):
    """'running' but claim_expires in the past => refuse (stale recycle)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="expired", assignee="w1")
        run_id = _claim_as_dispatcher(conn, tid, worker_pid=os.getpid())
        conn.execute(
            "UPDATE task_runs SET claim_expires=? WHERE id=?",
            (int(time.time()) - 1000, run_id),
        )
        conn.execute(
            "UPDATE tasks SET claim_expires=? WHERE id=?",
            (int(time.time()) - 1000, tid),
        )
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    before = _count_rows()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="probe-b", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert _count_rows() == before


def test_t1c_done_run_with_pins_refused(kanban_home, monkeypatch):
    """Run status 'done' (not 'running') though the pin is set => refuse."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="finished", assignee="w1")
        run_id = _claim_as_dispatcher(conn, tid, worker_pid=os.getpid())
        conn.execute("UPDATE task_runs SET status=? WHERE id=?", ("done", run_id))
        conn.execute("UPDATE tasks SET status=? WHERE id=?", ("done", tid))
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    before = _count_rows()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="probe-c", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert _count_rows() == before


def test_t1c2_other_pid_owns_run_refused(kanban_home, monkeypatch):
    """'running' but owned by a DIFFERENT pid (standalone probe inheriting the
    worker's pins): the exact wave-3 shape => refuse."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="otherpid", assignee="w1")
        _claim_as_dispatcher(conn, tid, worker_pid=os.getpid() + 1)
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    before = _count_rows()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="probe-c2", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert _count_rows() == before


# --- T2: dispatcher-style write (no worker pin) allowed ----------------------

def test_t2_dispatcher_write_allowed(kanban_home, monkeypatch):
    """No HERMES_KANBAN_TASK in a non-test dispatch tick => write passes even on
    an armed (production-shaped) board (I3)."""
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    before = _count_rows()
    tid = kb.create_task(kbc.connect(), title="dispatcher", assignee="w1")
    assert _count_rows() == before + 1
    assert tid


# --- T3: claimed worker with a valid live claim allowed ----------------------

def test_t3_claimed_worker_allowed(kanban_home, monkeypatch):
    """Claim pin resolves to a running run this PID owns => write succeeds even
    on an armed board (I1)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="claimed worker", assignee="w1")
        _claim_as_dispatcher(conn, tid, worker_pid=os.getpid())
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    before = _count_rows()
    with kbc.connect() as conn:
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET body=? WHERE id=?", ("sanctioned", tid))
    assert _count_rows() == before  # mutation only
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == "sanctioned"


def test_t3_claimed_worker_allowed_in_test_context(kanban_home, monkeypatch):
    """A genuinely claimed worker is allowed even in test context (I1) — the
    only live-board path that should pass under pytest."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="claimed in test", assignee="w1")
        _claim_as_dispatcher(conn, tid, worker_pid=os.getpid())
    _arm_live_board(monkeypatch, kanban_home, in_test=True)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with kbc.connect() as conn:
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET body=? WHERE id=?", ("ok", tid))
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == "ok"


def test_t3_claim_lock_is_dispatcher_not_required(kanban_home, monkeypatch):
    """Regression for the design deviation: ownership is keyed on worker_pid ==
    os.getpid(), NOT on claim_lock == _claimer_id(). A real dispatched worker's
    claim_lock carries the DISPATCHER's pid while worker_pid is the worker's
    pid; matching claim_lock would wrongly refuse the worker."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="claimlock mismatch", assignee="w1")
        # Claim with a claim_lock that will NEVER equal this process's
        # _claimer_id() — the old predicate would have refused.
        kb.claim_task(conn, tid, claimer=f"{_hostname()}:9999")
        run_id = kb._current_run_id(conn, tid)
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (os.getpid(), tid))
        conn.execute(
            "UPDATE task_runs SET worker_pid=? WHERE id=?", (os.getpid(), run_id)
        )
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with kbc.connect() as conn:
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET body=? WHERE id=?", ("ok", tid))
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == "ok"


# --- T4: hermetic / scratch path allowed in all cases ------------------------

def test_t4_hermetic_temp_path_allowed(kanban_home, monkeypatch):
    """No deny-root arm => the temp board is never mistaken for a production
    board, so writes pass even with a probe-shaped pin set (I4)."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_whatever_deadbeef")
    before = _count_rows()
    tid = kb.create_task(kbc.connect(), title="hermetic", assignee="w1")
    assert _count_rows() == before + 1
    assert tid


# --- T5: gateway-adoption split (submitted worker vs bound interpreter) ------
#
# 5aeba055f2 splits the claim across two processes: the dispatcher spawns a
# bare submitter (`hermes_cli.kanban_worker_client`, which holds
# `tasks.worker_pid` for reclaim/kill) while the adopted interpreter
# (`agent.managed_worker`, which runs the tools and evaluates this guard) is a
# DIFFERENT gateway child, bound by `gateway.session_kanban.bind_worker_context`
# via a durable `worker_bound` task event (its pid + process fingerprint). A
# standalone probe launched from a worker shell is a THIRD process: it has no
# `worker_bound` event binding its own pid+fingerprint, so it stays refused.

def _worker_bound(conn, tid, run_id, pid, started_at):
    kb._append_event(
        conn, tid, "worker_bound", {"pid": int(pid), "claim_lock": f"{_hostname()}:0", "started_at": started_at},
        run_id=int(run_id),
    )


def test_t5_adopted_interpreter_allowed(kanban_home, monkeypatch):
    """The bound interpreter pid differs from tasks.worker_pid (the submitted
    worker) yet the board binds THIS process => allow. This is the production
    shape of every `kanban_*` tool call in a dispatched worker."""
    from hermes_cli.kanban_db_dispatch import _process_fingerprint

    bound_pid = os.getpid()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="adopted", assignee="w1")
        run_id = _claim_as_dispatcher(conn, tid, worker_pid=os.getpid() + 1)
        _worker_bound(conn, tid, run_id, bound_pid, _process_fingerprint(bound_pid))
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    before = _count_rows()
    with kbc.connect() as conn:
        assert kg.is_claimed_worker(conn=conn) is True
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET body=? WHERE id=?", ("interpreter ok", tid))
    assert _count_rows() == before  # mutation only
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == "interpreter ok"


def test_t5_adopted_interpreter_probe_without_binding_refused(kanban_home, monkeypatch):
    """The split shape must NOT open the guard to the probe: a process with the
    worker's pins and a mismatching worker_pid but NO worker_bound binding for
    its own pid+fingerprint is still refused (T1c2 under 5aeba055f2)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="split probe", assignee="w1")
        _claim_as_dispatcher(conn, tid, worker_pid=os.getpid() + 1)
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    before = _count_rows()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kg.is_claimed_worker(conn=conn)
            kb.create_task(conn, title="probe-t5b", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert _count_rows() == before


def test_t5_adopted_interpreter_recycled_fingerprint_refused(kanban_home, monkeypatch):
    """A worker_bound event naming THIS pid but bound to a DIFFERENT process
    instance (pid recycled after a crash; fingerprint never matches) is not
    ownership: refuse."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="recycled bind", assignee="w1")
        run_id = _claim_as_dispatcher(conn, tid, worker_pid=os.getpid() + 1)
        _worker_bound(conn, tid, run_id, os.getpid(), "gone-boot-epoch|123")
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    before = _count_rows()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kg.is_claimed_worker(conn=conn)
            kb.create_task(conn, title="probe-t5c", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert _count_rows() == before


# --- Wave-1/2 regression: bare pytest must still refuse the live board -------

def test_regression_bare_pytest_live_board_refused(kanban_home, monkeypatch):
    """The original wave-1/2 guard: a bare pytest process (no claim pin) writing
    to an armed (production-shaped) board MUST still be refused."""
    _arm_live_board(monkeypatch, kanban_home, in_test=True)
    before = _count_rows()
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="bare pytest", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert _count_rows() == before


# --- Bypass escape hatch -----------------------------------------------------

def test_bypass_escape_hatch_allows_even_probe(kanban_home, monkeypatch):
    """HERMES_KANBAN_GUARD_BYPASS is honored (D8): with it set, even an
    off-claim probe-shaped write to an armed board passes."""
    _arm_live_board(monkeypatch, kanban_home, in_test=False)
    monkeypatch.setenv("HERMES_KANBAN_GUARD_BYPASS", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef0000")
    before = _count_rows()
    tid = kb.create_task(kbc.connect(), title="bypassed", assignee="w1")
    assert _count_rows() == before + 1
    assert tid
