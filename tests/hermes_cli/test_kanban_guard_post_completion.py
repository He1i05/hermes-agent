"""Post-completion lockout regression tests (t_e35a95a1).

The wave-3 claim guard refused a genuinely claimed worker's OWN writes as soon
as its card reached ``done``: ``complete_task`` NULLs ``tasks.current_run_id``
and ``tasks.worker_pid`` inside its MAIN txn, so ``is_claimed_worker`` is
already false for the post-commit tail it runs itself
(``_clear_failure_counter`` / ``recompute_ready`` each open their own
``write_txn``) and for every later write the same worker process makes. The
escaped ``RuntimeError`` made the TOOL report failure although the completion
had committed, and skipped ``_cleanup_workspace`` + the
``kanban_task_completed`` hook (93% of dispatched completions, measured live).

Both directions must hold at once, so every test here is paired:

    T5-allow  the claimed worker's post-completion writes SUCCEED
    T5-refuse a standalone probe (separate process / different pid) stays REFUSED
    T5-park   a SUPERSEDED worker (dispatcher reclaimed + respawned onto a new
              run) stays REFUSED — the graceful-park duty must not be weakened
    T5-mutant removing the fix makes the allow-direction test fail

Hermetic-only: every test uses the ``kanban_home`` fixture (temp board under a
per-test HERMES_HOME) and arms THAT board as the "live" one via
extra-deny-roots, so the guard runs its real production predicate without ever
touching ``~/.hermes/kanban.db``. Run via ``scripts/run_tests.sh`` (never plain
pytest against the live home).
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_guard as kg
from hermes_cli.plugins import get_plugin_manager


def _hostname() -> str:
    import socket

    try:
        return socket.gethostname() or "unknown"
    except Exception:
        return "unknown"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Temp board + stripped pins (mirrors test_kanban_guard_wave3)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv(kg.GUARD_ENV_BYPASS, raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    # The ledger is process-global by design; clear it per test so one test's
    # validated claim cannot leak an allowance into the next. ``raising=False``
    # keeps this file runnable against the PRE-FIX guard, which is what makes
    # the mutant/regression direction demonstrable.
    monkeypatch.setattr(kg, "_OWNED_WORKER_PINS", {}, raising=False)
    return db_path


def _arm_live_board(monkeypatch, db_path, *, in_test=False):
    """Arm the scratch board as production and (by default) simulate a real
    non-test worker process — the shape the lockout was measured in."""
    monkeypatch.setattr(kg, "_KANBAN_GUARD_EXTRA_DENY_ROOTS", (db_path.parent,))
    monkeypatch.setattr(kg, "_in_test_context", lambda: in_test)


def _claim_as_worker(conn, tid, *, worker_pid, claimer_pid=9999):
    """Claim the way the gateway dispatcher does, then stamp ``worker_pid`` the
    way ``_set_worker_pid`` does (claim_lock carries the DISPATCHER's pid)."""
    kb.claim_task(conn, tid, claimer=f"{_hostname()}:{claimer_pid}")
    run_id = kb._current_run_id(conn, tid)
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (int(worker_pid), tid))
    conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (int(worker_pid), run_id))
    return run_id


def _raw_query(sql: str, params: tuple = ()) -> tuple:
    """One-row read through a plain connection that bypasses the guarded
    ``connect`` path, so asserting on state never itself trips the guard."""
    conn = sqlite3.connect(str(os.environ["HERMES_KANBAN_DB"]))
    try:
        return conn.execute(sql, params).fetchone()
    finally:
        conn.close()


def _live_conn() -> sqlite3.Connection:
    """A caller-supplied connection in production SHAPE.

    ``row_factory = sqlite3.Row`` is load-bearing: without it the column-name
    reads inside ``is_claimed_worker`` raise, the I5 fail-safe swallows it, and
    every assertion reads "allowed" — the exact mis-read the discovery card
    warns about (t_e35a95a1 caveat). Callers close it.
    """
    conn = sqlite3.connect(str(os.environ["HERMES_KANBAN_DB"]))
    conn.row_factory = sqlite3.Row
    return conn


def _ledger() -> dict:
    """The owned-pin ledger; ``{}`` against a guard revision without one.

    ``getattr`` so the refusal-direction tests still assert their real subject
    (the guard's behaviour) rather than dying on an AttributeError when this
    file is run against the PRE-FIX guard for the mutant proof.
    """
    return getattr(kg, "_OWNED_WORKER_PINS", {})


def _pin_worker_env(monkeypatch, tid: str) -> int:
    """Stand in for a dispatcher-spawned worker: pin BOTH env vars the dispatcher
    exports at spawn (``kanban_db_dispatch`` sets ``HERMES_KANBAN_RUN_ID`` to the
    card's current run). Hand-merge (t_788c8b56): upstream 7c2b5b5981 refuses a
    worker-pinned run-lifecycle tool call whose run id cannot be resolved, so a
    fixture that models a worker but omits the run id no longer models anything
    real. Returns the pinned run id."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    run_id = _raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0]
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return run_id


# ── T5-allow: the claimed worker's own post-completion writes pass ──────────

def test_t5_worker_writes_succeed_after_own_completion(kanban_home, monkeypatch):
    """AC1: a genuinely claimed worker completes its card, then comments in the
    same run with no guard refusal.

    Pre-fix this raised ``kanban_write_guard`` from ``complete_task`` itself
    (the tail's own ``write_txn``) AND from the follow-up comment.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="post-completion comment", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        # A managed scratch workspace, so cleanup has something to remove.
        ws = kbw.resolve_workspace(kb.get_task(conn, tid))
        kbw.set_workspace_path(conn, tid, ws)
        (ws / "deliverable.txt").write_text("scratch\n")

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)

    with kbc.connect() as conn:
        # Hand-merge (t_788c8b56, F2): the pin's complete_task fence requires
        # explicit ownership proof for a live claim — name the run this
        # worker owns (what HERMES_KANBAN_RUN_ID resolves to in production).
        assert kb.complete_task(
            conn, tid, summary="done",
            expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0],
        ) is True

    # The completion committed...
    row = _raw_query(
        "SELECT status, current_run_id, worker_pid FROM tasks WHERE id=?", (tid,)
    )
    assert row[0] == "done"
    assert row[1] is None  # the NULLing that broke the live-claim gate
    assert row[2] is None

    # ...and the SAME process can still write to that card.
    with kbc.connect() as conn:
        cid = kb.add_comment(conn, tid, author="builder", body="post-completion note")
    assert cid

    with kbc.connect() as conn:
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET body=? WHERE id=?", ("tail", tid))
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).body == "tail"


def test_t5_cleanup_and_completed_hook_run_on_normal_completion(kanban_home, monkeypatch):
    """AC3: ``_cleanup_workspace`` and the ``kanban_task_completed`` hook
    observably run — they were skipped because the tail raised."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cleanup + hook", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        ws = kbw.resolve_workspace(kb.get_task(conn, tid))
        kbw.set_workspace_path(conn, tid, ws)
        (ws / "artifact.txt").write_text("scratch\n")
    assert ws.exists()

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)

    fired: list[dict] = []
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("kanban_task_completed", []).append(
        lambda **kw: fired.append(kw)
    )
    try:
        with kbc.connect() as conn:
            # Hand-merge (t_788c8b56, F2): the pin's complete_task fence requires
            # explicit ownership proof for a live claim — name the run this
            # worker owns (what HERMES_KANBAN_RUN_ID resolves to in production).
            assert kb.complete_task(
                conn, tid, summary="done",
                expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0],
            ) is True
    finally:
        mgr._hooks = saved

    assert not ws.exists(), "scratch workspace was not cleaned up (tail was skipped)"
    assert len(fired) == 1, f"kanban_task_completed did not fire once: {fired}"
    assert fired[0]["task_id"] == tid


# ── T5-refuse: the wave-3 vector stays closed ───────────────────────────────

def test_t5_standalone_probe_still_refused_after_a_real_completion(kanban_home, monkeypatch):
    """AC2: a probe inheriting the pins (different pid) is still refused, even
    in a process that has ALSO just legitimately completed a card.

    This is the load-bearing pairing: the allowance must key on the specific
    task+pid, not on "this process completed something".
    """
    with kbc.connect() as conn:
        mine = kb.create_task(conn, title="mine", assignee="builder")
        _claim_as_worker(conn, mine, worker_pid=os.getpid())
        # A second task whose live run is owned by a DIFFERENT pid: the exact
        # wave-3 shape a probe inheriting the worker's pins presents.
        other = kb.create_task(conn, title="someone else's", assignee="builder")
        _claim_as_worker(conn, other, worker_pid=os.getpid() + 1)

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, mine)
    with kbc.connect() as conn:
        # Hand-merge (t_788c8b56, F2): the pin's complete_task fence requires
        # explicit ownership proof for a live claim — name the run this
        # worker owns (what HERMES_KANBAN_RUN_ID resolves to in production).
        assert kb.complete_task(
            conn, mine, summary="done",
            expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (mine,))[0],
        ) is True

    before = _raw_query("SELECT COUNT(*) FROM tasks")[0]

    # Same process, pins pointed at the OTHER task (owned by a different pid).
    _pin_worker_env(monkeypatch, other)
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="probe row", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)

    # And the forged/unknown pin.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef0000")
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="probe row 2", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)

    assert _raw_query("SELECT COUNT(*) FROM tasks")[0] == before


def test_t5_probe_with_no_history_refused(kanban_home, monkeypatch):
    """A probe pointing at an ALREADY-COMPLETED card must be refused: the ledger
    entry is the only way in, and this process never validated a claim.

    This is the load-bearing counterpart to the allow-direction test — the SAME
    on-disk state (task done, run completed, ``current_run_id`` NULL) is
    writable by the worker that EARNED it and refused for a process that merely
    inherited the pins. That is precisely the wave-3 vector.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="completed elsewhere", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())

    # Complete it with the pin UNset (the plain-CLI I3 path), so this process
    # never runs the claim validator and therefore records nothing.
    _arm_live_board(monkeypatch, kanban_home)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kbc.connect() as conn:
        # Hand-merge (t_788c8b56, F2): the pin's complete_task fence requires
        # explicit ownership proof for a live claim — name the run this
        # worker owns (what HERMES_KANBAN_RUN_ID resolves to in production).
        assert kb.complete_task(
            conn, tid, summary="done elsewhere",
            expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0],
        ) is True
    assert _ledger() == {}, "no claim was validated in this process"
    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "done"

    # Now the probe arrives carrying the pins for that done card.
    _pin_worker_env(monkeypatch, tid)
    before = _raw_query("SELECT COUNT(*) FROM tasks WHERE id=?", (tid,))[0]
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.add_comment(conn, tid, author="probe", body="not mine")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert _raw_query("SELECT COUNT(*) FROM task_comments WHERE task_id=?", (tid,))[0] == 0
    assert _raw_query("SELECT COUNT(*) FROM tasks WHERE id=?", (tid,))[0] == before


# ── T5-park: a SUPERSEDED worker stays refused (graceful-park duty) ─────────

def test_t5_superseded_worker_refused_after_respawn(kanban_home, monkeypatch):
    """The dispatcher reclaimed this card and respawned onto a NEW run: the old
    worker must NOT regain write access (t5-graceful-stop-runbook §4 step 3).
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="superseded", assignee="builder")
        first_run = _claim_as_worker(conn, tid, worker_pid=os.getpid())

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)
    with kbc.connect() as conn:
        # Hand-merge (t_788c8b56, F2): the pin's complete_task fence requires
        # explicit ownership proof for a live claim — name the run this
        # worker owns (what HERMES_KANBAN_RUN_ID resolves to in production).
        assert kb.complete_task(
            conn, tid, summary="first run",
            expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0],
        ) is True
    assert tid in _ledger()

    # The dispatcher reclaims and a fresh run is spawned beside us.
    with kbc.connect() as conn:
        with kbc.write_txn(conn):
            new_run = conn.execute(
                "INSERT INTO task_runs (task_id, status, started_at, worker_pid) "
                "VALUES (?, 'running', ?, ?)",
                (tid, int(time.time()), os.getpid() + 7),
            ).lastrowid
            conn.execute(
                "UPDATE tasks SET status='running', current_run_id=?, worker_pid=? WHERE id=?",
                (new_run, os.getpid() + 7, tid),
            )
    # A genuinely superseded worker's write is refused.
    live = _live_conn()
    try:
        assert kg.is_claimed_worker(conn=live) is False
        assert kg._owns_ended_run(live, tid) is False
    finally:
        live.close()
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="stale worker write", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    assert first_run != new_run


def test_t5_reclaimed_outcome_refused(kanban_home, monkeypatch):
    """Run closed as ``reclaimed`` (this worker stopped owning the card): the
    allowance must not apply even with no newer run present."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reclaimed", assignee="builder")
        run_id = _claim_as_worker(conn, tid, worker_pid=os.getpid())

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)
    # Validate while live, so the ledger entry exists...
    live = _live_conn()
    try:
        assert kg.is_claimed_worker(conn=live) is True
    finally:
        live.close()
    assert tid in _ledger()
    # ...then close the run the way a reclaim does.
    with kbc.connect() as conn:
        with kbc.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', ended_at=? WHERE id=?",
                (int(time.time()), run_id),
            )
            conn.execute(
                "UPDATE tasks SET status='ready', current_run_id=NULL, worker_pid=NULL WHERE id=?",
                (tid,),
            )
    live = _live_conn()
    try:
        assert kg._owns_ended_run(live, tid) is False
    finally:
        live.close()
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.create_task(conn, title="reclaimed worker write", assignee="w1")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)


def test_t5_orchestrator_park_refused(kanban_home, monkeypatch):
    """A park's ``blocked`` run outcome must NOT grant the allowance.

    An orchestrator STOP-pivot park calls ``block_task``, which ends the run
    with ``outcome="blocked"`` — byte-identical on the run row to a worker's own
    ``kanban_block``. Admitting ``blocked`` would hand the parked worker write
    access to the card the orchestrator just took away, and to
    ``complete_task`` on it (``blocked`` is inside complete_task's accepted
    status set), directly violating the graceful-park contract
    (docs/runbooks/t5-graceful-stop-runbook.md §4 step 3).
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="parked", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)
    # Validate the live claim first, so a ledger entry DOES exist — the park
    # must still win.
    live = _live_conn()
    try:
        assert kg.is_claimed_worker(conn=live) is True
    finally:
        live.close()
    assert tid in _ledger()

    # The orchestrator parks the card (CLI-capable path: no worker pin).
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kbc.connect() as conn:
        assert kb.block_task(conn, tid, reason="parked: STOP pivot t_root", kind="dependency") is True
    assert _raw_query("SELECT outcome FROM task_runs WHERE task_id=?", (tid,))[0] == "blocked"

    _pin_worker_env(monkeypatch, tid)
    live = _live_conn()
    try:
        assert kg._owns_ended_run(live, tid) is False
    finally:
        live.close()
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.add_comment(conn, tid, author="builder", body="ignoring the park")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)
    # The acknowledgement must not have landed, and the task must not be closed.
    assert _raw_query("SELECT COUNT(*) FROM task_comments WHERE task_id=?", (tid,))[0] == 0
    # Hand-merge (t_788c8b56): upstream 42a778ab4b re-kinds a parent-less
    # dependency park to a sticky ``needs_input`` block, so the card lands in
    # the ``blocked`` lane. The park still WINS over the worker's retained
    # access — which is what this test asserts.
    assert _raw_query("SELECT status, block_kind FROM tasks WHERE id=?",
                      (tid,)) == ("blocked", "needs_input")


def test_t5_forked_descendant_refused(kanban_home, monkeypatch):
    """The ledger is process-local; a descendant that inherits the dict cannot
    use it because the recorded pid is re-checked against ``os.getpid()``."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="fork", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)
    with kbc.connect() as conn:
        # Hand-merge (t_788c8b56, F2): the pin's complete_task fence requires
        # explicit ownership proof for a live claim — name the run this
        # worker owns (what HERMES_KANBAN_RUN_ID resolves to in production).
        assert kb.complete_task(
            conn, tid, summary="done",
            expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0],
        ) is True
    live = _live_conn()
    try:
        assert kg._owns_ended_run(live, tid) is True
    finally:
        live.close()

    real = os.getpid
    monkeypatch.setattr(os, "getpid", lambda: real() + 1)  # emulate the child
    live = _live_conn()
    try:
        assert kg._owns_ended_run(live, tid) is False
    finally:
        live.close()
    with pytest.raises(RuntimeError):
        with kbc.connect() as conn:
            kb.create_task(conn, title="child write", assignee="w1")


# ── I3 must be untouched (AC5) ──────────────────────────────────────────────

def test_t5_dispatcher_write_path_unchanged(kanban_home, monkeypatch):
    """No worker pin => dispatcher / plain CLI writes freely (I3), before and
    after a completion in the same process."""
    _arm_live_board(monkeypatch, kanban_home)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert kb.create_task(kbc.connect(), title="dispatcher", assignee="w1")
    assert _ledger() == {}


def test_t5_pidless_local_worker_writes_after_own_completion(kanban_home, monkeypatch):
    """A locally-driven worker whose ``worker_pid`` was never stamped (the loose
    I1 allow path) must ALSO survive its own completion.

    The dispatcher only records a pid when ``spawn_fn`` reports one, so this
    process shape is real; leaving it out would keep this exact card's bug for
    that population.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="pidless", assignee="builder")
        kb.claim_task(conn, tid, claimer=f"{_hostname()}:{os.getpid()}")

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)
    with kbc.connect() as conn:
        # Hand-merge (t_788c8b56, F2): the pin's complete_task fence requires
        # explicit ownership proof for a live claim — name the run this
        # worker owns (what HERMES_KANBAN_RUN_ID resolves to in production).
        assert kb.complete_task(
            conn, tid, summary="done",
            expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0],
        ) is True  # no refusal
    with kbc.connect() as conn:
        assert kb.add_comment(conn, tid, author="builder", body="tail")  # no refusal
    assert _raw_query("SELECT COUNT(*) FROM task_comments WHERE task_id=?", (tid,))[0] == 1


# ── AC1 at the TOOL layer: the named verbs must not refuse ───────────────────

def test_t5_tool_verbs_work_after_completion(kanban_home, monkeypatch):
    """AC1 verbatim: after ``kanban_complete`` the same run succeeds at
    ``kanban_comment`` / ``kanban_heartbeat`` / ``kanban_attach`` with no guard
    refusal.

    Drives the real handlers rather than the DB API, because the defect's
    reported symptom was the TOOL returning a ``kanban_write_guard`` refusal
    (and ``_kanban_handler`` converts any escape into a ``tool_error`` string,
    so an exception alone would not have failed the old tests).
    """
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="tool verbs", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)
    monkeypatch.setenv("HERMES_PROFILE", "builder")

    out = kt._handle_complete({"summary": "done"})
    assert "kanban_write_guard" not in out, out
    assert json.loads(out).get("ok") is True, out
    # ``_worker_run_id`` pins expected_run_id only when the env carries it.
    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "done"

    out = kt._handle_comment({"task_id": tid, "body": "post-completion note"})
    assert "kanban_write_guard" not in out, out
    assert json.loads(out).get("ok") is True, out

    out = kt._handle_attach({
        "task_id": tid,
        "filename": "note.txt",
        "content_base64": base64.b64encode(b"attachment after completion").decode(),
    })
    assert "kanban_write_guard" not in out, out
    assert json.loads(out).get("ok") is True, out

    # The card is done, so heartbeat's own "not running" refusal is expected —
    # the point is that it must be THAT refusal, not a guard refusal.
    out = kt._handle_heartbeat({"note": "tail liveness"})
    assert "kanban_write_guard" not in out, out

    assert _raw_query("SELECT COUNT(*) FROM task_comments WHERE task_id=?", (tid,))[0] == 1
    assert _raw_query("SELECT COUNT(*) FROM task_attachments WHERE task_id=?", (tid,))[0] == 1


# ── Mutant: prove the tests fail without the fix ────────────────────────────

def test_t5_mutant_without_the_allowance_the_worker_is_locked_out(kanban_home, monkeypatch):
    """Mutant for AC4: disable the post-completion allowance (the pre-fix
    behaviour) and the allow-direction test's outcome must invert.

    Keeps the two directions honest in one place: the refusal is real and the
    allowance is what makes the worker's own writes succeed.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mutant", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)

    # MUTANT: the allowance is inert.
    monkeypatch.setattr(kg, "_owns_ended_run", lambda conn, task_pin=None: False)

    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            # Hand-merge (t_788c8b56, F2): name the run this worker owns (the
            # pin's fence requires explicit ownership proof for a live claim).
            kb.complete_task(
                conn, tid, summary="done",
                expected_run_id=_raw_query(
                    "SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0],
            )
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)

    # The completion still landed (the pre-fix trap: tool error, durable write).
    row = _raw_query("SELECT status, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "done"
    assert row[1] is None
    # ...and the follow-up write is refused too.
    with pytest.raises(RuntimeError):
        with kbc.connect() as conn:
            kb.add_comment(conn, tid, author="builder", body="locked out")


def test_t5_mutant_control_same_setup_with_fix_passes(kanban_home, monkeypatch):
    """Control for the mutant above: identical setup, allowance intact => both
    writes succeed. Pairs with it so a passing mutant cannot be mistaken for a
    passing fix."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="control", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())

    _arm_live_board(monkeypatch, kanban_home)
    _pin_worker_env(monkeypatch, tid)

    with kbc.connect() as conn:
        # Hand-merge (t_788c8b56, F2): the pin's complete_task fence requires
        # explicit ownership proof for a live claim — name the run this
        # worker owns (what HERMES_KANBAN_RUN_ID resolves to in production).
        assert kb.complete_task(
            conn, tid, summary="done",
            expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0],
        ) is True
    with kbc.connect() as conn:
        assert kb.add_comment(conn, tid, author="builder", body="allowed")
