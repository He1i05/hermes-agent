"""A live run cannot be closed out from under its owner (t_6c8342e0).

Incident (2026-09-15, card ``t_c5018dd9``): review run 894 was still LIVE —
``task_runs.ended_at`` NULL, worker pid 2035238 executing harnesses 19 minutes
later — when the card was force-closed to ``done``. The closing write went
through the sanctioned ``complete_task`` path (the ``completed`` event carries
``_completed_event_payload``'s ``result_len``+``summary`` shape on ``run_id``
894), driven by the *implementer's* process with its ``HERMES_KANBAN_TASK`` /
``HERMES_KANBAN_RUN_ID`` pins unset. Because ``expected_run_id`` was therefore
``None``, ``complete_task``'s UPDATE appended no ``AND current_run_id = ?``
clause, and its accepted status set (``running``/``ready``/``blocked``/
``review``) admitted the reviewing card. A caller that cannot prove it owns the
live run must not be able to close it.

The SECOND half of the incident is a guard defect with the larger blast radius:
after the worker's own ``review_requested`` handoff the dispatcher claimed the
review lane 34s later, at which point ``_owns_ended_run`` refused the worker on
its own card — including the read-only ``kanban_show`` and the ``kanban_comment``
handoff ack (measured, wiki-keeper session 20260915_105118_6d6362: 7 consecutive
``kanban_write_guard`` refusals at ``op=connect``). That lockout is what
manufactured the ``HERMES_KANBAN_GUARD_BYPASS=1`` export and, in the parent
incident, the hand-rolled direct-SQLite writers. A guard whose sanctioned path
is worse than its bypass will be bypassed.

Both directions are pinned here so neither fix can be traded for the other:

    T1-refuse   a live run cannot be closed by a caller without ownership
    T1-allow    the run's OWNER still completes (the sanctioned path works)
    T1-human    a parked ``review`` card with NO live run still completes (#54823)
    T2-window   the post-handoff window enumerated per verb (show/comment/
                heartbeat/attach/complete)
    T3-park     a SUPERSEDED worker stays refused (graceful-park contract)
    T-mutant    removing the ownership check makes the refuse-direction invert

Hermetic-only: every test uses the ``kanban_home`` fixture (temp board under a
per-test HERMES_HOME) and arms THAT board as the "live" one via extra-deny-roots,
so the guard runs its real production predicate without ever touching
``~/.hermes/kanban.db``. Run via ``scripts/run_tests.sh`` (never plain pytest
against the live home).
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_guard as kg


def _hostname() -> str:
    import socket

    try:
        return socket.gethostname() or "unknown"
    except Exception:
        return "unknown"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Temp board + stripped pins (mirrors test_kanban_guard_post_completion)."""
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
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    # ``getattr``/``raising=False`` keep this file runnable against the PRE-FIX
    # guard, which is what makes the mutant direction demonstrable.
    monkeypatch.setattr(kg, "_OWNED_WORKER_PINS", {}, raising=False)
    # The opened-run ledger is process-global for the same reason; clear it so
    # one test's validated claim cannot admit another's caller.
    monkeypatch.setattr(kb, "_OPENED_RUNS", {}, raising=False)
    return db_path


def _arm_live_board(monkeypatch, db_path, *, in_test=False):
    """Arm the scratch board as production and simulate a real non-test worker."""
    monkeypatch.setattr(kg, "_KANBAN_GUARD_EXTRA_DENY_ROOTS", (db_path.parent,))
    monkeypatch.setattr(kg, "_in_test_context", lambda: in_test)


def _claim_as_worker(conn, tid, *, worker_pid, claimer_pid=9999):
    """Claim the way the dispatcher does, then stamp ``worker_pid`` the way
    ``_set_worker_pid`` does (``claim_lock`` carries the DISPATCHER's pid)."""
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


def _events(tid: str, kind: str) -> list[str]:
    conn = sqlite3.connect(str(os.environ["HERMES_KANBAN_DB"]))
    try:
        return [
            r[0] for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind=?", (tid, kind)
            )
        ]
    finally:
        conn.close()


def _live_conn() -> sqlite3.Connection:
    """A caller-supplied connection in production SHAPE.

    Used for state setup/surgery once the board is ARMED as production: the
    guarded ``connect()`` would (correctly) refuse this process, but the guard
    only fires on the API boundary, so a plain connection is how the existing
    post-completion suite manipulates state to construct a scenario. Callers
    close it.
    """
    conn = sqlite3.connect(str(os.environ["HERMES_KANBAN_DB"]))
    conn.row_factory = sqlite3.Row
    return conn


def _live_review_run(conn, tid, *, reviewer_pid=None):
    """Hand the card off, then let the reviewer's run arrive — the exact 34s
    sequence that preceded the force-close.

    The reviewer's run is opened here only because a hermetic test cannot spawn a
    real dispatcher. In production the DISPATCHER (a different pid) opens it, so
    the worker's process holds no opened-run ledger entry for it — the
    discriminating fact of the incident. ``_simulate_dispatcher_claim`` restores
    that shape, and it is what makes the refuse-direction tests non-vacuous.
    """
    kb.request_review(
        conn, tid, summary="ready for review",
        expected_run_id=kb.get_task(conn, tid).current_run_id,
    )
    reviewer = kb.claim_review_task(conn, tid, claimer=f"{_hostname()}:4242")
    assert reviewer is not None, "the review lane was not claimable"
    run_id = kb._current_run_id(conn, tid)
    if reviewer_pid is not None:
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (int(reviewer_pid), tid))
        conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (int(reviewer_pid), run_id))
    _simulate_dispatcher_claim(tid)
    return run_id


def _simulate_dispatcher_claim(tid: str) -> None:
    """Drop this process's opened-run entry for *tid*.

    Models the production fact that the run was opened by the gateway dispatcher,
    not by the process under test. Without this, an in-process
    ``claim_review_task`` would make the caller the run's owner and every
    refuse-direction assertion below would pass vacuously.

    ``getattr``-guarded so this file also runs against the PRE-FIX tree, where
    neither helper exists: the T4 refuse-direction assertions must then fail
    BEHAVIOURALLY (the live run is displaced), not with an ``AttributeError`` —
    same honesty rule as
    ``test_t1_incident_reproduced_with_public_api_only``.
    """
    ledger = getattr(kb, "_OPENED_RUNS", None)
    if ledger is None:
        return  # pre-fix tree: no ledger exists, so nothing can be owned.
    board_fn = getattr(kb, "_board_identity", None)
    board = board_fn(_live_conn()) if board_fn is not None else None
    if board is not None:
        ledger.pop((board, tid), None)


# ── T1-refuse: a live run cannot be closed by a non-owner ───────────────────

def test_t1_incident_reproduced_with_public_api_only(kanban_home, monkeypatch):
    """AC1 RED/GREEN anchor: the incident, using ONLY pre-existing public API.

    Deliberately free of every helper this card introduces (no ``_board_identity``,
    no ``_simulate_dispatcher_claim``, no ``_completion_ownership_guard``), so
    running this file against the PRE-FIX guard fails BEHAVIOURALLY — the close
    LANDS and the live run is destroyed — rather than erroring on a missing
    attribute. That is what makes the mutant proof honest.

    The reviewer's run is opened by the DISPATCHER in production (a different
    process); a hermetic test cannot spawn one, so ``_OPENED_RUNS`` is cleared to
    that state. ``raising=False`` is load-bearing: on the pre-fix guard the
    attribute does not exist and is simply created, which is what keeps the RED
    direction a behavioural failure instead of an ``AttributeError``.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="incident anchor", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        kb.request_review(
            conn, tid, summary="ready",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        reviewer = kb.claim_review_task(conn, tid, claimer=f"{_hostname()}:4242")
        assert reviewer is not None
        live = kb._current_run_id(conn, tid)
        # The reviewer's run belongs to another process (the dispatcher's child).
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (os.getpid() + 1, tid))
        conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (os.getpid() + 1, live))

    # This process did NOT open that run — the dispatcher did.
    monkeypatch.setattr(kb, "_OPENED_RUNS", {}, raising=False)

    # The incident's write: sanctioned path, no run id, no ownership.
    with kbc.connect() as conn:
        ok = kb.complete_task(
            conn, tid,
            summary="Round-4 repairs: C1 trailing newline, C2 provenance annotations",
        )

    assert ok is False, (
        "PRE-FIX BEHAVIOUR: a run-id-less caller closed a live run it did not own"
    )
    row = _raw_query("SELECT status, completed_at, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "running"
    assert row[1] is None
    assert row[2] == live
    run = _raw_query("SELECT outcome, ended_at FROM task_runs WHERE id=?", (live,))
    assert run[0] is None, "the reviewer's run was closed out from under it"
    assert run[1] is None
    refused = _events(tid, "completion_refused_not_owner")
    assert len(refused) == 1, f"no auditable refusal event: {refused}"


def test_t1_live_review_run_cannot_be_closed_by_run_id_less_caller(kanban_home):
    """AC1: the incident, reproduced at the API. The implementer's process
    arrives with NO ``expected_run_id`` (its pins were unset) while the
    reviewer's run is live → refused, card untouched, audit event emitted."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mid-review close", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)

    # The exact pre-close state from the incident: running, live run, other pid.
    row = _raw_query(
        "SELECT status, current_run_id, worker_pid FROM tasks WHERE id=?", (tid,)
    )
    assert row[0] == "running"
    assert row[1] == live
    assert row[2] == os.getpid() + 1

    with kbc.connect() as conn:
        ok = kb.complete_task(
            conn, tid,
            summary="Round-4 repairs: C1 trailing newline, C2 provenance annotations",
        )

    assert ok is False, "a live run was closed by a caller with no ownership proof"
    after = _raw_query("SELECT status, completed_at, current_run_id FROM tasks WHERE id=?", (tid,))
    assert after[0] == "running", "the card was force-closed mid-review"
    assert after[1] is None
    assert after[2] == live, "the live run was closed out from under its owner"

    refused = _events(tid, "completion_refused_not_owner")
    assert len(refused) == 1, f"no auditable refusal event: {refused}"
    payload = json.loads(refused[0])
    assert payload["live_run_id"] == live
    assert payload["reason"]


def test_t1_live_run_cannot_be_closed_even_by_a_wrong_run_id(kanban_home):
    """A *stale* ``expected_run_id`` (the implementer's own, superseded run) is
    just as much a non-owner: the CAS must name the LIVE run, not any run."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale run id", assignee="builder")
        stale = _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)
    assert stale != live

    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, summary="mine, honestly", expected_run_id=stale) is False

    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "running"
    assert _raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0] == live


# ── T1-allow: the sanctioned paths still work ──────────────────────────────

def test_t1_reviewer_still_completes_its_own_live_run(kanban_home):
    """AC1 converse: the reviewer that OWNS the live run still approves. A fix
    that refuses this would be a lockout, not a guard."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reviewer approves", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid())

    with kbc.connect() as conn:
        assert kb.complete_task(
            conn, tid, summary="LGTM — approved after re-running the harnesses",
            expected_run_id=live,
        ) is True

    row = _raw_query("SELECT status, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "done"
    assert row[1] is None
    assert _events(tid, "completion_refused_not_owner") == []


def test_t1_implementer_still_completes_its_own_live_run(kanban_home):
    """The ordinary implementation completion is untouched: same pid owns the
    live run and passes its run id."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="implementer completes", assignee="builder")
        live = _claim_as_worker(conn, tid, worker_pid=os.getpid())

    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, summary="done", expected_run_id=live) is True
    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "done"


def test_t1_human_approval_of_parked_review_card_still_works(kanban_home):
    """AC1 human path: the #54823 shape — a card parked in ``review`` with NO
    active run (``request_review`` closed it) must stay completable by a human
    with no run id. The ownership rule keys on a LIVE run, never on status."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="approve me", assignee="worker")
        kb.claim_task(conn, tid)
        kb.request_review(
            conn, tid, summary="ready",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "review"
        assert kb.get_task(conn, tid).current_run_id is None

        assert kb.complete_task(conn, tid, summary="LGTM — merged", result="approved") is True
    row = _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))
    assert row[0] == "done"
    assert _events(tid, "completion_refused_not_owner") == []


def test_t1_dashboard_and_bulk_completion_of_unclaimed_cards_still_work(kanban_home):
    """The dashboard's ``done`` verb (and the bulk form) call ``complete_task``
    with no run id. On a card with no live run that must keep working."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="dashboard close", assignee="builder")
    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, result="shipped", summary="shipped") is True
    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "done"


def test_t1_locally_driven_caller_owns_the_run_it_opened(kanban_home):
    """The third ownership clause: a locally-driven caller that OPENED the run
    via ``claim_task``/``claim_review_task`` owns it even when the dispatcher
    never stamped a ``worker_pid`` (the CLI's and the test corpus's shape).

    Without this clause, every ordinary ``kb.claim_task(conn, tid, claimer=...)``
    + ``complete_task`` caller would be locked out — the regression this clause
    exists to prevent.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="locally driven", assignee="builder")
        claimed = kb.claim_task(conn, tid, claimer="builder:local")
        assert claimed is not None
        run_id = kb._current_run_id(conn, tid)
        # No worker_pid stamp at all: the "spawn_fn reported no pid" shape.
        assert kb.get_task(conn, tid).worker_pid is None

    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, summary="locally completed") is True

    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "done"
    assert run_id is not None
    assert _events(tid, "completion_refused_not_owner") == []


def test_t1_other_processes_claim_is_not_owned(kanban_home):
    """The clause is keyed on THIS process having opened the run. A run opened
    elsewhere is not owned — clear the ledger and the same caller is refused."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="someone else's run", assignee="builder")
        claimed = kb.claim_task(conn, tid, claimer="builder:local")
        assert claimed is not None
        run_id = kb._current_run_id(conn, tid)
        kb._OPENED_RUNS.clear()  # as if the claim had happened in another pid
        assert kb._completion_ownership_guard(conn, tid, None)[0] is False

    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, summary="not mine") is False

    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "running"
    assert _raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0] == run_id


# ── T2-window: the post-handoff window, enumerated per verb ────────────────

def test_t2_post_handoff_window_per_verb(kanban_home, monkeypatch):
    """AC2: after the implementer's ``review_requested`` AND the reviewer's
    dispatch onto a new run, enumerate the verbs. Writes that the reviewer now
    owns (complete/block) and the read + handoff-comment channel must be
    distinguishable — a guard that refuses ``kanban_show`` and ``kanban_comment``
    to the worker on its own card is what drove the bypass."""
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="handoff window", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        mine = kb._current_run_id(conn, tid)

    _arm_live_board(monkeypatch, kanban_home)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(mine))
    monkeypatch.setenv("HERMES_PROFILE", "builder")

    # The worker hands off; the reviewer's run arrives while this process lives.
    with kbc.connect() as conn:
        assert kb.request_review(
            conn, tid, summary="ready for review", expected_run_id=mine,
        ) is True
        assert _live_review_run_lane(conn, tid) is not None

    # READ — must never be refused by the guard.
    out = kt._handle_show({})
    assert "kanban_write_guard" not in out, out
    assert json.loads(out)["task"]["id"] == tid

    # COMMENT — the handoff channel on the worker's own card.
    out = kt._handle_comment({"task_id": tid, "body": "handoff ack after review dispatch"})
    assert "kanban_write_guard" not in out, out
    assert json.loads(out).get("ok") is True, out

    # HEARTBEAT — the card is running under the reviewer, so heartbeat_worker's
    # own "not running for you" refusal is expected; it must be THAT, not the guard.
    out = kt._handle_heartbeat({"note": "tail liveness"})
    assert "kanban_write_guard" not in out, out

    # ATTACH — same class as comment: a durable note on the worker's own card.
    out = kt._handle_attach({
        "task_id": tid,
        "filename": "tail.txt",
        "content_base64": base64.b64encode(b"post-handoff note").decode(),
    })
    assert "kanban_write_guard" not in out, out

    # COMPLETE — the reviewer owns the live run; the worker must NOT close it.
    out = kt._handle_complete({"summary": "sneaking a close past the reviewer"})
    assert json.loads(out).get("ok") is not True, out
    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "running"

    assert _raw_query("SELECT COUNT(*) FROM task_comments WHERE task_id=?", (tid,))[0] == 1
    assert _events(tid, "completion_refused_not_owner")


def _live_review_run_lane(conn, tid):
    """Claim the review lane beside a worker that has already handed off."""
    reviewer = kb.claim_review_task(conn, tid, claimer=f"{_hostname()}:4242")
    if reviewer is None:
        return None
    run_id = kb._current_run_id(conn, tid)
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (os.getpid() + 1, tid))
    conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (os.getpid() + 1, run_id))
    _simulate_dispatcher_claim(tid)
    return run_id


def test_t2_window_matrix_after_handoff_and_after_done(kanban_home, monkeypatch):
    """AC2 verbatim: the post-handoff window ENUMERATED, for each verb, in both
    windows — after ``review_requested`` (the reviewer's run has superseded ours)
    and after ``done`` (our own run is closed).

    Expected access, and why:

    | verb                | after review_requested | after done |
    |---------------------|------------------------|------------|
    | kanban_show (read)  | allow                  | allow      |
    | kanban_comment      | allow (handoff ack)    | allow      |
    | kanban_heartbeat    | allow (no-op: not ours)| allow-n/a  |
    | kanban_attach       | allow                  | allow      |
    | kanban_complete     | REFUSE (not our run)   | n/a (done) |
    | kanban_block        | REFUSE (not our run)   | n/a (done) |
    | kanban_request_changes | REFUSE (not our run)| n/a (done) |

    ``allow-n/a`` for heartbeat-after-done: ``heartbeat_worker`` refuses on its
    own "not running" rule; the assertion is that the refusal is NOT the guard's.

    The three run-ending verbs are enumerated side by side on purpose (round-1
    review Finding 3): the matrix previously concluded "complete -> REFUSE" while
    ``block`` / ``request_changes`` LANDS, i.e. its stated conclusion was the
    opposite of the real exposure. ``test_t4_*`` below carries the remaining
    run-ending verbs (schedule/archive/reclaim) plus the operator directions.
    """
    from tools import kanban_tools as kt

    # ── window 1: after review_requested, the reviewer's run live ──
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="matrix handoff", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        mine = kb._current_run_id(conn, tid)

    _arm_live_board(monkeypatch, kanban_home)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(mine))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.delenv(kg.GUARD_ENV_BYPASS, raising=False)

    with kbc.connect() as conn:
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=mine,
        ) is True
        assert _live_review_run_lane(conn, tid) is not None

    for label, call in (
        ("show", lambda: kt._handle_show({})),
        ("comment", lambda: kt._handle_comment({"task_id": tid, "body": "ack"})),
        ("heartbeat", lambda: kt._handle_heartbeat({"note": "tail"})),
        ("attach", lambda: kt._handle_attach({
            "task_id": tid, "filename": "n.txt",
            "content_base64": base64.b64encode(b"note").decode(),
        })),
    ):
        out = call()
        assert "kanban_write_guard" not in out, f"after-review_requested {label}: {out}"

    out = kt._handle_complete({"summary": "close it anyway"})
    assert '"ok": true' not in out, out
    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "running"

    # The other two run-ending TOOL verbs, in the same window.
    out = kt._handle_request_changes({"reason": "reject it anyway"})
    assert '"ok": true' not in out, out
    out = kt._handle_block({"reason": "park it anyway", "kind": "needs_input"})
    assert '"ok": true' not in out, out
    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "running"

    # ── window 2: after OUR OWN completion (the t_e35a95a1 window) ──
    with kbc.connect() as conn:
        tid2 = kb.create_task(conn, title="matrix done", assignee="builder")
        _claim_as_worker(conn, tid2, worker_pid=os.getpid())

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid2)
    monkeypatch.setenv(
        "HERMES_KANBAN_RUN_ID", str(_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid2,))[0]),
    )
    with kbc.connect() as conn:
        # Hand-merge (t_788c8b56, F2): the pin's complete_task fence
        # (upstream 0959224313) requires ownership PROOF for a live claim;
        # owner-by-pid alone is not accepted. Production workers always carry
        # HERMES_KANBAN_RUN_ID, so name the run the way the dispatcher does.
        assert kb.complete_task(
            conn, tid2, summary="done",
            expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid2,))[0],
        ) is True

    out = kt._handle_show({})
    assert "kanban_write_guard" not in out, out
    out = kt._handle_comment({"task_id": tid2, "body": "post-done note"})
    assert "kanban_write_guard" not in out and json.loads(out).get("ok") is True, out
    out = kt._handle_attach({
        "task_id": tid2, "filename": "d.txt",
        "content_base64": base64.b64encode(b"post-done").decode(),
    })
    assert "kanban_write_guard" not in out, out
    out = kt._handle_heartbeat({"note": "post-done"})
    assert "kanban_write_guard" not in out, out


# ── T4: EVERY run-ending verb, both actor directions ───────────────────────
#
# Round-1 review Finding 2: the invariant was enforced in ``complete_task`` ONLY.
# With the reviewer's live run owned by another pid, ``block_task`` /
# ``request_changes`` / ``schedule_task`` / ``reclaim_task`` / ``archive_task``
# each terminated that run — the defect class this card exists to close, and the
# opposite of what the T2 matrix concluded. One test per verb, plus the paired
# direction (the operator paths that MUST keep working), so neither half can be
# traded for the other.

#: ``(label, event_stem, callable(conn, tid))`` for every run-ending verb, called
#: at the API layer because that is the boundary the review measured. ``event_stem``
#: is what ``_RUN_END_VERBS`` maps the verb to — ``complete`` audits as
#: ``completion_refused_not_owner`` (the incident's documented kind), the rest as
#: ``<label>_refused_not_owner``.
_RUN_END_VERBS = (
    ("complete", "completion",
     lambda c, t: kb.complete_task(c, t, summary="sneak")),
    ("block", "block",
     lambda c, t: kb.block_task(c, t, reason="sneak", kind="needs_input")),
    ("request_changes", "changes",
     lambda c, t: kb.request_changes(c, t, reason="sneak")),
    ("schedule", "schedule",
     lambda c, t: kb.schedule_task(c, t, reason="sneak")),
    ("reclaim", "reclaim",
     lambda c, t: kb.reclaim_task(c, t, reason="sneak", signal_fn=lambda *_a: None)),
    ("archive", "archive",
     lambda c, t: kb.archive_task(c, t, signal_fn=lambda *_a: None)),
)


def _worker_sponsored_after_handoff(monkeypatch, tid, mine):
    """Put THIS process in the post-handoff worker shape for *tid*.

    Sets the worker pin (so the sponsored policy applies), arms the scratch board
    as production, records the validated-claim ledger entry a real claim leaves,
    and asserts the reviewer's live run is attributed to ANOTHER pid — the
    incident's discriminator. Without the ledger entry the guard would be refusing
    at ``op=connect`` and the test would be vacuous about the run-end seam.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(mine))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    kg._OWNED_WORKER_PINS[tid] = (os.getpid(), mine)


def test_t4_every_run_ending_verb_refuses_a_worker_that_does_not_own_the_run(
    kanban_home, monkeypatch,
):
    """AC1 (round-1 Finding 2): with the reviewer's live run owned by another pid,
    a worker-sponsored caller that does not own it is refused by EVERY run-ending
    verb — not just ``complete_task``."""
    for label, stem, call in _RUN_END_VERBS:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title=f"t4 refuse {label}", assignee="builder")
            _claim_as_worker(conn, tid, worker_pid=os.getpid())
            mine = kb._current_run_id(conn, tid)
            live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)

        _arm_live_board(monkeypatch, kanban_home)
        _worker_sponsored_after_handoff(monkeypatch, tid, mine)
        try:
            with kbc.connect() as conn:
                call(conn, tid)
        finally:
            kg._OWNED_WORKER_PINS.pop(tid, None)

        row = _raw_query(
            "SELECT status, current_run_id FROM tasks WHERE id=?", (tid,)
        )
        assert row[0] == "running", f"{label}: the card left `running`"
        assert row[1] == live, f"{label}: the live run was displaced"
        run = _raw_query("SELECT ended_at, outcome FROM task_runs WHERE id=?", (live,))
        assert run[0] is None, f"{label}: the reviewer's live run was ended ({run[1]!r})"
        assert run[1] is None, f"{label}: the reviewer's run got an outcome ({run[1]!r})"
        refused = _events(tid, f"{stem}_refused_not_owner")
        assert len(refused) == 1, f"{label}: no auditable refusal event: {refused}"
        payload = json.loads(refused[0])
        assert payload["live_run_id"] == live
        # The event KIND is the stem-derived name (found above via ``_events(...,
        # f"{stem}_refused_not_owner")``); the payload's ``verb`` is the raw verb,
        # so a reader can tell which API call the guard refused. Both pinned here so
        # the ``_RUN_END_VERBS`` mapping cannot drift from the documented kinds.
        assert payload["verb"] == label
        _disarm(monkeypatch, kanban_home)


def _disarm(monkeypatch, db_path):
    """Return the guard/board to its inert state between parametrized cases."""
    monkeypatch.setattr(kg, "_KANBAN_GUARD_EXTRA_DENY_ROOTS", ())
    monkeypatch.setattr(kg, "_in_test_context", lambda: True)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)


def test_t4_every_run_ending_verb_still_works_for_the_runs_owner(kanban_home):
    """The paired ALLOW direction: the same verbs, called by the process that owns
    the live run (it opened it), must all land. A fix that refuses these is a
    lockout, not a guard — and a lockout on ``block``/``schedule`` breaks the
    worker's own handoff story.

    ``request_changes`` is exercised separately: its contract requires an active
    run CLAIMED FROM ``review`` (``source_status == 'review'``), which the ready
    lane never stamps.
    """
    for label, call in _OWNER_RUN_END_VERBS:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title=f"t4 allow {label}", assignee="builder")
            kb.claim_task(conn, tid, claimer="builder:local")
            run_id = kb._current_run_id(conn, tid)
            assert run_id is not None
            # No worker_pid stamp: the locally-driven caller owns the run it opened.
            with kbc.connect() as c2:
                call(c2, tid)
        status = _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0]
        assert status != "running", f"{label}: the owner's own call did not land"
        assert _events(
            tid, f"{label}_refused_not_owner"
        ) == [], f"{label}: the owner was refused"

    # request_changes: the reviewer that OWNS the review-lane run still verdicts.
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t4 allow request_changes", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        mine = kb._current_run_id(conn, tid)
        kb.request_review(conn, tid, summary="ready", expected_run_id=mine)
        assert kb.claim_review_task(conn, tid, claimer="builder:reviewer") is not None
    with kbc.connect() as conn:
        ok, detail = kb.request_changes(conn, tid, reason="real rework")
    assert ok is True, detail
    assert _events(tid, "changes_refused_not_owner") == []


#: The owner-direction subset: every run-ending verb whose happy path does not
#: need a review-lane claim. ``request_changes`` is handled explicitly above.
_OWNER_RUN_END_VERBS = (
    ("complete", lambda c, t: kb.complete_task(c, t, summary="mine")),
    ("block", lambda c, t: kb.block_task(c, t, reason="mine", kind="needs_input")),
    ("schedule", lambda c, t: kb.schedule_task(c, t, reason="mine")),
    ("reclaim", lambda c, t: kb.reclaim_task(c, t, reason="mine", signal_fn=lambda *_a: None)),
    ("archive", lambda c, t: kb.archive_task(c, t, signal_fn=lambda *_a: None)),
)


def test_t4_operator_paths_are_not_locked_out(kanban_home, monkeypatch):
    """The T5 graceful-park contract, and the whole operator surface.

    ``docs/runbooks/t5-graceful-stop-runbook.md`` §3 step 3 parks an in-flight
    SIBLING card from a shell carrying no worker pin: ``hermes kanban block <id>
    dependency "<reason>"``. That MUST land against a card whose live run belongs
    to another process — requiring ``--force`` there would break the binding STOP
    procedure mid-incident. Same for the dashboard's reclaim/archive buttons and
    ``reassign_task(reclaim_first=True)``, all of which run in the dashboard server
    (also pin-less).

    ``force=True`` is the explicit override for the sponsored verbs, and it must
    really work: an override that does not exist sends the operator to
    ``HERMES_KANBAN_GUARD_BYPASS``, which is how the incident happened.
    """
    # ── the T5 park: no worker pin at all ──
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t5 park", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid() + 1)
        live = kb._current_run_id(conn, tid)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kbc.connect() as conn:
        assert kb.block_task(
            conn, tid, reason="STOP pivot t_root: resume via promote",
            kind="dependency",
        ) is True, "the T5 park did not land"
    # Hand-merge (t_788c8b56): upstream 42a778ab4b re-kinds a parent-less
    # ``dependency`` block to ``needs_input`` so ``recompute_ready`` cannot
    # respawn it context-free. This card carries NO parent, so the park lands
    # in the sticky ``blocked`` lane, not ``todo``. The operator LANE decision
    # is unaffected — it is keyed on the FILED kind (``requested_kind``),
    # which is why the pin-less park still lands at all.
    assert _raw_query("SELECT status, block_kind FROM tasks WHERE id=?",
                      (tid,)) == ("blocked", "needs_input")
    assert _raw_query("SELECT ended_at FROM task_runs WHERE id=?", (live,))[0] is not None

    # ── the operator reclaim / archive, also pin-less ──
    for label, setup_call, expect_status in (
        ("reclaim", lambda c, t: kb.reclaim_task(c, t, signal_fn=lambda *_a: None), "ready"),
        ("archive", lambda c, t: kb.archive_task(c, t, signal_fn=lambda *_a: None), "archived"),
    ):
        with kbc.connect() as conn:
            tid2 = kb.create_task(conn, title=f"t4 operator {label}", assignee="builder")
            _claim_as_worker(conn, tid2, worker_pid=os.getpid() + 1)
        with kbc.connect() as conn:
            assert setup_call(conn, tid2) is True, f"operator {label} did not land"
        assert _raw_query(
            "SELECT status FROM tasks WHERE id=?", (tid2,)
        )[0] == expect_status, label

    # ── the sponsored verb's explicit override ──
    with kbc.connect() as conn:
        tid3 = kb.create_task(conn, title="t4 forced block", assignee="builder")
        _claim_as_worker(conn, tid3, worker_pid=os.getpid())
        mine3 = kb._current_run_id(conn, tid3)
        live3 = _live_review_run(conn, tid3, reviewer_pid=os.getpid() + 1)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid3)
    kg._OWNED_WORKER_PINS[tid3] = (os.getpid(), mine3)
    try:
        with kbc.connect() as conn:
            assert kb.block_task(
                conn, tid3, reason="forced", kind="needs_input", force=True,
            ) is True
    finally:
        kg._OWNED_WORKER_PINS.pop(tid3, None)
    forced = _events(tid3, "block_forced_not_owner")
    assert len(forced) == 1, forced
    assert json.loads(forced[0])["live_run_id"] == live3
    assert _events(tid3, "block_refused_not_owner") == []


def test_t4_sponsored_policy_is_keyed_on_the_pinned_card(kanban_home, monkeypatch):
    """The sponsored policy is scoped to the card the worker is pinned to.

    A worker pinned to ``other`` end-running ``tid`` is NOT sponsored for it, so
    the operator policy applies and the call lands. That is deliberate: the pin
    means "this is MY card", and a pin for a different card tells the guard nothing
    about the intent behind this write. What actually protects a sibling run is the
    tool layer's ``_enforce_worker_task_ownership``, which refuses a dispatched
    worker an explicit foreign ``task_id`` before the DB is reached.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="not my card", assignee="builder")
        other = kb.create_task(conn, title="my card", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid() + 1)

    monkeypatch.setenv("HERMES_KANBAN_TASK", other)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999999")
    with kbc.connect() as conn:
        assert kb.block_task(
            conn, tid, reason="parked via operator lane", kind="dependency",
        ) is True
    # Hand-merge (t_788c8b56): upstream 42a778ab4b re-kinds a parent-less
    # ``dependency`` block to ``needs_input`` so ``recompute_ready`` cannot
    # respawn it context-free. This card carries NO parent, so the park lands
    # in the sticky ``blocked`` lane, not ``todo``. The operator LANE decision
    # is unaffected — it is keyed on the FILED kind (``requested_kind``),
    # which is why the pin-less park still lands at all.
    assert _raw_query("SELECT status, block_kind FROM tasks WHERE id=?",
                      (tid,)) == ("blocked", "needs_input")


def test_t4_claim_landing_between_the_precheck_and_the_cas_is_still_refused(
    kanban_home, monkeypatch,
):
    """The always-armed CAS, not just the pre-check.

    The pre-check answers against one read; a dispatcher reclaim + respawn can land
    before the in-txn UPDATE. The caller here is the run's genuine owner (it passes
    ``expected_run_id=<its own run>``, so the pre-check admits it legitimately), and
    a SUCCESSOR run appears in between. Without the in-txn re-read plus the
    ``current_run_id IS ?`` clause, the verdict would land on the successor run —
    the incident's exact shape, one race later.

    Modelled by swapping in a new run after the first read with a one-shot wrapper
    around ``_run_end_allowed``. The wrapper fires only on ``block_task``'s own two
    call sites (the seam is called directly there; ``_live_run_ownership_guard``
    underneath it is not wrapped).
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cas race", assignee="builder")
        mine = _claim_as_worker(conn, tid, worker_pid=os.getpid())

    # A worker-sponsored caller, so ``block`` is governed by the ownership policy
    # rather than the operator lane (an operator shell is admitted by design).
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(mine))

    real = kb._run_end_allowed
    state = {"calls": 0}

    def swap_after_first_read(conn, task_id, expected_run_id, *, verb, kind=None):
        verdict = real(conn, task_id, expected_run_id, verb=verb, kind=kind)
        state["calls"] += 1
        if state["calls"] == 1:
            assert verdict[0] is True, "the caller owns its own run; pre-check must pass"
            # The "dispatcher" now claims a NEW run for the reviewer.
            with kb.write_txn(conn):
                new_run = conn.execute(
                    "INSERT INTO task_runs (task_id, status, started_at, worker_pid) "
                    "VALUES (?, 'running', 1, ?)", (task_id, os.getpid() + 1),
                ).lastrowid
                conn.execute(
                    "UPDATE tasks SET status='running', current_run_id=?, "
                    "worker_pid=? WHERE id=?", (new_run, os.getpid() + 1, task_id),
                )
            state["new_run"] = new_run
        return verdict

    monkeypatch.setattr(kb, "_run_end_allowed", swap_after_first_read)
    with kbc.connect() as conn:
        # The successor run belongs to another process, so this must NOT land on it.
        assert kb.block_task(
            conn, tid, reason="race", kind="needs_input", expected_run_id=mine,
        ) is False

    row = _raw_query("SELECT status, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "running", "the successor's live run was displaced by a stale caller"
    assert row[1] == state["new_run"]
    assert _raw_query(
        "SELECT ended_at FROM task_runs WHERE id=?", (state["new_run"],)
    )[0] is None


def test_t4_request_changes_force_is_the_explicit_operator_override(kanban_home):
    """``request_changes(force=True)`` is the strict verb's escape hatch.

    Strict means "owner or explicit force", so the override must actually work —
    an override that does not exist sends the operator to the guard bypass hatch.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="forced verdict", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        mine = kb._current_run_id(conn, tid)
        kb.request_review(conn, tid, summary="ready", expected_run_id=mine)
        assert kb.claim_review_task(conn, tid, claimer="builder:reviewer") is not None
        live = kb._current_run_id(conn, tid)
        # Attribute the review run to another process (the dispatcher's child).
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (os.getpid() + 1, tid))
        conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (os.getpid() + 1, live))
    # The DISPATCHER opened that review run, not us — so we hold no ledger entry
    # for it and the ownership decision sees a genuinely unowned run.
    _simulate_dispatcher_claim(tid)

    with kbc.connect() as conn:
        ok, detail = kb.request_changes(conn, tid, reason="forced verdict", force=True)
    assert ok is True, detail
    forced = _events(tid, "changes_forced_not_owner")
    assert len(forced) == 1, forced
    assert json.loads(forced[0])["live_run_id"] == live
    assert _events(tid, "changes_refused_not_owner") == []


# ── T3-park: a superseded worker stays refused ─────────────────────────────

def test_t3_superseded_worker_stays_refused_after_respawn(kanban_home, monkeypatch):
    """The graceful-park contract (t5-graceful-stop-runbook §4 step 3): a worker
    the dispatcher reclaimed and respawned onto a new READY-lane run must NOT
    regain write access. The post-handoff allowance is keyed on the newer run
    being the REVIEWER arriving at MY handoff, never on 'a newer run exists'."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="superseded", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())

    _arm_live_board(monkeypatch, kanban_home)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with kbc.connect() as conn:
        # Hand-merge (t_788c8b56, F2): see the t2 note — a live claim needs
        # explicit ownership proof, so pass the run id the worker owns.
        assert kb.complete_task(
            conn, tid, summary="first run",
            expected_run_id=_raw_query("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0],
        ) is True
    assert tid in kg._OWNED_WORKER_PINS

    # A fresh READY-lane run is spawned beside us (reclaim/respawn, not review).
    # Setup goes through a plain connection: the board is armed as production,
    # so the guarded ``connect`` correctly refuses this process.
    setup = _live_conn()
    try:
        with kbc.write_txn(setup):
            setup.execute(
                "UPDATE task_runs SET status='reclaimed', outcome='reclaimed' WHERE task_id=?",
                (tid,),
            )
            new_run = setup.execute(
                "INSERT INTO task_runs (task_id, status, started_at, worker_pid) "
                "VALUES (?, 'running', ?, ?)",
                (tid, int(__import__("time").time()), os.getpid() + 7),
            ).lastrowid
            setup.execute(
                "UPDATE tasks SET status='running', current_run_id=?, worker_pid=? WHERE id=?",
                (new_run, os.getpid() + 7, tid),
            )
            # A ready-lane claim carries NO ``source_status`` (only the review
            # lane stamps it) — that is the clause under test.
            setup.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, 'claimed', ?, ?)",
                (tid, new_run, json.dumps({"lock": f"{_hostname()}:4242"}), int(__import__("time").time())),
            )
        live = _live_conn()
    finally:
        setup.close()

    try:
        assert kg._owns_ended_run(live, tid) is False
        assert kg._owns_handed_off_card(live, tid) is False, (
            "a ready-lane respawn must stay refused (graceful-park contract)"
        )
    finally:
        live.close()

    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.add_comment(conn, tid, author="builder", body="ignoring the respawn")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)


def test_t3_parked_worker_stays_refused(kanban_home, monkeypatch):
    """An orchestrator STOP-pivot park writes outcome ``blocked`` — byte-identical
    on the run row to the worker's own ``kanban_block`` — and must stay refused."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="parked", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())

    _arm_live_board(monkeypatch, kanban_home)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    live = _live_conn()
    try:
        assert kg.is_claimed_worker(conn=live) is True
    finally:
        live.close()
    assert tid in kg._OWNED_WORKER_PINS

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kbc.connect() as conn:
        assert kb.block_task(conn, tid, reason="parked: STOP pivot", kind="dependency") is True

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    live = _live_conn()
    try:
        assert kg._owns_ended_run(live, tid) is False
        # A park is not a handoff: the allowance must not admit it either.
        assert kg._owns_handed_off_card(live, tid) is False
    finally:
        live.close()
    with pytest.raises(RuntimeError) as ei:
        with kbc.connect() as conn:
            kb.add_comment(conn, tid, author="builder", body="ignoring the park")
    assert kg.KANBAN_WRITE_GUARD_MARK in str(ei.value)


# ── T5: the legacy-backfill run is owned by whoever ran the migration ──────
#
# Round-1 review Finding 1 (regression): the ownership guard refused the
# SANCTIONED legacy-backfill completion. The migration synthesizes a run row via a
# path that never went through ``_claim_and_open_run``, so none of the ownership
# clauses matched — task row has no ``worker_pid`` (that IS the legacy shape) and
# no claim was made by this process — and ``_OPENED_RUNS`` still held a stale entry
# for the run the caller had deleted. End state: card stuck ``running`` with an
# OPEN synthesized run, which is exactly the fingerprint the T5 watchdog alerts on.

def test_t5_legacy_backfilled_run_is_completable_by_its_opener(kanban_home):
    """The corpus regression, isolated: backfill then complete must close cleanly.

    Mirrors ``test_kanban_core_functionality.py::
    test_migration_backfills_inflight_run_for_legacy_db`` and adds the two
    properties that regression hinges on: the synthesized run is REGISTERED as
    opened by the opener, and a STALE ledger entry for the deleted run does not
    survive to admit (or block) the new one.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="legacy inflight", assignee="worker")
        kb.claim_task(conn, tid)
        stale_run = kb._current_run_id(conn, tid)
        assert stale_run is not None
        # Simulate the pre-runs schema: drop the run row and the pointer.
        with kb.write_txn(conn):
            conn.execute("DELETE FROM task_runs WHERE task_id = ?", (tid,))
            conn.execute("UPDATE tasks SET current_run_id = NULL WHERE id = ?", (tid,))
        assert kb.list_runs(conn, tid) == []

    kb.init_db()  # the migration backfill

    with kbc.connect() as conn:
        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1
        backfilled = runs[0].id
        assert backfilled != stale_run, "the backfill must synthesize a NEW run"
        assert kb.get_task(conn, tid).current_run_id == backfilled
        # The synthesized run IS owned by this process — the fix under test.
        assert kb._live_run_ownership_guard(conn, tid, None)[0] is True, (
            "the backfilled run is unowned, so the card is un-completable"
        )
        assert kb.complete_task(conn, tid, result="done", summary="ok") is True

    row = _raw_query("SELECT status, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "done"
    assert row[1] is None, "the card is `done` with an open run pointer (T5 fingerprint)"
    run = _raw_query("SELECT outcome, ended_at FROM task_runs WHERE id=?", (backfilled,))
    assert run[0] == "completed", f"the backfilled run was not closed: {run!r}"
    assert run[1] is not None, "the backfilled run stayed OPEN (T5 fingerprint)"
    assert _events(tid, "completion_refused_not_owner") == []


def test_t5_stale_ledger_entry_does_not_admit_a_deleted_run(kanban_home):
    """A stale ``_OPENED_RUNS`` entry must not admit a run this process never opened.

    The entry names a run id that no longer resolves to the card (deleted by the
    migration, or a re-claim by another process). Honouring it would let an
    unrelated later run — one whose id happened to be recorded — be treated as
    owned, so :func:`_opened_run_matches` drops it instead.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale ledger", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:local")
        stale = kb._current_run_id(conn, tid)
        board = kb._board_identity(conn)
        assert kb._OPENED_RUNS.get((board, tid)) == stale
        # Delete the run row out from under the ledger (direct-SQL / migration).
        with kb.write_txn(conn):
            conn.execute("DELETE FROM task_runs WHERE task_id = ?", (tid,))
            conn.execute("UPDATE tasks SET current_run_id = NULL WHERE id = ?", (tid,))
        other_run = conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at) VALUES (?, 'running', 1)",
            (tid,),
        ).lastrowid
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ?", (other_run, tid),
            )
        # The deleted run's entry must NOT be read as owning the new run.
        assert kb._opened_run_matches(conn, tid, other_run) is False
        assert (board, tid) not in kb._OPENED_RUNS, (
            "the stale ledger entry survived"
        )
        # And the ownership seam agrees: this process does not own `other_run`.
        assert kb._live_run_ownership_guard(conn, tid, None)[0] is False


# ── Mutant: prove the refuse-direction inverts without the fix ─────────────

def test_mutant_without_the_ownership_check_the_live_run_is_closed(kanban_home, monkeypatch):
    """AC1 mutant: inert the ownership predicate and the refuse-direction test
    must invert — the live run is closed by the run-id-less caller, which is the
    precise pre-fix behaviour of the incident."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mutant", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)

    # MUTANT: the invariant is inert. Pre-fix, `expected_run_id=None` appended NO
    # CAS clause, so the terminal UPDATE matched on `status` alone and closed
    # whatever run the row carried. Returning the row's OWN live run id
    # reproduces that exactly: the pre-check passes and the CAS is satisfied, so
    # the close lands.
    monkeypatch.setattr(
        kb, "_completion_ownership_guard",
        lambda conn, task_id, expected_run_id: (
            True,
            conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (task_id,)).fetchone()[0],
        ),
    )

    with kbc.connect() as conn:
        ok = kb.complete_task(conn, tid, summary="Mutant: no ownership check")

    # Pre-fix: the close LANDS and the reviewer's live run is destroyed.
    assert ok is True
    row = _raw_query("SELECT status, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "done"
    assert row[1] is None
    run = _raw_query("SELECT status, outcome, ended_at FROM task_runs WHERE id=?", (live,))
    assert run[1] == "completed"
    assert run[2] is not None


def test_mutant_control_same_setup_with_fix_stays_open(kanban_home):
    """Control: identical setup, fix intact => the card is NOT closed."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="control", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)

    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, summary="no proof of ownership") is False

    row = _raw_query("SELECT status, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "running"
    assert row[1] == live


def test_incident_actor_shape_is_refused(kanban_home):
    """The incident's exact discriminator: the reviewer's run was opened by a
    DIFFERENT process (the dispatcher), so neither the pid clause nor the
    opened-run ledger matches. Modelled by clearing the ledger entry after the
    reviewer's run is opened — the state the implementer's process was in."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="dispatcher-opened review", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)
        # Simulate the dispatcher (a different process) having opened run `live`.
        kb._OPENED_RUNS.clear()
        owner_ok, cas_run = kb._completion_ownership_guard(conn, tid, None)
        assert owner_ok is False, "a dispatcher-opened run was treated as owned"
        assert cas_run == live

    with kbc.connect() as conn:
        assert kb.complete_task(
            conn, tid,
            summary="Round-4 repairs: C1 trailing newline, C2 provenance annotations",
        ) is False

    row = _raw_query("SELECT status, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "running"
    assert row[1] == live


# ── The human override: explicit, audited, and not the default ─────────────

def test_force_closes_a_live_run_and_audits_the_override(kanban_home):
    """``force=True`` is the operator's escape hatch (CLI ``--force``, dashboard
    ``done``). It must actually work — an override that does not exist just sends
    the operator back to ``HERMES_KANBAN_GUARD_BYPASS`` — and it must be
    distinguishable from an ordinary handoff in the event log."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="forced close", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)

    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, summary="operator forced this", force=True) is True

    row = _raw_query("SELECT status, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "done"
    assert row[1] is None
    forced = _events(tid, "completion_forced_not_owner")
    assert len(forced) == 1, forced
    assert json.loads(forced[0])["live_run_id"] == live
    # A forced close is NOT a refusal: the refusal event must not be emitted.
    assert _events(tid, "completion_refused_not_owner") == []


def test_force_does_not_weaken_the_default(kanban_home):
    """The paired negative: the SAME caller without ``force`` is refused. Paired
    so a passing force test cannot be mistaken for the invariant being off."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unforced", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)

    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, summary="not forced") is False

    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "running"
    assert _events(tid, "completion_forced_not_owner") == []
    assert len(_events(tid, "completion_refused_not_owner")) == 1


def test_refusal_detail_names_the_run_and_the_ways_forward(kanban_home):
    """The refusal must be ACTIONABLE, not "unknown id or terminal state".

    The incident operator, unable to tell why the sanctioned verb stopped
    working, exported ``HERMES_KANBAN_GUARD_BYPASS``. The message must name the
    live run and both legitimate exits (ownership, reclaim, --force)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="detail", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)

        detail = kb.completion_refusal_detail(conn, tid, None)
        assert detail, "a would-be refusal produced no explanation"
        assert str(live) in detail
        assert "--force" in detail
        assert "reclaim" in detail

        # The allowed directions must NOT produce a refusal detail.
        assert kb.completion_refusal_detail(conn, tid, live) is None
        assert kb.completion_refusal_detail(conn, tid, None) is not None


def test_refusal_detail_is_silent_when_completion_would_succeed(kanban_home):
    """Never raise a false alarm: a card with no live run has no refusal."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no live run", assignee="builder")
        assert kb.completion_refusal_detail(conn, tid, None) is None
        kb.claim_task(conn, tid)
        assert kb.completion_refusal_detail(
            conn, tid, kb.get_task(conn, tid).current_run_id,
        ) is None


# ── The guard-bypass hatch must not be the path of least resistance ────────

def test_bypass_is_not_required_to_read_your_own_handed_off_card(kanban_home, monkeypatch):
    """AC3: the lockout that manufactured the bypass. After the handoff and the
    reviewer's dispatch, the worker's OWN ``kanban_show`` and ``kanban_comment``
    must work through the sanctioned path — with ``HERMES_KANBAN_GUARD_BYPASS``
    explicitly UNSET. If this needs the bypass, the incident recurs by design."""
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no bypass needed", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        mine = kb._current_run_id(conn, tid)

    _arm_live_board(monkeypatch, kanban_home)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(mine))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    # The bypass must NOT be what makes this work.
    monkeypatch.delenv(kg.GUARD_ENV_BYPASS, raising=False)
    monkeypatch.setattr(kg, "_KANBAN_GUARD_BYPASS", False, raising=False)

    with kbc.connect() as conn:
        assert kb.request_review(
            conn, tid, summary="ready for review", expected_run_id=mine,
        ) is True
        assert _live_review_run_lane(conn, tid) is not None

    assert kg._bypassed() is False, "the bypass hatch is armed; test is vacuous"

    out = kt._handle_show({})
    assert "kanban_write_guard" not in out, out

    out = kt._handle_comment({"task_id": tid, "body": "handoff ack, no bypass"})
    assert "kanban_write_guard" not in out, out
    assert json.loads(out).get("ok") is True, out


# ── T6: the PIN-LESS caller shape — the incident's own, one test per verb ──
#
# Round-2 review Finding 1: the T4 group above covers only the worker-SPONSORED
# shape, while the incident's closer was the implementer's process with
# ``HERMES_KANBAN_TASK`` / ``HERMES_KANBAN_RUN_ID`` UNSET — indistinguishable
# from an operator shell by env alone. With round-1's design the SPONSORED branch
# admitted that caller unconditionally, so ``block`` (any kind) / ``schedule`` /
# ``reclaim`` / ``archive`` each displaced a live review run SILENTLY (no
# ``*_refused_not_owner``, no ``*_forced_not_owner``), and ``reclaim``/``archive``
# additionally SIGTERM'd the reviewer — which the T5 contract forbids ("STOP is
# NEVER a dispatcher kill and never a hard SIGTERM").
#
# The policy is now: the pin-less lane admits ONLY the runbook's own verb+kind
# (``block`` + ``kind="dependency"``, audited), plus the operator RECOVERY verbs
# ``reclaim``/``archive`` whose termination is their documented contract (also
# audited). Everything else is refused like any other non-owner.
#
# One test per verb, asserting the DECIDED outcome — so a later change to the
# operator lane has to state which of these it is changing.


def _pinless_review_card(monkeypatch) -> tuple[str, int]:
    """A card mid-review whose live run belongs to ANOTHER process.

    Returns ``(task_id, live_run_id)``. The caller is put in the pin-less shape
    (both pins deleted) — the incident's discriminator.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="pin-less caller", assignee="builder")
        _claim_as_worker(conn, tid, worker_pid=os.getpid())
        live = _live_review_run(conn, tid, reviewer_pid=os.getpid() + 1)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    return tid, live


def _assert_run_still_live(tid: str, live: int, label: str) -> None:
    row = _raw_query("SELECT status, current_run_id FROM tasks WHERE id=?", (tid,))
    assert row[0] == "running", f"{label}: the card left `running`"
    assert row[1] == live, f"{label}: the live run was displaced"
    run = _raw_query("SELECT ended_at, outcome FROM task_runs WHERE id=?", (live,))
    assert run[0] is None, f"{label}: the reviewer's live run was ENDED ({run[1]!r})"
    assert run[1] is None, f"{label}: the reviewer's run got an outcome ({run[1]!r})"


def test_t6_pinless_complete_is_still_refused(kanban_home, monkeypatch):
    """``complete`` — STRICT, so the pin-less shape changes nothing. This is the
    AC1 half round 2 already had right; pinned here so the T6 group states the
    whole matrix rather than only the newly-closed verbs."""
    tid, live = _pinless_review_card(monkeypatch)
    with kbc.connect() as conn:
        assert kb.complete_task(conn, tid, summary="sneak") is False
    _assert_run_still_live(tid, live, "complete")
    assert len(_events(tid, "completion_refused_not_owner")) == 1


def test_t6_pinless_request_changes_is_still_refused(kanban_home, monkeypatch):
    """``request_changes`` — STRICT for the same reason."""
    tid, live = _pinless_review_card(monkeypatch)
    with kbc.connect() as conn:
        ok, _detail = kb.request_changes(conn, tid, reason="sneak")
    assert ok is False
    _assert_run_still_live(tid, live, "request_changes")
    assert len(_events(tid, "changes_refused_not_owner")) == 1


def test_t6_pinless_schedule_is_refused_and_audited(kanban_home, monkeypatch):
    """``schedule`` from a pin-less caller: REFUSED (round-2 Finding 1 — it used
    to displace the review run silently). §3 of the T5 runbook never schedules,
    so nothing legitimate needs this admission."""
    tid, live = _pinless_review_card(monkeypatch)
    with kbc.connect() as conn:
        assert kb.schedule_task(conn, tid, reason="later") is False
    _assert_run_still_live(tid, live, "schedule")
    assert len(_events(tid, "schedule_refused_not_owner")) == 1
    assert _events(tid, "schedule_displaced_not_owner") == []


def test_t6_pinless_block_without_dependency_is_refused_and_audited(
    kanban_home, monkeypatch,
):
    """``block`` with any kind OTHER than ``dependency``: REFUSED. Only the T5
    park is the documented operator lane; a generic human block is not, and it
    used to land silently against a live reviewer."""
    for kind in ("needs_input", "capability", "transient", None):
        tid, live = _pinless_review_card(monkeypatch)
        with kbc.connect() as conn:
            assert kb.block_task(
                conn, tid, reason="park it", kind=kind,
            ) is False, f"kind={kind!r} landed from a pin-less caller"
        _assert_run_still_live(tid, live, f"block kind={kind!r}")
        assert len(_events(tid, "block_refused_not_owner")) == 1, kind
        assert _events(tid, "block_displaced_not_owner") == [], kind


def test_t6_pinless_dependency_park_still_lands_and_is_audited(
    kanban_home, monkeypatch,
):
    """The T5 graceful-park contract (``docs/runbooks/t5-graceful-stop-runbook.md``
    §3 step 3): a pin-less orchestrator shell parks an in-flight SIBLING with
    ``hermes kanban block <id> dependency "<reason>"``, and that MUST land — it is
    a binding runbook step and requiring ``--force`` would break it mid-incident.

    New in round 2: the park is AUDITED as ``block_displaced_not_owner`` (with
    ``via="operator_park"``), because the trail previously read as an ordinary
    dependency park and a displaced reviewer was indistinguishable from a
    legitimate stop (round-2 review Finding 1)."""
    tid, live = _pinless_review_card(monkeypatch)
    with kbc.connect() as conn:
        assert kb.block_task(
            conn, tid, reason="STOP pivot t_root: resume via promote",
            kind="dependency",
        ) is True, "the T5 park did not land"
    # The park is SUPPOSED to end the run — that is the graceful stop.
    # Hand-merge (t_788c8b56): upstream 42a778ab4b re-kinds a parent-less
    # ``dependency`` block to ``needs_input`` so ``recompute_ready`` cannot
    # respawn it context-free. This card carries NO parent, so the park lands
    # in the sticky ``blocked`` lane, not ``todo``. The operator LANE decision
    # is unaffected — it is keyed on the FILED kind (``requested_kind``),
    # which is why the pin-less park still lands at all.
    assert _raw_query("SELECT status, block_kind FROM tasks WHERE id=?",
                      (tid,)) == ("blocked", "needs_input")
    assert _raw_query("SELECT ended_at FROM task_runs WHERE id=?", (live,))[0] is not None
    displaced = _events(tid, "block_displaced_not_owner")
    assert len(displaced) == 1, f"the park left no displacement record: {displaced}"
    payload = json.loads(displaced[0])
    assert payload["live_run_id"] == live
    assert payload["via"] == "operator_park"
    assert payload["worker_sponsored"] is False
    # The park is NOT a refusal and NOT a force override: honest naming matters
    # here, or the audit trail overstates what an operator did.
    assert _events(tid, "block_refused_not_owner") == []
    assert _events(tid, "block_forced_not_owner") == []


def test_t6_pinless_reclaim_lands_but_is_audited_not_silent(
    kanban_home, monkeypatch,
):
    """``reclaim`` from a pin-less caller: LANDS (it IS the operator's recovery
    verb, and its SIGTERM is pinned by ``test_kanban_core_functionality``'s
    ``killed == [signal.SIGTERM]``) — but it is now AUDITED.

    Round-2 Finding 1's second consequence was that this displacement was SILENT:
    ``reclaim``/``archive`` hard-killed a live reviewer with no ``*_forced_not_owner``
    and no ``*_refused_not_owner``, so the trail looked like an ordinary release.
    The termination is unchanged; only its visibility is."""
    tid, live = _pinless_review_card(monkeypatch)
    sent: list[tuple[int, int]] = []
    with kbc.connect() as conn:
        assert kb.reclaim_task(
            conn, tid, reason="operator recovery", signal_fn=lambda p, s: sent.append((p, s)),
        ) is True
    assert _raw_query("SELECT ended_at FROM task_runs WHERE id=?", (live,))[0] is not None
    # The reviewer's pid was signalled — the documented contract, kept.
    assert sent and sent[0][0] == os.getpid() + 1, sent
    displaced = _events(tid, "reclaim_displaced_not_owner")
    assert len(displaced) == 1, f"the reclaim left no displacement record: {displaced}"
    payload = json.loads(displaced[0])
    assert payload["live_run_id"] == live
    assert payload["via"] == "operator_recovery"
    assert _events(tid, "reclaim_forced_not_owner") == []
    assert _events(tid, "reclaim_refused_not_owner") == []


def test_t6_pinless_archive_lands_but_is_audited_not_silent(
    kanban_home, monkeypatch,
):
    """``archive`` from a pin-less caller: same disposition as ``reclaim`` —
    lands (dashboard/bulk cleanup), SIGTERM kept, displacement recorded."""
    tid, live = _pinless_review_card(monkeypatch)
    sent: list[tuple[int, int]] = []
    with kbc.connect() as conn:
        assert kb.archive_task(
            conn, tid, signal_fn=lambda p, s: sent.append((p, s)),
        ) is True
    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "archived"
    assert sent and sent[0][0] == os.getpid() + 1, sent
    displaced = _events(tid, "archive_displaced_not_owner")
    assert len(displaced) == 1, f"the archive left no displacement record: {displaced}"
    assert json.loads(displaced[0])["via"] == "operator_recovery"
    assert _events(tid, "archive_forced_not_owner") == []


def test_t6_operator_lane_verbs_still_land_on_an_UNOWNED_but_free_card(
    kanban_home, monkeypatch,
):
    """The paired negative that keeps the narrowing honest: the operator lane is
    narrowed only where a LIVE run would be displaced. A pin-less caller still
    schedules / blocks-generically / archives a card with NO live run, and no
    displacement event is invented for it."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    with kbc.connect() as conn:
        free = kb.create_task(conn, title="no live run", assignee="builder")
        assert kb.schedule_task(conn, free, reason="later") is True
        assert _raw_query("SELECT status FROM tasks WHERE id=?", (free,))[0] == "scheduled"
    for kind in ("schedule", "block", "archive"):
        assert _events(free, f"{kind}_displaced_not_owner") == [], kind
        assert _events(free, f"{kind}_refused_not_owner") == [], kind

    with kbc.connect() as conn:
        free2 = kb.create_task(conn, title="no live run either", assignee="builder")
        assert kb.block_task(conn, free2, reason="human", kind="needs_input") is True
        assert _raw_query("SELECT status FROM tasks WHERE id=?", (free2,))[0] == "blocked"


# ── T7: the runbook's LITERAL command must reach the typed park ────────────
#
# The T5 runbook §3 step 3 writes the park as
# ``hermes kanban block <task_id> dependency "<reason>"`` — the kind POSITIONAL.
# ``--kind`` is a separate flag, so the kind word lands in ``reason`` and
# ``--kind`` stays unset. Measured on the untouched round-2 tip
# (``runbook_cmd_probe.py``): that literal command yielded ``kind=None``, so the
# "typed park" emitted the human ``blocked`` event instead of the promised
# ``todo`` / ``dependency_wait``. Pre-existing, and it matters more after the
# narrowing (the pin-less lane is now scoped on the kind), so the CLI accepts the
# documented form. These tests pin BOTH spellings end to end through the real
# parser + handler, because a runbook step must not depend on the operator
# remembering a flag that the runbook never mentions.

def _parse_and_run_block(argv: list[str]):
    """Run ``hermes kanban block ...`` through the REAL parser + handler."""
    import argparse

    from hermes_cli import kanban_parser as kp
    from hermes_cli.kanban import _cmd_block

    root = argparse.ArgumentParser(prog="hermes")
    sub = root.add_subparsers(dest="command")
    parser = kp.build_parser(sub)
    ns = parser.parse_args(["block", *argv])
    return _cmd_block(ns)


def test_t7_runbook_positional_kind_reaches_the_dependency_park(
    kanban_home, monkeypatch,
):
    """The runbook's exact form: ``block <id> dependency "<reason>"``.

    Asserts the TYPED park's own contract — not merely that the verb returned
    ok. Merely returning ok was the pre-existing trap: the kind word landed in
    ``reason`` and ``--kind`` stayed unset, so the park was untyped.

    Hand-merge (t_788c8b56): the CARD here has no parent, so upstream's
    re-kind (42a778ab4b) records it as a sticky ``needs_input`` park in the
    ``blocked`` lane with ``requested_kind``/``rekind_reason`` on the event.
    The typed park is what is asserted — the trail must name the requested
    kind, and the runbook command must be accepted by the real parser.
    """
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="runbook park", assignee="builder")
        kb.claim_task(conn, tid, claimer="host:9999")

    rc = _parse_and_run_block([tid, "dependency", "STOP pivot t_root: resume via promote"])
    assert rc == 0, "the runbook's own command failed"
    # Hand-merge (t_788c8b56): upstream 42a778ab4b re-kinds a parent-less
    # ``dependency`` block to ``needs_input`` so ``recompute_ready`` cannot
    # respawn it context-free. This card carries NO parent, so the park lands
    # in the sticky ``blocked`` lane, not ``todo``. The operator LANE decision
    # is unaffected — it is keyed on the FILED kind (``requested_kind``),
    # which is why the pin-less park still lands at all.
    assert _raw_query("SELECT status, block_kind FROM tasks WHERE id=?",
                      (tid,)) == ("blocked", "needs_input"), (
        "the typed park did not land as the re-kinded needs_input park"
    )
    events = [
        r[0] for r in sqlite3.connect(str(os.environ["HERMES_KANBAN_DB"])).execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,),
        )
    ]
    # Hand-merge (t_788c8b56): with upstream's re-kind the parent-less typed
    # park records a ``blocked`` event that NAMES the requested kind and the
    # re-kind reason — so the typed park is still provable from the trail
    # (it did not fall into the generic human block lane; the payload says
    # what was asked for). ``dependency_wait`` is the with-an-open-parent
    # outcome, exercised by upstream's own control test.
    assert "blocked" in events, f"no park event at all: {events}"
    payloads = [
        json.loads(r[0]) for r in sqlite3.connect(str(os.environ["HERMES_KANBAN_DB"])).execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='blocked' ORDER BY id", (tid,),
        )
    ]
    assert payloads, f"the park left no blocked payload: {payloads}"
    last = payloads[-1]
    assert last.get("requested_kind") == "dependency", last
    assert last.get("rekind_reason") == "no_open_parent", last
    assert last.get("kind") == "needs_input", last


def test_t7_explicit_kind_flag_still_wins(kanban_home, monkeypatch):
    """``--kind needs_input`` must NOT be overridden by a leading kind-looking
    REASON word, or a human block whose prose starts with a kind name would be
    silently re-typed."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="explicit kind", assignee="builder")
        kb.claim_task(conn, tid, claimer="host:9999")

    rc = _parse_and_run_block([tid, "dependency", "--kind", "needs_input"])
    assert rc == 0
    assert _raw_query("SELECT status FROM tasks WHERE id=?", (tid,))[0] == "blocked", (
        "--kind needs_input was overridden by the leading reason word"
    )
