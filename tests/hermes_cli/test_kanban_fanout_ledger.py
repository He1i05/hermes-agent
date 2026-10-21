"""Content-addressed fan-out guard (design t_82610a64 D1/D4/D6 items 4-5, 9).

The defect shape: a card is decomposed, blocked twice by the unblock-loop breaker,
and lands back in ``triage`` with a ``decomposed`` marker. Two "recovery" designs
were measured and REJECTED before this one:

  * clearing the marker on re-entry re-fans the SAME body — a 2-child root became
    6 board rows with 5 parents and duplicate child titles;
  * an event-keyed guard is wrong in both directions: ``specify_triage_task``
    appends ``specified`` unconditionally (payload NULL on a no-op) and
    ``edit_task`` records ``edited {"fields": ["body"]}`` without comparing bytes,
    so a no-op specify OPENS it and a real ``edit --body`` leaves it SHUT.

What this suite pins is the surviving form: the guard compares
``sha256(tasks.body)`` against the ``body_sha`` values recorded in the card's
``decomposed`` events, so a fan-out happens exactly once per body CONTENT.

Also pinned here (D6 item 9): ``triage`` stays inert — no dispatch lane returns a
triage id, ``recompute_ready`` never promotes one, and the two specifier verbs
keep their signatures. Those three non-changes are load-bearing: they are what
makes the breaker terminate the block loop at all.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_graph import (
    REFUSAL_ALREADY_FANNED_OUT,
    REFUSAL_LEGACY_DIGESTLESS,
    REFUSAL_NOT_IN_TRIAGE,
    body_sha,
    decompose_triage_task,
    fanout_ledger,
    fanout_refusal,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kbc.connect()
    try:
        yield c
    finally:
        c.close()


# ── helpers ────────────────────────────────────────────────────────────────
def _triage(conn, title="rough idea", body="BODY v1", **kw):
    return kb.create_task(conn, title=title, body=body, assignee="default", triage=True, **kw)


def _back_to_triage(conn, tid):
    """Simulate re-entry: the card is stranded in triage again (what the breaker does)."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))


def _decompose(conn, tid, titles=("child a", "child b")):
    return decompose_triage_task(
        conn, tid, root_assignee="default",
        children=[{"title": t, "assignee": "builder"} for t in titles],
        author="test",
    )


def _events(conn, tid, kind):
    return conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (tid, kind),
    ).fetchall()


def _children(conn, root_id):
    return conn.execute(
        "SELECT p.id, p.title FROM task_links l JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? ORDER BY p.title", (root_id,),
    ).fetchall()


def _fanned_out(conn, title="fanned", body="BODY v1"):
    """A card whose current body already fanned out, stranded back in triage."""
    tid = _triage(conn, title=title, body=body)
    kids = _decompose(conn, tid)
    assert kids
    _back_to_triage(conn, tid)
    return tid, kids


# ── the ledger ─────────────────────────────────────────────────────────────
def test_decomposed_event_records_the_body_digest(conn):
    """D6 item 4: the fan-out records body_sha; the ledger is the guard's input."""
    tid = _triage(conn, body="BODY: two documents")
    kids = _decompose(conn, tid)
    assert kids

    events = _events(conn, tid, "decomposed")
    assert len(events) == 1
    payload = json.loads(events[0]["payload"])
    assert payload["child_ids"] == kids
    assert payload["body_sha"] == hashlib.sha256(b"BODY: two documents").hexdigest()
    assert fanout_ledger(conn, tid) == [(events[0]["id"], payload["body_sha"])]


def test_same_body_is_refused_after_reentry_to_triage(conn):
    """The measured duplicate: an unchanged stranded card must NOT fan out again."""
    tid, _kids = _fanned_out(conn)
    before_links = len(_children(conn, tid))

    assert fanout_refusal(conn, tid) == REFUSAL_ALREADY_FANNED_OUT
    assert _decompose(conn, tid, titles=("dup 1", "dup 2")) is None

    assert len(_events(conn, tid, "decomposed")) == 1, "a refusal must not fan out"
    titles = [r["title"] for r in _children(conn, tid)]
    assert len(titles) == before_links and "dup 1" not in titles


def test_noop_specify_does_not_open_the_guard(conn):
    """B1 (round 1): `specified` is appended unconditionally -> content decides."""
    tid, _ = _fanned_out(conn, body="BODY v1")
    # A no-op: identical title and identical body bytes.
    assert kb.specify_triage_task(
        conn, tid, title=kb.get_task(conn, tid).title, body="BODY v1",
        assignee="default", author="operator",
    )
    assert _events(conn, tid, "specified"), "specify still writes its audit event"
    _back_to_triage(conn, tid)
    assert fanout_refusal(conn, tid) == REFUSAL_ALREADY_FANNED_OUT


def test_real_body_edit_opens_the_guard_and_fans_out_once(conn):
    """B2 (round 1): a real `edit --body` must open the guard."""
    tid, _ = _fanned_out(conn, body="BODY v1")
    assert kb.edit_task(conn, tid, body="BODY v2: one document instead")
    assert fanout_refusal(conn, tid) is None
    assert _decompose(conn, tid, titles=("v2 child",))

    titles = [r["title"] for r in _children(conn, tid)]
    assert "v2 child" in titles
    digests = [sha for _, sha in fanout_ledger(conn, tid)]
    assert len(digests) == 2 and len(set(digests)) == 2
    # And the new body is spent too, exactly like the first one.
    _back_to_triage(conn, tid)
    assert fanout_refusal(conn, tid) == REFUSAL_ALREADY_FANNED_OUT


def test_reverting_to_an_already_fanned_body_is_refused(conn):
    """A -> B -> A: exactly-once per body CONTENT, not per transition."""
    tid, _ = _fanned_out(conn, body="BODY A")
    assert kb.edit_task(conn, tid, body="BODY B")
    assert fanout_refusal(conn, tid) is None
    assert _decompose(conn, tid, titles=("B child",))
    _back_to_triage(conn, tid)
    assert kb.edit_task(conn, tid, body="BODY A")
    assert fanout_refusal(conn, tid) == REFUSAL_ALREADY_FANNED_OUT


def test_legacy_digestless_marker_fails_closed(conn):
    """A pre-digest `decomposed` event: the fanned-out body is unknowable."""
    tid = _triage(conn, body="BODY legacy")
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'decomposed', ?, 0)",
        (tid, json.dumps({"child_ids": ["t_x"], "root_assignee": "default"})),
    )
    conn.commit()
    assert fanout_refusal(conn, tid) == REFUSAL_LEGACY_DIGESTLESS
    assert _decompose(conn, tid) is None, "fail closed: no fan-out on an unknowable body"
    # A real body change cannot open a digest that was never recorded either.
    assert kb.edit_task(conn, tid, body="BODY legacy (rewritten)")
    assert fanout_refusal(conn, tid) == REFUSAL_LEGACY_DIGESTLESS


def test_commit_aborts_when_the_body_changed_after_the_prompt(conn):
    """The race: body edited between the decomposer's read and the commit.

    The guard is COMMIT-side and is only as good as the digest the caller
    carries: passing the digest of a body that is no longer there must abort
    with no child and no ``decomposed`` event, so the new body stays fannable
    instead of being pinned forever by a digest nobody's children describe.
    """
    tid = _triage(conn, body="BODY at prompt time")
    assert kb.edit_task(conn, tid, body="BODY edited mid-flight")
    # The digest the decomposer's prompt actually carried no longer matches.
    assert decompose_triage_task(
        conn, tid, root_assignee="default",
        children=[{"title": "stale child", "assignee": "builder"}],
        expected_body_sha=body_sha("BODY at prompt time"),
    ) is None
    assert not _children(conn, tid), "no child may describe a body nobody has"
    assert not _events(conn, tid, "decomposed"), "no digest recorded for an aborted fan-out"
    # The current body is still fannable: pass ITS digest.
    kids = decompose_triage_task(
        conn, tid, root_assignee="default",
        children=[{"title": "current child", "assignee": "builder"}],
        expected_body_sha=body_sha("BODY edited mid-flight"),
    )
    assert kids


def test_matching_expected_digest_does_not_change_the_allow_path(conn):
    """The happy path is untouched: correct digest -> fan-out as before."""
    tid = _triage(conn, body="BODY stable")
    kids = decompose_triage_task(
        conn, tid, root_assignee="default",
        children=[{"title": "child", "assignee": "builder"}],
        expected_body_sha=body_sha("BODY stable"),
    )
    assert kids and kb.get_task(conn, tid).status == "todo"


def test_not_in_triage_and_missing_cards_are_reported(conn):
    tid = _triage(conn, body="BODY")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))
    assert fanout_refusal(conn, tid) == REFUSAL_NOT_IN_TRIAGE
    assert fanout_refusal(conn, "t_does_not_exist") == REFUSAL_NOT_IN_TRIAGE


def test_refusals_never_delete_or_rewrite_the_audit_trail(conn):
    """D6 item 10: the ledger is append-only — no marker clearing, ever."""
    tid, _ = _fanned_out(conn, body="BODY v1")
    before = [(r["id"], r["payload"]) for r in _events(conn, tid, "decomposed")]
    for _ in range(3):
        _back_to_triage(conn, tid)
        assert _decompose(conn, tid, titles=("dup",)) is None
    after = [(r["id"], r["payload"]) for r in _events(conn, tid, "decomposed")]
    assert after == before


# ── D6 item 9: the deliberate NON-changes ──────────────────────────────────
def test_triage_never_appears_in_a_dispatch_lane(conn):
    """`triage` is inert to dispatch — this is what makes the breaker terminate."""
    tid = _triage(conn, body="BODY")
    _breaker(conn, tid)
    assert kb.get_task(conn, tid).status == "triage"
    for status in ("ready", "review"):
        ids = [r["id"] for r in kbd._lane_rows(conn, status)]
        assert tid not in ids, f"triage card leaked into the {status} dispatch lane"


def test_recompute_ready_does_not_promote_a_triage_card(conn):
    """All parents terminal is not enough: triage is outside the scan tuple."""
    parent = kb.create_task(conn, title="parent", assignee="default")
    tid = _triage(conn, body="BODY")
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (parent, tid),
        )
    kb.complete_task(conn, parent, summary="done")
    kb.recompute_ready(conn)
    assert kb.get_task(conn, tid).status == "triage"


def test_specifier_verbs_keep_their_behaviour(conn):
    """`specify_triage_task` / `edit_task` are read-only for this design."""
    tid = _triage(conn, body="BODY")
    assert kb.specify_triage_task(conn, tid, body="BODY new", author="operator")
    # The specifier's own mover: triage -> todo, promoted to ready by the
    # recompute_ready pass it triggers (never touched by this design).
    assert kb.get_task(conn, tid).status in ("todo", "ready")
    assert kb.specify_triage_task(conn, tid, body="BODY new") is False  # not in triage
    assert kb.edit_task(conn, tid, body="BODY edited")
    assert _events(conn, tid, "edited")


# ── the incident shape, end to end ─────────────────────────────────────────
def _breaker(conn, tid):
    """Drive the unblock-loop breaker: claim -> block -> unblock -> claim -> block."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    assert kb.claim_task(conn, tid, claimer="builder") is not None
    kb.block_task(conn, tid, reason="recurring cause", kind="needs_input")
    kb.unblock_task(conn, tid)
    assert kb.claim_task(conn, tid, claimer="builder") is not None
    kb.block_task(conn, tid, reason="recurring cause", kind="needs_input")
    assert kb.get_task(conn, tid).status == "triage"


def test_incident_shape_is_refused_and_visible_not_duplicated(conn):
    """t_7f9693e9 shape: decompose -> breaker -> triage.

    Refused on re-decompose (no duplicate graph), still visible to the watchdog's
    predicate inputs (triage + no claim + a `block_loop_detected` timestamp), and
    recoverable only by changing the body.
    """
    tid = _triage(conn, body="BODY incident")
    kids = _decompose(conn, tid)
    assert kids
    for kid in kids:
        kb.complete_task(conn, kid, summary="child done")
    kb.recompute_ready(conn)
    _breaker(conn, tid)

    task = kb.get_task(conn, tid)
    assert task.status == "triage" and task.claim_lock is None
    breaker_ev = conn.execute(
        "SELECT id, created_at FROM task_events WHERE task_id = ? AND kind = 'block_loop_detected' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    assert breaker_ev is not None, "the breaker must leave the predicate its timestamp"

    assert fanout_refusal(conn, tid) == REFUSAL_ALREADY_FANNED_OUT
    assert _decompose(conn, tid, titles=("duplicate",)) is None
    assert "duplicate" not in [r["title"] for r in _children(conn, tid)]

    assert kb.edit_task(conn, tid, body="BODY incident (revised)")
    assert fanout_refusal(conn, tid) is None
    assert _decompose(conn, tid, titles=("revised child",))
