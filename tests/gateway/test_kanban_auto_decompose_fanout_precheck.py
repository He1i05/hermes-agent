"""The auto-decompose tick's fan-out pre-check (design t_82610a64 D6 item 6 / D2).

Before the guard, a triage card whose body had already fanned out was
re-discovered only AFTER ``decompose_task`` had spent an aux LLM call, and it
consumed one of the ``auto_decompose_per_tick`` slots on every 60 s tick,
forever, reporting only at DEBUG. Two things must now hold, and they pull in
opposite directions:

  * a SPENT card costs no slot and no aux call, and says so ONCE at INFO;
  * nothing else changes: a card that is merely not-in-triage at pre-check time
    (or a pre-check that cannot read at all) still reaches ``decompose_task``,
    because a pre-check may save work but may never change behaviour.

Hermetic board in ``tmp_path``; the aux LLM is faked at the module boundary the
tick imports from, exactly as the multiplex secret-scope test does.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import body_sha


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb_mod: ["default"])
    # The INFO-once memo is process-global: start every test with it empty.
    monkeypatch.setattr(kwd, "_FANOUT_REFUSAL_LOGGED", set())
    kb.init_db()
    return h


@pytest.fixture
def conn(home):
    c = kbc.connect()
    try:
        yield c
    finally:
        c.close()


class _FakeDecompose:
    """Records every ``decompose_task`` call; the tick must not make one for a spent card."""

    def __init__(self, triage_ids):
        self._ids = list(triage_ids)
        self.calls: list[str] = []

    def list_triage_ids(self):
        return list(self._ids)

    def decompose_task(self, task_id, author=None):
        self.calls.append(task_id)
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def _tick(fake, per_tick=5):
    import hermes_cli

    real = sys.modules.get("hermes_cli.kanban_decompose")
    sys.modules["hermes_cli.kanban_decompose"] = fake
    old_attr = getattr(hermes_cli, "kanban_decompose", None)
    hermes_cli.kanban_decompose = fake
    try:
        return _dispatcher().auto_decompose_tick(per_tick)
    finally:
        if real is not None:
            sys.modules["hermes_cli.kanban_decompose"] = real
        else:
            sys.modules.pop("hermes_cli.kanban_decompose", None)
        if old_attr is not None:
            hermes_cli.kanban_decompose = old_attr


def _spent_card(conn, title="spent", body="BODY spent"):
    """In triage with a `decomposed` marker whose digest IS the current body."""
    tid = kb.create_task(conn, title=title, body=body, assignee="default", triage=True)
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'decomposed', ?, 0)",
        (tid, json.dumps({"child_ids": ["t_x"], "root_assignee": "default",
                          "body_sha": body_sha(body)})),
    )
    conn.commit()
    return tid


def _legacy_card(conn, title="legacy"):
    tid = kb.create_task(conn, title=title, body="BODY legacy", assignee="default", triage=True)
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'decomposed', ?, 0)",
        (tid, json.dumps({"child_ids": ["t_y"], "root_assignee": "default"})),
    )
    conn.commit()
    return tid


# ── the saving ─────────────────────────────────────────────────────────────
def test_spent_triage_card_costs_no_slot_and_no_aux_call(conn, caplog):
    spent = _spent_card(conn)
    legacy = _legacy_card(conn)
    fresh = kb.create_task(conn, title="fresh", body="BODY fresh", assignee="default", triage=True)
    fake = _FakeDecompose([spent, legacy, fresh])

    with caplog.at_level(logging.INFO, logger=kwd.logger.name):
        assert _tick(fake, per_tick=1) == 1, "the fresh card took the only slot"

    assert fake.calls == [fresh], "a spent/legacy card must not reach decompose_task"
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "refused before the aux call" in text
    assert "not charged to auto_decompose_per_tick" in text
    # One INFO line per card, then DEBUG: the second card is still announced.
    infos = [r for r in caplog.records if r.levelno == logging.INFO
             and "refused before the aux call" in r.getMessage()]
    assert len(infos) == 2, text


def test_refusal_info_is_logged_once_per_card(conn, caplog):
    spent = _spent_card(conn)
    fake = _FakeDecompose([spent])

    with caplog.at_level(logging.INFO, logger=kwd.logger.name):
        assert _tick(fake) == 0
        first = len([r for r in caplog.records if r.levelno == logging.INFO
                     and "refused before the aux call" in r.getMessage()])
        caplog.clear()
        assert _tick(fake) == 0
        second = len([r for r in caplog.records if r.levelno == logging.INFO
                      and "refused before the aux call" in r.getMessage()])

    assert (first, second) == (1, 0), "a spent card is news once, not every 60 s tick"
    assert fake.calls == []


# ── the non-change ────────────────────────────────────────────────────────
def test_ordinary_triage_card_still_reaches_decompose_task(conn):
    """A card that never fanned out is NOT refused — the pre-check only saves work."""
    fresh = kb.create_task(conn, title="fresh", body="BODY fresh", assignee="default", triage=True)
    fake = _FakeDecompose([fresh])
    assert _tick(fake) == 1
    assert fake.calls == [fresh]


def test_not_in_triage_card_still_reaches_decompose_task(conn):
    """The pre-check narrows to SPENT codes only: a stale listing still reports precisely.

    `decompose_task` (not the pre-check) owns the not-in-triage message and the
    aux-call decision for every case other than a spent body — flipping the
    condition back to `refusal is not None` makes this fixture fail.
    """
    moved = kb.create_task(conn, title="moved", body="BODY moved", assignee="default", triage=True)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (moved,))
    fake = _FakeDecompose([moved])
    assert _tick(fake) == 1
    assert fake.calls == [moved]


def test_unreadable_precheck_falls_through(conn, monkeypatch):
    """A pre-check failure must never block a legitimate fan-out."""
    fresh = kb.create_task(conn, title="fresh", body="BODY fresh", assignee="default", triage=True)
    fake = _FakeDecompose([fresh])

    def boom(board=None):
        raise RuntimeError("board unavailable")

    monkeypatch.setattr(kwd._kbc(), "connect", boom)
    assert _tick(fake) == 1
    assert fake.calls == [fresh], "the failure must be silent to the fan-out path"
