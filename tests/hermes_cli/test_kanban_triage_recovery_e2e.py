"""The exact incident shape, end to end, across the two actors (design t_82610a64 D6 item 8).

One hermetic board, the ``t_7f9693e9`` shape, and both halves of the fix driven
against it:

  board: triage card -> decomposed -> children complete -> breaker fires twice
         -> the card is parked in ``triage`` with a ``decomposed`` marker;
  guard: an unchanged body is REFUSED (no duplicate graph), a no-op ``specify``
         keeps it refused, a real ``edit --body`` opens it and the NEW body's
         children are created, a digest-less legacy marker fails closed;
  predicate: silent while the breaker event is fresh, fires after
         ``TRIAGE_STRANDED_SECONDS`` via ``--now``, and a later ``commented`` /
         ``specified`` event does NOT silence it.

The watchdog script is an operator script that lives outside this checkout
(``~/.hermes/scripts/``); it is located by ``KANBAN_RELEASE_WATCHDOG`` (default
that path) and the predicate assertions SKIP when it is not installed, so the
suite stays green on a machine that does not run the watchdog.

No test touches the live ``kanban.db``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import (
    REFUSAL_ALREADY_FANNED_OUT,
    REFUSAL_LEGACY_DIGESTLESS,
    decompose_triage_task,
    fanout_refusal,
)

WATCHDOG = Path(os.environ.get("KANBAN_RELEASE_WATCHDOG",
                               str(Path.home() / ".hermes" / "scripts" / "kanban-release-watchdog.py")))
TRIAGE_T = 3600  # fixture threshold; the production default is 3 h

needs_watchdog = pytest.mark.skipif(
    not WATCHDOG.is_file(), reason=f"operator watchdog script not installed at {WATCHDOG}",
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return h


@pytest.fixture
def conn(home):
    c = kbc.connect()
    try:
        yield c
    finally:
        c.close()


def _watchdog(home, now, *, state_name="wd_state.json"):
    """One tick of the real script; returns the fresh alert keys ([] == silent).

    ``state_name`` is how a test asks "would a FIRST tick at this clock see it?":
    the script's ``state["seen"]`` suppresses a repeat of the same transition key,
    so two observations that must both report the same key need separate state
    files. (The repeat-suppression itself is pinned by the watchdog suite.)
    """
    state = home / "kanban" / state_name
    reg = home / "kanban" / "release_edges.json"  # absent == no declared edges
    state.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "HERMES_HOME": str(home),
        "WATCHDOG_TRIAGE_SECONDS": str(TRIAGE_T),
        "WATCHDOG_REGISTRY": str(reg),
    })
    env.pop("WATCHDOG_STATE", None)
    proc = subprocess.run(
        [sys.executable, str(WATCHDOG), str(home / "kanban.db"), "--now", str(int(now)),
         "--json", "--state", str(state), "--registry", str(reg)],
        capture_output=True, text=True, env=env, timeout=180, check=False,
    )
    assert proc.returncode == 0, f"watchdog rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert "Traceback" not in proc.stderr, proc.stderr
    if not proc.stdout.strip():
        return []
    return [a["key"] for a in json.loads(proc.stdout)["alerts"]]


def _drive_breaker(conn, tid):
    """claim -> block -> unblock -> claim -> block, i.e. BLOCK_RECURRENCE_LIMIT."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    assert kb.claim_task(conn, tid, claimer="builder") is not None
    kb.block_task(conn, tid, reason="recurring cause", kind="needs_input")
    kb.unblock_task(conn, tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    assert kb.claim_task(conn, tid, claimer="builder") is not None
    kb.block_task(conn, tid, reason="recurring cause", kind="needs_input")
    assert kb.get_task(conn, tid).status == "triage"


def _children(conn, root_id):
    return [r["title"] for r in conn.execute(
        "SELECT p.title FROM task_links l JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? ORDER BY p.title", (root_id,))]


def _incident(conn, body="BODY incident: two documents"):
    """t_7f9693e9: triage -> decomposed -> children done -> breaker -> triage."""
    tid = kb.create_task(conn, title="incident", body=body, assignee="default", triage=True)
    kids = decompose_triage_task(
        conn, tid, root_assignee="default",
        children=[{"title": "child 1", "assignee": "builder"},
                  {"title": "child 2", "assignee": "builder"}],
        author="auto-decomposer",
    )
    assert kids
    for kid in kids:
        kb.complete_task(conn, kid, summary="child done")
    kb.recompute_ready(conn)
    _drive_breaker(conn, tid)
    task = kb.get_task(conn, tid)
    assert task.status == "triage" and task.claim_lock is None
    return tid, kids


@needs_watchdog
def test_incident_shape_end_to_end(conn, home):
    tid, _kids = _incident(conn)
    breaker_ev = conn.execute(
        "SELECT id, created_at FROM task_events WHERE task_id = ? AND kind = 'block_loop_detected' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    assert breaker_ev is not None, "the breaker must leave the predicate its only timestamp"
    fresh_now = int(breaker_ev["created_at"]) + 60

    # 1. Predicate: silent while fresh, fires once after T, keyed on the breaker event.
    assert _watchdog(home, fresh_now) == []
    assert _watchdog(home, int(breaker_ev["created_at"]) + TRIAGE_T + 60) == [
        f"{tid}:{breaker_ev['id']}"
    ]

    # 2. Guard: the unchanged stranded body is refused — no duplicate graph.
    assert fanout_refusal(conn, tid) == REFUSAL_ALREADY_FANNED_OUT
    assert decompose_triage_task(
        conn, tid, root_assignee="default",
        children=[{"title": "duplicate", "assignee": "builder"}],
    ) is None
    assert _children(conn, tid) == ["child 1", "child 2"]

    # 3. A no-op `specify` writes its audit event but does NOT open the guard —
    #    and it does not silence the predicate either.
    original = kb.get_task(conn, tid)
    assert kb.specify_triage_task(
        conn, tid, title=original.title, body=original.body, assignee="default",
        author="operator",
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
    assert fanout_refusal(conn, tid) == REFUSAL_ALREADY_FANNED_OUT
    assert _watchdog(home, int(breaker_ev["created_at"]) + TRIAGE_T + 120,
                     state_name="wd_after_event.json") == [
        f"{tid}:{breaker_ev['id']}"
    ], "a later `specified`/`commented` event must not silence the alert"

    # 4. A real body change opens the guard and fans out the NEW body's children.
    assert kb.edit_task(conn, tid, body="BODY incident v2: one document")
    assert fanout_refusal(conn, tid) is None
    new_kids = decompose_triage_task(
        conn, tid, root_assignee="default",
        children=[{"title": "v2 child", "assignee": "builder"}],
        author="auto-decomposer",
    )
    assert new_kids
    assert "v2 child" in _children(conn, tid)
    digests = [r["payload"] for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'decomposed' ORDER BY id",
        (tid,))]
    assert len(digests) == 2, "one `decomposed` event per fan-out, append-only"
    assert all("body_sha" in json.loads(d) for d in digests)


def test_legacy_digestless_marker_refuses(conn):
    """Migrate by message, not by backfill: an unknowable body stays refused."""
    tid = kb.create_task(conn, title="legacy", body="BODY legacy", assignee="default", triage=True)
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'decomposed', ?, ?)",
        (tid, json.dumps({"child_ids": ["t_x"], "root_assignee": "default"}), int(time.time())),
    )
    conn.commit()
    assert fanout_refusal(conn, tid) == REFUSAL_LEGACY_DIGESTLESS
    assert decompose_triage_task(
        conn, tid, root_assignee="default",
        children=[{"title": "never", "assignee": "builder"}],
    ) is None
    assert _children(conn, tid) == []
