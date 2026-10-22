"""The recovery defect at the CLI boundary, in a board the test fully owns (t_1a6c571b).

The defect (root card ``t_82e367bf``): a card that is decomposed, then blocks
twice and is parked in ``triage`` by the unblock-loop breaker, is invisible to
every actor -- ``decompose`` refused it with one vague sentence, ``dispatch``
never enumerates ``triage``, and the release watchdog only scanned ``blocked``.
``t_7f9693e9`` sat that way for 11 h.

What the two sibling suites already pin, and this file does NOT re-assert:

* ``tests/hermes_cli/test_kanban_fanout_ledger.py`` -- the guard's semantics at
  the function level (digest ledger, legacy fail-closed, body-changed abort);
* ``tests/hermes_cli/test_kanban_triage_recovery_e2e.py`` -- the incident shape
  in-process, against the real watchdog script via ``--now``;
* ``~/.hermes/scripts/tests/test_kanban_release_watchdog_triage.py`` -- the
  predicate's conjuncts on a minimal hand-built board.

What is only observable through the verbs an operator (or cron) actually runs,
and is therefore pinned here:

  1. ``hermes kanban decompose <id>`` on a spent, stranded card refuses with the
     per-code operator text AND never pays for an ``auxiliary.kanban_decomposer``
     call (the pre-check runs before the LLM call);
  2. a digest-less legacy marker renders its own code ("fail closed"), not the
     generic "already decomposed or moved out of triage";
  3. ``hermes kanban edit <id> --body ...`` then ``decompose`` fans out the NEW
     body exactly once -- and a re-entry into ``triage`` of that same new body is
     refused again, with no aux call and no duplicate children;
  4. ``hermes kanban dispatch --dry-run`` does not see the triage card at all,
     while the SAME card is spawnable once ``hermes kanban specify`` promotes it
     out of ``triage`` -- the recovery path, end to end;
  5. the watchdog run exactly as cron runs it (no ``--now``, no flags) alerts on
     a breaker episode older than the production 3 h threshold, once, and is
     silent on the next tick.

Hermeticity contract. Every non-pytest process here is started with a fully
specified environment (the ``env -i`` shape): only ``PATH``/``HOME``/
``HERMES_HOME``/``TMPDIR``/``PYTHONPATH``/locale/``TZ``/hash-seed, with
``HOME`` and ``HERMES_HOME`` under pytest's ``tmp_path``. No inherited
``HERMES_KANBAN_*``, no ``HERMES_DELEGATED_CHILD_CONTEXT``, no credentials, so a
stray aux call cannot silently reach a real provider. The live
``~/.hermes/kanban.db`` is never opened; the board under test is asserted to live
under ``tmp_path``.

The only doubled boundary is the network: the CLI driver replaces
``agent.auxiliary_client.call_llm`` with a stub that records each call and returns
a canned reply, so a fan-out can be driven through the real verb without an API
key. Argparse, the handlers, the DB layer, the commit-time digest guard and the
watchdog are all the real code -- and the driver asserts that the imported
``hermes_cli`` is this checkout's, not an installed copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

WATCHDOG = Path(os.environ.get(
    "KANBAN_RELEASE_WATCHDOG",
    str(Path.home() / ".hermes" / "scripts" / "kanban-release-watchdog.py"),
))
needs_watchdog = pytest.mark.skipif(
    not WATCHDOG.is_file(), reason=f"operator watchdog script not installed at {WATCHDOG}",
)

BODY = "BODY incident: two documents"
BODY_V2 = "BODY incident v2: one document"


# ── the helper scripts the subprocesses run ────────────────────────────────
# Kept as source strings so the whole harness is one reviewable file.

_CLI_DRIVER = '''\
"""Run the real ``hermes`` CLI; double only ``call_llm`` when a reply is given."""
import json
import os
import sys

hit_path = os.environ.pop("KANBAN_TEST_AUX_HIT", "")
reply_path = os.environ.pop("KANBAN_TEST_AUX_REPLY", "")
if reply_path:
    from types import SimpleNamespace

    import agent.auxiliary_client as _aux

    def call_llm(*args, **kwargs):
        if hit_path:
            with open(hit_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"task": kwargs.get("task")}) + "\\n")
        with open(reply_path, encoding="utf-8") as fh:
            return SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content=fh.read())),
            ])

    _aux.call_llm = call_llm

from hermes_cli.main import main

sys.argv = ["hermes"] + sys.argv[1:]
sys.exit(main())
'''

_SETUP = '''\
"""Build the t_7f9693e9 shape (or the legacy-marker shape) on a fresh board."""
import hashlib
import json
import sys

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose
from hermes_cli import kanban_db_graph
from hermes_cli.kanban_db_graph import decompose_triage_task

BODY = "BODY incident: two documents"


def drive_breaker(conn, tid):
    """claim -> block -> unblock -> claim -> block: BLOCK_RECURRENCE_LIMIT."""
    for _ in range(2):
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        assert kb.claim_task(conn, tid, claimer="builder") is not None
        kb.block_task(conn, tid, reason="recurring cause", kind="needs_input")
        if kb.get_task(conn, tid).status == "triage":
            return
        kb.unblock_task(conn, tid)
    raise SystemExit("the breaker did not park the card in triage")


def legacy(conn):
    import time
    tid = kb.create_task(conn, title="legacy", body="BODY legacy", assignee="default", triage=True)
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'decomposed', ?, ?)",
        (tid, json.dumps({"child_ids": ["t_x"], "root_assignee": "default"}), int(time.time())),
    )
    conn.commit()
    return {"task_id": tid, "legacy": True}


def incident(conn):
    tid = kb.create_task(conn, title="incident", body=BODY, assignee="default", triage=True)
    kids = decompose_triage_task(
        conn, tid, root_assignee="default",
        children=[{"title": "child 1", "assignee": "default"},
                  {"title": "child 2", "assignee": "default"}],
        author="auto-decomposer",
    )
    assert kids, "fixture fan-out failed"
    for kid in kids:
        kb.complete_task(conn, kid, summary="child done")
    kb.recompute_ready(conn)
    drive_breaker(conn, tid)
    task = kb.get_task(conn, tid)
    assert task.status == "triage" and task.claim_lock is None, task.status
    return {"task_id": tid, "child_ids": list(kids),
            "children": {"child 1": "done", "child 2": "done"}}


def main(argv):
    kb.init_db()
    with kbc.connect_closing() as conn:
        payload = legacy(conn) if "--legacy" in argv else incident(conn)
    payload["paths"] = {
        "hermes_cli": kanban_decompose.__file__,
        "kanban_db_graph": kanban_db_graph.__file__,
    }
    # The digest of the body that fanned out, so a test can prove the ledger is
    # content-addressed rather than a boolean marker.
    payload["body_sha"] = hashlib.sha256(BODY.encode()).hexdigest()
    print(json.dumps(payload))


if __name__ == "__main__":
    main(sys.argv[1:])
'''

_REENTER = '''\
"""Put a card back into ``triage`` -- the re-entry every test here is about."""
import json
import sqlite3
import sys

db, tid = sys.argv[1], sys.argv[2]
conn = sqlite3.connect(db)
conn.execute("UPDATE tasks SET status = 'triage', claim_lock = NULL WHERE id = ?", (tid,))
conn.commit()
print(json.dumps({"task_id": tid, "status": "triage"}))
'''

_BACKDATE = '''\
"""Age the newest ``block_loop_detected`` event by N seconds.

The cron tick takes its clock from ``time.time()`` (no ``--now``), so the breaker
episode is aged where the predicate reads it.
"""
import json
import sqlite3
import sys

db, seconds = sys.argv[1], int(sys.argv[2])
conn = sqlite3.connect(db)
row = conn.execute(
    "SELECT id, task_id FROM task_events WHERE kind = 'block_loop_detected' "
    "ORDER BY id DESC LIMIT 1"
).fetchone()
assert row, "fixture has no breaker event"
conn.execute("UPDATE task_events SET created_at = created_at - ? WHERE id = ?", (seconds, row[0]))
conn.commit()
print(json.dumps({"event_id": row[0], "task_id": row[1], "aged_seconds": seconds}))
'''


# ── harness ────────────────────────────────────────────────────────────────
class Board:
    """A hermetic ``HERMES_HOME`` plus the scripts a test drives it with."""

    def __init__(self, root: Path):
        self.root = root
        self.home = root / "home"
        self.home.mkdir(parents=True, exist_ok=True)
        # ``HOME`` and ``HERMES_HOME`` are DELIBERATELY disjoint. The kanban write
        # guard (`hermes_cli.kanban_guard`) reads the platform root from
        # `os.path.expanduser("~")`, i.e. from ``HOME``: pointing ``HERMES_HOME`` at
        # ``<HOME>/.hermes`` would make this fixture board indistinguishable from
        # the production board and the guard would (correctly) refuse every write
        # from a test-context process.
        self.user_home = root / "user-home"
        self.user_home.mkdir(parents=True, exist_ok=True)
        self.db = self.home / "kanban.db"
        self.scripts = {}
        for name, source in (("cli.py", _CLI_DRIVER), ("setup.py", _SETUP),
                             ("reenter.py", _REENTER), ("backdate.py", _BACKDATE)):
            path = root / name
            path.write_text(source, encoding="utf-8")
            self.scripts[name] = path

    # The ``env -i`` shape: nothing inherited, everything a child needs named
    # here. No HERMES_KANBAN_*, no HERMES_DELEGATED_CHILD_CONTEXT, no secrets.
    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.user_home),
            "HERMES_HOME": str(self.home),
            "TMPDIR": str(self.home),
            "PYTHONPATH": str(REPO_ROOT),
            "LANG": "C.UTF-8",
            "TZ": "UTC",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env.update(extra)
        return env

    def run(self, script: str, *args, env_extra: dict | None = None,
            timeout: int = 240) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            [sys.executable, str(self.scripts[script]), *[str(a) for a in args]],
            env=self.env(**(env_extra or {})),
            cwd=str(self.user_home), capture_output=True, text=True, timeout=timeout,
        )
        return proc

    def cli(self, *argv, aux_reply: str | None = None, **kw) -> subprocess.CompletedProcess:
        extra = {}
        if aux_reply is not None:
            reply = self.root / f"reply_{len(list(self.root.glob('reply_*.json')))}.json"
            reply.write_text(aux_reply, encoding="utf-8")
            extra["KANBAN_TEST_AUX_REPLY"] = str(reply)
            extra["KANBAN_TEST_AUX_HIT"] = str(self.hits)
        return self.run("cli.py", "kanban", *argv, env_extra=extra, **kw)

    @property
    def hits(self) -> Path:
        return self.root / "aux_hits.log"

    def aux_calls(self) -> list[dict]:
        if not self.hits.exists():
            return []
        return [json.loads(line) for line in self.hits.read_text(encoding="utf-8").splitlines() if line]

    def setup(self, *args) -> dict:
        proc = self.run("setup.py", *args)
        assert proc.returncode == 0, f"fixture failed\n{proc.stdout}\n{proc.stderr}"
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def sql(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(query, params).fetchall()
        finally:
            conn.close()

    def task(self, tid: str) -> sqlite3.Row:
        return self.sql("SELECT * FROM tasks WHERE id = ?", (tid,))[0]

    def children(self, tid: str) -> list[str]:
        return [r["title"] for r in self.sql(
            "SELECT p.title FROM task_links l JOIN tasks p ON p.id = l.parent_id "
            "WHERE l.child_id = ? ORDER BY p.title", (tid,))]

    def events(self, tid: str, kind: str) -> list[sqlite3.Row]:
        return self.sql("SELECT * FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
                        (tid, kind))


@pytest.fixture
def board(tmp_path) -> Board:
    return Board(tmp_path)


# ── 1. the fixture really is the incident shape ────────────────────────────
def test_hermetic_board_reproduces_the_incident_shape(board):
    """The shape, plus the two harness invariants every later assertion needs."""
    fixture = board.setup()
    tid = fixture["task_id"]

    # The board is under tmp_path, never the live one.
    assert str(board.db).startswith(str(board.root))
    assert board.db.is_file()
    assert board.db != Path(os.path.expanduser("~")) / ".hermes" / "kanban.db"

    # The child process imported THIS checkout: the fix must be the code under test.
    for name, path in fixture["paths"].items():
        assert Path(path).resolve().is_relative_to(REPO_ROOT), (name, path)

    # The env -i contract: nothing kanban-shaped, nothing inherited.
    env = board.env()
    assert not [k for k in env if k.startswith("HERMES_KANBAN_")]
    assert "HERMES_DELEGATED_CHILD_CONTEXT" not in env
    assert env["HERMES_HOME"].startswith(str(board.root))
    # HOME and HERMES_HOME must stay disjoint or the write guard treats this
    # fixture board as the production one and refuses every write.
    assert env["HOME"] != str(Path(env["HERMES_HOME"]).parent)

    row = board.task(tid)
    assert row["status"] == "triage", "the breaker must park the card in triage"
    assert row["claim_lock"] is None
    assert (row["block_kind"], row["block_recurrences"]) == ("needs_input", 2)

    # One fan-out, content-addressed: the ledger carries the body's digest.
    decomposed = board.events(tid, "decomposed")
    assert len(decomposed) == 1
    assert json.loads(decomposed[0]["payload"])["body_sha"] == fixture["body_sha"]
    assert board.children(tid) == ["child 1", "child 2"]
    assert {r["status"] for r in board.sql(
        "SELECT status FROM tasks WHERE id IN (%s)" % ",".join("?" * len(fixture["child_ids"])),
        tuple(fixture["child_ids"]))} == {"done"}

    # The breaker left exactly the timestamp the watchdog predicate keys on.
    breaker = board.events(tid, "block_loop_detected")
    assert len(breaker) == 1
    assert json.loads(breaker[0]["payload"])["recurrences"] == 2


# ── 2. decompose: refused before the aux call, with per-code text ──────────
def test_cli_decompose_refuses_the_stranded_card_before_paying_for_aux(board):
    fixture = board.setup()
    tid = fixture["task_id"]

    proc = board.cli("decompose", tid,
                     aux_reply='{"fanout": true, "tasks": [{"title": "duplicate"}]}')

    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    text = proc.stderr
    assert "already fanned out for this body" in text
    # The operator text must name the ONLY actions that can recover this card.
    assert f"hermes kanban edit {tid} --body" in text
    assert "archive + re-create" in text
    assert "already decomposed or moved out of triage" not in text

    assert board.aux_calls() == [], "the pre-check must run BEFORE the aux call"
    assert board.children(tid) == ["child 1", "child 2"]
    assert len(board.events(tid, "decomposed")) == 1
    assert board.task(tid)["status"] == "triage"


def test_cli_decompose_refuses_a_legacy_digestless_marker(board):
    """Migration by message, never by backfill: an unknowable body fails closed."""
    fixture = board.setup("--legacy")
    tid = fixture["task_id"]

    proc = board.cli("decompose", tid,
                     aux_reply='{"fanout": true, "tasks": [{"title": "never"}]}')

    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "predates body digests" in proc.stderr
    assert "fail closed" in proc.stderr
    assert "Archive + re-create" in proc.stderr
    assert board.aux_calls() == []
    assert board.children(tid) == []


# ── 3. recovery path: a new body fans out exactly once ─────────────────────
def test_cli_edit_body_fans_out_once_and_a_re_entry_refuses_again(board):
    fixture = board.setup()
    tid = fixture["task_id"]

    edit = board.cli("edit", tid, "--body", BODY_V2)
    assert edit.returncode == 0, (edit.stdout, edit.stderr)

    fan = board.cli("decompose", tid, aux_reply=json.dumps({
        "fanout": True,
        "tasks": [{"title": "v2 child a", "assignee": "default"},
                  {"title": "v2 child b", "assignee": "default"}],
    }))
    assert fan.returncode == 0, (fan.stdout, fan.stderr)
    assert "Decomposed" in fan.stdout and "2 children" in fan.stdout
    assert [c["task"] for c in board.aux_calls()] == ["kanban_decomposer"], \
        "exactly one aux call for the new body"
    assert board.children(tid) == ["child 1", "child 2", "v2 child a", "v2 child b"]

    decomposed = board.events(tid, "decomposed")
    assert len(decomposed) == 2, "one `decomposed` event per fan-out, append-only"
    digests = [json.loads(ev["payload"])["body_sha"] for ev in decomposed]
    assert len(set(digests)) == 2, "the ledger is keyed on body content, not a boolean"
    assert hashlib.sha256(BODY_V2.encode()).hexdigest() in digests

    # The card re-enters triage (a second breaker episode, or an operator park):
    # the SAME new body must be refused, and nothing may be paid for or written.
    reenter = board.run("reenter.py", board.db, tid)
    assert reenter.returncode == 0, reenter.stderr
    again = board.cli("decompose", tid, aux_reply=json.dumps({
        "fanout": True, "tasks": [{"title": "duplicate v2"}]}))
    assert again.returncode == 1, (again.stdout, again.stderr)
    assert "already fanned out for this body" in again.stderr
    assert len(board.aux_calls()) == 1, "the re-entry must not reach the aux client"
    assert board.children(tid) == ["child 1", "child 2", "v2 child a", "v2 child b"]
    assert len(board.events(tid, "decomposed")) == 2


# ── 4. dispatch: blind to triage, spawnable once specified ────────────────
def test_cli_dispatch_is_blind_to_triage_and_spawns_the_specified_card(board):
    fixture = board.setup()
    tid = fixture["task_id"]

    before = board.cli("dispatch", "--dry-run", "--json")
    assert before.returncode == 0, (before.stdout, before.stderr)
    first = json.loads(before.stdout)
    assert first["spawned"] == [], "triage is inert: no lane may claim it"
    assert first["skipped_nonspawnable"] == [], "triage is not even enumerated"
    row = board.task(tid)
    assert (row["status"], row["claim_lock"]) == ("triage", None)

    # The recovery path an operator actually has: specify -> promote out of triage.
    spec = board.cli("specify", tid, aux_reply=json.dumps({
        "title": "incident (specified)",
        "body": "BODY incident v3: one document, one reviewer",
    }))
    assert spec.returncode == 0, (spec.stdout, spec.stderr)
    assert "Specified" in spec.stdout
    assert [c["task"] for c in board.aux_calls()] == ["triage_specifier"]
    row = board.task(tid)
    # `specify` promotes out of the inert column; with every dependency already
    # terminal the promotion lands in ready (todo would mean parents still open).
    assert row["status"] in {"todo", "ready"}, "specify promotes out of the inert column"
    assert row["body"] != BODY, "the spec rewrote the body, which is what opens the guard"

    after = board.cli("dispatch", "--dry-run", "--json")
    assert after.returncode == 0, (after.stdout, after.stderr)
    second = json.loads(after.stdout)
    assert [s["task_id"] for s in second["spawned"]] == [tid], \
        "the same card is dispatchable the moment it leaves triage"
    row = board.task(tid)
    assert row["status"] == "ready"
    assert row["claim_lock"] is None, "--dry-run must not spawn or claim anything"


# ── 5. the watchdog, exactly as cron runs it (no --now, no flags) ─────────
@needs_watchdog
def test_cron_tick_alerts_on_a_stranded_card_and_dedupes(board):
    fixture = board.setup()
    tid = fixture["task_id"]

    def cron_tick() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(WATCHDOG)],
            env=board.env(), cwd=str(board.home),
            capture_output=True, text=True, timeout=240,
        )

    def digest() -> str:
        return hashlib.sha256(board.db.read_bytes()).hexdigest()

    fresh = cron_tick()
    assert fresh.returncode == 0, fresh.stderr
    assert fresh.stdout.strip() == "", "a fresh breaker episode must be silent"

    aged = board.run("backdate.py", board.db, 4 * 3600)
    assert aged.returncode == 0, aged.stderr
    before = digest()

    fired = cron_tick()
    assert fired.returncode == 0, fired.stderr
    assert "Traceback" not in fired.stderr
    assert "KANBAN-WATCHDOG TRIAGE-STRANDED" in fired.stdout
    assert tid in fired.stdout
    assert "no live claim" in fired.stdout
    assert "notify only" in fired.stdout
    assert "archive + re-create" in fired.stdout
    assert f"hermes kanban edit {tid}" in fired.stdout

    again = cron_tick()
    assert again.returncode == 0, again.stderr
    assert again.stdout.strip() == "", "one alert per breaker episode"
    assert digest() == before, "the tick must not write to the board"
    row = board.task(tid)
    assert (row["status"], row["claim_lock"]) == ("triage", None)
