"""#69283: kanban write guard prevents tests from writing to real ~/.hermes."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_guard as kg

# These probe the kanban guard against the real root on purpose. Re-derived onto
# the post-refactor tree: upstream's b897d79ba2 (tests/home_io_guard.py) guards
# every real-home path, so the opt-out marker it added is required here too.
pytestmark = pytest.mark.allow_real_home_io


def test_connect_succeeds_under_test_home(tmp_path, monkeypatch):
    """When HERMES_HOME is a temp dir, kanban connect succeeds normally."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kbc.connect()
    try:
        assert str(kanban_db.kanban_db_path()).startswith(str(home))
    finally:
        conn.close()


def test_connect_raises_when_kanban_home_is_real_root(monkeypatch):
    """When kanban paths resolve to the REAL root, connect raises RuntimeError."""
    import tests.conftest as _conftest

    monkeypatch.setattr(
        kanban_db, "kanban_home", lambda: _conftest._REAL_KANBAN_ROOT
    )
    monkeypatch.setattr(
        kanban_db,
        "kanban_db_path",
        lambda board=None: _conftest._REAL_KANBAN_ROOT / "kanban.db",
    )
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect()


def test_connect_raises_for_explicit_db_path_under_real_root():
    """Explicit db_path pointing under the real root is also refused."""
    import tests.conftest as _conftest

    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect(_conftest._REAL_KANBAN_ROOT / "kanban.db")


def test_env_pin_cannot_escape_hermes_home(tmp_path, monkeypatch):
    """An env pin (HERMES_KANBAN_DB) pointing at a board UNDER the REAL root is
    refused even though HERMES_HOME was redirected to a tempdir — the pin wins
    over HERMES_HOME, so the guard must compare against the real root, not the
    redirected home. This is leak #2 (design card t_d46b036c); the test-side
    ``_kanban_write_guard`` monkeypatch is import-order dependent and would miss
    a write that reaches ``connect()`` before it arms, so the in-code guard must
    block it."""
    import tests.conftest as _conftest

    live_pin = _conftest._REAL_KANBAN_ROOT / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_pin))
    # HERMES_HOME is redirected to a tempdir; the pin still wins.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "sandboxed"))
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kbc.connect()


def test_direct_sqlite_conn_write_refused(tmp_path):
    """A caller-supplied sqlite3.Connection to a board under the REAL root is
    refused at the write boundary (write_txn) — the one shape the test-side
    deny-list cannot catch because ``connect()`` is never called (V5 in the
    design card)."""
    import tests.conftest as _conftest

    conn = sqlite3.connect(str(_conftest._REAL_KANBAN_ROOT / "kanban.db"))
    conn.row_factory = sqlite3.Row
    try:
        with pytest.raises(RuntimeError, match="kanban_write_guard"):
            with kanban_db.write_txn(conn):
                conn.execute(
                    "INSERT INTO tasks (id,title,status,created_at) "
                    "VALUES ('t_guard_invariant','probe','triage',0)"
                )
    finally:
        conn.close()


def test_scratch_under_real_root_still_allowed(tmp_path):
    """A scratch DB under the real root must NOT be refused by the in-code guard
    — a naive \"under ~/.hermes\" deny-list destroys legitimate scratch boards
    (design card V6/V7). The guard's board-SHAPE predicate allows
    ``<root>/kanban/workspaces/<task>/*.db``.

    Note the test-side conftest fixture uses a deliberately broader deny-list
    (defence-in-depth) and would intercept ``kbc.connect(scratch)``, so this
    drives the in-code guard at its unit surface and proves a scratch write is
    allowed through a caller-supplied connection (the shape the in-code guard
    owns). The schema is seeded with a direct execute so neither guard runs."""
    import tests.conftest as _conftest

    scratch = _conftest._REAL_KANBAN_ROOT / "kanban" / "workspaces" / "t_guard" / "scratch.db"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    # Force the test-context signal so the in-code guard is armed for the write.
    os.environ["PYTEST_CURRENT_TEST"] = "probe (call)"
    try:
        # The in-code guard must NOT consider this a live board.
        kg.assert_not_live_board(scratch, op="connect")  # no raise => allowed
        # Seed the schema directly (skips kbc.connect, which the test-side
        # fixture would intercept for a path under the real root).
        seed = sqlite3.connect(str(scratch))
        seed.row_factory = sqlite3.Row
        try:
            seed.executescript(kanban_db.SCHEMA_SQL)
            kbc._migrate_add_optional_columns(seed)
            seed.commit()
        finally:
            seed.close()
        # A real kanban write through a caller-supplied connection must work.
        conn = sqlite3.connect(str(scratch))
        conn.row_factory = sqlite3.Row
        try:
            tid = kanban_db.create_task(conn, title="scratch task")
            assert tid
        finally:
            conn.close()
    finally:
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        for suffix in ("", "-wal", "-shm", "-init.lock"):
            p = Path(str(scratch) + suffix)
            if p.exists():
                p.unlink(missing_ok=True)
        if scratch.parent.exists():
            scratch.parent.rmdir()


def test_sibling_boards_and_profiles_allowed(tmp_path, monkeypatch):
    """A hermetic temp HERMES_HOME and a named board under it are unaffected."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    kanban_db._INITIALIZED_PATHS.clear()
    conn = kbc.connect()  # default board at <home>/kanban.db
    try:
        tid = kanban_db.create_task(conn, title="hermetic task")
        assert kanban_db.get_task(conn, tid) is not None
    finally:
        conn.close()


def test_guard_is_inert_outside_test_context(tmp_path, monkeypatch):
    """No test-context signal => the guard must not fire (production stays
    writable). Narrowly: the group of signals is empty when the marker is
    removed and the guard reports no live board."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.delenv("HERMES_TEST_ISOLATION", raising=False)
    # The in-process guard must not consider this a test context.
    assert kg._in_test_context() is False
