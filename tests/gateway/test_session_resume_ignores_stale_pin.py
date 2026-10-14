"""Integration: session-resume model override after a gateway restart.

Regression for the reported bug — a TUI session resumed on the parent's
stale runtime model (deepseek) instead of config.model.default after a
gateway restart.

Root cause (approved Option B): the stale pin is written into the fork-child
row AT BRANCH CREATION time (``tui_gateway/server.py:_ensure_session_db_row``
and the CLI ``/branch``), inherited from the parent's ``model_override``. A
restart merely exposes an already-bad row. Under the fix, fork-child rows are
seeded from the config default, so a restart / resume of that row re-adopts
config.model.default while genuine subagent children keep their delegated pin.

This test drives the REAL ``_ensure_session_db_row`` against a REAL SQLite
SessionDB (no mocks on the DB layer) and simulates a gateway restart by
reopening the store from the same file, then asserting the persisted row is
clean (config default, no stale deepseek pin, fork marker intact).
"""

import os

import pytest

from hermes_state import SessionDB
from tui_gateway import server

DEEPSEEK = "deepseek-ai/DeepSeek-V4-Flash-0731"
CONFIG_DEFAULT = "z-ai/glm-5.3-flash"


def _cfg(row):
    """model_config is persisted as a JSON string; decode for assertions."""
    import json

    raw = row.get("model_config")
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


@pytest.fixture
def session_db(tmp_path, monkeypatch):
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    db_path = tmp_path / ".hermes" / "state.db"
    db = SessionDB(db_path=db_path)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: CONFIG_DEFAULT)
    yield db, db_path
    db.close()


def test_fork_child_row_survives_restart_on_config_default(session_db):
    """A fork-child row written from the bug input (parent + source='tui' +
    stale deepseek override) persists config.model.default — so a gateway
    restart + resume adopts the default, not the stale deepseek pin."""
    db, db_path = session_db
    # FK: the parent row must exist before a child can reference it.
    db.create_session("parent", "tui", model=CONFIG_DEFAULT)

    # Bug input: TUI branch child whose in-flight override carries deepseek.
    server._ensure_session_db_row(
        {
            "session_key": "child",
            "source": "tui",
            "parent_session_id": "parent",
            "model_override": {"model": DEEPSEEK, "provider": "openrouter"},
        }
    )

    # Simulate a gateway restart: reopen the store from the same file.
    db.close()
    db2 = SessionDB(db_path=db_path)
    try:
        row = db2.get_session("child")
        assert row is not None
        # The stale deepseek pin is NOT baked into the row.
        assert row["model"] == CONFIG_DEFAULT
        assert row["model"] != DEEPSEEK
        # No override-derived provider leaked into the persisted model_config.
        assert _cfg(row).get("provider") is None
        assert _cfg(row).get("model") is None
        # Fork marker preserved so the prompt-cache fence survives the restart.
        assert _cfg(row)["_branched_from"] == "parent"
        assert row["parent_session_id"] == "parent"
    finally:
        db2.close()


def test_subagent_child_keeps_delegated_pin_across_restart(session_db):
    """Valid delegation chains are preserved: a genuine subagent child still
    persists the orchestrator-chosen model across a restart."""
    db, db_path = session_db
    # FK: the parent row must exist before a child can reference it.
    db.create_session("parent", "tui", model=CONFIG_DEFAULT)

    server._ensure_session_db_row(
        {
            "session_key": "subchild",
            "source": "subagent",
            "parent_session_id": "parent",
            "model_override": {"model": DEEPSEEK, "provider": "openrouter"},
        }
    )

    db.close()
    db2 = SessionDB(db_path=db_path)
    try:
        row = db2.get_session("subchild")
        assert row is not None
        assert row["model"] == DEEPSEEK
        assert _cfg(row)["model"] == DEEPSEEK
        assert _cfg(row)["provider"] == "openrouter"
        # The delegated pin (deepseek) survived the restart — the delegation
        # chain is intact, unlike the fork-child default-seeding above.
    finally:
        db2.close()
