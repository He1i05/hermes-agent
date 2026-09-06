"""Prompt-cache regression: the fork-fence markers must survive the Option B fix.

The session-resume model-override fix changes the *model column seeding* of
fork-child rows, but it must NOT touch the ``_branched_from`` / ``_delegate_from``
lineage markers. Those markers are what ``agent/prompt_cache_scope.py`` uses (via
``hermes_state.SessionDB._is_explicit_fork_child_row``) to fence the prompt-cache
key across the fork boundary. If the fix dropped them, sibling branches / delegate
children would share a cache key and cross-contaminate prompt state.

This drives the REAL ``_ensure_session_db_row`` against a REAL SessionDB and
asserts the fence predicate still classifies the rows correctly.
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
    db = SessionDB(db_path=tmp_path / ".hermes" / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: CONFIG_DEFAULT)
    # FK: the parent row must exist before any child can reference it.
    db.create_session("parent", "tui", model=CONFIG_DEFAULT)
    yield db
    db.close()


def test_fork_child_keeps_branched_marker_fence(session_db):
    """A /branch-style fork child still carries ``_branched_from`` after the fix,
    so the prompt-cache fork fence still classifies it as an explicit fork."""
    server._ensure_session_db_row(
        {
            "session_key": "child",
            "source": "tui",
            "parent_session_id": "parent",
            "model_override": {"model": DEEPSEEK, "provider": "openrouter"},
        }
    )
    row = session_db.get_session("child")
    assert row is not None
    assert _cfg(row)["_branched_from"] == "parent"
    assert session_db._is_explicit_fork_child_row(row) is True


def test_subagent_delegate_marker_fence_preserved(session_db):
    """A genuine subagent child's ``_delegate_from`` marker (written by
    delegate_tool, untouched by this fix) still triggers the fork fence."""
    server._ensure_session_db_row(
        {
            "session_key": "subchild",
            "source": "subagent",
            "parent_session_id": "parent",
            "model_override": {"model": DEEPSEEK, "provider": "openrouter"},
        }
    )
    # Simulate delegate_tool stamping _delegate_from on the child row (it does
    # this on _session_init_model_config independently of _ensure_session_db_row).
    session_db.patch_session_model_config(
        "subchild", {"_delegate_from": "parent"}
    )
    row = session_db.get_session("subchild")
    assert row is not None
    assert row["model"] == DEEPSEEK  # delegation chain model preserved
    assert _cfg(row)["_delegate_from"] == "parent"
    assert session_db._is_explicit_fork_child_row(row) is True


def test_plain_session_is_not_a_fork(session_db):
    """A non-fork session with no parent / no marker must NOT be fenced."""
    server._ensure_session_db_row(
        {"session_key": "solo", "source": "tui", "model_override": None}
    )
    row = session_db.get_session("solo")
    assert session_db._is_explicit_fork_child_row(row) is False
