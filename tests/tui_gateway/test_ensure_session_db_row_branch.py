"""Unit tests: _ensure_session_db_row must seed fork-child rows from the config
default, NOT the parent's runtime model override.

Regression for the session-resume model-override bug (approved Option B):

  * A /branch-style fork child (parent_session_id set, source not in
    {subagent, tool}) previously inherited the parent's runtime model — e.g. a
    delegated OpenRouter deepseek pin — which then survived gateway restarts
    and silently overrode config.model.default.
  * Under the fix the fork child is seeded from ``_resolve_model()`` (the config
    default), while the ``_branched_from`` / ``_delegate_from`` fork marker is
    preserved so the prompt-cache fork fence (agent/prompt_cache_scope.py)
    keeps working.
  * Genuine subagent / tool children still inherit the orchestrator-chosen
    model (delegate_tool stamps _delegate_from on them intentionally) — the
    fix's predicate must NOT trip for them.
"""

import pytest

from tui_gateway import server

DEEPSEEK = "deepseek-ai/DeepSeek-V4-Flash-0731"
CONFIG_DEFAULT = "z-ai/glm-5.3-flash"


def _run(session, monkeypatch):
    created = []

    class _FakeDB:
        def create_session(
            self,
            key,
            source=None,
            model=None,
            model_config=None,
            parent_session_id=None,
            cwd=None,
            profile_name=None,
        ):
            created.append(
                {
                    "key": key,
                    "source": source,
                    "model": model,
                    "model_config": model_config,
                    "parent_session_id": parent_session_id,
                }
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: CONFIG_DEFAULT)
    server._ensure_session_db_row(session)
    assert len(created) == 1
    return created[0]


def test_fork_child_seeds_config_default_not_parent_override(monkeypatch):
    """A /branch-style fork child (parent set, source='tui') with a stale
    deepseek override in flight must land on the config default, and must NOT
    persist the override-derived provider/endpoint."""
    row = _run(
        {
            "session_key": "child",
            "source": "tui",
            "parent_session_id": "parent",
            "model_override": {"model": DEEPSEEK, "provider": "openrouter"},
        },
        monkeypatch,
    )

    assert row["model"] == CONFIG_DEFAULT
    # The override-derived provider/endpoint must be dropped for a fork child.
    assert row["model_config"].get("provider") is None
    assert row["model_config"].get("model") is None
    # Fork marker preserved for the prompt-cache fence.
    assert row["model_config"]["_branched_from"] == "parent"
    assert row["parent_session_id"] == "parent"


def test_fork_child_cli_source_seeds_config_default(monkeypatch):
    """source='cli' fork children behave the same as 'tui' (any non-subagent
    lineage)."""
    row = _run(
        {
            "session_key": "child",
            "source": "cli",
            "parent_session_id": "parent",
            "model_override": {"model": DEEPSEEK, "provider": "openrouter"},
        },
        monkeypatch,
    )
    assert row["model"] == CONFIG_DEFAULT


def test_subagent_child_inherits_override(monkeypatch):
    """Negative: a genuine subagent child must STILL inherit the
    orchestrator-chosen model — the fix's guard must not trip for it."""
    row = _run(
        {
            "session_key": "child",
            "source": "subagent",
            "parent_session_id": "parent",
            "model_override": {"model": DEEPSEEK, "provider": "openrouter"},
        },
        monkeypatch,
    )
    assert row["model"] == DEEPSEEK
    assert row["model_config"]["model"] == DEEPSEEK
    assert row["model_config"]["provider"] == "openrouter"
    # The fork marker is stamped for any parent_session_id (subagent children
    # included) so the prompt-cache fence classifies them correctly.
    assert row["model_config"]["_branched_from"] == "parent"


def test_tool_child_inherits_override(monkeypatch):
    """Negative: source='tool' children also bypass the guard."""
    row = _run(
        {
            "session_key": "child",
            "source": "tool",
            "parent_session_id": "parent",
            "model_override": {"model": DEEPSEEK, "provider": "openrouter"},
        },
        monkeypatch,
    )
    assert row["model"] == DEEPSEEK


def test_plain_session_without_parent_keeps_override(monkeypatch):
    """A non-fork session (no parent_session_id) keeps its explicit override."""
    row = _run(
        {
            "session_key": "solo",
            "source": "tui",
            "model_override": {"model": DEEPSEEK, "provider": "openrouter"},
        },
        monkeypatch,
    )
    assert row["model"] == DEEPSEEK
    assert row["model_config"]["model"] == DEEPSEEK
    assert row["model_config"]["provider"] == "openrouter"
    assert "_branched_from" not in row["model_config"]
