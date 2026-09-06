"""Regression test: /branch seeds the new session row from the config default,
NOT the parent's runtime model.

The CLI /branch command previously wrote ``self.model`` (the parent's live
runtime model) into the child row. When the parent was running a delegated /
fallback pin (e.g. deepseek via OpenRouter routing), that pin was baked into
the branch row and survived gateway restarts, silently overriding
config.model.default. Under the fix the branch lands on the ambient config
default (``_resolve_model_default()``), while the ``_branched_from`` fork marker
is preserved for the prompt-cache fence.
"""

import os
from datetime import datetime

import pytest


def _cfg(row):
    """model_config is persisted as a JSON string; decode for assertions."""
    import json

    raw = row.get("model_config")
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


@pytest.fixture
def session_db(tmp_path):
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / ".hermes" / "test_sessions.db")
    yield db
    db.close()


DEEPSEEK = "deepseek-ai/DeepSeek-V4-Flash-0731"
CONFIG_DEFAULT = "z-ai/glm-5.3-flash"
PARENT_ID = "20260403_120000_abc123"


@pytest.fixture
def cli_instance(session_db):
    from unittest.mock import MagicMock

    from cli import HermesCLI

    cli = MagicMock()
    cli._session_db = session_db
    cli.session_id = PARENT_ID
    # Parent was running a transient delegated pin.
    cli.model = DEEPSEEK
    cli.max_turns = 90
    cli.reasoning_config = {"enabled": True, "effort": "medium"}
    cli.session_start = datetime.now()
    cli._pending_title = None
    cli._resumed = False
    cli.agent = None
    cli.conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    # Config default is glm; bind the REAL helper so it reads cli.config.
    cli.config = {"model": {"default": CONFIG_DEFAULT}}
    cli._resolve_model_default = HermesCLI._resolve_model_default.__get__(cli)

    session_db.create_session(session_id=PARENT_ID, source="cli", model=cli.model)
    session_db.set_session_title(PARENT_ID, "Parent Session")
    return cli


def test_branch_seeds_config_default_not_parent_runtime_model(cli_instance, session_db):
    from cli import HermesCLI

    HermesCLI._handle_branch_command(cli_instance, "/branch")

    new_id = cli_instance.session_id
    assert new_id != PARENT_ID
    row = session_db.get_session(new_id)
    assert row is not None
    # The branch lands on the config default, NOT the parent's deepseek pin.
    assert row["model"] == CONFIG_DEFAULT
    # Fork marker preserved so the prompt-cache fence still works.
    assert _cfg(row)["_branched_from"] == PARENT_ID
    assert row["parent_session_id"] == PARENT_ID


def test_branch_child_resume_would_adopt_config_default(cli_instance, session_db):
    """After /branch + restart, a resume of the child row sees the config
    default (no stale deepseek pin), so model_override is empty and the next
    sync adopts config.model.default."""
    from cli import HermesCLI

    HermesCLI._handle_branch_command(cli_instance, "/branch")
    row = session_db.get_session(cli_instance.session_id)
    # No override-derived provider persisted on the child.
    assert _cfg(row).get("provider") is None
    assert _cfg(row).get("model") is None


def test_resolve_model_default_prefers_env_over_config(monkeypatch):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"model": {"default": CONFIG_DEFAULT}}
    monkeypatch.setenv("HERMES_MODEL", "env/model")
    assert HermesCLI._resolve_model_default(cli) == "env/model"


def test_resolve_model_default_uses_config_default():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"model": {"default": CONFIG_DEFAULT}}
    assert HermesCLI._resolve_model_default(cli) == CONFIG_DEFAULT


def test_resolve_model_default_handles_string_model():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"model": "z-ai/glm-5.3-flash"}
    assert HermesCLI._resolve_model_default(cli) == CONFIG_DEFAULT
