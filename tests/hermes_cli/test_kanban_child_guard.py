"""The eval ``--child`` guard must refuse on the RESOLVED kanban DB path.

Regression for the ``HERMES_KANBAN_DB`` bypass (2026-09-13): the first guard
iteration validated ``HERMES_HOME`` only, but ``kanban_db.kanban_db_path()``
gives ``HERMES_KANBAN_DB`` precedence over ``HERMES_HOME`` — and
``HERMES_KANBAN_DB=<real root>/kanban.db`` is exported into *every* kanban
worker session. So ``HERMES_HOME=<temp>`` plus an inherited
``HERMES_KANBAN_DB`` passed the guard while the child still wrote the live
board (24 probe rows leaked, 3 spawned as real workers).

Cross-platform parity (t_8cc5bff4): the guard's *real-root* side was
``Path.home() / ".hermes"``, while production's
``hermes_constants._get_platform_default_hermes_home()`` answers
``%LOCALAPPDATA%/hermes`` on ``win32``. On Windows the pin therefore never
matched the real root and a live-board child was NOT refused. The win32 cases
below simulate that branch the same way ``tests/test_hermes_constants.py`` does
(patch ``sys.platform`` + ``LOCALAPPDATA``), with ``HOME`` redirected to a
throwaway fake home so even a regression cannot reach the operator's board.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_constants
from hermes_constants import _get_platform_default_hermes_home, get_default_hermes_root

from evals import kanban_child_guard as guard

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_guardmod_{name}", EVALS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """A throwaway OS home, so the guard's real root never leaves the tmpdir."""
    home = tmp_path / "fakehome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    return home


# ── the regression: the case the old guard missed ──────────────────────────

def test_refuses_live_db_pin_under_isolated_hermes_home(fake_home, tmp_path, monkeypatch):
    """HERMES_HOME=<temp> + HERMES_KANBAN_DB=<real root>/kanban.db must refuse."""
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(isolated))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(fake_home / ".hermes" / "kanban.db"))

    reason = guard.refusal_reason(REPO_ROOT)
    assert reason is not None, "isolated HERMES_HOME + live DB pin must be refused"
    assert str(fake_home / ".hermes" / "kanban.db") in reason
    assert "effective kanban DB path" in reason


def test_refuses_kanban_home_pin(fake_home, tmp_path, monkeypatch):
    """The sibling path-pin variable is covered by the same resolved-path rule."""
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(isolated))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(fake_home / ".hermes"))

    reason = guard.refusal_reason(REPO_ROOT)
    assert reason is not None
    assert str(fake_home / ".hermes" / "kanban.db") in reason


def test_refuses_named_board_under_real_root(fake_home, tmp_path, monkeypatch):
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(isolated))
    monkeypatch.setenv(
        "HERMES_KANBAN_DB", str(fake_home / ".hermes" / "kanban" / "boards" / "probe" / "kanban.db")
    )
    assert guard.refusal_reason(REPO_ROOT) is not None


def test_refuses_current_board_file_pointing_at_real_root(fake_home, tmp_path, monkeypatch):
    """Resolution through <root>/kanban/current is also an escape vector.

    A hand-written ``current`` file selects a named board *within the kanban
    root*, so the vector needs the root itself pinned at the real home
    (``HERMES_KANBAN_HOME``) — the resolved path is then what catches it.
    """
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(isolated))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(fake_home / ".hermes"))
    current = fake_home / ".hermes" / "kanban" / "current"
    probe_board = fake_home / ".hermes" / "kanban" / "boards" / "probe"
    probe_board.mkdir(parents=True)
    (probe_board / "board.json").write_text("{}")
    current.write_text("probe\n", encoding="utf-8")

    reason = guard.refusal_reason(REPO_ROOT)
    assert reason is not None
    assert "probe" in reason


# ── the pre-existing conditions must keep refusing ─────────────────────────

def test_refuses_unset_hermes_home(fake_home, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    reason = guard.refusal_reason(REPO_ROOT)
    assert reason is not None
    assert "unset" in reason


def test_refuses_hermes_home_at_real_root(fake_home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(fake_home / ".hermes"))
    reason = guard.refusal_reason(REPO_ROOT)
    assert reason is not None
    assert "inside the real" in reason


def test_refuses_hermes_home_inside_real_root(fake_home, monkeypatch):
    profile_home = fake_home / ".hermes" / "profiles" / "builder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    assert guard.refusal_reason(REPO_ROOT) is not None


# ── the allow path: the wrapper's own child env must still run ─────────────

def test_allows_isolated_home_without_kanban_pins(fake_home, tmp_path, monkeypatch):
    """Exactly the env the parent wrapper hands the child."""
    wrapper_home = tmp_path / "wrapper"
    harness = wrapper_home / "hermes"
    harness.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(harness))

    assert guard.refusal_reason(REPO_ROOT) is None
    assert guard.effective_db_path(REPO_ROOT) == (harness / "kanban.db").resolve()


def test_board_slug_alone_is_not_an_escape_vector(fake_home, tmp_path, monkeypatch):
    """Over-refusal control: HERMES_KANBAN_BOARD does not move the root.

    ``kanban_home()`` follows ``get_default_hermes_root()``, which honours
    ``HERMES_HOME``. A guard that refused on the variable NAME instead of the
    resolved path would refuse this legitimate isolated run.
    """
    harness = tmp_path / "wrapper" / "hermes"
    harness.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(harness))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    assert guard.refusal_reason(REPO_ROOT) is None


def test_require_isolated_child_exits_nonzero_with_message_shape(fake_home, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(fake_home / ".hermes" / "kanban.db"))
    with pytest.raises(SystemExit) as excinfo:
        guard.require_isolated_child(REPO_ROOT)
    message = str(excinfo.value)
    assert message.startswith("refusing --child: ")
    assert message.endswith(guard._REFUSAL_TAIL)


def test_require_isolated_child_returns_none_when_isolated(fake_home, tmp_path, monkeypatch):
    harness = tmp_path / "wrapper" / "hermes"
    harness.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(harness))
    assert guard.require_isolated_child(REPO_ROOT) is None


# ── Windows parity: the real-root side must follow the win32 branch ────────

@pytest.fixture
def fake_win32(fake_home, tmp_path, monkeypatch):
    """Simulate production's win32 branch: ``%LOCALAPPDATA%`` is the native base.

    Same idiom as ``tests/test_hermes_constants.py``'s windows cases —
    ``sys.platform`` and ``LOCALAPPDATA`` patched in-process, with the OS home
    already redirected to the throwaway fake home by ``fake_home``.
    """
    local_appdata = tmp_path / "LocalAppData"
    (local_appdata / "hermes").mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
    return local_appdata


def test_real_root_is_localappdata_hermes_on_win32(fake_win32):
    """The defect: the guard answered ``~/.hermes`` on Windows, never the real root."""
    assert guard._real_hermes_root(REPO_ROOT) == (fake_win32 / "hermes").resolve()
    assert guard._platform_home_replica() == (fake_win32 / "hermes").resolve()


def test_refuses_win32_localappdata_db_pin(fake_win32, tmp_path, monkeypatch):
    """A pin under the Windows platform root must be refused, not silently allowed."""
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    live_db = fake_win32 / "hermes" / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(isolated))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_db))

    reason = guard.refusal_reason(REPO_ROOT)
    assert reason is not None, "win32 live-board pin must be refused (t_8cc5bff4)"
    assert str(live_db) in reason
    assert "effective kanban DB path" in reason


def test_refuses_win32_kanban_home_pin(fake_win32, tmp_path, monkeypatch):
    """``HERMES_KANBAN_HOME`` under the Windows platform root refuses too."""
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(isolated))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(fake_win32 / "hermes"))

    reason = guard.refusal_reason(REPO_ROOT)
    assert reason is not None
    assert str(fake_win32 / "hermes" / "kanban.db") in reason


def test_refuses_win32_unset_hermes_home(fake_win32, monkeypatch):
    """On Windows the unset-home refusal must name the %LOCALAPPDATA% root."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    reason = guard.refusal_reason(REPO_ROOT)
    assert reason is not None
    assert str(fake_win32 / "hermes") in reason


def test_refuses_win32_hermes_home_inside_platform_root(fake_win32, monkeypatch):
    """``HERMES_HOME`` at/under %LOCALAPPDATA%/hermes is the live root, not isolation."""
    monkeypatch.setenv("HERMES_HOME", str(fake_win32 / "hermes" / "profiles" / "builder"))
    assert guard.refusal_reason(REPO_ROOT) is not None


def test_win32_allows_isolated_home_without_pins(fake_win32, tmp_path, monkeypatch):
    """Over-refusal control: the wrapper's own temp harness must still run on Windows."""
    harness = tmp_path / "wrapper" / "hermes"
    harness.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(harness))

    assert guard.refusal_reason(REPO_ROOT) is None
    assert guard.effective_db_path(REPO_ROOT) == (harness / "kanban.db").resolve()


def test_win32_without_localappdata_uses_home_appdata_local(fake_home, monkeypatch):
    """Empty ``LOCALAPPDATA`` falls back to ``~/AppData/Local/hermes`` (production parity)."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    expected = (fake_home / "AppData" / "Local" / "hermes").resolve()

    assert guard._platform_home_replica() == expected
    assert guard._real_hermes_root(REPO_ROOT) == expected


# ── real-root parity with production on BOTH platform branches ─────────────

@pytest.mark.parametrize("win32", [True, False], ids=["win32", "posix"])
def test_real_root_agrees_with_production_on_both_branches(win32, fake_home, tmp_path, monkeypatch):
    """The guard's real root == production's root whenever HERMES_HOME cannot move it.

    ``HERMES_HOME`` unset (and, below, set under the native root) is the only
    regime where ``get_default_hermes_root()`` is the root the guard must match:
    with an *out-of-root* ``HERMES_HOME`` production follows the env by design,
    while the guard deliberately keeps the platform root (see the next test).
    """
    if win32:
        local_appdata = tmp_path / "LocalAppData"
        local_appdata.mkdir(parents=True)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        native = local_appdata / "hermes"
    else:
        monkeypatch.setattr(sys, "platform", "linux")
        native = fake_home / ".hermes"

    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    assert guard._real_hermes_root(REPO_ROOT) == native.resolve()
    assert guard._real_hermes_root(REPO_ROOT) == get_default_hermes_root().resolve()
    assert guard._real_hermes_root(REPO_ROOT) == Path(_get_platform_default_hermes_home()).resolve()

    # HERMES_HOME inside the native root (normal or profile mode): still the root.
    profile = native / "profiles" / "builder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    assert guard._real_hermes_root(REPO_ROOT) == get_default_hermes_root().resolve() == native.resolve()


@pytest.mark.parametrize("win32", [True, False], ids=["win32", "posix"])
def test_real_root_ignores_hermes_home_by_design(win32, fake_home, tmp_path, monkeypatch):
    """An out-of-root HERMES_HOME must NOT move the guard's real root.

    ``get_default_hermes_root()`` follows it (Docker/custom root), but that is
    exactly the variable the leak vectors redirect — and the variable the
    sanctioned parent wrapper sets to its own temp harness, so a
    ``HERMES_HOME``-aware guard would make the wrapper's own child self-refuse.
    """
    if win32:
        local_appdata = tmp_path / "LocalAppData"
        local_appdata.mkdir(parents=True)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        native = local_appdata / "hermes"
    else:
        monkeypatch.setattr(sys, "platform", "linux")
        native = fake_home / ".hermes"

    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(isolated))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)

    assert guard._real_hermes_root(REPO_ROOT) == native.resolve()
    assert get_default_hermes_root().resolve() != native.resolve()  # production follows env


@pytest.mark.parametrize("win32", [True, False], ids=["win32", "posix"])
def test_platform_home_replica_agrees_with_production(win32, fake_home, tmp_path, monkeypatch):
    """Replica vs production for the platform root (the fail-closed fallback path)."""
    if win32:
        local_appdata = tmp_path / "LocalAppData"
        local_appdata.mkdir(parents=True)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
    else:
        monkeypatch.setattr(sys, "platform", "linux")

    production = guard._production_platform_home(REPO_ROOT)
    assert production is not None
    assert guard._platform_home_replica() == production


def test_real_root_falls_back_to_replica_when_import_unavailable(fake_win32, monkeypatch):
    """Fail-closed: an unimportable ``--repo`` still yields the win32 root."""
    monkeypatch.setattr(guard, "_production_platform_home", lambda repo=None: None)
    assert guard._real_hermes_root(REPO_ROOT) == (fake_win32 / "hermes").resolve()


def test_win32_replica_kanban_home_uses_platform_root(fake_win32, tmp_path, monkeypatch):
    """``_kanban_home_replica`` must share the platform-root derivation, not re-literal it."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(guard, "_production_db_path", lambda repo=None: None)

    assert guard._kanban_home_replica() == (fake_win32 / "hermes").resolve()
    assert guard._documented_db_path() == (fake_win32 / "hermes" / "kanban.db").resolve()


# ── resolver parity: production's resolver vs the stdlib replica ───────────

VECTORS = [
    {},
    {"HERMES_KANBAN_DB": "/tmp/x/kanban.db"},
    {"HERMES_KANBAN_DB": "relative-board/kanban.db"},
    {"HERMES_HOME": "/tmp/x/hermes"},
    {"HERMES_HOME": "/tmp/x/hermes", "HERMES_KANBAN_DB": "/tmp/y/kanban.db"},
    {"HERMES_HOME": "/tmp/x/profiles/builder"},
    {"HERMES_KANBAN_HOME": "/tmp/kb"},
    {"HERMES_KANBAN_BOARD": "default"},
    {"HERMES_KANBAN_BOARD": "named-board"},
    {"HERMES_KANBAN_BOARD": "default", "HERMES_HOME": "/tmp/x/hermes"},
    {"HERMES_KANBAN_HOME": "/tmp/kb", "HERMES_KANBAN_BOARD": "named-board"},
]


@pytest.mark.parametrize("vector", VECTORS)
def test_documented_replica_agrees_with_production_resolver(vector, fake_home, monkeypatch):
    for key in ("HERMES_HOME", "HERMES_KANBAN_HOME", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)
    for key, value in vector.items():
        monkeypatch.setenv(key, value)

    production = guard._production_db_path(REPO_ROOT)
    monkeypatch.setattr(guard, "_production_db_path", lambda repo=None: None)
    replica = guard._documented_db_path()
    assert production is not None, vector
    assert replica == production, (vector, production, replica)


def test_documented_replica_agrees_for_existing_named_board(fake_home, monkeypatch):
    """The named-board branch must agree too — only an *existing* board is honoured."""
    root = fake_home / ".hermes"
    board = root / "kanban" / "boards" / "named"
    board.mkdir(parents=True)
    (board / "board.json").write_text("{}")
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "named")

    production = guard._production_db_path(REPO_ROOT)
    monkeypatch.setattr(guard, "_production_db_path", lambda repo=None: None)
    replica = guard._documented_db_path()
    assert production == (board / "kanban.db").resolve()
    assert replica == production


def test_refusal_reason_is_fail_closed_when_resolution_fails(fake_home, tmp_path, monkeypatch):
    harness = tmp_path / "wrapper" / "hermes"
    harness.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(harness))
    monkeypatch.setattr(guard, "_production_db_path", lambda repo=None: None)
    monkeypatch.setattr(guard, "_documented_db_path", lambda: None)
    reason = guard.refusal_reason(REPO_ROOT)
    assert reason is not None
    assert "could not be resolved" in reason


# ── both evals must delegate to the shared guard, not their old string check ─

@pytest.mark.parametrize("eval_name", ["kanban_phase_cascade", "kanban_graph_identity"])
def test_eval_child_path_delegates_to_shared_guard(eval_name):
    """No eval may keep its own copy of the guard logic."""
    source = (EVALS / f"{eval_name}.py").read_text(encoding="utf-8")
    assert "kanban_child_guard" in source, f"{eval_name} must use the shared guard"
    assert "require_isolated_child(" in source, f"{eval_name} must call the shared guard"
    # The superseded env-var-only check must be gone, not merely shadowed: the
    # refusal string and its local variables live in the shared module alone.
    assert "child_home" not in source, f"{eval_name} still holds the env-var-only check"
    assert '"refusing --child' not in source, f"{eval_name} still holds its own refusal string"
    assert "sys.exit(" not in source, f"{eval_name} still exits on its own judgement"


def test_eval_child_entrypoint_actually_refuses_live_db_pin(fake_home, tmp_path):
    """End-to-end: run the real eval child with the live-board pin, expect rc!=0.

    HOME is redirected to the throwaway fake home and the pin points inside it,
    so even a guard regression writes a discarded board — never the operator's.
    """
    script = EVALS / "kanban_graph_identity.py"
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(fake_home),
        "HERMES_HOME": str(isolated),
        "HERMES_KANBAN_DB": str(fake_home / ".hermes" / "kanban.db"),
        "TZ": "UTC",
        "LANG": "C.UTF-8",
    }
    result = subprocess.run(
        [sys.executable, str(script), "--repo", str(REPO_ROOT), "--child"],
        env=env, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "refusing --child" in (result.stdout + result.stderr)


def test_eval_wrapper_still_runs_clean(fake_home, tmp_path):
    """The sanctioned parent wrapper must still exit 0 (its own temp HERMES_HOME)."""
    script = EVALS / "kanban_phase_cascade.py"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(fake_home),
        "TZ": "UTC",
        "LANG": "C.UTF-8",
    }
    result = subprocess.run(
        [sys.executable, str(script), "--repo", str(REPO_ROOT)],
        env=env, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
