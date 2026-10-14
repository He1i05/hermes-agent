"""Guard regression: the live-flagless ``kanban_pr_acceptance_live`` eval must
refuse the ambient live-board pin before it writes anything.

The defect (t_a573d3f4): the eval writes a real SQLite kanban DB, and the temp
``HERMES_HOME`` its own fixture monkeypatches in is NOT isolation —
``kanban_db.kanban_db_path()`` gives ``HERMES_KANBAN_DB`` precedence over
``HERMES_HOME``, and every kanban worker session exports
``HERMES_KANBAN_DB=<real root>/kanban.db``. So the eval wrote the operator's
live board (reproduced: rc=0, 6 rows, no complaint). The in-code
``hermes_cli.kanban_guard`` cannot cover it: ``_in_test_context()`` requires
``PYTEST_CURRENT_TEST`` / ``PYTEST_VERSION`` / ``HERMES_TEST_ISOLATION``, none of
which exist when the eval is launched directly from a worker shell, so it fails
open by design.

Two twins of the leak vector are exercised, because each has a different failure
cost:

* a **fake-root twin** — ``HOME`` redirected into a tmpdir and the pin aimed at
  THAT fake root, so a guard regression writes a throwaway board. This is the
  primary assertion;
* a **live-target twin** — the real ``~/.hermes/kanban.db`` pin, bracketed by
  read-only ``file:...?mode=ro`` snapshots. It runs only AFTER the fake-root twin
  has already proved refusal, and a snapshot mismatch fails the test, so a
  regression is caught rather than silently committed.

Every child is spawned with an explicitly built env that scrubs
``PYTEST_CURRENT_TEST`` / ``PYTEST_VERSION`` / ``HERMES_TEST_ISOLATION`` on
purpose: that is the realistic worker-shell condition, and it guarantees these
tests exercise the eval's OWN guard rather than the library's test-context guard.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SCRIPT = EVALS / "kanban_pr_acceptance_live.py"

#: The real live board this eval must never touch. Derived from the OS home, not
#: from ``HERMES_HOME`` — the variable the leak vectors redirect.
LIVE_DB = (Path.home() / ".hermes" / "kanban.db").resolve()

_TIMEOUT = 300


def _child_env(**overrides: str) -> dict:
    """A realistic worker-shell env: scrubbed of every pytest/guard marker.

    Built from scratch (never ``os.environ.copy()``) so the child cannot inherit
    ``PYTEST_CURRENT_TEST`` / ``PYTEST_VERSION`` / ``HERMES_TEST_ISOLATION`` /
    ``HERMES_KANBAN_GUARD_BYPASS``. Those markers would arm
    ``hermes_cli.kanban_guard`` in the child and let every negative case pass for
    the wrong reason.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TZ": "UTC",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(overrides)
    return env


def _run(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(REPO_ROOT)],
        env=env, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=_TIMEOUT,
    )


def _live_snapshot() -> dict:
    """Read-only snapshot of the live board (count + newest row + id digest)."""
    if not LIVE_DB.exists():
        return {}
    conn = sqlite3.connect(LIVE_DB.as_uri() + "?mode=ro", uri=True)
    try:
        ids = [row[0] for row in conn.execute("SELECT id FROM tasks ORDER BY id")]
        total, newest = conn.execute("SELECT count(*), max(created_at) FROM tasks").fetchone()
    finally:
        conn.close()
    return {
        "ids_sha256": hashlib.sha256(",".join(ids).encode()).hexdigest(),
        "tasks_total": total,
        "max_created_at": newest,
    }


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("_pr_acceptance_live_mod", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ── the regression: the pin must be refused, in both twins ─────────────────

def test_refuses_pinned_board_and_writes_zero_rows(tmp_path, monkeypatch):
    """``HERMES_HOME``=<isolated> + ``HERMES_KANBAN_DB``=<fake root> must refuse.

    ``HOME`` is redirected to a throwaway fake home and the pin points INSIDE it,
    so if this assertion ever stops holding the write lands in a discarded board
    — never the operator's. The fake-root twin runs first; the live-target twin
    below is only reached once refusal is demonstrated, so a regression cannot
    reach the real board from this test.
    """
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    fake_root = fake_home / ".hermes"
    pinned = fake_root / "kanban.db"

    result = _run(_child_env(
        HOME=str(fake_home),
        HERMES_HOME=str(isolated),
        HERMES_KANBAN_DB=str(pinned),
        HERMES_KANBAN_BOARD="default",
    ))
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert "refusing:" in output, output
    assert str(pinned.resolve()) in output, output
    assert "effective kanban DB path" in output, output
    # The eval's own guard fired, not the library's test-context guard.
    assert "kanban_write_guard" not in output, output
    # Zero rows: the pinned DB must not even have been created.
    assert not pinned.exists(), f"the eval wrote {pinned}"

    # ── live-target twin (opt-in by file existence), bracketed read-only ──
    if not LIVE_DB.exists():
        pytest.skip("no live board on this host; fake-root twin already asserted")
    before = _live_snapshot()
    live = _run(_child_env(
        HOME=str(Path.home()),
        HERMES_HOME=str(isolated),
        HERMES_KANBAN_DB=str(LIVE_DB),
        HERMES_KANBAN_BOARD="default",
    ))
    live_output = live.stdout + live.stderr
    assert live.returncode != 0, live_output
    assert "refusing:" in live_output, live_output
    assert str(LIVE_DB) in live_output, live_output
    assert "kanban_write_guard" not in live_output, live_output
    assert _live_snapshot() == before, "the live board changed during the refusal probe"


@pytest.mark.parametrize(
    "vector",
    [
        {"HERMES_KANBAN_HOME": ".hermes"},
        {"HERMES_KANBAN_DB": ".hermes/kanban/boards/probe/kanban.db"},
    ],
    ids=["kanban-home-pin", "named-board-pin"],
)
def test_refuses_sibling_path_pins(tmp_path, vector):
    """The predicate is the RESOLVED path, so sibling pins are closed too.

    A guard that matched the variable NAME would pass these; the shared
    ``refusal_reason`` resolves whatever production would resolve, which is what
    makes ``HERMES_KANBAN_HOME`` and a named-board ``HERMES_KANBAN_DB`` covered
    by construction.
    """
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    env_overrides = {key: str(fake_home / value) for key, value in vector.items()}

    result = _run(_child_env(
        HOME=str(fake_home), HERMES_HOME=str(isolated), **env_overrides,
    ))
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "refusing:" in output, output
    assert str(fake_home / ".hermes") in output, output


def test_refuses_unset_and_real_hermes_home(tmp_path):
    """The pre-existing conditions keep refusing (unset / pointing at the real root)."""
    for overrides in (
        {},  # no HOME, no HERMES_HOME: resolution would fall back to the real root
        {"HOME": str(Path.home()), "HERMES_HOME": str((Path.home() / ".hermes").resolve())},
    ):
        result = _run(_child_env(**overrides))
        output = result.stdout + result.stderr
        assert result.returncode != 0, output
        assert "refusing:" in output, output


# ── the allow path: the sanctioned isolated run must still work ─────────────

def test_isolated_home_runs_and_reports(tmp_path):
    """A provably isolated run exits 0 with the normal JSON report.

    Hand-merge (t_788c8b56): the eval now passes ``summary=`` evidence,
    because upstream 6504b665ba added ``_gate_empty_completion`` (a
    completion with no result/summary is refused) inside the sync range.
    The probe's subject — the PR-acceptance receipt path — is unchanged.

    Over-refusal control: the guard must not block the legitimate invocation.
    """
    home = tmp_path / "wh"
    before = _live_snapshot()
    result = _run(_child_env(HOME=str(home), HERMES_HOME=str(home / "hermes")))
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    report = json.loads(result.stdout)
    assert sorted(report) == ["reports", "requests"]
    assert [row["case"] for row in report["reports"]] == [
        "failure", "cancelled", "timed_out", "success", "CAS-success", "CAS-failure",
    ]
    assert _live_snapshot() == before, "the isolated run touched the live board"


def test_board_slug_alone_is_not_an_escape_vector(tmp_path):
    """``HERMES_KANBAN_BOARD`` pins the SLUG, not the root — must not over-refuse."""
    home = tmp_path / "wh"
    result = _run(_child_env(
        HOME=str(home), HERMES_HOME=str(home / "hermes"), HERMES_KANBAN_BOARD="default",
    ))
    assert result.returncode == 0, result.stdout + result.stderr


# ── the guard must be on the CLI path only, and reuse the shared predicate ──

def test_import_is_inert(tmp_path, monkeypatch):
    """Importing the eval must not arm a refusal or touch a board.

    A module-scope guard would make this module unloadable as a library; the
    check lives inside ``main()`` instead.
    """
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    pinned = fake_home / ".hermes" / "kanban.db"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(isolated))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))

    module = _load_eval_module()  # must not raise SystemExit
    assert callable(module.main)
    assert not pinned.exists()


def test_guard_reuses_the_shared_predicate_and_runs_first():
    """No second copy of the resolution logic; armed before any board access."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "kanban_child_guard" in source
    assert "refusal_reason(" in source
    # Armed before the first board access — `connect()` / `kanban_db` import.
    assert source.index("guard_refusal(root)") < source.index("from hermes_cli import kanban_db as kb")
    # Not at module scope: the guard call lives inside `main()`.
    assert source.index("def main()") < source.index("guard_refusal(root)")
