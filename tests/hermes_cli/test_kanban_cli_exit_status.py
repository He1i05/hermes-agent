"""Regression coverage for Kanban CLI process exit status propagation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _spawn_timeout() -> int:
    """Ceiling for one spawned ``hermes kanban`` CLI, not a cost.

    A warm launch needs ~11 s here (measured on the loaded 4-core Pi: cold interpreter plus
    the checkout's imports). The hardcoded 30 s expired under 4-way sweep load in
    t_97673f26's round-3 run, before any exit-status assertion could be reached — the same
    slow-host budget class as the descendant/dialogue ceilings. Env-tunable for sweeps that
    run on a busier host; the file's own runner ceiling still bounds a genuine hang.
    """
    return max(1, int(os.environ.get("KANBAN_CLI_SPAWN_TIMEOUT", "120")))


def _run_hermes(home: Path, *args: str, marker: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_HOME"] = str(home)
    # A test-owned HERMES_HOME makes this checkout a borrowing data root, so the CLI would
    # re-sync its dependencies per launch (>60 s measured); the suite's switch must survive
    # the copy for the spawned process (t_238f2d3f).
    env["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    for name in (
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if marker:
        env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"
    else:
        env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=_spawn_timeout(),
    )


def test_delegated_child_kanban_cli_refusal_returns_nonzero_exit_status(tmp_path):
    """A printed Kanban mutation refusal must not look like CLI success."""
    home = tmp_path / "hermes"
    home.mkdir()

    created = _run_hermes(home, "kanban", "create", "exit status probe", "--json")
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]

    refused = _run_hermes(
        home,
        "kanban",
        "comment",
        task_id,
        "must be refused",
        marker=True,
    )

    assert refused.returncode == 1
    assert "delegate_task" in refused.stderr
