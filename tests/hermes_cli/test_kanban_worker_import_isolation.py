"""Regression: the kanban worker's install tree must precede cwd on ``sys.path``.

``_default_spawn`` launches the worker with ``cwd=<workspace>`` and
``-m hermes_cli.kanban_worker_client``. ``-m`` prepends the process cwd to
``sys.path``, *ahead of* every ``PYTHONPATH`` entry — including the install-tree
pin ``pin_hermes_tree_on_pythonpath`` applies. On a ``worktree`` workspace the
cwd is a full checkout, so ``hermes_cli`` and ``hermes_bootstrap`` resolved from
the worktree, whose PM install key has no committed dependency generation:
the worker died at boot with

    hermes: no dependency environment is committed for this install

and never reached its first tool call. A workspace containing its own
``hermes_cli/`` package shadowed ours in exactly the same way.

``_worker_argv`` therefore carries ``-P`` (``sys.flags.safe_path``), which drops
the implicit cwd entry while ``cwd`` stays the workspace.

These tests plant decoys in a cwd that mimics that workspace and assert the
install tree still wins. They fail on the pre-fix argv (no ``-P``) and pass with
it, which is the point: the decoys are inert unless cwd is consulted first.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


DECOY_UNIQUE = "DECOY-hermes-cli-shadowed-the-install"


def _make_task(kb, *, workspace_path: str):
    return kb.Task(
        id="t_argv_shadow",
        title="install tree precedes cwd",
        body=None,
        assignee="w",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="worktree",
        workspace_path=workspace_path,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=1,
    )


def _isolate_from_host_payload_manifest(monkeypatch):
    """Keep the spawn's PATH build off the host's real hermes home.

    This checkout is deployed directly inside a hermes home
    (``~/.hermes/hermes-agent``), so ``pm.environments._payload_manifest`` finds a
    ``manifest.json`` beside it and reads it while ``build_subprocess_env`` resolves
    the hermes bin dir — I/O the suite's home guard (``tests/home_io_guard.py``)
    refuses by design. CI checkouts sit beside no payload manifest, so "not a sealed
    payload" is the same answer there; it is irrelevant to sys.path ordering, which
    is what these tests are about.
    """
    from pm import environments as pm_env

    monkeypatch.setattr(pm_env, "_payload_manifest", lambda root: None)


def _capture_spawn(monkeypatch, tmp_path, workspace):
    """Run ``_default_spawn`` against a fake Popen and return what it would exec.

    Returns ``(argv, env, cwd)`` exactly as handed to ``subprocess.Popen`` — the
    real production shape, not a re-derivation of it.
    """
    _isolate_from_host_payload_manifest(monkeypatch)
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        log = kwargs.get("stdout")
        if log is not None and hasattr(log, "close"):
            log.close()
        return FakeProc()

    # Scoped: the tests below spawn real interpreters through subprocess themselves.
    with monkeypatch.context() as scoped:
        scoped.setattr(subprocess, "Popen", fake_popen)
        kbd._default_spawn(_make_task(kb, workspace_path=str(workspace)), str(workspace))
    return captured["cmd"], captured["env"], captured["cwd"]


def _interpreter_tail(argv):
    """Everything after the interpreter: its options, then the module selector.

    ``_restart_safe_worker_argv`` may wrap the command in ``systemd-run ... --``,
    so locate the interpreter instead of assuming index 0.
    """
    try:
        index = argv.index(sys.executable)
    except ValueError:  # a systemd-run wrapper may carry a resolved interpreter path
        candidates = [i for i, part in enumerate(argv) if os.path.basename(part).startswith("python")]
        assert candidates, f"no interpreter in spawn argv: {argv}"
        index = candidates[-1]
    return argv[: index + 1], argv[index + 1 :]


def _with_module(argv, module):
    """The same command, with the worker module swapped for a probe module."""
    head, tail = _interpreter_tail(argv)
    assert tail[-2:] == ["-m", "hermes_cli.kanban_worker_client"], (
        f"unexpected spawn argv shape: {argv}"
    )
    return [*head, *tail[:-2], "-m", module]


def _plant_decoy(workspace):
    """A cwd standing in for a worktree: a `hermes_cli` package, a `hermes_bootstrap` module."""
    pkg = workspace / "hermes_cli"
    pkg.mkdir(parents=True)
    pkg.joinpath("__init__.py").write_text(
        f"print({DECOY_UNIQUE!r})\nraise RuntimeError({DECOY_UNIQUE!r})\n", encoding="utf-8"
    )
    workspace.joinpath("hermes_bootstrap.py").write_text(
        f"print({DECOY_UNIQUE!r})\nraise RuntimeError({DECOY_UNIQUE!r})\n", encoding="utf-8"
    )
    return workspace


def _probe_module(which: str) -> str:
    return textwrap.dedent(
        f"""
        import json, os, sys
        print("PROBE=" + json.dumps({{"which": {which!r}, "cwd": os.getcwd(),
                                     "sys_path": sys.path[:4]}}))
        """
    )


def test_worker_argv_drops_the_implicit_cwd_entry(monkeypatch, tmp_path):
    """``-P`` is in the spawn argv, and ``-I`` is not (it would ignore PYTHONPATH)."""
    workspace = _plant_decoy(tmp_path / "ws")
    argv, env, cwd = _capture_spawn(monkeypatch, tmp_path, workspace)

    _head, tail = _interpreter_tail(argv)
    assert tail[-1] == "hermes_cli.kanban_worker_client", f"spawn argv: {argv}"
    assert tail[-2] == "-m", f"spawn argv: {argv}"
    assert "-P" in tail, f"spawn argv must carry -P: {argv}"
    assert "-I" not in tail, f"-I implies -E and would drop the PYTHONPATH pin: {argv}"
    # The workspace is still the working directory — only sys.path order changed.
    assert cwd == str(workspace)
    # And the install tree is still pinned ahead of everything on PYTHONPATH.
    if env.get("PYTHONPATH"):
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(Path(__file__).resolve().parents[2])


def test_spawned_module_resolves_from_install_tree_not_cwd(monkeypatch, tmp_path):
    """The captured argv's own interpreter options must keep cwd off ``sys.path``.

    Runs the real interpreter options from ``_worker_argv`` with a probe module in
    place of the worker: the probe exists twice — once in the decoy cwd, once on
    the PYTHONPATH the dispatcher pins — and the pinned copy must win.
    """
    workspace = _plant_decoy(tmp_path / "ws")
    argv, env, cwd = _capture_spawn(monkeypatch, tmp_path, workspace)

    pinned = tmp_path / "pinned"
    pinned.mkdir()
    pinned.joinpath("zzprobe.py").write_text(_probe_module("install"), encoding="utf-8")
    workspace.joinpath("zzprobe.py").write_text(_probe_module("decoy"), encoding="utf-8")

    probe_env = dict(env)
    probe_env["PYTHONPATH"] = os.pathsep.join([str(pinned), probe_env["PYTHONPATH"]])
    # The real interpreter options from the spawn argv, with only the module swapped.
    probe_argv = _with_module(argv, "zzprobe")

    result = subprocess.run(
        probe_argv, cwd=cwd, env=probe_env, capture_output=True, text=True, timeout=180,
        check=False,
    )
    out = result.stdout + result.stderr
    assert DECOY_UNIQUE not in out, f"cwd shadowed the pinned tree:\n{out}"
    assert result.returncode == 0, out
    payload = json.loads(next(line for line in out.splitlines() if line.startswith("PROBE="))[6:])
    assert payload["which"] == "install", out
    # The process still ran *in* the workspace; only sys.path changed.
    assert os.path.realpath(payload["cwd"]) == os.path.realpath(str(workspace)), out
    assert os.path.realpath(cwd) == os.path.realpath(str(workspace))
    spath = [os.path.realpath(p) for p in payload["sys_path"] if p]
    assert os.path.realpath(str(workspace)) not in spath, f"cwd reached sys.path: {spath}"


def test_worker_client_boots_from_a_worktree_shaped_cwd(monkeypatch, tmp_path):
    """End to end: the real worker module boots from a decoy-worktree cwd.

    The decoys raise on import, so any output carrying their marker means the
    workspace beat the install tree. Reaching the owner client's own refusal
    proves the install's ``hermes_cli`` *and* ``hermes_bootstrap`` (whose
    ``project_root`` is derived from its own file location) were the ones loaded.
    """
    workspace = _plant_decoy(tmp_path / "ws")
    argv, env, cwd = _capture_spawn(monkeypatch, tmp_path, workspace)

    # Boot only: with no HERMES_KANBAN_TASK the client refuses immediately instead
    # of waiting on an owner (OWNER_WAIT_S). The board paths are redirected to a
    # scratch dir so nothing this probe does can touch a live board.
    probe_env = dict(env)
    probe_env.pop("HERMES_KANBAN_TASK", None)
    probe_env.pop("HERMES_KANBAN_RUN_ID", None)
    probe_env.pop("HERMES_KANBAN_CLAIM_LOCK", None)
    probe_env["HERMES_KANBAN_DB"] = str(tmp_path / "probe.db")
    probe_env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(tmp_path / "probe-workspaces")

    result = subprocess.run(
        argv, cwd=cwd, env=probe_env, capture_output=True, text=True, timeout=300, check=False
    )
    out = result.stdout + result.stderr
    assert DECOY_UNIQUE not in out, f"cwd shadowed the install tree:\n{out}"
    assert "no dependency environment is committed" not in out, (
        "the worker resolved the workspace checkout as its project root:\n" + out
    )
    assert "Kanban owner refused" in out, f"the install's worker client never booted:\n{out}"
