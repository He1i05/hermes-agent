"""A real bootstrap reader pins its generation before GC can select victims."""
import json
import os
import subprocess
import sys


def test_bootstrap_lease_survives_selection_change(tmp_path, monkeypatch):
    from pm.environments import install_state_dir, runtime_facts_path, site_packages
    from hermes_cli.runtime_state import collect_generations

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    state = install_state_dir(repo)
    for name in ("first", "second", "unused"):
        venv = state / "environments" / name / "venv"
        venv.mkdir(parents=True)
        # pyvenv.cfg first: the layout keys its site-packages path off the recorded version.
        (venv / "pyvenv.cfg").write_text("version = 3.11")
        site_packages(venv).mkdir(parents=True)
        (venv.parent / ".lease-managed").touch()
    legacy = state / "environments" / "old-unleased"
    legacy.mkdir()

    def select(name):
        environment = state / "environments" / name / "venv"
        runtime_facts_path(repo).write_text(json.dumps({"packages": {"venv": {"environment": str(environment)}}}))
        return environment

    first = select("first")
    code = '''
import sys
from pathlib import Path
from pm.environments import activate_dependencies
activate_dependencies(Path(sys.argv[1]))
print("ready", flush=True)
sys.stdin.readline()
'''
    child = subprocess.Popen([sys.executable, "-c", code, str(repo)], env=dict(os.environ),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        select("second")
        collect_generations(repo, min_age_seconds=0)
        assert first.is_dir()
        assert not (state / "environments" / "unused").exists()
        assert legacy.is_dir()
    finally:
        child.communicate("done\n", timeout=15)
    assert child.returncode == 0
    collect_generations(repo, min_age_seconds=0)
    assert not first.exists()
    assert (state / "environments" / "second").is_dir()
    assert legacy.is_dir()


def test_gc_max_age_is_the_operators_way_out_of_a_young_orphan(tmp_path, monkeypatch):
    """`hermes pm gc --max-age 0` reclaims a crash orphan; the launch path keeps its 24h guard.

    A half-built generation is indistinguishable from an orphan until its facts record
    lands, so lowering the guard on the launch path is forbidden (plan W6/C10).
    """
    import argparse

    from hermes_cli.runtime_state import collect_generations
    from pm import cli
    from pm.environments import install_state_dir

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    state = install_state_dir(repo)
    young = state / "environments" / "young" / "venv"
    young.mkdir(parents=True)
    (young / "pyvenv.cfg").write_text("version = 3.11", encoding="utf-8")
    (young.parent / ".lease-managed").touch()

    assert collect_generations(repo) == []  # the launch path refuses a young generation
    assert young.parent.is_dir()

    forwarded = {}

    def record(project, **kwargs):
        forwarded.clear()
        forwarded.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.runtime_state.collect_generations", record)
    monkeypatch.setattr("pm.runtime.collect_runtime_generations", lambda root: [])
    monkeypatch.setattr("pm.install_states.collect_orphan_install_states", lambda root: [])
    monkeypatch.setattr(cli, "_gc_store", lambda store, facts: (0, 0))

    assert cli.cmd_gc(argparse.Namespace(max_age=0.0)) == 0
    assert forwarded == {"min_age_seconds": 0.0}
    assert cli.cmd_gc(argparse.Namespace(max_age=None)) == 0
    assert forwarded == {}  # the default stays collect_generations' own 86400 s
