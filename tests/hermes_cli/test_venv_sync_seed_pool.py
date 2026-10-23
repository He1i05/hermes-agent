"""Seed-first borrowed launches and the borrower-only publish hook (plan W4/W5).

The pool is a launch-path optimization: a hit must skip the build entirely, a miss must
leave today's path byte-for-byte, and the owner root must never touch pool code (IV4).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

import pm
from hermes_cli import _launchers, venv_sync
from pm import install


class _StubLock:
    """The launch lock is not what these tests are about; record acquisition instead."""

    def __init__(self, **kwargs) -> None:
        self.acquired = False

    def acquire(self) -> bool:
        self.acquired = True
        return True

    def acquire_checkout(self, _root) -> bool:
        return True

    def release(self) -> None:
        return None


@pytest.fixture
def borrowed(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'example'\nversion = '0.0.0'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "install-stamp.json").write_text('{"updateMechanism": "self"}', encoding="utf-8")
    owner = tmp_path / "owner"
    owner.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(install.POOL_DISABLE_ENV, raising=False)
    store_python = tmp_path / "store" / "cpython-3.14" / "bin" / "python3"
    store_python.parent.mkdir(parents=True)
    store_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _root: store_python)
    monkeypatch.setattr(venv_sync, "collect_superseded_generations", lambda _root: None)
    return root, owner, store_python


def test_seed_hit_skips_the_build_and_relaunches(borrowed, monkeypatch):
    root, owner, store_python = borrowed
    seeded, synced = [], []
    monkeypatch.setattr(install, "seed_from_generation_pool",
                        lambda project_root, **kwargs: seeded.append((project_root, kwargs)) or True)
    monkeypatch.setattr(venv_sync, "_sync_source_dependencies", lambda *args, **kwargs: synced.append(kwargs))

    assert venv_sync._prepare_borrowed_launch(root, owner, current=False) == store_python
    assert seeded == [(root, {"owner": owner})]
    assert synced == []  # the whole point: no build, no uv


def test_pool_miss_runs_todays_sync_path(borrowed, monkeypatch):
    root, owner, store_python = borrowed
    from hermes_cli import update_lock

    monkeypatch.setattr(update_lock, "UpdateLock", _StubLock)
    monkeypatch.setattr(install, "seed_from_generation_pool", lambda *args, **kwargs: False)
    synced = []
    monkeypatch.setattr(venv_sync, "_sync_source_dependencies",
                        lambda *args, **kwargs: synced.append(kwargs))
    monkeypatch.setattr(pm, "venv_is_current", lambda **kwargs: True)

    assert venv_sync._prepare_borrowed_launch(root, owner, current=False) == store_python
    assert synced == [{"arm": False, "borrowed_from": owner}]


def test_seed_failure_never_blocks_the_launch(borrowed, monkeypatch):
    root, owner, store_python = borrowed
    from hermes_cli import update_lock

    monkeypatch.setattr(update_lock, "UpdateLock", _StubLock)
    monkeypatch.setattr(venv_sync, "_sync_source_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kwargs: True)

    def explode(*args, **kwargs):
        raise RuntimeError("a broken pool may never fail a launch")

    monkeypatch.setattr(install, "seed_from_generation_pool", explode)
    assert venv_sync._prepare_borrowed_launch(root, owner, current=False) == store_python


def test_kill_switch_short_circuits_before_any_stamp_work(borrowed, monkeypatch):
    root, owner, _store_python = borrowed
    monkeypatch.setenv(install.POOL_DISABLE_ENV, "1")
    monkeypatch.setattr(install, "_seed_expected",
                        lambda *args, **kwargs: pytest.fail("the kill switch must decide first"))
    assert install.seed_from_generation_pool(root, owner=owner) is False
    assert install.publish_generation_to_pool(root, owner=owner) is None


def _stub_sync_inputs(monkeypatch):
    """Everything a sync does EXCEPT publish: the wiring is what these tests assert."""
    import pm.client
    import pm.extras
    from pm import features

    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)
    monkeypatch.setattr(pm, "sync_venv", lambda *args, **kwargs: None)
    monkeypatch.setattr(pm.extras, "legacy_selection", lambda _root: None)
    monkeypatch.setattr(venv_sync, "refuse_foreign_owned_venv", lambda _root: None)
    # The frozen-feature policy reads <bundle>/enabled-features.json beside the checkout, i.e.
    # the real home in a test run; pin the "no bundle" answer (the common case).
    monkeypatch.setattr(features, "read_features", lambda: None)


def _stub_sync(monkeypatch, published):
    _stub_sync_inputs(monkeypatch)
    monkeypatch.setattr(install, "publish_generation_to_pool",
                        lambda project_root, **kwargs: published.append((project_root, kwargs)))


def _committed_generation(root: Path, *, files: int = 9):
    """Record a committed generation for *root*, the way a real sync commits one."""
    import uuid

    from pm.environments import install_state_dir, runtime_facts_path
    from pm.lock import Facts
    from pm.packages import Venv

    generation = install_state_dir(root) / "environments" / uuid.uuid4().hex
    venv = generation / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = {Path(sys.base_prefix) / 'bin'}\nversion = 3.14.7\n", encoding="utf-8")
    (generation / "workspace").mkdir()
    (generation / "workspace" / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (generation / ".lease-managed").touch()
    for index in range(files):
        (venv / "bin" / f"shim{index}").write_text(
            '#!/bin/sh\nexec "$(dirname "$0")/python" "$@"\n', encoding="utf-8")
    python = venv / "bin" / "python"
    python.write_text('#!/bin/sh\necho "$(cd "$(dirname "$0")/.." && pwd)"\n', encoding="utf-8")
    python.chmod(0o755)
    stamp = Venv(root).expected_stamp([])
    Facts(runtime_facts_path(root)).record_state(
        "venv", stamp, [], environment=venv, resolved_lock=generation / "workspace" / "uv.lock")
    return stamp, generation


def test_a_borrowed_sync_publishes_its_generation(borrowed, monkeypatch):
    root, owner, _store_python = borrowed
    published = []
    _stub_sync(monkeypatch, published)
    (root / ".update-incomplete").touch()

    venv_sync._sync_source_dependencies(root, arm=False, borrowed_from=owner)

    # The owner is part of the call, not an optional nicety: `generation_pool_root(root,
    # owner=None)` answers None, so a call without it publishes nothing (round-1 blocking).
    assert published == [(root, {"owner": owner})]
    # The checkout's markers describe the owner's environment, not this one.
    assert (root / ".update-incomplete").exists()


def test_an_owner_sync_publishes_nothing(borrowed, monkeypatch):
    root, _owner, _store_python = borrowed
    published = []
    _stub_sync(monkeypatch, published)
    (root / ".update-incomplete").touch()

    venv_sync._sync_source_dependencies(root, arm=False)

    assert published == []
    assert not (root / ".update-incomplete").exists()


# --- round 1 review: the W5 wiring, driven with the REAL publish ------------------------------

def test_the_borrowed_sync_wiring_publishes_through_the_real_publish(borrowed, monkeypatch):
    """``_sync_source_dependencies`` must reach publish WITH the owner, for real.

    Round-1 review, blocking: the hook called ``publish_generation_to_pool(project_root=root)``
    without ``owner=``, so ``generation_pool_root(root, owner=None)`` answered ``None`` (the
    owner root never pools with itself), publish returned ``None`` and no entry was ever
    written -- every borrowing launch a permanent miss. The suites stayed green because each
    one either monkeypatched publish or passed an explicit ``pool=``, bypassing that line.
    This drives the production call shape with the real publish, nothing patched but the
    sync's own inputs.
    """
    root, owner, _store_python = borrowed
    _stub_sync_inputs(monkeypatch)
    stamp, generation = _committed_generation(root)

    # The defect itself, reproduced on the same root: without the owner this is a no-op.
    assert install.publish_generation_to_pool(project_root=root) is None
    assert not (owner / "generation-pool").exists()

    venv_sync._sync_source_dependencies(root, arm=False, borrowed_from=owner)

    pool = owner / "generation-pool"
    entry = pool / stamp
    assert entry.is_dir()
    manifest = install.read_pool_manifest(pool, stamp)
    assert manifest is not None and manifest["stamp"] == stamp
    # hardlinked from the borrowing root's own committed generation: no data was copied
    assert (os.stat(entry / "venv" / "bin" / "python").st_ino
            == os.stat(generation / "venv" / "bin" / "python").st_ino)
    assert os.stat(entry / "venv" / "bin" / "python").st_nlink >= 2
    assert install._verify_clone(entry) is None  # the entry is a launchable environment


def test_a_published_generation_seeds_a_fresh_borrowing_root(borrowed, monkeypatch, tmp_path):
    """The join: A builds -> the wired publish -> a genuinely fresh B seeds from the entry."""
    from pm.environments import install_state_dir, runtime_facts_path
    from pm.lock import Facts
    from pm.packages import Venv

    root, owner, _store_python = borrowed
    _stub_sync_inputs(monkeypatch)
    stamp, _generation_dir = _committed_generation(root)

    venv_sync._sync_source_dependencies(root, arm=False, borrowed_from=owner)
    entry = owner / "generation-pool" / stamp
    assert entry.is_dir()

    # B: another data root for the same checkout, with no dependency state of its own.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home-b"))
    assert not runtime_facts_path(root).is_file()
    assert Venv(root).expected_stamp([]) == stamp  # same tree+lockfile -> same expected stamp

    assert install.seed_from_generation_pool(root, owner=owner) is True

    fact = Facts(runtime_facts_path(root)).get("venv")
    assert fact["stamp"] == stamp
    environment = Path(fact["environment"]).resolve()
    generations = (install_state_dir(root) / "environments").resolve()
    # A9: the selection is inside B's own install state, never a pool path
    assert environment.is_relative_to(generations)
    assert "generation-pool" not in environment.parts
    cloned = [path for path in environment.rglob("*") if path.is_file()]
    assert len(cloned) >= 10
    assert all(os.stat(path).st_nlink >= 2 for path in cloned)  # A5: nothing was copied
    # the entry is shared, never consumed by the seed
    assert (entry / "manifest.json").is_file()
    assert install.read_pool_manifest(owner / "generation-pool", stamp) is not None


def test_owner_launch_calls_no_pool_code(tmp_path, monkeypatch):
    root = tmp_path / "owner-checkout"
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'example'\nversion = '0.0.0'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "install-stamp.json").write_text('{"updateMechanism": "self"}', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    from pm import features

    # A frozen bundle's enabled-features.json sits beside the checkout -- the real home in a
    # test run. Pin the no-bundle answer (see _stub_sync_inputs).
    monkeypatch.setattr(features, "read_features", lambda: None)
    store_python = tmp_path / "store" / "bin" / "python3"
    store_python.parent.mkdir(parents=True)
    store_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _root: store_python)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kwargs: True)
    monkeypatch.setattr("pm.environments.owning_home_root", lambda _root: None)
    monkeypatch.setattr(venv_sync, "publish_launchers", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(install, "seed_from_generation_pool", lambda *args, **kwargs: calls.append("seed") or True)
    monkeypatch.setattr(install, "publish_generation_to_pool", lambda *args, **kwargs: calls.append("publish"))

    assert venv_sync.prepare_launch(root, []) == store_python
    assert calls == []

    # A6/A12: with a generation already committed, an owner launch leaves the recorded state
    # byte-identical and creates no second stamp -- the pool may not re-record, re-stamp or
    # shadow the owner's own home on the way past.
    from pm.environments import install_state_dir, runtime_facts_path

    _committed_generation(root)
    facts = runtime_facts_path(root)
    before = hashlib.sha256(facts.read_bytes()).hexdigest()
    listing = sorted(str(path.relative_to(install_state_dir(root))) for path in install_state_dir(root).rglob("*"))

    assert venv_sync.prepare_launch(root, []) == store_python
    assert calls == []
    assert hashlib.sha256(facts.read_bytes()).hexdigest() == before
    assert sorted(str(path.relative_to(install_state_dir(root)))
                  for path in install_state_dir(root).rglob("*")) == listing
