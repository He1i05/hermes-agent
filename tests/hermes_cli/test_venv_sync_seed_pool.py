"""Seed-first borrowed launches and the borrower-only publish hook (plan W4/W5).

The pool is a launch-path optimization: a hit must skip the build entirely, a miss must
leave today's path byte-for-byte, and the owner root must never touch pool code (IV4).
"""

from __future__ import annotations

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
    import hermes_cli.update_lock as update_lock

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
    import hermes_cli.update_lock as update_lock

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


def _stub_sync(monkeypatch, published):
    import pm.client
    import pm.extras

    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)
    monkeypatch.setattr(pm, "sync_venv", lambda *args, **kwargs: None)
    monkeypatch.setattr(pm.extras, "legacy_selection", lambda _root: None)
    monkeypatch.setattr(venv_sync, "refuse_foreign_owned_venv", lambda _root: None)
    monkeypatch.setattr(install, "publish_generation_to_pool",
                        lambda project_root, **kwargs: published.append(project_root))


def test_a_borrowed_sync_publishes_its_generation(borrowed, monkeypatch):
    root, owner, _store_python = borrowed
    published = []
    _stub_sync(monkeypatch, published)
    (root / ".update-incomplete").touch()

    venv_sync._sync_source_dependencies(root, arm=False, borrowed_from=owner)

    assert published == [root]
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


def test_owner_launch_calls_no_pool_code(tmp_path, monkeypatch):
    root = tmp_path / "owner-checkout"
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'example'\nversion = '0.0.0'\n", encoding="utf-8")
    (root / "install-stamp.json").write_text('{"updateMechanism": "self"}', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
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
