"""What "< 2 s on a borrowed root" can honestly mean -- and what it cannot.

The card behind the generation pool (t_a06eae46) bounded a borrowing launch at
"< 2 s per launch". Three runs of measurement decided how that line may be tested
(t_b4925ffb benchmark, t_c3eab9d5 plan, re-measured on this host):

* a cold build of a fresh borrowing root is **236 s / 752 MB**. Asserting < 2 s on
  it would be a false claim, so every build here is stubbed and the *structure* is
  asserted instead: the preparation path selected for a borrowed root must reach a
  usable interpreter without calling ``_sync_source_dependencies`` (no ``uv``, no
  build);
* the currency decision on a COMMITTED borrowing root is ~5 ms in process
  (``pm.install.venv_is_current``); the delegated launch-shape probe costs ~0.06 s.
  A warm launch therefore skips dependency work in well under the line;
* a pool hit replaces the build with a hardlink clone: ~0.15 s for this fixture's
  11-file generation (~2.33 s for a real 20,873-file one, benchmark t_b4925ffb).

The < 2 s assertions below are therefore scoped to the launch-preparation path and
to the currency decision at fixture scale, with the measured value printed in the
failure message. The owner-root guard (A6) lives beside the pool's own suite
(``test_venv_sync_seed_pool.py``) and is extended there rather than duplicated here.

Timing hygiene (plan §9.3): three runs, median, fixture-scale operations only, so a
20 % load swing cannot flip the result.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
import uuid
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
    """A borrowing checkout, the data root that owns it, and a store interpreter."""
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
    # The frozen-feature policy reads <bundle>/enabled-features.json beside the checkout,
    # i.e. the REAL home in a test run; pin the "no bundle" answer (the common case), the
    # same way the sibling pool suite does.
    from pm import features

    monkeypatch.setattr(features, "read_features", lambda: None)
    return root, owner, store_python


def _committed_generation(root: Path, *, files: int = 11):
    """Record a committed generation for *root*, the way a real sync commits one.

    Same shape as the sibling pool suite's helper: lease marker, ``pyvenv.cfg``, a
    resolved lock and relocatable shims, so publish/seed/verify all see a launchable
    environment instead of a directory of fixtures.
    """
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
        (venv / "bin" / f"shim{index}").write_text("#!/bin/sh\n", encoding="utf-8")
    python = venv / "bin" / "python"
    python.write_text('#!/bin/sh\necho "$(cd "$(dirname "$0")/.." && pwd)"\n', encoding="utf-8")
    python.chmod(0o755)
    stamp = Venv(root).expected_stamp([])
    Facts(runtime_facts_path(root)).record_state(
        "venv", stamp, [], environment=venv, resolved_lock=generation / "workspace" / "uv.lock")
    return stamp, generation


def _median_seconds(callable_, runs: int = 3) -> tuple[float, list[float], object]:
    samples = []
    value = None
    for _ in range(runs):
        start = time.perf_counter()
        value = callable_()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), samples, value


# ── 1. the fast path: a pool hit prepares the launch without paying the build ────
def test_a_pool_hit_prepares_a_borrowed_launch_in_under_two_seconds(borrowed, monkeypatch, tmp_path):
    """The whole point of the pool: a fresh borrowing root launches on a hardlink clone.

    The < 2 s line is asserted on the *preparation* (seed + relaunch decision), and the
    zero-build part is asserted structurally, because the build it replaces is 236 s at
    real scale and is never allowed to run here.
    """
    root, owner, store_python = borrowed
    stamp, generation = _committed_generation(root)
    assert install.publish_generation_to_pool(project_root=root, owner=owner) is not None
    assert (owner / "generation-pool" / stamp).is_dir()

    builds = []
    monkeypatch.setattr(venv_sync, "_sync_source_dependencies",
                        lambda *args, **kwargs: builds.append(kwargs))
    # ``current=False`` is what a real launch passes for a borrowing root: the delegated
    # currency probe cannot certify a borrowing home, so the seed is what makes it current.
    median, samples, target = _median_seconds(
        lambda: _prepare_into_fresh_home(monkeypatch, tmp_path, root, owner), runs=3)

    assert target["target"] == store_python
    assert builds == [], "a pool hit must not reach the build path"
    assert median < 2.0, f"pool-hit preparation took {median:.3f}s (runs: {samples})"
    # The timing was measured on a real seed, not on a no-op: the last root is committed
    # to a clone inside its OWN install state.
    from pm.environments import install_state_dir, runtime_facts_path
    from pm.lock import Facts

    fact = Facts(runtime_facts_path(root)).get("venv")
    environment = Path(fact["environment"]).resolve()
    assert fact["stamp"] == stamp
    assert environment.is_relative_to(install_state_dir(root).resolve())
    assert "generation-pool" not in environment.parts
    cloned = [path for path in environment.rglob("*") if path.is_file()]
    assert cloned and all(os.stat(path).st_nlink >= 2 for path in cloned)  # nothing copied


def _prepare_into_fresh_home(monkeypatch, tmp_path, root, owner):
    """One launch's preparation against a data root that has never held a generation."""
    index = len(list(tmp_path.glob("home-b*")))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / f"home-b{index}"))
    start = time.perf_counter()
    target = venv_sync._prepare_borrowed_launch(root, owner, current=False)
    return {"target": target, "seconds": time.perf_counter() - start}


# ── 2. the decision itself: cheaper than the line by three orders of magnitude ───
def test_the_currency_decision_on_a_committed_borrowing_root_is_under_two_seconds(borrowed):
    """The "sync" a warm borrowed launch pays is a decision, and the decision is ~5 ms.

    Deliberately the in-process engine (``pm.install.venv_is_current``), not the
    delegated seam: a borrowing home has no PM runtime to answer the seam, so the seam
    answers False there and cannot be used as evidence that a sync is unnecessary.
    """
    root, _owner, _store_python = borrowed
    _committed_generation(root)

    median, samples, current = _median_seconds(
        lambda: install.venv_is_current(project_root=root), runs=3)

    assert current is True
    assert median < 2.0, f"currency decision took {median:.3f}s (runs: {samples})"


# ── 3. a warm launch does no dependency work at all ─────────────────────────────
def test_a_warm_borrowed_launch_runs_neither_the_seed_nor_a_sync(borrowed, monkeypatch):
    """``current=True`` is the warm-launch path: no clone, no build, no uv.

    Measured warm launches are 5.6-8.8 s end to end, and that time is git identity plus
    CLI import, not dependency work -- so what this test bounds is the preparation step,
    which must stay far below the launch budget.
    """
    root, owner, store_python = borrowed
    seeded, builds = [], []
    monkeypatch.setattr(install, "seed_from_generation_pool",
                        lambda *args, **kwargs: seeded.append(kwargs) or True)
    monkeypatch.setattr(venv_sync, "_sync_source_dependencies",
                        lambda *args, **kwargs: builds.append(kwargs))

    start = time.perf_counter()
    target = venv_sync._prepare_borrowed_launch(root, owner, current=True)
    elapsed = time.perf_counter() - start

    assert target == store_python  # a different interpreter: relaunch once
    assert seeded == [] and builds == []
    assert elapsed < 2.0, f"warm preparation took {elapsed:.3f}s"


# ── 4. live update: a borrowing launch never runs the owner's tail ──────────────
def test_a_borrowing_launch_runs_no_owner_tail_and_keeps_the_markers(borrowed, monkeypatch):
    """A borrowed root must not finish, arm or clear an update that is not its own.

    The tail mutates the checkout (launchers, products, maintenance) and is the owner's
    obligation. A borrowing launch that ran it would rebuild launchers another data root
    is serving, so this pins both halves: the tail never runs, and every marker it would
    have touched survives the launch.
    """
    root, owner, _store_python = borrowed
    from hermes_cli import update_lock

    monkeypatch.setattr(update_lock, "UpdateLock", _StubLock)
    monkeypatch.setattr(install, "seed_from_generation_pool", lambda *args, **kwargs: False)
    monkeypatch.setattr(venv_sync, "_sync_source_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kwargs: True)
    tails = []
    monkeypatch.setattr(venv_sync, "_finish_source_update",
                        lambda *args, **kwargs: tails.append(kwargs))

    legacy = [root / ".update-incomplete", root / ".lazy-refresh-incomplete"]
    for marker in legacy:
        marker.touch()
    pending = venv_sync.completion_pending_path(root)
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.touch()

    venv_sync._prepare_borrowed_launch(root, owner, current=False)

    assert tails == [], "the owner's tail obligation is not the borrower's to run"
    assert all(marker.exists() for marker in legacy)
    assert pending.exists()


# ── 5. the kill switch: today's path, exactly, and it is still not a build here ──
def test_the_kill_switch_takes_the_unchanged_path_without_a_pool(borrowed, monkeypatch):
    """Rolling the pool back must not leave a half-seeded root behind."""
    root, owner, store_python = borrowed
    from hermes_cli import update_lock

    monkeypatch.setenv(install.POOL_DISABLE_ENV, "1")
    monkeypatch.setattr(update_lock, "UpdateLock", _StubLock)
    _committed_generation(root)
    monkeypatch.setattr(venv_sync, "_sync_source_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kwargs: True)

    assert venv_sync._prepare_borrowed_launch(root, owner, current=False) == store_python
    assert not (owner / "generation-pool").exists()
