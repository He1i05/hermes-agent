"""The stamp-keyed generation pool: layout, publish, eviction and seed.

Plan: PLAN-verify-only-sync-and-generation-pool.md W1-W3. Every launch-path anomaly must
answer "miss": a broken pool may make a launch slow, never failed (IV5).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path

import pytest

from pm import install
from pm.environments import install_state_dir, runtime_facts_path
from pm.lock import Facts


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A checkout root and a borrowing data root that share a filesystem with the pool."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(install.POOL_ENV, raising=False)
    monkeypatch.delenv(install.POOL_DISABLE_ENV, raising=False)
    monkeypatch.delenv(install.POOL_MAX_BYTES_ENV, raising=False)
    # The frozen-feature policy reads <bundle>/enabled-features.json beside the checkout; in a
    # test the checkout is the real install, so pin the "no bundle" answer (the common case).
    from pm import features

    monkeypatch.setattr(features, "read_features", lambda: None)
    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'example'\nversion = '0.0.0'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\nrevision = 3\n", encoding="utf-8")
    return root, home


def _generation(state: Path, *, body: str | None = None, files: int = 4) -> Path:
    """A managed generation: lease marker, interpreter, resolved lock, relocatable shims."""
    generation = state / "environments" / uuid.uuid4().hex
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
    # The interpreter probe must answer with ITS OWN prefix, so a clone is verifiable while
    # the shim content stays free of any absolute generation path (A10).
    python = venv / "bin" / "python"
    python.write_text(body if body is not None else
                      '#!/bin/sh\necho "$(cd "$(dirname "$0")/.." && pwd)"\n', encoding="utf-8")
    python.chmod(0o755)
    return generation


def _stamp(root: Path) -> str:
    from pm.packages import Venv

    return Venv(root).expected_stamp([])


def _record(root: Path, generation: Path, stamp: str, extras: list[str] | None = None) -> None:
    Facts(runtime_facts_path(root)).record_state(
        "venv", stamp, list(extras or []), environment=generation / "venv",
        resolved_lock=generation / "workspace" / "uv.lock")


def _pool_entry(pool: Path, stamp: str, generation: Path) -> Path:
    """A pool entry written the way publish writes it, minus the hardening."""
    entry = pool / stamp
    install._clone(generation, entry)
    manifest = install._manifest(generation, stamp=stamp, extras=[], members="",
                                 target="linux-arm64", python="3.14.7")
    (entry / install._POOL_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return entry


def _published(tmp_path: Path, root: Path, *, files: int = 4) -> tuple[Path, str, Path]:
    state = install_state_dir(root)
    stamp = _stamp(root)
    generation = _generation(state, files=files)
    _record(root, generation, stamp)
    pool = tmp_path / "pool"
    entry = install.publish_generation_to_pool(root, owner=tmp_path / "owner", pool=pool)
    assert entry == pool / stamp
    return pool, stamp, generation


# --- W1: root resolution, device check, locking, manifest read -----------------------------

def test_pool_root_resolution_never_creates_the_pool(sandbox, tmp_path, monkeypatch):
    root, _home = sandbox
    owner = tmp_path / "owner"
    assert install.generation_pool_root(root, owner=None) is None
    assert install.generation_pool_root(root, owner=root) is None

    monkeypatch.setenv(install.POOL_DISABLE_ENV, "yes")
    assert install.generation_pool_root(root, owner=owner) is None
    monkeypatch.delenv(install.POOL_DISABLE_ENV)

    assert install.generation_pool_root(root, owner=owner) == owner.resolve() / "generation-pool"
    assert not (owner / "generation-pool").exists()

    override = tmp_path / "elsewhere"
    monkeypatch.setenv(install.POOL_ENV, str(override))
    assert install.generation_pool_root(root, owner=owner) == override
    monkeypatch.setenv(install.POOL_ENV, "relative/pool")
    assert install.generation_pool_root(root, owner=owner) is None


def test_same_device_rejects_a_pool_on_another_filesystem(sandbox, tmp_path):
    root, _home = sandbox
    shared = Path("/dev/shm")
    if not shared.is_dir() or os.stat(shared).st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("no second filesystem available")
    assert install._pool_same_device(shared / "pool", install_state_dir(root)) is False
    assert install._pool_same_device(tmp_path / "pool", install_state_dir(root)) is True


def test_entry_lock_yields_false_instead_of_waiting(tmp_path):
    pool = tmp_path / "pool"
    with install._pool_lock(pool, "abc", exclusive=True) as held:
        assert held is True
        with install._pool_lock(pool, "abc", exclusive=True) as contender:
            assert contender is False
        with install._pool_lock(pool, "abc", exclusive=False) as reader:
            assert reader is False
    with install._pool_lock(pool, "abc", exclusive=False) as reader:
        assert reader is True


def test_manifest_rejections_all_answer_miss(sandbox, tmp_path):
    root, _home = sandbox
    stamp = _stamp(root)
    generation = _generation(install_state_dir(root))
    base = tmp_path / "base"
    _pool_entry(base, stamp, generation)
    assert install.read_pool_manifest(base, stamp) is not None
    assert install.read_pool_manifest(base, "0" * 64) is None
    assert install.read_pool_manifest(base, "../escape") is None

    def case(name: str, mutate) -> Path:
        pool = tmp_path / name
        shutil.copytree(base, pool, symlinks=True)  # a real copy: each case mutates its own bytes
        mutate(pool)
        return pool

    assert install.read_pool_manifest(
        case("malformed", lambda p: (p / stamp / "manifest.json").write_text("{not json")), stamp) is None
    assert install.read_pool_manifest(
        case("incomplete", lambda p: (p / stamp / install._POOL_INCOMPLETE).touch()), stamp) is None
    assert install.read_pool_manifest(
        case("suspect", lambda p: ((p / install._POOL_SUSPECT_DIR).mkdir(),
                                   (p / install._POOL_SUSPECT_DIR / stamp).touch())), stamp) is None
    assert install.read_pool_manifest(
        case("unmanaged", lambda p: (p / stamp / "workspace" / "uv.lock").unlink()), stamp) is None

    def drop_bytes(pool: Path) -> None:
        manifest = json.loads((pool / stamp / "manifest.json").read_text())
        manifest.pop("bytes")
        (pool / stamp / "manifest.json").write_text(json.dumps(manifest))

    assert install.read_pool_manifest(case("missing-field", drop_bytes), stamp) is None

    def foreign(pool: Path) -> None:
        manifest = json.loads((pool / stamp / "manifest.json").read_text())
        manifest["stamp"] = "f" * 64
        (pool / stamp / "manifest.json").write_text(json.dumps(manifest))

    assert install.read_pool_manifest(case("foreign-stamp", foreign), stamp) is None


def test_generation_of_requires_the_three_managed_markers(sandbox):
    root, _home = sandbox
    generation = _generation(install_state_dir(root))
    fact = {"environment": str(generation / "venv"),
            "resolved_lock": str(generation / "workspace" / "uv.lock")}
    assert install._generation_of(fact) == generation.resolve()
    (generation / "workspace" / "uv.lock").unlink()
    assert install._generation_of(fact) is None
    # selected_venv's in-tree fallback is never pool material (measured 417,380 files)
    assert install._generation_of({"environment": str(root / "venv"),
                                   "resolved_lock": str(root / "venv" / "uv.lock")}) is None


# --- W2: publish, atomicity, eviction, relocation ------------------------------------------

def test_publish_hardlinks_a_hardened_entry_and_leaves_facts_alone(sandbox, tmp_path):
    root, _home = sandbox
    state = install_state_dir(root)
    stamp = _stamp(root)
    generation = _generation(state)
    _record(root, generation, stamp)
    facts_before = runtime_facts_path(root).read_bytes()

    pool = tmp_path / "pool"
    entry = install.publish_generation_to_pool(root, owner=tmp_path / "owner", pool=pool)

    assert entry == pool / stamp
    manifest = install.read_pool_manifest(pool, stamp)
    assert manifest is not None and manifest["stamp"] == stamp
    assert manifest["files"] >= 6 and manifest["bytes"] > 0
    # no data blocks were copied: the entry shares the generation's inodes
    assert os.stat(entry / "venv" / "bin" / "python").st_nlink >= 2
    # the pool is read-only for everyone, including us
    assert stat.S_IMODE(os.stat(entry).st_mode) == 0o555
    assert stat.S_IMODE(os.stat(entry / "manifest.json").st_mode) == 0o444
    assert stat.S_IMODE(os.stat(entry / "venv" / "bin" / "python").st_mode) & 0o200 == 0
    # relocation: no shim carries its own generation path (A10)
    for path in (entry / "venv" / "bin").iterdir():
        assert str(generation) not in path.read_text(encoding="utf-8")
    # publishing never rewrites this install's own record (A6/A12)
    assert runtime_facts_path(root).read_bytes() == facts_before
    assert not any((pool / install._POOL_TMP_DIR).iterdir())


def test_publish_twice_is_a_no_op(sandbox, tmp_path):
    root, _home = sandbox
    pool, stamp, _generation_dir = _published(tmp_path, root)
    before = (pool / stamp / "manifest.json").stat().st_mtime_ns
    assert install.publish_generation_to_pool(root, owner=tmp_path / "owner", pool=pool) is None
    assert (pool / stamp / "manifest.json").stat().st_mtime_ns == before


def test_publish_treats_a_lost_rename_race_as_published(sandbox, tmp_path, monkeypatch):
    root, _home = sandbox
    state = install_state_dir(root)
    stamp = _stamp(root)
    _record(root, _generation(state), stamp)
    pool = tmp_path / "pool"

    def lose(_source, _target):
        raise OSError(39, "Directory not empty")

    monkeypatch.setattr(install.os, "replace", lose)
    assert install.publish_generation_to_pool(root, owner=tmp_path / "owner", pool=pool) == pool / stamp
    assert not any((pool / install._POOL_TMP_DIR).iterdir())


def test_publish_skips_a_cross_device_pool_without_copying(sandbox, tmp_path, monkeypatch, capsys):
    root, _home = sandbox
    shared = Path("/dev/shm")
    if not shared.is_dir() or os.stat(shared).st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("no second filesystem available")
    stamp = _stamp(root)
    _record(root, _generation(install_state_dir(root)), stamp)
    pool = shared / f"hermes-pool-{uuid.uuid4().hex}"
    assert install.publish_generation_to_pool(root, owner=root.parent, pool=pool) is None
    assert not pool.exists()
    assert "another filesystem" in capsys.readouterr().err


def test_eviction_keeps_the_current_entry_and_drops_the_oldest(sandbox, tmp_path, monkeypatch):
    root, _home = sandbox
    pool, first, _generation_dir = _published(tmp_path, root)
    second_root = tmp_path / "checkout-two"
    second_root.mkdir()
    (second_root / "uv.lock").write_text("version = 1\nrevision = 99\n", encoding="utf-8")
    second_root_stamp = _stamp(second_root)
    assert second_root_stamp != first
    _record(second_root, _generation(install_state_dir(second_root)), second_root_stamp)

    old = 1_600_000_000
    os.utime(pool / first, (old, old))
    size_of_second = json.loads((pool / first / "manifest.json").read_text())["bytes"]
    monkeypatch.setenv(install.POOL_MAX_BYTES_ENV, str(size_of_second))

    assert install.publish_generation_to_pool(second_root, owner=tmp_path / "owner", pool=pool) == pool / second_root_stamp
    assert not (pool / first).exists()
    assert (pool / second_root_stamp).is_dir()


def test_hardened_tree_is_removable_only_after_softening(tmp_path):
    generation = _generation(tmp_path / "state")
    entry = tmp_path / "entry"
    install._clone(generation, entry)
    install._harden(entry)
    with pytest.raises(PermissionError):
        shutil.rmtree(entry)
    install._soften_dirs(entry)
    shutil.rmtree(entry)
    assert not entry.exists()


# --- W3: seed ------------------------------------------------------------------------------

def test_seed_materializes_a_verified_clone_in_the_roots_own_state(sandbox, tmp_path):
    root, _home = sandbox
    pool, stamp, _generation_dir = _published(tmp_path, root)
    shutil.rmtree(install_state_dir(root))  # a fresh borrowing root: no local environment

    assert install.seed_from_generation_pool(root, owner=tmp_path / "owner", pool=pool) is True

    fact = Facts(runtime_facts_path(root)).get("venv")
    assert set(fact) == {"stamp", "extras", "environment", "resolved_lock"}
    assert fact["stamp"] == stamp
    environment = Path(fact["environment"]).resolve()
    generations = (install_state_dir(root) / "environments").resolve()
    assert environment.is_relative_to(generations)
    assert Path(fact["resolved_lock"]).is_file()
    # the clone is a hardlink of the pool entry, not a copy of its data blocks
    assert os.stat(environment / "bin" / "python").st_nlink >= 2
    # and it is collectable: clone directories were softened after the hardlink copy (C7)
    shutil.rmtree(environment.parent)
    assert not environment.parent.exists()
    # A12: no second currency source — the install state gains a generation, never a stamp file
    names = {path.name for path in install_state_dir(root).iterdir()}
    assert not {name for name in names if "stamp" in name or name in {"venv-sync.json", "stamp"}}


def test_seed_misses_leave_nothing_behind(sandbox, tmp_path, monkeypatch):
    root, _home = sandbox
    pool = tmp_path / "pool"
    # no pool at all
    assert install.seed_from_generation_pool(root, owner=tmp_path / "owner", pool=pool) is False
    generations = install_state_dir(root) / "environments"
    assert not generations.exists() or not any(generations.iterdir())

    # a pool entry for a different stamp is a miss, not an error
    other = tmp_path / "other"
    other.mkdir()
    (other / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    _pool_entry(pool, _stamp(other), _generation(tmp_path / "source-state"))
    assert install.seed_from_generation_pool(root, owner=tmp_path / "owner", pool=pool) is False
    assert not generations.exists() or not any(generations.iterdir())


def test_seed_verification_failure_marks_the_entry_suspect(sandbox, tmp_path, capsys):
    root, _home = sandbox
    stamp = _stamp(root)
    source = _generation(tmp_path / "source-state", body="#!/bin/sh\nexit 3\n")
    pool = tmp_path / "pool"
    _pool_entry(pool, stamp, source)

    assert install.seed_from_generation_pool(root, owner=tmp_path / "owner", pool=pool) is False
    assert (pool / install._POOL_SUSPECT_DIR / stamp).is_file()
    generations = install_state_dir(root) / "environments"
    assert not any(generations.iterdir())
    assert "import probe" in capsys.readouterr().err
    # the suspect marker makes the next attempt a cheap miss, without re-probing
    assert install.read_pool_manifest(pool, stamp) is None


def test_seed_never_selects_a_pool_path(sandbox, tmp_path):
    from pm.environments import _recorded_venv

    root, _home = sandbox
    pool, stamp, _generation_dir = _published(tmp_path, root)
    shutil.rmtree(install_state_dir(root))
    assert install.seed_from_generation_pool(root, owner=tmp_path / "owner", pool=pool) is True
    assert _recorded_venv(root) is not None  # the clone, inside this root's own state

    record = json.loads(runtime_facts_path(root).read_text(encoding="utf-8"))
    record["packages"]["venv"]["environment"] = str(pool / stamp / "venv")
    runtime_facts_path(root).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside this install"):
        _recorded_venv(root)


def test_seed_without_a_pool_makes_no_change(sandbox, tmp_path):
    root, _home = sandbox
    stamp = _stamp(root)
    _record(root, _generation(install_state_dir(root)), stamp)
    before = runtime_facts_path(root).read_bytes()
    assert install.seed_from_generation_pool(root, owner=tmp_path / "owner") is False
    assert runtime_facts_path(root).read_bytes() == before


# --- IV4/A6: the owner root never touches the pool -----------------------------------------

def test_owner_root_calls_no_pool_code(sandbox, tmp_path):
    root, home = sandbox
    stamp = _stamp(root)
    _record(root, _generation(install_state_dir(root)), stamp)
    before = runtime_facts_path(root).read_bytes()

    assert install.generation_pool_root(root, owner=None) is None
    assert install.seed_from_generation_pool(root) is False
    assert install.publish_generation_to_pool(root) is None
    assert not (home / "generation-pool").exists()
    assert runtime_facts_path(root).read_bytes() == before


def test_kill_switch_disables_both_paths(sandbox, tmp_path, monkeypatch):
    root, _home = sandbox
    pool, _stamp_value, _generation_dir = _published(tmp_path, root)
    shutil.rmtree(install_state_dir(root))
    monkeypatch.setenv(install.POOL_DISABLE_ENV, "1")
    assert install.seed_from_generation_pool(root, owner=tmp_path / "owner", pool=pool) is False
    assert install.publish_generation_to_pool(root, owner=tmp_path / "owner", pool=pool) is None
    assert not (install_state_dir(root) / "environments").exists() or not any(
        (install_state_dir(root) / "environments").iterdir())
