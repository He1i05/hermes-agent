"""ensure(): make the installed state match the lockfile for a package,
and hand back its composed environment."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pm import paths
from pm.downloader import DownloadPaused, ProgressFn
from pm.filesystem import native, remove_tree, retry_held
from pm.lock import Facts, Lockfile
from pm.package import InstallError, Package, Runner, StatePackage, compose_env
from pm.plugin_inputs import Candidates, Members, PluginInput, Selection, StagedUpdate
from pm.registry import get_package, walk
from pm.store import Store, current_target, merge_tree, tree_digest

LOG = logging.getLogger(__name__)

# ``progress(stage, done, total, label)`` reports download/unpack/verify;
# multi-archive labels follow lockfile order.


def _prepare_artifacts(package, store, scratch, artifacts, version, target, *,
                       progress=None, pause_event=None, download_progress=None):
    def tick(done, total, ranges):
        if progress is not None:
            active = next(reversed(ranges))
            index = next(i for i, artifact in enumerate(artifacts) if artifact["url"] == active)
            label = f"{index + 1}/{len(artifacts)}" if len(artifacts) > 1 else ""
            progress("download", done, total, label)
        if download_progress is not None:
            download_progress(done, total, ranges)

    staged = scratch / "tree"
    archives = store.fetch_many(artifacts, scratch, progress=tick, pause_event=pause_event)
    for index, archive in enumerate(archives):
        if pause_event is not None and pause_event.is_set():
            raise DownloadPaused("install paused")
        if progress is not None:
            label = f"{index + 1}/{len(artifacts)}" if len(artifacts) > 1 else ""
            progress("unpack", 0, 0, label)
        # unpack empties its destination; extract additional archives separately.
        destination = staged if index == 0 else scratch / f"extra-{index}"
        package.unpack(archive, destination, target)
        if index:
            merge_tree(destination, staged)
    package.stage(store, staged, version, target)
    return staged


def _lockfile() -> Lockfile:
    return Lockfile(paths.lockfile_path())


def _facts() -> Facts:
    return Facts(paths.facts_path())


def _store() -> Store:
    return Store(paths.store_root())


def _heal_exec_bit(binary: Path) -> bool:
    """agent-browser entries staged before stage() set the exec bit sit at 0644 and fail every
    launch with PermissionError. Modes are not part of the pinned digest, so restore it in place;
    an entry we cannot chmod (sealed store) is treated as not installed."""
    if os.name == "nt" or os.access(binary, os.X_OK):
        return True
    try:
        binary.chmod(binary.stat().st_mode | 0o111)
    except OSError:
        return False
    return os.access(binary, os.X_OK)


def _installed_location(package: Package, lockfile: Lockfile, target: str, *,
                        verify: bool = False, allow_outdated: bool = False,
                        roots: tuple[Path, ...] | None = None):
    """Prefer the current pin. An explicit read may retain a prior PM install."""
    search_roots = dict.fromkeys(roots if roots is not None else (paths.store_root(), paths.writable_store_root()))
    fallback = None
    for root in search_roots:
        store = Store(root)
        facts = _facts() if root == paths.store_root() else Facts(root / "facts.json")
        fact = facts.get(package.name)
        if not fact or not facts.installed(package.name, None, store.root):
            continue
        binary = package.binary(store.entry(fact["entry"]), target)
        if binary is not None and not binary.is_file():
            continue
        if binary is not None and target == current_target() and not _heal_exec_bit(binary):
            continue
        if verify and not _entry_verified(package, fact, store, target):
            continue
        if facts.installed(package.name, lockfile.version(package.name), store.root,
                           _identity(lockfile, package.name, target)):
            return facts, store
        if (allow_outdated and fact.get("target") == target
                and fact.get("artifacts") and fact.get("digest")):
            fallback = facts, store
    return fallback


@dataclass(frozen=True)
class InstalledPackage:
    path: Path
    version: str
    binary: Path | None


def installed_package(name: str, *, allow_outdated: bool = False) -> InstalledPackage | None:
    """Read the selected PM entry without installing or changing its facts."""
    package = get_package(name)
    if package.internal:
        raise ValueError(f"{name} is internal PM tooling, not an application package")
    target = current_target()
    location = _installed_location(package, _lockfile(), target, allow_outdated=allow_outdated)
    if location is None:
        return None
    facts, store = location
    fact = facts.get(name)
    entry = store.entry(fact["entry"])
    binary = package.binary(entry, target)
    # Callers execute and compare these paths outside PM: the ordinary spelling, not the store's.
    return InstalledPackage(Path(native(entry)), fact["version"],
                            Path(native(binary)) if binary is not None else None)


def uv_launcher(name: str) -> Path | None:
    """PM's installed ``uv``/``uvx`` for a user-declared MCP stdio ``command:``, so a bare
    ``uvx`` server runs the packaged uv, never the user's. Read-only; Hermes's own Python
    work still goes through PM operations, not this executable."""
    if name not in ("uv", "uvx"):
        raise ValueError(f"{name!r} is not a uv launcher")
    package = get_package("uv")
    target = current_target()
    location = _installed_location(package, _lockfile(), target)
    if location is None:
        return None
    facts, store = location
    binary = package.binary(store.entry(facts.get("uv")["entry"]), target)
    launcher = binary.with_name(name + binary.suffix) if binary is not None else None
    return Path(native(launcher)) if launcher is not None and launcher.is_file() else None


def _identity(lockfile: Lockfile, name: str, target: str):
    """The identity the lock currently pins for `name` on `target`:
    (target, tuple(artifact sha256s)) — or None when the lock pins no
    artifacts (nothing digest-bound to compare)."""
    artifacts = lockfile.artifacts(name, target)
    if not artifacts:
        return None
    return (target, tuple(a["sha256"] for a in artifacts))


def lazy_installs_allowed() -> bool:
    """Policy: may pm install things on demand right now?

    HERMES_DISABLE_LAZY_INSTALLS is an internal bridge var set by the
    official Docker image and the hermetic test harness. The user-facing
    setting is security.allow_lazy_installs in config.yaml; a config
    system that fails to load counts as ALLOWED only when hermes_cli is
    genuinely absent (bootstrap) — config errors fail closed.
    """
    import os

    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    try:
        from hermes_cli.config import cfg_get, load_config_readonly, require_readable_config_before_write
    except ModuleNotFoundError as exc:
        return exc.name in {"hermes_cli", "hermes_cli.config"}
    except ImportError:
        return False
    try:
        # The normal loader falls back to defaults on invalid YAML. A broken
        # security policy must not grant permission to acquire dependencies.
        require_readable_config_before_write()
        return cfg_get(load_config_readonly(), "security", "allow_lazy_installs", default=True) is True
    except Exception:
        return False


def enabled_extras() -> list[str]:
    """The venv extras recorded in the installed state."""
    fact = Facts(paths.runtime_facts_path()).get("venv") or _facts().get("venv") or {}
    return list(fact.get("extras", []))


def is_installed(name: str) -> bool:
    return _installed_location(get_package(name), _lockfile(), current_target()) is not None


def sealed() -> bool:
    """Sealed = bundled LAYOUT only: the store sits beside the bundle
    manifest and is read-only as shipped."""
    return (paths.store_root().parent / "manifest.json").is_file()


def _refuse_lazy(name: str, what: str) -> InstallError:
    from pm import receipt

    error = InstallError(
        name,
        f"not installed and lazy installs are disabled: {what}",
        "enable security.allow_lazy_installs or run `hermes pm install`",
    )
    receipt.record_refusal("lazy-install", str(error))
    return error


def _remove_entry(store: Store, entry_name: str, *, attempts: int = 5) -> None:
    """Remove a replaced or failed entry, retrying transient Windows holds.

    Corruption may leave a file where the directory belonged. Failure
    must propagate so recovery never claims to have removed surviving bytes.
    """
    import time

    entry = store.entry(entry_name)
    for attempt in range(attempts):
        try:
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink(missing_ok=True)
            else:
                remove_tree(entry)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


_SET_ASIDE_PREFIX = ".reclaim-"


def _discard_entry(store: Store, entry_name: str) -> None:
    """Drop a tree a finished publish or restore left behind as garbage.

    A running Hermes process can keep the replaced interpreter's DLLs mapped
    far past the retry window, and failing here reports a completed install
    as broken (#124807). Windows still allows renaming a tree with mapped
    images, so it moves to a `.reclaim-*` name: its slot is free for the next
    publish, no restore ever picks it up, and the next install or
    `hermes pm gc` deletes it once the hold is gone.
    """
    import uuid

    try:
        _remove_entry(store, entry_name)
        return
    except OSError as e:
        error = e
    try:
        store.entry(entry_name).rename(store.entry(f"{_SET_ASIDE_PREFIX}{uuid.uuid4().hex}"))
    except OSError:
        LOG.warning("could not remove or set aside %s: %s", entry_name, error)
        return
    LOG.warning("%s is still in use (%s); set aside for the next install or `hermes pm gc`",
                entry_name, error)


def _reclaim_set_aside(store: Store) -> int:
    """Delete set-aside trees whose hold is gone; the caller holds the store lock."""
    removed = 0
    for item in sorted(store.root.glob(f"{_SET_ASIDE_PREFIX}*")):
        try:
            _remove_entry(store, item.name, attempts=1)
        except OSError:
            continue
        removed += 1
    return removed


def _remove_downloads(store: Store, artifacts: list[dict]) -> None:
    """Release this package's archives after publication, under its store lock."""
    for artifact in artifacts:
        _discard_entry(store, f"fetch-{artifact['sha256']}")


def _entry_verified(package: Package, fact: dict, store: Store, target: str) -> bool:
    """Explicit installs re-check realized bytes; startup keeps its cheap facts check."""
    entry = store.entry(fact["entry"])
    try:
        return not package.verify(entry, target) and fact.get("digest") == tree_digest(entry)
    except OSError:
        return False


def _restore_previous_entry(store: Store, entry, previous) -> None:
    """Keep both versions recoverable until the restore rename succeeds."""
    import uuid

    displaced = store.entry(f".displaced-{uuid.uuid4().hex}")
    had_entry = entry.exists() or entry.is_symlink()
    if had_entry:
        retry_held(lambda: entry.rename(displaced))
    try:
        retry_held(lambda: previous.rename(entry))
    except BaseException:
        if had_entry:
            retry_held(lambda: displaced.rename(entry))
        raise
    if had_entry:
        _discard_entry(store, displaced.name)


@contextmanager
def _publish_entry(package, store, staged, entry, previous_entry, target):
    """Keep rollback live through the caller's native facts commit, if any."""
    if entry.exists() or entry.is_symlink():
        retry_held(lambda: entry.rename(previous_entry))
    try:
        store.publish(staged, entry.name)
        reason = package.verify(entry, target)
        if reason:
            raise InstallError(package.name, f"published entry failed verification: {reason}")
        yield
    except BaseException:
        if previous_entry.exists():
            _restore_previous_entry(store, entry, previous_entry)
        raise
    if previous_entry.exists():
        _discard_entry(store, previous_entry.name)


def _settle_previous_entry(package, store, entry, previous_entry, previous, target) -> None:
    """Finish or undo a publication an earlier install left behind."""
    if not previous_entry.exists():
        return
    # Facts commit last. Stages have no host-side commit record, so
    # an interrupted stage always restores its prior usable bytes.
    if (previous and previous.get("entry") == entry.name
            and _entry_verified(package, previous, store, target)):
        _discard_entry(store, previous_entry.name)
    else:
        _restore_previous_entry(store, entry, previous_entry)


def _entry_current(package, lockfile, facts, store, entry, previous, version, pin, target) -> bool:
    if facts is not None:
        return previous is not None and facts.installed(
            package.name, version, store.root, _identity(lockfile, package.name, target)
        ) and _entry_verified(package, previous, store, target)
    try:
        recorded = (entry / ".pm-stage-pin.json").read_text(encoding="utf-8-sig")
    except OSError:
        recorded = None
    return recorded == pin and not package.verify(entry, target)


def _copy_verified_source(package, lockfile, copy_from, staged, version, target) -> None:
    source_facts, source_store = copy_from
    source = source_facts.get(package.name)
    if (not source_facts.installed(package.name, version, source_store.root,
                                   _identity(lockfile, package.name, target))
            or not _entry_verified(package, source, source_store, target)):
        raise InstallError(package.name, "bundled copy source failed verification")
    shutil.copytree(source_store.entry(source["entry"]), staged, symlinks=True)
    if tree_digest(staged) != source["digest"]:
        raise InstallError(package.name, "copied bytes do not match the bundled source")


def _log_repair(package, previous, version, artifacts) -> None:
    """Work item 6: replacing an ESTABLISHED fact is a repair — log it,
    no transaction system, no receipt file."""
    if not previous or "entry" not in previous:
        return
    old_artifact = (previous.get("artifacts") or ["?"])[0]
    old = f"{previous.get('version', '?')}/{str(old_artifact)[:12]}"
    new = f"{version}/{artifacts[0]['sha256'][:12]}" if artifacts else version
    LOG.info("repair: %s re-realized %s -> %s", package.name, old, new)


def _install(
    package: Package,
    lockfile: Lockfile,
    facts: Facts | None,
    store: Store,
    target: str,
    progress=None,
    pause_event: threading.Event | None = None,
    download_progress: ProgressFn | None = None,
    *,
    copy_from: tuple[Facts, Store] | None = None,
    _lock_held: bool = False,
    _fresh_copy: bool = False,
) -> Path:
    """Realize one pin. Host installs commit facts; cross-target stages carry a marker."""
    version = lockfile.version(package.name)
    if version is None:
        raise InstallError(
            package.name, "not in the lockfile", "add it with `hermes pm lock --bump`"
        )

    reason = package.missing_reason(target)
    if reason is not None:
        raise InstallError(package.name, f"unavailable on {target}: {reason}", "none")

    entry_name = package.store_entry(version, target)
    entry = store.entry(entry_name)
    if getattr(package, "pin_only", False):
        return entry
    artifacts = lockfile.artifacts(package.name, target)
    pin = json.dumps({"target": target, "sha256": [a["sha256"] for a in artifacts]})

    with nullcontext() if _lock_held else store.install_lock():
        if pause_event is not None and pause_event.is_set():
            raise DownloadPaused("install paused")
        if facts is not None:
            facts.reload()
        previous = facts.get(package.name) if facts is not None else None
        previous_entry = store.entry(f".previous-{'stage-' if facts is None else ''}{entry_name}")
        _settle_previous_entry(package, store, entry, previous_entry, previous, target)
        _reclaim_set_aside(store)
        if (_entry_current(package, lockfile, facts, store, entry, previous, version, pin, target)
                and not _fresh_copy):
            _remove_downloads(store, artifacts)
            return entry
        if not artifacts:
            raise InstallError(
                package.name,
                f"no artifact for {target} in the lockfile",
                "run `hermes pm lock --bump` for this package",
            )
        with store.scratch() as scratch:
            staged = scratch / "tree"
            try:
                if copy_from is not None:
                    _copy_verified_source(package, lockfile, copy_from, staged, version, target)
                else:
                    staged = _prepare_artifacts(package, store, scratch, artifacts, version, target,
                                                progress=progress, pause_event=pause_event,
                                                download_progress=download_progress)
                if pause_event is not None and pause_event.is_set():
                    raise DownloadPaused("install paused")
                if progress is not None:
                    progress("verify", 0, 0, "")
                reason, remedy = package.verify(staged, target), ""
                if reason:
                    reason, remedy = package.repair_staged_verification(staged, target, reason)
                if reason:
                    raise InstallError(package.name, f"staged entry failed verification: {reason}", remedy)
                if facts is None:
                    (staged / ".pm-stage-pin.json").write_text(pin, encoding="utf-8")
                with _publish_entry(package, store, staged, entry, previous_entry, target):
                    if facts is not None:
                        facts.record(
                            package.name, version, entry_name, package.env(entry, target), store.root,
                            target=target, artifacts=[a["sha256"] for a in artifacts],
                            digest=tree_digest(entry),
                        )
                _remove_downloads(store, artifacts)
            except (InstallError, DownloadPaused):
                raise
            except Exception as e:
                raise InstallError(package.name, f"install failed: {e}") from e

        _log_repair(package, previous, version, artifacts)
    return entry


def stage_only(
    name: str, target: str, progress=None, *,
    pause_event: threading.Event | None = None,
    download_progress: ProgressFn | None = None,
) -> Path:
    """Realize a cross-target pin without host facts or an executable Runner."""
    return _install(get_package(name), _lockfile(), None, _store(), target,
                    progress=progress, pause_event=pause_event, download_progress=download_progress)


class _InstallOperation:
    """Validity lasts only while this operation holds the publication lock."""

    def __init__(self) -> None:
        self.stack = ExitStack()
        self.store: Store | None = None
        self.checked: set[tuple[str, str | None, str, str]] = set()

    def lock(self) -> Store:
        if self.store is None:
            self.store = Store(paths.writable_store_root())
            self.stack.enter_context(self.store.install_lock())
        return self.store

    def close(self) -> None:
        self.checked.clear()
        self.store = None
        self.stack.close()


@contextmanager
def _install_operation():
    operation = _InstallOperation()
    try:
        yield operation
    finally:
        operation.close()


def ensure(
    name: str,
    *,
    base_env: Optional[dict] = None,
    explicit: bool = False,
    verify: bool = True,
    progress=None,
    pause_event: threading.Event | None = None,
    download_progress: ProgressFn | None = None,
    _operation: _InstallOperation | None = None,
) -> Runner:
    """``explicit`` marks a deliberate install command (`hermes pm
    install`, `hermes pm bundle`) — those ARE the remedy the lazy-install
    policy names, so the policy does not apply to them.

    ``verify`` re-hashes an already-recorded entry and repairs it when the
    bytes moved. A deliberate install keeps that check. Shell activation
    and the source-build tail (moments after the install/update's own PM
    step) pass ``False``. It trusts the recorded digest, the same check
    startup uses, because hashing every tool tree costs seconds per call.

    ``progress(stage, done, total, label)`` reports the slow parts of an
    install to a UI, including ordered multi-archive labels.
    """
    if isinstance(get_package(name), StatePackage):
        if _operation is not None:
            # Python construction can provision tools itself. Drop both the
            # lock and its validity before entering that independent operation.
            _operation.close()
        sync_venv(explicit=explicit)
        return Runner(name, compose_env([], base=base_env))

    lockfile = _lockfile()
    target = current_target()
    chain = walk([name])
    checked = _operation.checked if _operation is not None else set()
    if _operation is not None:
        _operation.lock()
    missing = []
    for package in chain:
        identity = (package.name, lockfile.version(package.name), target,
                    json.dumps(_identity(lockfile, package.name, target), sort_keys=True))
        if identity in checked:
            continue
        if _installed_location(package, lockfile, target, verify=explicit and verify) is None:
            missing.append(package)
        else:
            checked.add(identity)
    if missing and not explicit and not lazy_installs_allowed():
        raise _refuse_lazy(name, ", ".join(p.name for p in missing))
    if missing:
        store = _operation.lock() if _operation is not None else Store(paths.writable_store_root())
        facts = _facts() if store.root == _store().root else Facts(store.root / "facts.json")
        for package in missing:
            # Publication may change entries; do not carry observations across it.
            checked.clear()
            _install(package, lockfile, facts, store, target, progress=progress,
                     pause_event=pause_event, download_progress=download_progress,
                     _lock_held=_operation is not None)
    return Runner(name, env_for(name, base_env=base_env))


def env_for(*names: str, base_env: Optional[dict] = None) -> dict[str, str]:
    """Composed env of already-installed packages only. Never installs,
    never raises on missing packages — they contribute nothing."""
    lockfile = _lockfile()
    target = current_target()
    diffs: list[dict] = []
    for name in names:
        try:
            chain = walk([name])
        except KeyError:
            continue
        for package in chain:
            if package.internal:
                continue
            location = _installed_location(package, lockfile, target)
            if location:
                facts, store = location
                diffs.append(facts.env_for(package.name, store.root))
    return compose_env(diffs, base=base_env)


def _runtime_state_matches(fact: dict, stamp: str, *, project_root: Path | None = None) -> bool:
    if not isinstance(fact, dict) or fact.get("stamp") != stamp:
        return False
    from pm.environments import selected_venv

    try:
        environment = selected_venv(paths.repo_root() if project_root is None else project_root)
    except (OSError, RuntimeError, ValueError):
        return False
    recorded = fact.get("environment")
    if recorded is not None and (not isinstance(recorded, str) or Path(recorded).resolve() != environment):
        return False
    return (environment / "pyvenv.cfg").is_file()


def _member_inputs(plugins: PluginInput | None) -> dict:
    """The ``plugin_dirs`` argument of the venv package; empty means config discovery."""
    from pm.publication import candidate_members

    if isinstance(plugins, Candidates):
        return {"plugin_dirs": candidate_members(plugins.dirs)}
    if isinstance(plugins, Members):
        return {"plugin_dirs": plugins.dirs}
    if plugins is None:
        return {}
    raise TypeError(f"{type(plugins).__name__} changes plugin state; only a sync may carry it")


def _extra_key(name: str) -> str:
    """The PEP 685 name uv matches an extra by (``foo_bar`` is ``foo-bar``)."""
    import re

    return re.sub(r"[-_.]+", "-", name).lower()


def _still_declared(package, recorded: list[str]) -> list[str]:
    """The recorded extras this tree still declares.

    An extra the source removed (``hindsight``) would otherwise ride the ledger
    into every later ``uv sync`` and fail it with "Extra is not defined". Only
    recorded extras are pruned; an explicitly requested unknown extra still fails.
    Membership uses PEP 685 names (uv matches ``foo_bar`` to ``foo-bar``); the
    recorded spelling is what reaches uv.
    """
    from pm.features import declared_extras

    root = package.project_root()
    if not (root / "pyproject.toml").is_file():
        return list(recorded)
    declared = {_extra_key(extra) for extra in declared_extras(root)}
    return [extra for extra in recorded if _extra_key(extra) in declared]


def venv_is_current(*, extras: list[str] | None = None, plugins: Members | Candidates | None = None,
                    project_root: Path | None = None) -> bool:
    """Probe the requested union without changing recorded dependency state."""
    from pm.environments import runtime_facts_path
    from pm.packages import Venv

    root = paths.repo_root() if project_root is None else Path(project_root).absolute()
    package = get_package("venv") if project_root is None else Venv(root)
    fact = Facts(runtime_facts_path(root), strict=True).get("venv")
    if fact is None:
        fact = Facts(paths.facts_path(), strict=True).get("venv")
    if fact is None:
        return False
    if (not isinstance(fact, dict) or not isinstance(fact.get("stamp"), str) or not fact["stamp"]
            or not isinstance(fact.get("extras"), list)
            or any(not isinstance(extra, str) for extra in fact["extras"])):
        raise ValueError("invalid recorded dependency state")
    enabled = sorted(set(_still_declared(package, fact["extras"])) | set(extras or []))
    stamp = package.expected_stamp(enabled, **_member_inputs(plugins))
    return _runtime_state_matches(fact, stamp, project_root=root)


# --- the generation pool ------------------------------------------------------------
# A borrowing data root (a test's temporary HERMES_HOME, a per-task home) builds its
# dependency environment from nothing: a full ``uv sync`` over pyproject.toml + uv.lock,
# minutes and hundreds of megabytes, for a generation whose content the recorded stamp
# already determines. The pool is a read-only cache of finished generations keyed by that
# stamp; a borrower HARDLINKS one into its own install state instead of building it, and
# records it exactly as a sync would. Nothing here may ever fail a launch: every anomaly
# on the launch path is a miss, because a slow launch is a launch (plan §6, IV5).

POOL_ENV = "HERMES_GENERATION_POOL"                      # absolute pool root override
POOL_DISABLE_ENV = "HERMES_DISABLE_GENERATION_POOL"      # kill switch: today's behaviour
POOL_MAX_BYTES_ENV = "HERMES_GENERATION_POOL_MAX_BYTES"  # retention budget
_POOL_DEFAULT_MAX_BYTES = 4 * 1024**3
_POOL_MANIFEST = "manifest.json"
_POOL_INCOMPLETE = ".incomplete"
_POOL_SUSPECT_DIR = ".suspect"
_POOL_LOCKS_DIR = ".locks"
_POOL_TMP_DIR = "tmp"
_POOL_TMP_MAX_AGE = 3600.0
_POOL_SEED_LOCK_TIMEOUT = 2.0


def _pool_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _pool_enabled(project_root: Path, *, owner: Path | None) -> bool:
    """Cheap pre-check: env flags and owning-root identity only. Never touches the disk."""
    if _pool_truthy(os.environ.get(POOL_DISABLE_ENV)):
        return False
    if owner is None:
        return False  # the owner root IS the source of truth: it never seeds from itself
    try:
        return Path(project_root).resolve() != Path(owner).resolve()
    except OSError:
        return False


def generation_pool_root(project_root: Path, *, owner: Path | None = None) -> Path | None:
    """The pool this launch may use, or ``None`` (kill switch, owner root, bad override).

    Resolution order: ``HERMES_GENERATION_POOL`` (absolute) -> ``<owner>/generation-pool``.
    Never creates the pool: a read does not provision state, and a launch that finds no
    pool simply builds as it always did.
    """
    if not _pool_enabled(project_root, owner=owner):
        return None
    override = os.environ.get(POOL_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_absolute() else None
    return Path(owner).resolve() / "generation-pool"


def _pool_device(path: Path) -> int | None:
    """``st_dev`` of *path* or of its nearest existing ancestor; ``None`` when unreadable."""
    for candidate in (Path(path), *Path(path).parents):
        try:
            return os.stat(candidate).st_dev
        except OSError:
            continue
    return None


def _pool_same_device(pool: Path, target: Path) -> bool:
    """``st_dev`` equality: cross-device hardlinks raise EXDEV, so the pool must be disabled."""
    pool_device, target_device = _pool_device(pool), _pool_device(target)
    return pool_device is not None and pool_device == target_device


def _pool_safe_name(name: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in name)[:120]


@contextmanager
def _pool_lock(pool: Path, name: str, *, exclusive: bool):
    """flock a pool-scoped lock file (``<pool>/.locks/<name>.lock``); yield False, never wait.

    Lock files live at the pool root because a published entry directory is 0555 and cannot
    hold a created file. A platform without ``fcntl`` (Windows) yields True: entries are
    content-addressed and published by rename, so the remaining race is benign, while the
    retention budget and the 0555 modes still bound the damage.
    """
    try:
        import fcntl
    except ImportError:
        yield True
        return
    try:
        locks = Path(pool) / _POOL_LOCKS_DIR
        locks.mkdir(parents=True, exist_ok=True)
        fd = os.open(locks / f"{_pool_safe_name(name)}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        yield False
        return
    try:
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)


def read_pool_manifest(pool: Path, stamp: str) -> dict | None:
    """The validated manifest for *stamp*, or ``None`` — never raises.

    Absent, unreadable, malformed, still-incomplete, marked suspect, carrying a foreign
    field type, or a directory without the three markers a managed generation always has
    all answer ``None`` (a pool miss). A broken pool may not break a launch.
    """
    if not isinstance(stamp, str) or not stamp or not stamp.isalnum():
        return None
    try:
        entry = Path(pool) / stamp
        if (not entry.is_dir() or (entry / _POOL_INCOMPLETE).exists()
                or (Path(pool) / _POOL_SUSPECT_DIR / stamp).exists()):
            return None
        if not _managed_generation(entry):
            return None
        manifest = json.loads((entry / _POOL_MANIFEST).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get("schema") != 1 or manifest.get("stamp") != stamp:
        return None
    extras = manifest.get("extras")
    if (not isinstance(extras, list) or any(not isinstance(extra, str) for extra in extras)
            or not isinstance(manifest.get("members_stamp"), str)
            or not isinstance(manifest.get("target"), str)
            or not isinstance(manifest.get("python"), str)
            or not isinstance(manifest.get("files"), int)
            or not isinstance(manifest.get("bytes"), int)):
        return None
    return manifest


def _managed_generation(generation: Path) -> bool:
    """A generation PM built and finished: lease marker, interpreter, resolved lock.

    ``selected_venv`` falls back to the in-tree ``venv/`` for an install PM has never
    touched; a seed must never clone that (measured: 417,380 files, no ``workspace/uv.lock``).
    """
    try:
        generation = Path(generation)
        return ((generation / ".lease-managed").is_file()
                and (generation / "venv" / "pyvenv.cfg").is_file()
                and (generation / "workspace" / "uv.lock").is_file())
    except OSError:
        return False


def _generation_of(fact: dict) -> Path | None:
    """The managed generation a recorded ``venv`` fact names, or ``None`` (C5 guard)."""
    if not isinstance(fact, dict):
        return None
    environment, resolved = fact.get("environment"), fact.get("resolved_lock")
    if not isinstance(environment, str) or not isinstance(resolved, str):
        return None
    try:
        generation = Path(environment).resolve().parent
        if not _managed_generation(generation):
            return None
        if Path(resolved).resolve() != generation / "workspace" / "uv.lock":
            return None
    except (OSError, ValueError):
        return None
    return generation


def _pool_discard(path: Path) -> None:
    """Remove a clone or an entry, softening directory modes first (a 0555 tree raises EACCES)."""
    _soften_dirs(path)
    shutil.rmtree(path, ignore_errors=True)


def _soften_dirs(path: Path) -> None:
    """chmod every directory under *path* 0755, bottom-up.

    ``copytree`` preserves the entry's 0555 directory mode, and ``rmtree`` of a 0555 tree
    raises PermissionError — an unremovable generation, i.e. the orphan bug class. Files
    stay as they are: a writable parent directory is all that removal needs.
    """
    try:
        walker = os.walk(path, topdown=False, followlinks=False)
        for current, _dirs, _files in walker:
            try:
                Path(current).chmod(0o755)
            except OSError:
                continue
    except OSError:
        return


def _harden(entry: Path, *, root: bool = True) -> None:
    """Seal what this tree owns: directories 0555, publish-written files 0444/0555.

    ``chmod`` is per-inode and ``_clone`` hardlinks, so a cloned file IS the source
    generation's file. Sealing cloned files read-only would seal the *publishing* root's
    own selected environment with them (same inode), and the two can never be separated
    again — restoring the source's modes would unseal the entry, because the mode lives on
    the shared inode. It would also buy nothing: the source's owner can chmod it back. So
    only inodes this tree owns are sealed: the directories (``copytree`` creates those
    fresh) and files publish wrote itself (``st_nlink == 1``, i.e. ``manifest.json``).
    Structural immutability is what the mode can actually deliver here — a 0555 directory
    can neither gain, lose nor rename an entry, and a 0555 entry root cannot be swapped by
    a publishing race. File *content* immutability rests on the generation invariant
    instead: nothing writes into a committed generation in place.

    ``root=False`` leaves the tree's own top directory writable: ``rename(2)`` of a 0555
    directory raises EACCES on Linux (measured), and publish makes the entry visible by
    renaming its staging tree into place. The caller seals the root straight after.
    """
    import stat as statmod

    for current, dirs, files in os.walk(entry, topdown=False, followlinks=False):
        for name in files:
            path = Path(current) / name
            try:
                info = path.lstat()
                if statmod.S_ISLNK(info.st_mode) or info.st_nlink > 1:
                    continue  # a symlink, or an inode shared with the source generation
                path.chmod(0o555 if info.st_mode & 0o111 else 0o444)
            except OSError:
                continue
        for name in dirs:
            path = Path(current) / name
            try:
                if not path.is_symlink():
                    path.chmod(0o555)
            except OSError:
                continue
    if not root:
        return
    try:
        Path(entry).chmod(0o555)
    except OSError:
        pass


def _clone(source: Path, destination: Path) -> None:
    """Materialize *source* at *destination* as hardlinks (zero additional data blocks)."""
    shutil.copytree(source, destination, symlinks=True, copy_function=os.link, dirs_exist_ok=True)
    _soften_dirs(destination)


def _verify_clone(generation: Path) -> str | None:
    """``None`` when the clone is usable, else a reason string.

    The interpreter probe imports the packages a launch actually needs; ``yaml`` is NOT
    installed (YAML is the vendored ``hermes_yaml``), so probing it would fail every clone.
    """
    import subprocess

    from pm.environments import venv_python

    venv = Path(generation) / "venv"
    config = venv / "pyvenv.cfg"
    try:
        lines = config.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return "seeded environment has no readable pyvenv.cfg"
    home = None
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip() == "home" and value.strip():
            home = Path(value.strip())
    if home is not None and not home.is_dir():
        return f"seeded environment's interpreter home is missing: {home}"
    python = venv_python(venv)
    if not python.exists():
        return f"seeded environment has no interpreter: {python}"
    probe = "import sys, ruamel.yaml, hermes_yaml, hermes_cli; print(sys.prefix)"
    try:
        result = subprocess.run([str(python), "-I", "-c", probe], capture_output=True, text=True,
                                timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"seeded environment cannot start its interpreter: {exc}"
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()
        return "seeded environment failed its import probe" + (f": {tail[-1]}" if tail else "")
    printed = [line.strip() for line in (result.stdout or "").strip().splitlines() if line.strip()]
    prefix = printed[-1] if printed else ""
    try:
        matches = bool(prefix) and Path(prefix).resolve() == venv.resolve()
    except OSError:
        matches = False
    return None if matches else f"seeded interpreter reports {prefix!r}, not {venv}"


def _manifest(generation: Path, *, stamp: str, extras: list[str], members: str,
              target: str, python: str) -> dict:
    """The entry's provenance record: one rglob pass for ``files``/``bytes``."""
    import time

    files = total = 0
    for item in Path(generation).rglob("*"):
        try:
            if item.is_symlink() or item.is_dir():
                continue
            total += item.stat().st_size
        except OSError:
            continue
        files += 1
    return {"schema": 1, "stamp": stamp, "extras": list(extras), "members_stamp": members,
            "target": target, "python": python, "files": files, "bytes": total,
            "created_at": int(time.time())}


def _pool_manifest_fields() -> tuple[str, str, str]:
    """``(members_stamp, target, python pin)`` as far as a pre-venv process can tell.

    Diagnostic only — the stamp is the key and the seed re-derives it. Each field is
    best-effort because this runs in the bootstrap interpreter, where the config readers
    that plugin discovery needs are not installed yet.
    """
    members = target = python = ""
    try:
        from pm.workspace import enabled_member_dirs, members_stamp

        members = members_stamp(enabled_member_dirs())
    except Exception:
        pass
    try:
        from pm.store import current_target

        target = current_target()
    except Exception:
        pass
    try:
        from pm.lock import Lockfile
        from pm.paths import lockfile_path

        python = Lockfile(lockfile_path()).version("python") or ""
    except Exception:
        pass
    return members, target, python


def _pool_note(message: str) -> None:
    """One stderr line: the launch continues, the operator can see why the pool was skipped."""
    import sys

    try:
        print(f"hermes: generation pool: {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _pool_max_bytes() -> int:
    try:
        return int(os.environ.get(POOL_MAX_BYTES_ENV, "") or _POOL_DEFAULT_MAX_BYTES)
    except ValueError:
        return _POOL_DEFAULT_MAX_BYTES


def _sweep_pool_tmp(pool: Path) -> None:
    """Drop abandoned staging trees; the caller holds the entry lock."""
    import time

    tmp_root = Path(pool) / _POOL_TMP_DIR
    try:
        items = list(tmp_root.iterdir())
    except OSError:
        return
    for item in items:
        try:
            if time.time() - item.stat().st_mtime < _POOL_TMP_MAX_AGE:
                continue
        except OSError:
            continue
        _pool_discard(item)


def _evict(pool: Path, *, keep: str) -> None:
    """Enforce the retention budget, oldest first, never touching *keep* or a busy entry.

    Safe by construction: clones are hardlinks, so removing an entry never invalidates one.
    """
    budget = _pool_max_bytes()
    if budget <= 0:
        return
    entries = []
    total = 0
    try:
        items = list(Path(pool).iterdir())
    except OSError:
        return
    for item in items:
        if item.name.startswith(".") or not item.is_dir():
            continue
        manifest = read_pool_manifest(pool, item.name)
        if manifest is None:
            continue
        try:
            entries.append((item.stat().st_mtime, item, int(manifest.get("bytes") or 0)))
        except OSError:
            continue
        total += entries[-1][2]
    if total <= budget:
        return
    for _mtime, item, size in sorted(entries, key=lambda row: row[0]):
        if total <= budget:
            break
        if item.name == keep:
            continue
        with _pool_lock(pool, item.name, exclusive=True) as held:
            if not held:
                continue
            _pool_discard(item)
            total -= size


def publish_generation_to_pool(project_root: Path, *, owner: Path | None = None,
                               pool: Path | None = None) -> Path | None:
    """Publish the generation this root just committed, keyed by its recorded stamp.

    Returns the published entry path, or ``None`` when there is nothing to publish (owner
    root, kill switch, no record, already published, cross-device, lock busy, any error).
    Never raises. Atomic by rename: readers only ever see a complete, hardened entry.
    """
    import errno
    import uuid

    try:
        from pm.environments import install_state_dir, runtime_facts_path

        if _pool_truthy(os.environ.get(POOL_DISABLE_ENV)):
            return None  # the kill switch is the rollback contract: today's behaviour, exactly
        root = Path(project_root).absolute()
        pool = Path(pool) if pool is not None else generation_pool_root(root, owner=owner)
        if pool is None:
            return None
        fact = Facts(runtime_facts_path(root)).get("venv")
        stamp, extras = (fact.get("stamp"), fact.get("extras")) if isinstance(fact, dict) else (None, None)
        if (not isinstance(stamp, str) or not stamp
                or not isinstance(extras, list) or any(not isinstance(extra, str) for extra in extras)):
            return None
        generation = _generation_of(fact)
        if generation is None:
            return None
        entry = pool / stamp
        if entry.is_dir():
            return None
        if not _pool_same_device(pool, install_state_dir(root)):
            _pool_note(f"{pool} is on another filesystem from this data root; not publishing")
            return None
        pool.mkdir(parents=True, exist_ok=True)
        with _pool_lock(pool, stamp, exclusive=True) as held:
            if not held or entry.is_dir():
                return None
            _sweep_pool_tmp(pool)
            staged = pool / _POOL_TMP_DIR / uuid.uuid4().hex
            try:
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.mkdir()
                (staged / _POOL_INCOMPLETE).touch()
                _clone(generation, staged)
                members, target, python = _pool_manifest_fields()
                manifest = _manifest(generation, stamp=stamp, extras=sorted(extras),
                                     members=members, target=target, python=python)
                (staged / _POOL_MANIFEST).write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                (staged / _POOL_INCOMPLETE).unlink()
                _harden(staged, root=False)
            except BaseException as exc:
                _pool_discard(staged)
                if not isinstance(exc, Exception):
                    raise
                if isinstance(exc, OSError) and exc.errno == errno.EXDEV:
                    _pool_note(f"{pool} is on another filesystem; not publishing")
                return None
            try:
                os.replace(staged, entry)
            except OSError as exc:
                # ENOTEMPTY/EEXIST: a concurrent publish won — the desired state, not an error.
                _pool_discard(staged)
                if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    return None
                return entry
            try:
                entry.chmod(0o555)
            except OSError:
                pass  # the entry is complete and read-only below its root; sealing is best effort
            _evict(pool, keep=stamp)
            return entry
    except Exception:
        return None


def _seed_expected(package, fact: dict | None) -> tuple[list[str], str]:
    """``(enabled extras, expected stamp)`` exactly as a sync would derive them."""
    shipped, _frozen = _feature_policy(None, repair=False)
    recorded = fact.get("extras") if isinstance(fact, dict) else None
    if not isinstance(recorded, list) or any(not isinstance(extra, str) for extra in recorded):
        recorded = []
    enabled = sorted(set(_still_declared(package, recorded)) | set(shipped or []))
    return enabled, package.expected_stamp(enabled, **_member_inputs(None))


def _mark_pool_suspect(pool: Path, stamp: str) -> None:
    """Record a failed clone at the pool ROOT: a hardened entry cannot hold a created file."""
    try:
        suspects = Path(pool) / _POOL_SUSPECT_DIR
        suspects.mkdir(parents=True, exist_ok=True)
        (suspects / stamp).write_text("clone verification failed\n", encoding="utf-8")
    except OSError:
        pass


def _seeded_state_ok(root: Path, stamp: str, generation: Path) -> bool:
    """The in-process verification the caller turns into ``current`` (plan §2.2).

    The delegated probe cannot certify a borrowing home (it has no PM runtime to answer
    it), so the seed certifies its own write: the record names this stamp, the environment
    sits inside this root's own install state, and its ``pyvenv.cfg`` is present.
    """
    from pm.environments import install_state_dir, runtime_facts_path

    try:
        fact = Facts(runtime_facts_path(root), strict=True).get("venv")
        if not isinstance(fact, dict) or fact.get("stamp") != stamp:
            return False
        environment = Path(str(fact.get("environment") or "")).resolve()
        generations = (install_state_dir(root) / "environments").resolve()
        return (environment == (Path(generation) / "venv").resolve()
                and environment.is_relative_to(generations)
                and (environment / "pyvenv.cfg").is_file())
    except (OSError, RuntimeError, ValueError):
        return False


def seed_from_generation_pool(project_root: Path, *, owner: Path | None = None,
                              pool: Path | None = None) -> bool:
    """Materialize this root's expected-stamp entry into its OWN install state.

    ``True`` only when the seed is complete and verified in-process; ``False`` on every
    anomaly, so the caller falls back to building exactly as it does today. The clone and
    the record both land inside ``<this home>/installs/<key>/``: a pool path is never
    selected, so ``_recorded_venv``'s containment guard keeps holding.
    """
    import uuid

    try:
        from hermes_cli.runtime_state import runtime_lock
        from pm.environments import flush_before_selecting, install_state_dir, runtime_facts_path
        from pm.packages import Venv

        if _pool_truthy(os.environ.get(POOL_DISABLE_ENV)):
            return False  # the kill switch is the rollback contract: today's behaviour, exactly
        root = Path(project_root).absolute()
        pool = Path(pool) if pool is not None else generation_pool_root(root, owner=owner)
        if pool is None:
            return False
        state = install_state_dir(root)
        if not _pool_same_device(pool, state):
            return False  # a copy would move 752 MB of data blocks; build instead
        with runtime_lock(root, timeout=_POOL_SEED_LOCK_TIMEOUT) as held:
            if not held:
                return False  # a build owns this install; the next launch will seed
            package = Venv(root)
            fact = Facts(runtime_facts_path(root)).get("venv") or Facts(paths.facts_path()).get("venv")
            enabled, stamp = _seed_expected(package, fact)
            if read_pool_manifest(pool, stamp) is None:
                return False
            entry = pool / stamp
            with _pool_lock(pool, stamp, exclusive=False) as shared:
                if not shared:
                    return False
                generations = state / "environments"
                generations.mkdir(parents=True, exist_ok=True)
                generation = generations / uuid.uuid4().hex
                try:
                    _clone(entry, generation)
                except BaseException as exc:
                    _pool_discard(generation)
                    if not isinstance(exc, Exception):
                        raise
                    _pool_note(f"could not clone the cached environment: {exc}")
                    return False
                try:
                    reason = _verify_clone(generation)
                    if reason is not None:
                        _mark_pool_suspect(pool, stamp)
                        _pool_discard(generation)
                        _pool_note(reason)
                        return False
                    flush_before_selecting()
                    Facts(runtime_facts_path(root)).record_state(
                        "venv", stamp, enabled,
                        environment=generation / "venv",
                        resolved_lock=generation / "workspace" / "uv.lock",
                    )
                except BaseException as exc:
                    _pool_discard(generation)
                    if not isinstance(exc, Exception):
                        raise
                    _pool_note(f"could not record the cached environment: {exc}")
                    return False
                if _seeded_state_ok(root, stamp, generation):
                    return True
                _pool_discard(generation)
                return False
    except Exception as exc:
        _pool_note(f"seed skipped: {exc}")
        return False


def _feature_policy(extras: Optional[list[str]], *, repair: bool) -> tuple[list[str] | None, list[str] | None]:
    """Refuse extras this platform or a frozen bundle cannot carry; return (shipped, frozen)."""
    from pm.features import read_features

    if extras:
        from pm.extras import extra_supported
        unsupported = [extra for extra in extras
                       if not extra_supported(extra, importable=lambda _: False)]
        if unsupported:
            raise InstallError("venv", f"extras {unsupported} are not supported by this Python/platform",
                               "choose a supported provider; no dependency environment was changed")
    shipped = read_features()
    frozen = shipped
    # Without a frozen declaration, explicit source setup needs no policy
    # read: the config loader initializes/chmods unrelated user state.
    if frozen is not None and not repair and lazy_installs_allowed():
        frozen = None
    if frozen is not None and extras:
        outside = sorted(set(extras) - set(frozen))
        if outside:
            raise _refuse_lazy(
                "venv",
                f"extras {outside} are outside this bundle's frozen feature "
                "set (security.allow_lazy_installs is false)",
            )
    return shipped, frozen


@contextmanager
def _venv_install_lock(*, patient: bool):
    """Hold the dependency lock, or refuse when an impatient caller would queue."""
    from pm import receipt
    from hermes_cli.runtime_state import INSTALL_LOCK_TIMEOUT_SECONDS, runtime_lock

    # Holding this lock means rebuilding the whole dependency environment, which takes tens of
    # seconds on a bundle. Only an install the user asked for may queue for it; an opportunistic
    # one (a lazy extra at first use, the only non-explicit caller) refuses instead of holding
    # a sibling profile's backend off its port behind a rebuild it did not request.
    with runtime_lock(paths.repo_root(), timeout=None if patient else INSTALL_LOCK_TIMEOUT_SECONDS) as held:
        if not held:
            error = InstallError(
                "venv",
                f"another Hermes process is installing dependencies (waited {INSTALL_LOCK_TIMEOUT_SECONDS:.0f}s)",
                "retry in a moment, or run `hermes pm install` to install explicitly",
            )
            receipt.record_refusal("install-busy", str(error))
            raise error
        yield


def _publication(plugins: PluginInput | None):
    """Snapshot a plugin state change under the held lock, or None for member-only inputs."""
    from pm.publication import PluginSelection, StagedPlugin

    if isinstance(plugins, Selection):
        return PluginSelection(dict(plugins.data))
    if isinstance(plugins, StagedUpdate):
        return StagedPlugin(dict(plugins.data))
    return None


def _publish_inactive(change) -> None:
    """A disabled plugin's code changes without touching the dependency environment."""
    from pm import receipt
    from hermes_cli.runtime_state import finish_publication, recover_publication

    try:
        change.publish(paths.repo_root())
        finish_publication(paths.repo_root())
    except BaseException:
        recover_publication(paths.repo_root())
        raise
    receipt.record_venv_rebuild(False, "inactive plugin")


def _target_selection(package, fact: dict, *, extras, inputs: dict, repair: bool, shipped, frozen):
    """Return (enabled extras, expected stamp, package inputs) this sync must reach."""
    if repair:
        if fact and (not isinstance(fact.get("extras"), list)
                     or any(not isinstance(extra, str) for extra in fact["extras"])
                     or not isinstance(fact.get("stamp"), str) or not fact["stamp"]):
            raise InstallError("venv", "recorded dependency selection is incomplete; refusing to change its graph")
        enabled = list(fact.get("extras", frozen if frozen is not None else ["all"]))
        stamp = fact.get("stamp") or package.expected_stamp(enabled, plugin_dirs=[])
        return enabled, stamp, {"repair": True}
    # The first writable generation replaces, rather than layers on,
    # the payload. Retain its extras until a recorded selection owns them.
    enabled = sorted(set(_still_declared(package, fact.get("extras", shipped or []))) | set(extras or []))
    return enabled, package.expected_stamp(enabled, **inputs), inputs


def _commit_selection(package, facts: Facts, change, *, enabled: list[str], stamp: str, inputs: dict,
                      current: bool, repair: bool, explicit: bool, skip_invalid_secondary: bool = False) -> None:
    """Build (unless current), publish the plugin change, then record the selection."""
    from pm import receipt
    from hermes_cli.runtime_state import finish_publication, recover_publication

    try:
        result = {} if current else (package.apply(enabled, explicit=explicit,
                                                   skip_invalid_secondary=skip_invalid_secondary, **inputs) or {})
        if not repair and package.expected_stamp(enabled, **inputs) != stamp:
            raise ValueError("Dependency inputs changed while preparing publication; retry.")
        if change is not None:
            change.publish(paths.repo_root())
        if not current:
            if result.get("environment") is not None:
                from pm.environments import flush_before_selecting
                flush_before_selecting()
            facts.record_state("venv", stamp, enabled, **result)
        if change is not None:
            finish_publication(paths.repo_root())
        receipt.record_venv_rebuild(not current, "already in sync" if current else "")
    except BaseException:
        recover_publication(paths.repo_root())
        raise


def sync_venv(extras: Optional[list[str]] = None, *, explicit: bool = False,
              plugins: PluginInput | None = None, repair: bool = False,
              evict_incompatible_plugins: bool = False) -> None:
    """Make the venv match uv.lock + the enabled extras. Extras union into
    the installed state (one ledger); no-op when the stamp already matches.
    ``repair`` restores the recorded dependency graph into a fresh generation,
    bypassing both that shortcut and config discovery. It cannot add features.
    ``explicit`` marks a deliberate install command (`hermes pm install`,
    `hermes update`) — those are the remedy the lazy-install policy points
    at, so the policy does not apply to them. ``plugins`` names the one
    source of plugin members (see pm.plugin_inputs); None discovers them from config.
    ``evict_incompatible_plugins`` is the update's contract: a discovered plugin that
    keeps the environment from building is disabled instead of failing the sync
    (see pm.plugin_eviction).

    Lazy installs OFF = the frozen feature set: when
    security.allow_lazy_installs is false AND the bundle's
    enabled-features.json exists, the feature list is FROZEN to that file
    — requested extras outside it are refused, and plugin members are
    never installed (the bundle IS the install).

    Receipt contract: EVERY outcome writes a receipt. ``begin`` fires
    BEFORE the frozen/lazy refusals (a refusal is a recorded ``failed``
    outcome, not a silent raise); finalize runs in FINALLY — no-op syncs
    ("ok" with ``venv_rebuild`` false) and refusals ("failed") both get
    a receipt. Plugin selection data is discovered and published by the worker
    under this lock; no executable transaction phases cross the process boundary."""
    from pm import receipt

    token = receipt.begin("sync")
    outcome = "failed"
    try:
        if repair and (extras is not None or plugins is not None):
            raise ValueError("repair restores the recorded environment; it cannot change features or plugins")
        if evict_incompatible_plugins and (repair or plugins is not None or not explicit):
            raise ValueError("only an explicit sync of the discovered plugin selection may disable plugins")
        shipped, frozen = _feature_policy(extras, repair=repair)
        package = get_package("venv")
        from hermes_cli.runtime_state import recover_publication
        from pm.publication import StagedPlugin
        with _venv_install_lock(patient=explicit or repair):
            recover_publication(paths.repo_root())
            change = _publication(plugins)
            if isinstance(change, StagedPlugin) and not change.active:
                _publish_inactive(change)
            elif evict_incompatible_plugins:
                from pm.plugin_eviction import sync_evicting

                facts = Facts(paths.runtime_facts_path())
                fact = facts.get("venv") or _facts().get("venv") or {}
                sync_evicting(package, facts, fact, extras=extras, shipped=shipped, frozen=frozen, explicit=explicit)
            else:
                inputs = {"plugin_dirs": change.members} if change is not None else _member_inputs(plugins)
                facts = Facts(paths.runtime_facts_path(), strict=repair)
                fact = facts.get("venv") or _facts().get("venv") or {}
                enabled, stamp, inputs = _target_selection(package, fact, extras=extras, inputs=inputs,
                                                           repair=repair, shipped=shipped, frozen=frozen)
                current = not repair and _runtime_state_matches(fact, stamp)
                if not current and not repair and not explicit and not lazy_installs_allowed():
                    raise _refuse_lazy("venv", str(extras) if extras else "venv out of sync")
                receipt.record_feature_list(enabled)
                _commit_selection(package, facts, change, enabled=enabled, stamp=stamp, inputs=inputs,
                                  current=current, repair=repair, explicit=explicit)
        outcome = "ok"
    except BaseException as exc:
        receipt.record_step("dependency-sync", False, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt.finalize(outcome, 0 if outcome == "ok" else 1, token=token)


def drift(*, include_venv: bool = True) -> dict[str, str]:
    """Cheap stamp comparisons of the installed state
    against the lockfile. Maps package names to reasons; empty means healthy. Never
    installs, never touches the network. An install pm has never touched
    (no installed-state file) reports nothing — pm only vouches for what
    it installed. Lockfile packages this build doesn't know (version skew
    during a partial update) are skipped, not fatal."""
    if not paths.facts_path().is_file() and not paths.runtime_facts_path().is_file():
        return {}

    problems: dict[str, str] = {}
    lockfile = _lockfile()
    facts = _facts()
    target = current_target()
    for name in lockfile.names():
        try:
            package = get_package(name)
        except KeyError:
            continue
        if package.optional or package.internal:
            continue
        if package.missing_reason(target) is not None:
            continue
        if _installed_location(package, lockfile, target) is None:
            problems[name] = "not installed or outdated"
    try:
        venv = get_package("venv")
    except KeyError:
        venv = None
    if include_venv and venv is not None and (paths.runtime_facts_path().is_file() or facts.get("venv") is not None):
        try:
            if not venv_is_current():
                problems["venv"] = "out of sync with uv.lock"
        except (OSError, RuntimeError, ValueError) as exc:
            problems["venv"] = str(exc)
    return problems


def check(*, include_venv: bool = True) -> list[str]:
    """Human-readable startup diagnostics. Use drift() for package identities."""
    return [f"{name}: {reason}" for name, reason in drift(include_venv=include_venv).items()]


def _dependents_first(names: list[str]) -> list[str]:
    """*names* with each package moved ahead of the packages it depends on.

    A package that extends its dependency must win on PATH: node's own dir also holds node's
    bundled npm and npx, so with node first every ``npm`` ran that copy, not the pinned npm.
    Unrelated packages keep their order.
    """
    order = list(names)
    for name in names:
        try:
            deps = {package.name for package in walk([name])} - {name}
        except KeyError:
            continue
        positions = [order.index(dep) for dep in deps if dep in order]
        if positions and min(positions) < order.index(name):
            order.remove(name)
            order.insert(min(positions), name)
    return order


def _store_path_dirs() -> list[str]:
    """Composed PATH dirs of all installed (non-internal, on_path) store
    packages, each package ahead of its dependencies (``_dependents_first``),
    deduped. Includes optional packages that are
    *installed* (facts say so) — an installed git/gh must be on PATH even
    though it's not in the root closure. A package whose dependency chain is
    not fully installed contributes nothing: store npm over a missing store
    node would run the user's node, a partial toolchain. Never installs."""

    lockfile = _lockfile()
    target = current_target()
    locations: dict[str, object] = {}

    def located(package):
        if package.name not in locations:
            locations[package.name] = _installed_location(package, lockfile, target)
        return locations[package.name]

    dirs: list[str] = []
    for name in _dependents_first(lockfile.names()):
        try:
            package = get_package(name)
            chain = walk([name])
        except KeyError:
            continue
        if package.internal:
            continue
        if not getattr(package, "on_path", True):
            continue
        if package.missing_reason(target) is not None:
            continue
        location = located(package)
        if location is None or any(located(dep) is None for dep in chain
                                   if dep.missing_reason(target) is None):
            continue
        facts, store = location
        env = facts.env_for(name, store.root)
        path_dirs = env.get("PATH") or []
        if isinstance(path_dirs, str):
            path_dirs = [path_dirs]
        for directory in path_dirs:
            if directory and native(directory) not in dirs:
                dirs.append(native(directory))
    return dirs


def activate(*, allow_incomplete: bool = False) -> list[str]:
    """Make the installed store usable: put its tool dirs at the FRONT of
    os.environ['PATH'] so reactive `shutil.which('git'|'node'|'ffmpeg'|...)`
    resolves the bundled binaries, never a user copy that sorts earlier.
    Return `check()`'s problems, or an empty list on success, so startup
    callers can report the verdict without checking the store twice.

    Drift still activates every package whose whole chain is installed
    (`_store_path_dirs`): refusing all of them handed every tool to the
    user's PATH copies. A damaged package, and anything depending on it,
    stays off PATH so no toolchain is served half from the store.

    ``allow_incomplete`` is the install-time exception: tools are published
    before the venv sync, so the venv verdict is not computed at all.

    This is the ONE sanctioned global PATH write: PATH is the discovery
    contract every `which` reads, not a tool-specific env leak. Store-first
    unconditionally — pinned bundled versions win on dev machines too.
    """
    import os

    # The venv verdict is discarded here, and computing it imports application
    # config readers (ruamel) that the bare update interpreter does not carry.
    problems = check(include_venv=not allow_incomplete)
    path = os.environ.get("PATH", "")
    first = store_first_path(path)
    if first != path:
        os.environ["PATH"] = first
    return problems


def store_first_path(path: str) -> str:
    """``path`` with the installed store's tool dirs moved to the front, for
    Hermes's own children whose PATH gets other dirs prepended after activate."""
    import os

    dirs = _store_path_dirs()
    if not dirs:
        return path
    store = {d.lower() for d in dirs}
    rest = [p for p in path.split(os.pathsep) if p and p.lower() not in store]
    return os.pathsep.join([*dirs, *rest])
