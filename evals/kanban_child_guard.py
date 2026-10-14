"""Shared ``--child`` isolation guard for the DB-writing kanban evals.

Why this module exists
----------------------
``evals/kanban_phase_cascade.py`` and ``evals/kanban_graph_identity.py`` both
have a ``--child`` path that writes a **real SQLite kanban DB**, and both are
re-exec'd by their parent wrapper with a temp ``HERMES_HOME``. The first
iteration of the guard (2026-09-12 leak hardening) validated ``HERMES_HOME``
**only** — but kanban path resolution does not read ``HERMES_HOME`` for the DB
path at all. The DB path comes from ``kanban_db.kanban_db_path()``:

    ``HERMES_KANBAN_DB`` -> ``HERMES_KANBAN_BOARD`` -> ``<root>/kanban/current``
    -> ``default``, where ``<root>`` is ``HERMES_KANBAN_HOME`` else
    ``get_default_hermes_root()``.

``HERMES_KANBAN_DB`` is exported into *every* kanban worker session
(``HERMES_KANBAN_DB=<real root>/kanban.db``), so ``HERMES_HOME=<temp>`` with an
inherited ``HERMES_KANBAN_DB`` PASSED the old guard while the write still landed
on the operator's live board. A bare string check on ``HERMES_KANBAN_DB`` would
only move the hole to the next path-pin variable (``HERMES_KANBAN_HOME``,
``HERMES_KANBAN_BOARD``, ``<root>/kanban/current``, a future pin added to
production).

So the predicate is the **effective resolved DB path**, not the env var names:
whatever production would resolve becomes the thing we refuse. A future pin is
covered by construction, because it routes through the same resolver.

Two resolvers, deliberately
---------------------------
1. :func:`_production_db_path` imports ``hermes_cli.kanban_db.kanban_db_path``
   from the ``--repo`` under test — the authoritative answer, no duplication.
2. :func:`_documented_db_path` is a stdlib-only replica of the documented
   resolution order, used ONLY if (1) cannot be imported (e.g. the child is
   invoked before the repo is importable). It refuses fail-closed rather than
   fail-open, and ``tests/hermes_cli/test_kanban_child_guard.py`` asserts the
   two resolvers agree across the full env vector set, so drift is caught.

The REAL root side of the comparison gets the same treatment
------------------------------------------------------------
:func:`_real_hermes_root` is the platform-native default home
(``hermes_constants._get_platform_default_hermes_home()``: ``%LOCALAPPDATA%/hermes``
on ``win32``, ``~/.hermes`` elsewhere), imported from the ``--repo`` under test
with :func:`_platform_home_replica` as the fail-closed fallback. A POSIX-only
derivation (``Path.home() / ".hermes"``) made the guard compare against the
wrong root on Windows, so a live-board pin never matched and was not refused
(under-refusal — the leak class this module exists for).

It is deliberately NOT ``get_default_hermes_root()``: that function honours
``HERMES_HOME``, which is both the variable the leak vectors redirect and the
variable the *sanctioned* parent wrapper sets to its own temp harness — a
``HERMES_HOME``-aware root would make the wrapper's own child self-refuse.

Refusal style is preserved from the original guards: ``sys.exit`` with a message
starting ``refusing --child: `` and ending in the original tail sentence, so
existing greps and exit-code expectations keep matching.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

#: Kept byte-identical to the guards' original tail sentence — operators grep
#: for it and the wrapper's contract is described by it.
_REFUSAL_TAIL = (
    "this child writes a real kanban DB and must only run under the parent "
    "wrapper's temp HERMES_HOME"
)

#: Kept in sync with ``hermes_cli.kanban_db._BOARD_SLUG_RE``.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")

_DEFAULT_BOARD = "default"


def _platform_home_replica() -> Path:
    """Stdlib replica of ``hermes_constants._get_platform_default_hermes_home()``.

    Mirrors the win32 ``%LOCALAPPDATA%`` branch exactly (including the
    ``Path.home() / "AppData" / "Local"`` fallback when the variable is empty),
    so the guard's notion of "the real root" cannot silently diverge if the
    ``--repo`` import is unavailable. A POSIX-only ``Path.home() / ".hermes"``
    here was the defect: on Windows the operator's live board lives at
    ``%LOCALAPPDATA%/hermes``, so the pin was never matched and never refused.
    """
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def _production_platform_home(repo: Optional[Path]) -> Optional[Path]:
    """``hermes_constants._get_platform_default_hermes_home()`` from the ``--repo``.

    Same anti-drift approach as :func:`_production_db_path`: the guard imports the
    platform-root function it must agree with rather than re-implementing the
    platform branch. ``None`` means "could not import" (the replica answers).

    Deliberately the *platform* default, not ``get_default_hermes_root()``: the
    latter honours ``HERMES_HOME``, and an isolated ``HERMES_HOME`` is precisely
    what the sanctioned parent wrapper sets — a ``HERMES_HOME``-aware real root
    would make the wrapper's own child refuse itself.
    """
    try:
        if repo is not None:
            text = str(Path(repo).resolve())
            if text not in sys.path:
                sys.path.insert(0, text)
        from hermes_constants import _get_platform_default_hermes_home

        return Path(_get_platform_default_hermes_home())
    except Exception:
        return None


def _real_hermes_root(repo: Optional[Path] = None) -> Path:
    """The REAL platform-default Hermes root (``%LOCALAPPDATA%/hermes`` on Windows).

    Derived from the OS home / platform, never from ``HERMES_HOME`` —
    ``HERMES_HOME`` is exactly the variable the leak vectors redirect.
    """
    native = _production_platform_home(repo)
    if native is None:
        native = _platform_home_replica()
    return native.resolve()


def _production_db_path(repo: Optional[Path]) -> Optional[Path]:
    """Effective kanban DB path per **production's own resolver**, or ``None``.

    Importing the resolver from the ``--repo`` under test is the point: the
    guard cannot drift from the thing it guards. ``None`` means "could not
    import", which the caller resolves via the documented-order replica.
    """
    try:
        if repo is not None:
            text = str(Path(repo).resolve())
            if text not in sys.path:
                sys.path.insert(0, text)
        from hermes_cli.kanban_db import kanban_db_path

        return Path(kanban_db_path()).expanduser().resolve()
    except Exception:
        return None


def _kanban_home_replica() -> Path:
    """``kanban_home()``: ``HERMES_KANBAN_HOME`` else ``get_default_hermes_root()``.

    The native fallback goes through :func:`_real_hermes_root`, NOT a second
    ``Path.home() / ".hermes"`` literal: the two must answer the same root on
    every platform or the replica can allow what ``refusal_reason`` refuses.
    """
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    native = _real_hermes_root()
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        try:
            resolved = Path(env_home).expanduser().resolve()
        except OSError:
            return native
        if resolved == native or native in resolved.parents:
            # Under ~/.hermes this is a normal root or a profiles/<name> home;
            # ``get_default_hermes_root`` returns the platform root in both cases.
            return native
        return resolved.parent.parent if resolved.parent.name == "profiles" else resolved
    return native


def _board_exists_replica(root: Path, slug: str) -> bool:
    """``kanban_db.board_exists``: ``default`` always exists, else board.json/kanban.db."""
    if slug == _DEFAULT_BOARD:
        return True
    board_dir = root / "kanban" / "boards" / slug
    return (board_dir / "board.json").exists() or (board_dir / "kanban.db").exists()


def _documented_db_path() -> Optional[Path]:
    """Stdlib replica of the documented resolution order (fallback resolver).

    ``HERMES_KANBAN_DB`` -> ``HERMES_KANBAN_BOARD`` -> ``<root>/kanban/current``
    -> ``default``. Mirrors ``kanban_db._board_path`` / ``get_current_board``:
    a malformed, stale or non-existent slug falls through to the default board,
    never raises (a hand-edited ``current`` file must not break the dispatcher).
    """
    try:
        root = _kanban_home_replica()
        override = os.environ.get("HERMES_KANBAN_DB", "").strip()
        if override:
            return Path(override).expanduser().resolve()
        slug = os.environ.get("HERMES_KANBAN_BOARD", "").strip().lower()
        if slug and not _SLUG_RE.match(slug):
            slug = ""
        if slug and not _board_exists_replica(root, slug):
            slug = ""
        if not slug:
            current = root / "kanban" / "current"
            if current.exists():
                candidate = current.read_text(encoding="utf-8").strip().lower()
                if _SLUG_RE.match(candidate) and _board_exists_replica(root, candidate):
                    slug = candidate
        if slug and slug != _DEFAULT_BOARD:
            return (root / "kanban" / "boards" / slug / "kanban.db").resolve()
        return (root / "kanban.db").resolve()
    except Exception:
        return None


def effective_db_path(repo: Optional[Path] = None) -> Optional[Path]:
    """The kanban DB path this child process would actually write, or ``None``."""
    resolved = _production_db_path(repo)
    if resolved is not None:
        return resolved
    return _documented_db_path()


def refusal_reason(repo: Optional[Path] = None) -> Optional[str]:
    """Why a ``--child`` run must be refused, or ``None`` when it is safe.

    Two independent conditions, both fail-closed:

    * ``HERMES_HOME`` is unset or resolves inside the real platform root
      (``%LOCALAPPDATA%/hermes`` on Windows, ``~/.hermes`` elsewhere) — the
      original check, since an unset home would fall back to the live default;
    * the **effective resolved kanban DB path** lands under that real root,
      whatever env var produced it.

    The second condition is the regression this module exists for: an isolated
    ``HERMES_HOME`` plus an inherited ``HERMES_KANBAN_DB`` (standard in every
    kanban worker session) satisfied the first check and still wrote the live
    board.
    """
    real_root = _real_hermes_root(repo)

    home = os.environ.get("HERMES_HOME", "").strip()
    if not home:
        return (
            "HERMES_HOME is unset, so kanban resolution would fall back to the "
            f"real {real_root}"
        )
    try:
        child_home = Path(home).expanduser().resolve()
    except OSError:
        child_home = None
    if child_home is None or child_home == real_root or real_root in child_home.parents:
        return f"HERMES_HOME={home} resolves inside the real {real_root}"

    db_path = effective_db_path(repo)
    if db_path is None:
        return (
            "the effective kanban DB path could not be resolved, so isolation "
            "cannot be proven"
        )
    if db_path == real_root or real_root in db_path.parents:
        return (
            f"the effective kanban DB path {db_path} lands under the real "
            f"{real_root}; HERMES_KANBAN_DB / HERMES_KANBAN_HOME / "
            f"HERMES_KANBAN_BOARD win over HERMES_HOME, so an isolated "
            f"HERMES_HOME is not isolation"
        )
    return None


def require_isolated_child(repo: Optional[Path] = None) -> None:
    """Exit non-zero unless this ``--child`` run is provably DB-isolated."""
    reason = refusal_reason(repo)
    if reason is not None:
        sys.exit(f"refusing --child: {reason}; {_REFUSAL_TAIL}")
