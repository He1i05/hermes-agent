"""Pre-dispatch guard: refuse to spawn a repo-touching worker while the deploy
checkout is off-lane or dirty.

The deploy checkout (``<hermes-home>/hermes-agent``) is the tree systemd runs:
every newly spawned worker, every ``hermes`` CLI call and every no-agent cron
script imports it from disk. A card that says "create a feature branch from the
deploy tree and commit there" therefore serves unreviewed WIP to the live
install with no restart and no rollback marker (incident 2026-11-10, measured at
branch ``upstream/…`` + 15 uncommitted files for ~30 minutes).

``~/.hermes/scripts/deploy_tree_guard.py`` already *detects* that state, but it
runs on a 6-hourly cron — far too late. This module is the prevention half: a
named, sticky refusal at the spawn seam in
:func:`hermes_cli.kanban_db_dispatch._dispatch_lane_task`.

Two facts are asserted about the deploy root, worded with
``deploy_tree_guard.py``'s vocabulary so an operator greps once:

* ``git -C <root> rev-parse --abbrev-ref HEAD`` is ``carry/*`` or ``main`` —
  otherwise ``deploy-tree-off-lane branch=<b> head=<sha8>``;
* ``git -C <root> status --porcelain`` is empty — otherwise
  ``deploy-tree-dirty files=<n>``.

A checkout that exists but whose git state cannot be read is refused as
``deploy-tree-unreadable root=<path>``: for a card that can write there, an
unprovable tree is not a safe tree.

Scope is deliberately narrow: only cards that can touch that checkout are
asserted, because a global assertion wedges the whole board on update-day (when
the tree legitimately moves) and during the repair itself.

* workspace kind ``worktree`` whose resolved workspace sits inside the deploy
  root — the dispatcher materialises ``<repo>/.worktrees/<task-id>``, so the
  checkout is what the git commands in that card write to; or
* a card body that *names* the checkout. That is a text heuristic on purpose:
  the population at risk is exactly the cards that name the tree, and it covers
  hand-authored cards the decomposer never sees. Absolute, ``~/…`` and
  ``$HOME/…`` forms are normalised, so writing the path another way does not
  slip past the guard.

Environment (read at call time, never at import, so tests and a dispatcher
restart are not needed to change either):

``HERMES_KANBAN_DEPLOY_ROOT``
    Absolute path of the deploy checkout to assert. Defaults to
    ``<hermes-home>/hermes-agent`` — the conventional install of the Hermes home
    the dispatcher serves (never a hardcoded ``~/.hermes``). An unset env var on
    an install whose checkout lives elsewhere leaves the guard inert.
``HERMES_ALLOW_OFFLANE_DEPLOY_TREE``
    ``1``/``true``/``yes``/``on`` bypasses the check for every card — the
    dispatcher consults :func:`deploy_tree_bypass` before acting on a reason.
    This is the escape hatch a repo-touching *repair* card needs — otherwise the
    guard blocks its own fix. The dispatcher records a ``deploy_tree_bypassed``
    event (naming the state it overrode) on the card it lets through, so a
    bypass is never invisible.

The probes are read-only (``rev-parse``/``status``) and bound to the deploy root
itself: ``git -C <dir>`` walks up to an enclosing repository, so the toplevel is
verified against the root before any branch is believed.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Mapping
from typing import Optional

DEPLOY_ROOT_ENV = "HERMES_KANBAN_DEPLOY_ROOT"
ALLOW_OFFLANE_ENV = "HERMES_ALLOW_OFFLANE_DEPLOY_TREE"

# ``<hermes-home>/hermes-agent`` — the conventional install directory. The home
# is resolved through hermes_constants so a profile or a test-scoped HERMES_HOME
# never turns into a hardcoded path.
DEPLOY_SUBDIR = "hermes-agent"

# The lanes a deploy checkout may sit on: the newest carry/* integration lane,
# or the ``main`` upstream mirror.
LANE_PREFIX = "carry/"
LANE_MAIN = "main"

_GIT_TIMEOUT_SECONDS = 15
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Absolute and ``~``/``$VAR`` prefixed path tokens in free-form card prose, so a
# path written in another form is normalised before it is compared.
_PATH_TOKEN_RE = re.compile(r"""[~$][^\s`'"()\[\]{}<>|,;]*|/[^\s`'"()\[\]{}<>|,;]*""")


def _env_of(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def _truthy(raw: Optional[str]) -> bool:
    return str(raw or "").strip().lower() in _TRUTHY


def _git(root: Path, *args: str) -> Optional[str]:
    """Run ``git -C <root> <args>``; ``None`` on any failure (never raises)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout or ""


def _resolved(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError):
        return path


def _inside(path: Optional[str | Path], root: Path) -> bool:
    """True when *path* is *root* or sits below it (both resolved)."""
    if path is None or str(path).strip() == "":
        return False
    candidate = _resolved(Path(str(path)).expanduser())
    return candidate == root or root in candidate.parents


def deploy_tree_root(
    *, root: Optional[str | Path] = None, env: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    """Resolve the deploy checkout to assert, or ``None`` when there is none.

    Order: the explicit *root*, then ``HERMES_KANBAN_DEPLOY_ROOT``, then
    ``<hermes-home>/hermes-agent``. ``None`` means "no checkout to assert here"
    (missing path, or not itself a git toplevel) — the guard is then inert rather
    than blocking every repo-touching card on a host without that install.
    """
    env = _env_of(env)
    if root is None:
        raw = str(env.get(DEPLOY_ROOT_ENV) or "").strip()
        if raw:
            root = Path(os.path.expandvars(os.path.expanduser(raw)))
        else:
            from hermes_constants import get_default_hermes_root

            root = get_default_hermes_root() / DEPLOY_SUBDIR
    path = _resolved(Path(root).expanduser())
    if not path.is_dir():
        return None
    toplevel = _git(path, "rev-parse", "--show-toplevel")
    if toplevel is None or _resolved(Path(toplevel.strip())) != path:
        return None
    return path


def _root_forms(root: Path) -> set[str]:
    """Every spelling of *root* a card body may plausibly use."""
    forms = {str(root), os.path.normpath(str(root))}
    try:
        home = _resolved(Path.home())
        forms.add("~/" + str(root.relative_to(home)))
    except (ValueError, OSError):
        pass
    return {form for form in forms if form}


def names_deploy_root(body: Optional[str], root: Path) -> bool:
    """True when a card body names *root*, in any normalised spelling."""
    if not body:
        return False
    text = str(body)
    if any(form in text for form in _root_forms(root)):
        return True
    for token in _PATH_TOKEN_RE.findall(text):
        expanded = os.path.expandvars(os.path.expanduser(token))
        if os.path.isabs(expanded) and _inside(expanded, root):
            return True
    return False


def card_touches_deploy_tree(task, workspace: Optional[str | Path], root: Path) -> bool:
    """True when spawning *task* could write to the deploy checkout *root*.

    Either the resolved worktree workspace sits inside the checkout, or the card
    body names it. Everything else (scratch cards, unrelated repos, ``dir``
    workspaces) is out of scope.
    """
    root = _resolved(Path(root))
    kind = str(getattr(task, "workspace_kind", "") or "").strip()
    if kind == "worktree" and (
        _inside(workspace, root) or _inside(getattr(task, "workspace_path", None), root)
    ):
        return True
    return names_deploy_root(getattr(task, "body", None), root)


def deploy_tree_bypass(
    task, workspace: Optional[str | Path], *,
    root: Optional[str | Path] = None, env: Optional[Mapping[str, str]] = None,
    reason: Optional[str] = None,
) -> Optional[dict]:
    """The escape-hatch event payload when this spawn is explicitly allowed.

    ``None`` unless the card is in scope AND ``HERMES_ALLOW_OFFLANE_DEPLOY_TREE``
    is truthy. The caller passes the refusal *reason* it overrode and appends the
    returned payload as a ``deploy_tree_bypassed`` event, so a bypass always
    leaves a trace on the card — including which tree state it let through.
    """
    env = _env_of(env)
    resolved = deploy_tree_root(root=root, env=env)
    if resolved is None:
        return None
    if not _truthy(env.get(ALLOW_OFFLANE_ENV)):
        return None
    if not card_touches_deploy_tree(task, workspace, resolved):
        return None
    payload = {
        "root": str(resolved),
        "via": f"{ALLOW_OFFLANE_ENV}={env.get(ALLOW_OFFLANE_ENV)}",
        "detail": "off-lane/dirty deploy-tree precondition bypassed for this spawn",
    }
    if reason:
        payload["reason"] = reason
    return payload


def deploy_tree_precondition(
    task, workspace: Optional[str | Path], *,
    root: Optional[str | Path] = None, env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """``None`` when spawning *task* is safe, else a named refusal reason.

    Pure observation: read-only git probes, no writes, no events. The escape
    hatch is the caller's business (:func:`deploy_tree_bypass`) — this reports
    the tree's actual state, so a bypass can name what it overrode.
    """
    env = _env_of(env)
    resolved = deploy_tree_root(root=root, env=env)
    if resolved is None:
        return None
    if not card_touches_deploy_tree(task, workspace, resolved):
        return None

    branch = _git(resolved, "rev-parse", "--abbrev-ref", "HEAD")
    head = _git(resolved, "rev-parse", "HEAD")
    if branch is None or head is None:
        return f"deploy-tree-unreadable root={resolved}"
    branch = branch.strip()
    if branch != LANE_MAIN and not branch.startswith(LANE_PREFIX):
        return f"deploy-tree-off-lane branch={branch} head={head.strip()[:8]}"

    status = _git(resolved, "status", "--porcelain")
    if status is None:
        return f"deploy-tree-unreadable root={resolved}"
    files = sum(1 for line in status.splitlines() if line.strip())
    if files:
        return f"deploy-tree-dirty files={files}"
    return None
