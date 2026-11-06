"""M13 completion-evidence gate — **SHADOW ONLY**.

One refusal class (``R2``): *a completion claims a code/file change but no cited
commit resolves into the card's own tree*. Evidence is the completion payload plus
``git merge-base --is-ancestor`` per card, never the M3 verification ledger (that
ledger holds 0 ``verification_events``, has no ``task_id``/``run_id`` column, and its
feeding classifier accepts 0 of 4,601 verification-shaped invocations — see
``~/.hermes/data/m13-gate/M13_decision_packet.md`` §4).

Flag: ``kanban.completion_gate`` (per ``HERMES_HOME``, therefore per profile, because
the gate runs in the worker process that completes the card).

``off`` (default)
    Mode resolution only. No evidence read, no ``git`` subprocess, no event.
``shadow``
    Evaluate every completion and append one ``completion_gate_shadow`` event whose
    payload carries ``would_refuse``. **Shadow never refuses**: nothing is raised,
    nothing is mutated, no worker-visible text is produced. The completion completes
    byte-identically to the un-gated path.
``enforce``
    **RESERVED — UNPROVEN and UNTESTED.** Accepted as a value so the flag has the
    three states the card names, and so a pre-registered promotion has somewhere to
    land, but it currently behaves exactly like ``shadow`` and records
    ``enforce_ignored: true``. No refusal path ships in this commit: the 14-day
    shadow window that gates promotion (``M13_decision_packet.md`` §6 — ≥150
    evaluations, ≥25 ``would_refuse`` hits, ≥26/30 hand-audit precision, exactly 0
    exempt-class hits, ≤10 s median added latency, ≤66 crashes/11d) has not run.

Exempt classes (explicit, each recorded in the event so exempt-class purity is
measurable — the promotion rule requires *exactly 0* would_refuse hits on them):

``review_approval``
    The completion approves a card out of ``review``. Structural, derived from
    status: 43–48 % of 11d completions are review approvals, so a gate that sees
    them refuses nearly half the board's closures. Same precedent as
    ``kanban_db._gate_empty_completion``.
``adoption_evidence_waiver``
    The completion metadata carries an ``adoption_evidence`` key. Presence is the
    waiver; the builder protocol already uses it for probe waivers
    (``{"waiver": "no-adoption-risk"}``) and the ``probe_absent_fail_open`` literal.
``no_implementable_artifact``
    Nothing a commit could live in: the card declares ``read_only: true``, or no
    resolution domain (see below) is a git working tree. Research cards, decision
    packets and wiki prose land here.

The hook never raises. Any internal failure is recorded as a
``completion_gate_error`` event and the completion proceeds unchanged — a shadow
instrument that can wedge a card is worse than no instrument.

Latency is measured per evaluation and carried on the event (``elapsed_ms``); the
promotion gate reads the median against a ≤10 s budget.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "EXEMPT_CLASSES",
    "GATE_CONFIG_KEY",
    "GATE_MODES",
    "REFUSAL_CLASS",
    "GateDecision",
    "configured_mode",
    "evaluate_completion",
    "gate_completion",
    "reset_mode_cache",
]

#: ``kanban.<key>`` — read per call from the *worker's* config home.
GATE_CONFIG_KEY = "completion_gate"

#: Accepted flag values. ``enforce`` is reserved: accepted, documented, unimplemented.
GATE_MODES = ("off", "shadow", "enforce")

DEFAULT_MODE = "off"

EXEMPT_REVIEW_APPROVAL = "review_approval"
EXEMPT_ADOPTION_WAIVER = "adoption_evidence_waiver"
EXEMPT_NO_ARTIFACT = "no_implementable_artifact"
EXEMPT_CLASSES = (
    EXEMPT_REVIEW_APPROVAL,
    EXEMPT_ADOPTION_WAIVER,
    EXEMPT_NO_ARTIFACT,
)

#: The single refusal class this build evaluates.
REFUSAL_CLASS = "R2_unresolvable_code_change"

EVENT_SHADOW = "completion_gate_shadow"
EVENT_ERROR = "completion_gate_error"

# --- evidence-scan guards (packet §5: cap the scan, subprocesses are superlinear) ---
_TEXT_SCAN_CHARS = 64 * 1024
_MAX_SHA_CANDIDATES = 5
_MAX_DOMAINS = 5
#: How far :func:`_toplevel` walks up from a path token to an existing directory.
_MAX_PARENT_WALK = 6
_GIT_TIMEOUT_S = 5.0

#: Metadata keys whose value is a commit reference.
_SHA_KEYS = frozenset(
    {"commit", "commits", "commit_sha", "head", "head_sha", "base_commit", "sha"}
)
#: Metadata keys whose presence means "this card changed files".
_CHANGED_KEYS = ("changed_files", "files_changed")
#: Metadata keys naming the tree the card worked in (worker-declared, card-scoped).
_REPO_KEYS = ("repo", "repo_path", "workspace_repo", "repo_root")

_SHA_TEXT = re.compile(
    r"(?i)\bcommit(?:ted)?(?:\s+(?:as|at|on|in|hash|sha|to))?\s*[:=]?\s*`?([0-9a-f]{7,40})`?"
)
#: A bare backticked hex token is only a commit *candidate* at git's abbreviation
#: lengths. 32- and 64-char hex are md5/sha256 digests (checksums, corpus hashes) and
#: treating one as a cited commit would manufacture a refusal.
_SHA_BACKTICK = re.compile(r"`([0-9a-f]{7,12})`")
_HEX_RUN = re.compile(r"[0-9a-f]{7,40}")
_ABS_PATH = re.compile(r"(/[A-Za-z0-9._@+-]+(?:/[A-Za-z0-9._@+-]+)+)")

#: Trees that are never a card's repo: the agent's own install + data roots.
#: Attributing a scratch workspace path to these would check a sha against the
#: wrong tree (this host has 13 ``hermes_cli`` checkouts).
_NEVER_A_DOMAIN = frozenset(
    p for p in (str(Path.home()), str(Path.home() / ".hermes")) if p
)


# --------------------------------------------------------------------------- #
# flag resolution — hot, per call
# --------------------------------------------------------------------------- #
def resolve_mode(raw: Any) -> str:
    """Map a raw config value onto :data:`GATE_MODES`; anything unrecognised is ``off``.

    Fail-closed toward *not measuring* rather than toward refusing: an unreadable or
    mistyped flag must never create a refusal, and ``off`` is the only value that
    guarantees that.
    """
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in GATE_MODES:
            return value
    return DEFAULT_MODE


def configured_mode() -> str:
    """``kanban.completion_gate`` for *this process's* config home, read per call.

    ``load_config_readonly`` is stat-signature cached (``hermes_cli/config.py``
    ``_load_config_impl`` → ``config_cache_version``), so an edit is visible to the
    next completion without a restart. A config that cannot be loaded falls through
    to ``off``.
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return DEFAULT_MODE
    kanban = cfg.get("kanban") if isinstance(cfg, Mapping) else None
    if not isinstance(kanban, Mapping):
        return DEFAULT_MODE
    return resolve_mode(kanban.get(GATE_CONFIG_KEY))


def reset_mode_cache() -> None:
    """Test seam: drop the config caches so the next read re-stats the file."""
    try:
        from hermes_cli import config as _config

        _config._LOAD_CONFIG_CACHE.clear()
        _config._RAW_CONFIG_CACHE.clear()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# decision record
# --------------------------------------------------------------------------- #
@dataclass
class GateDecision:
    """One evaluation. ``as_dict`` is what lands on the event payload."""

    mode: str
    evaluated: bool = False
    exempt_class: Optional[str] = None
    refusal_class: Optional[str] = None
    would_refuse: bool = False
    reason: str = ""
    claims_code_change: bool = False
    sha_candidates: list = field(default_factory=list)
    resolution_domains: list = field(default_factory=list)
    resolved_sha: Optional[str] = None
    enforce_ignored: bool = False
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# git probes (cap: _GIT_TIMEOUT_S per call, _MAX_DOMAINS domains)
# --------------------------------------------------------------------------- #
def _git_env() -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/"),
        "LC_ALL": "C",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    return env


def _git(repo: str, *args: str) -> tuple[int, str]:
    """Run one read-only git verb; ``(returncode, stdout)``, ``(1, "")`` on any failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", repo, "-c", "core.fsmonitor=false", *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            env=_git_env(),
            check=False,
        )
    except Exception:
        return 1, ""
    return proc.returncode, proc.stdout.strip()


def _toplevel(path: str) -> Optional[str]:
    """Git working-tree root attributed to ``path``, or ``None``.

    A payload path token is often a file (``.../wiki/page.md``) or already deleted
    (a cleaned-up scratch workspace), so attribution walks up to the nearest existing
    directory first — bounded to :data:`_MAX_PARENT_WALK` levels, and never past a
    root the agent owns outright (see :data:`_NEVER_A_DOMAIN`).
    """
    if not path:
        return None
    candidate = path if os.path.isdir(path) else os.path.dirname(path)
    for _ in range(_MAX_PARENT_WALK):
        if not candidate or candidate == os.path.dirname(candidate):
            return None
        if os.path.isdir(candidate):
            break
        candidate = os.path.dirname(candidate)
    if not os.path.isdir(candidate):
        return None
    rc, out = _git(candidate, "rev-parse", "--show-toplevel")
    if rc != 0 or not out:
        return None
    root = out.strip()
    if not root or root in _NEVER_A_DOMAIN:
        return None
    return root


def _is_ancestor(repo: str, sha: str) -> Optional[str]:
    """Full sha when ``sha`` resolves to a commit contained in ``repo``'s HEAD.

    Ancestry, never equality: HEAD advances mid-run (the packet records a real
    instance of a commit landing during review of the same card).
    """
    rc, out = _git(repo, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
    if rc != 0 or not out:
        return None
    full = out.strip()
    rc, _ = _git(repo, "merge-base", "--is-ancestor", full, "HEAD")
    if rc != 0:
        return None
    return full


# --------------------------------------------------------------------------- #
# payload parsing
# --------------------------------------------------------------------------- #
def _capped_text(*parts: Optional[str]) -> str:
    joined = " ".join(p for p in parts if isinstance(p, str) and p)
    return joined[:_TEXT_SCAN_CHARS]


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                yield item
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)


def _metadata_shas(metadata: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in metadata.items():
        if not isinstance(key, str) or key.lower() not in _SHA_KEYS:
            continue
        for text in _strings(value):
            out.extend(_HEX_RUN.findall(text.lower())[:3])
    return out


def _text_shas(text: str) -> list[str]:
    out = [m.lower() for m in _SHA_TEXT.findall(text)]
    out.extend(m.lower() for m in _SHA_BACKTICK.findall(text))
    return out


def _changed_files(metadata: Mapping[str, Any]) -> list[str]:
    for key in _CHANGED_KEYS:
        value = metadata.get(key)
        if isinstance(value, (list, tuple)) and any(
            isinstance(v, str) and v.strip() for v in value
        ):
            return [v for v in value if isinstance(v, str) and v.strip()]
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def _dedup_capped(values: Iterable[str], limit: int) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
        if len(seen) >= limit:
            break
    return seen


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _resolution_domains(
    workspace_path: Optional[str],
    metadata: Mapping[str, Any],
    text: str,
    *,
    scope: str = "card",
) -> list[tuple[str, str]]:
    """Card-scoped trees a cited sha may resolve into, ``[(kind, root)]``.

    Kinds, in precedence order:

    ``workspace``
        The card's own workspace tree — the evidence source the packet names. For a
        managed ``scratch`` card this is usually *not* a git tree, which is why the
        other two kinds exist at all.
    ``declared``
        A repo path the worker declared in completion metadata (``repo``,
        ``repo_path``, ``workspace_repo``, ``repo_root``).
    ``payload``
        An absolute path named in the completion payload that is a git working tree.
        Still card-scoped — it is the tree the card said it worked in — but never a
        host-wide search: an unmatched sha is not attributed to some other checkout.

    ``scope="workspace"`` restricts resolution to the ``workspace`` kind: that is the
    literal M13 wording, and the replay harness reports it beside the card-scoped
    result so the coverage cost of the strict reading stays visible.
    """
    domains: list[tuple[str, str]] = []

    def add(kind: str, path: Optional[str]) -> None:
        if len(domains) >= _MAX_DOMAINS:
            return
        root = _toplevel(path or "")
        if root and all(root != existing for _, existing in domains):
            domains.append((kind, root))

    add("workspace", workspace_path)
    if scope == "workspace":
        return domains
    for key in _REPO_KEYS:
        for value in _strings(metadata.get(key)):
            add("declared", value)
    for token in _dedup_capped(_ABS_PATH.findall(text), _MAX_DOMAINS):
        add("payload", token)
    return domains


def evaluate_completion(
    *,
    prior_status: Optional[str],
    workspace_path: Optional[str] = None,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    mode: Optional[str] = None,
    domain_scope: str = "card",
) -> GateDecision:
    """Evaluate one completion against R2. Pure: touches no DB, mutates nothing.

    ``mode`` defaults to :func:`configured_mode`; callers that must not read config
    (the replay harness) pass it explicitly. ``domain_scope="workspace"`` narrows
    evidence to the card's workspace tree alone (the literal M13 wording).
    """
    started = time.perf_counter()
    decision = GateDecision(mode=resolve_mode(mode if mode is not None else configured_mode()))
    if decision.mode == "off":
        decision.reason = "gate_off"
        decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return decision

    decision.evaluated = True
    if decision.mode == "enforce":
        # Reserved value, no behaviour: recorded so the shadow window can see that
        # someone flipped it, and the completion still passes unchanged.
        decision.enforce_ignored = True

    md: Mapping[str, Any] = metadata if isinstance(metadata, Mapping) else {}

    # E1 — review approvals are structural, and a gate that sees them refuses
    # 43–48 % of the board's closures.
    if prior_status == "review":
        decision.exempt_class = EXEMPT_REVIEW_APPROVAL
        decision.reason = "completion approves a card out of review"
        decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return decision

    # E3 — an adoption_evidence waiver is explicit, so honour it before any git work.
    if "adoption_evidence" in md:
        decision.exempt_class = EXEMPT_ADOPTION_WAIVER
        decision.reason = "completion carries an adoption_evidence waiver"
        decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return decision

    if md.get("read_only") is True:
        decision.exempt_class = EXEMPT_NO_ARTIFACT
        decision.reason = "card declares read_only: no implementable artifact"
        decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return decision

    text = _capped_text(summary, result)
    shas = _dedup_capped(
        _metadata_shas(md) + _text_shas(text), _MAX_SHA_CANDIDATES
    )
    changed = _changed_files(md)
    decision.sha_candidates = shas
    decision.claims_code_change = bool(shas) or bool(changed)

    domains = _resolution_domains(workspace_path, md, text, scope=domain_scope)
    decision.resolution_domains = [f"{kind}:{root}" for kind, root in domains]

    if not domains:
        decision.exempt_class = EXEMPT_NO_ARTIFACT
        decision.reason = "no git working tree for this card: nothing a sha could live in"
        decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return decision

    if not decision.claims_code_change:
        decision.reason = "no code-change claim in the payload"
        decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return decision

    for sha in shas:
        for kind, root in domains:
            full = _is_ancestor(root, sha)
            if full:
                decision.resolved_sha = full
                decision.reason = f"cited commit resolves in the card's {kind} tree"
                decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
                return decision
    decision.refusal_class = REFUSAL_CLASS
    decision.would_refuse = True
    decision.reason = (
        "completion claims a code/file change but no cited commit resolves into the "
        "card's own tree"
        if shas
        else "completion claims changed files but cites no commit"
    )
    decision.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return decision


# --------------------------------------------------------------------------- #
# DB hook
# --------------------------------------------------------------------------- #
def _prior_status(conn, task_id: str) -> tuple[Optional[str], Optional[str]]:
    row = conn.execute(
        "SELECT status, workspace_path FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return None, None
    return row["status"], row["workspace_path"]


def gate_completion(
    conn,
    task_id: str,
    *,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[GateDecision]:
    """H1 hook: evaluate a completion and record the shadow decision; never raise.

    Returns the decision (``None`` when the gate is off, or when the evaluation
    itself failed). The event is written in its own transaction, exactly like
    :func:`hermes_cli.kanban_db._gate_created_cards` does, so it survives the
    completion's own txn either way.
    """
    try:
        mode = configured_mode()
        if mode == "off":
            return None
        decision = GateDecision(mode=mode)
        try:
            prior_status, workspace_path = _prior_status(conn, task_id)
            decision = evaluate_completion(
                prior_status=prior_status,
                workspace_path=workspace_path,
                result=result,
                summary=summary,
                metadata=metadata,
                mode=mode,
            )
        except Exception as exc:
            decision = GateDecision(
                mode=mode, evaluated=False, error=f"{type(exc).__name__}: {exc}"[:300]
            )
        _record(conn, task_id, decision)
        return decision
    except Exception:
        return None


def _record(conn, task_id: str, decision: GateDecision) -> None:
    """Append the shadow event; swallow a write failure (shadow must not block)."""
    payload = decision.as_dict()
    kind = EVENT_ERROR if decision.error else EVENT_SHADOW
    try:
        # Imported here: ``kanban_db`` imports this module for the H1 hook, so a
        # module-level import would be a cycle.
        from hermes_cli import kanban_db as _kb

        with _kb.write_txn(conn):
            _kb._append_event(conn, task_id, kind, payload)
    except Exception:
        pass
