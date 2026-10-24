"""Kanban explain payload — the board's answer to "why is this card here?".

One **derived** object per task (never stored, never a second copy of the budget):
failure class and its evidence, retry budget, cooldown, owner-loss reason, the
blocking children, and the abandon plan with its refusal reasons. The plan is
computed by the SAME function the ``abandon`` verb uses
(:func:`hermes_cli.kanban_failure_class.blocking_class` over the same rows), so the
badge and the gate cannot disagree.

Cost contract: :func:`board_explain` is a bulk loader — the board's tasks plus one
``task_runs`` query plus one ``task_links`` query, regardless of board size (the
shape ``plugin_api._compute_task_diagnostics`` already uses).

Read-only: nothing here writes to the board.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from hermes_cli import kanban_failure_class as fc


def _row_get(row: Any, col: str, default: Any = None) -> Any:
    return fc._row_get(row, col, default)


def _kanban_cfg(explicit: Optional[dict] = None) -> dict:
    """The ``kanban`` config section, best-effort (a broken config never 500s a board)."""
    if isinstance(explicit, dict):
        return explicit
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        section = cfg.get("kanban") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


# --- bulk loaders ------------------------------------------------------------

def _placeholders(ids: list[str]) -> str:
    return ",".join("?" for _ in ids)


def _load_tasks(conn, ids: Iterable[str]) -> dict[str, dict]:
    ordered = [i for i in dict.fromkeys(ids) if i]
    if not ordered:
        return {}
    out: dict[str, dict] = {}
    for row in conn.execute(
        f"SELECT * FROM tasks WHERE id IN ({_placeholders(ordered)})", tuple(ordered)
    ).fetchall():
        out[row["id"]] = dict(row)
    return out


def _load_runs(conn, ids: Iterable[str]) -> dict[str, list]:
    ordered = [i for i in dict.fromkeys(ids) if i]
    if not ordered:
        return {}
    out: dict[str, list] = {tid: [] for tid in ordered}
    for row in conn.execute(
        f"SELECT * FROM task_runs WHERE task_id IN ({_placeholders(ordered)}) "
        "AND ended_at IS NOT NULL ORDER BY ended_at DESC, id DESC",
        tuple(ordered),
    ).fetchall():
        out.setdefault(row["task_id"], []).append(row)
    return out


def _load_children(conn, ids: Iterable[str]) -> dict[str, list[dict]]:
    """``{root_id: [{id, status}, ...]}`` for the fan-in direction only.

    ``task_links.parent_id`` holds the fan-out child and ``child_id`` the root
    (``decompose_triage_task`` links ``_link(conn, cid, root)``), so the subtree is
    ``WHERE child_id = <root>``. Archived children are included: their status is
    what makes them ``already_terminal`` for the plan.
    """
    ordered = [i for i in dict.fromkeys(ids) if i]
    if not ordered:
        return {}
    out: dict[str, list[dict]] = {tid: [] for tid in ordered}
    for row in conn.execute(
        f"SELECT l.child_id AS root_id, l.parent_id AS child_id, t.status AS status "
        f"FROM task_links l JOIN tasks t ON t.id = l.parent_id "
        f"WHERE l.child_id IN ({_placeholders(ordered)}) ORDER BY l.parent_id",
        tuple(ordered),
    ).fetchall():
        out.setdefault(row["root_id"], []).append(
            {"id": row["child_id"], "status": row["status"]}
        )
    return out


# --- plan (shared with the verb) --------------------------------------------

def abandon_plan(
    root_id: str,
    *,
    tasks: dict[str, dict],
    runs: dict[str, list],
    children: dict[str, list[dict]],
    now: Optional[int] = None,
) -> dict:
    """The abandon plan for ``root_id``: scope, left-in-place, refusal reasons.

    Pure function over the bulk rows (:mod:`hermes_cli.kanban_db.abandon_task`
    builds the same three maps and applies exactly this plan), so
    ``explain.abandon`` and the verb's decision are one computation.

    Refusal reasons are exhaustive — the UI can show a disabled button with a
    reason instead of a 409 the operator has to decode.
    """
    now = int(now if now is not None else time.time())
    root_row = tasks.get(root_id) or {}
    root_status = root_row.get("status") or "missing"
    blocked_by: list[str] = []
    scope: list[str] = []
    left_in_place: list[str] = []
    already_terminal: list[str] = []

    if root_status == "missing":
        blocked_by.append("unknown_root")
    elif root_status == "archived":
        blocked_by.append("already_archived")
    if root_status == "running":
        blocked_by.append("root_running")
    if root_status == "review":
        blocked_by.append("root_review")

    # Transitive walk in the fan-in direction, bounded (kanban_failure_class).
    seen: set[str] = {root_id}
    frontier = [root_id]
    descendants: list[dict] = []
    for _ in range(8):
        nxt: list[str] = []
        for node in frontier:
            for child in children.get(node, []):
                cid = child["id"]
                if cid in seen:
                    continue
                seen.add(cid)
                descendants.append(child)
                nxt.append(cid)
        if not nxt:
            break
        frontier = nxt

    for child in descendants:
        cid, status = child["id"], child.get("status") or "missing"
        if status in fc.TERMINAL_STATUSES:
            already_terminal.append(cid)
            continue
        if status == "running":
            blocked_by.append(f"running_descendant:{cid}")
            continue
        if status == "review":
            blocked_by.append(f"review_descendant:{cid}")
            continue
        klass = fc.classify_run(_first_booking(runs.get(cid, [])))
        if klass == fc.FAILURE_INFRA:
            scope.append(cid)
        elif klass == fc.FAILURE_CARD:
            blocked_by.append(f"card_class_descendant:{cid}")
        else:
            blocked_by.append(f"unclassified_descendant:{cid}")

    return {
        "eligible": not blocked_by,
        "root": root_id,
        "root_status": root_status,
        "scope": scope,
        "left_in_place": left_in_place,
        "already_terminal": already_terminal,
        "blocked_by": blocked_by,
        "command": f"hermes kanban abandon {root_id} --reason '<why>'",
        "decided_at": now,
        # The classified descendant list, so the verb can compute a FORCED scope
        # without parsing the refusal strings back into ids.
        "descendants": [
            {
                "id": child["id"],
                "status": child.get("status") or "missing",
                "failure_class": fc.classify_run(_first_booking(runs.get(child["id"], []))),
            }
            for child in descendants
        ],
    }


def _first_booking(rows: list):
    """Newest non-neutral, non-success run from an already-sorted run list."""
    for row in rows:
        outcome = _row_get(row, "outcome", None) or ""
        if outcome in fc.NEUTRAL_OUTCOMES:
            continue
        if outcome in fc.SUCCESS_OUTCOMES:
            return None
        return row
    return None


# --- cooldown (mirrors check_respawn_guard) ---------------------------------

def cooldown_state(
    latest_run: Any,
    *,
    now: int,
    rl_cooldown: int,
) -> dict:
    """``{kind, until, seconds_remaining}`` for the newest ended run.

    Mirrors ``kanban_db_dispatch.check_respawn_guard`` line for line: the same
    outcomes, the same ``rl_cooldown > 0`` precondition, and the same
    ``_coerce_infrastructure_cooldown`` clamp (60..3600, falling back to the
    rate-limit window) the guard applies to the run's stamped spacing. The board
    therefore cannot advertise a cooldown the guard is not applying. An elapsed
    cooldown reads as ``kind=None`` with ``seconds_remaining=0`` — never a stale
    negative number.
    """
    empty = {"kind": None, "until": None, "seconds_remaining": 0}
    if latest_run is None:
        return empty
    from hermes_cli import kanban_db_dispatch as kdd

    outcome = _row_get(latest_run, "outcome", None) or ""
    meta = fc._meta(latest_run)
    if outcome in ("spawn_failed", "crashed") and meta.get("infrastructure"):
        if rl_cooldown <= 0:
            return empty
        kind = "infrastructure_cooldown"
        window = kdd._coerce_infrastructure_cooldown(
            meta.get("infrastructure_cooldown_seconds"), default=rl_cooldown
        )
    elif outcome == "rate_limited":
        if rl_cooldown <= 0:
            return empty
        kind, window = "rate_limit_cooldown", rl_cooldown
    else:
        return empty
    ended_at = _row_get(latest_run, "ended_at", None)
    if ended_at is None or window <= 0:
        return empty
    until = int(ended_at) + int(window)
    remaining = max(0, until - int(now))
    if remaining <= 0:
        return {"kind": None, "until": until, "seconds_remaining": 0}
    return {"kind": kind, "until": until, "seconds_remaining": remaining}


# --- the payload -------------------------------------------------------------

def _explain_for(
    task_id: str,
    *,
    tasks: dict[str, dict],
    runs: dict[str, list],
    children: dict[str, list[dict]],
    kanban_cfg: dict,
    now: int,
) -> dict:
    row = tasks.get(task_id) or {}
    task_runs = runs.get(task_id, [])
    latest_run = task_runs[0] if task_runs else None
    booking = _first_booking(task_runs)
    klass = fc.classify_run(booking) if booking is not None else None
    evidence = None
    if booking is not None:
        meta = fc._meta(booking)
        error = (
            _row_get(booking, "error", None)
            or _text(meta.get("worker_output"))
        )
        source = None
        if meta.get("infrastructure") or any(
                meta.get(k) for k in ("protocol_violation", "work_present", "terminal_provider")):
            source = "run_metadata"
        elif fc.classify_error_text(error) is not None:
            source = "error_text"
        evidence = {
            "run_id": _row_get(booking, "id", None),
            "at": _row_get(booking, "ended_at", None),
            "error": _text(error)[:500] or None,
            "source": source,
        }

    limit, limit_source = _resolve_failure_limit(row.get("max_retries"), kanban_cfg)
    used = int(row.get("consecutive_failures") or 0)
    # ``infra_used`` is DERIVED from the run history, not from a stored counter: the
    # classifier's own design rejected a per-card infrastructure bound (it re-parked
    # innocent cards), so there is no counter to store — the runs are the source of
    # truth. Metadata stamps first (the landed booking path), text fallback for runs
    # booked before the classifier existed, so the count can never disagree with
    # ``failure_class``.
    infra_used = sum(1 for run in task_runs if fc.classify_run(run) == fc.FAILURE_INFRA)
    infra_limit = _resolve_infra_limit(kanban_cfg)
    retry_budget = {
        "used": used,
        "limit": limit,
        "remaining": max(0, limit - used),
        "limit_source": limit_source,
        "infra_used": infra_used,
        "infra_limit": infra_limit,
        "infra_remaining": max(0, infra_limit - infra_used),
        # The landed classifier retries a host-caused death SPACED and forever, by
        # design: it never parks a card the host killed. The allowance is shown so an
        # operator can see a host that keeps eating the same card, not enforced.
        "infra_enforced": False,
    }

    owner_loss = None
    if klass == fc.FAILURE_INFRA and booking is not None:
        meta = fc._meta(booking)
        raw = _text(meta.get("worker_output")) or _text(_row_get(booking, "error", None))
        variant = fc.host_loss_variant(raw) or "owner_refused_other"
        owner_loss = {
            "variant": variant,
            "text": fc.OWNER_LOSS_TEXT.get(variant, "the host caused this failure"),
            "run_id": _row_get(booking, "id", None),
            "at": _row_get(booking, "ended_at", None),
        }

    blocking_children = []
    for child in children.get(task_id, []):
        if child.get("status") in fc.TERMINAL_STATUSES:
            continue
        blocking_children.append({
            "id": child["id"],
            "status": child.get("status"),
            "failure_class": fc.classify_run(_first_booking(runs.get(child["id"], []))),
        })

    root = fc.nearest_fan_in_root_from(task_id, children=children)
    plan = abandon_plan(
        root or task_id, tasks=tasks, runs=runs, children=children, now=now,
    )

    return {
        "failure_class": klass,
        "failure_evidence": evidence,
        "retry_budget": retry_budget,
        "cooldown": cooldown_state(
            latest_run,
            now=now,
            rl_cooldown=_rl_cooldown(),
        ),
        "owner_loss": owner_loss,
        "blocking_children": blocking_children,
        "abandon": plan,
        "classifier": {
            # Fail-closed contract: when the landed predicate is absent every
            # failure_class is None and nothing is abandonable, and this flag says why.
            "deployed": fc.classifier_deployed(),
            "predicate": "kanban_db_dispatch._host_loss_class (landed 56683f8fe2)",
        },
        "computed_at": now,
    }


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _resolve_failure_limit(task_max_retries: Any, kanban_cfg: dict) -> tuple[int, str]:
    from hermes_cli import kanban_db as kb

    return kb.resolve_failure_limit(task_max_retries, kanban_cfg)


def _resolve_infra_limit(kanban_cfg: dict) -> int:
    """``kanban.infrastructure_retry_limit`` (default 5) — INFORMATIONAL only.

    The landed classifier deliberately enforces no per-card bound (a bound re-parked
    cards the host — not the card — had killed), so this number never blocks a card.
    It is the threshold at which the board starts telling the operator that the host
    keeps eating this card. ``retry_budget.infra_enforced`` is always ``False``.
    """
    raw = (kanban_cfg or {}).get("infrastructure_retry_limit")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 5


def _rl_cooldown() -> int:
    from hermes_cli import kanban_db as kb

    return kb._resolve_rate_limit_cooldown_seconds()


def task_explain(conn, task, *, kanban_cfg: Optional[dict] = None,
                 now: Optional[int] = None) -> dict:
    """The §4 payload for one task (``task`` = a ``Task`` or a row/dict)."""
    task_id = _row_get(task, "id", None)
    cfg = _kanban_cfg(kanban_cfg)
    now_ts = int(now if now is not None else time.time())
    ids = [task_id] if task_id else []
    if task_id:
        root = fc.nearest_fan_in_root(conn, task_id)
        if root:
            ids.append(root)
            ids.extend(fc.subtree_ids(conn, root))
        ids.extend(fc.fan_in_children(conn, task_id))
    tasks = _load_tasks(conn, ids)
    # The caller's object is authoritative for the task itself (it may carry a
    # freshly computed status the row set does not, e.g. an in-flight txn view).
    if task_id and task_id not in tasks:
        tasks[task_id] = dict(task) if isinstance(task, dict) else {
            k: _row_get(task, k) for k in ("id", "status", "max_retries", "consecutive_failures")
        }
    return _explain_for(
        task_id, tasks=tasks, runs=_load_runs(conn, ids),
        children=_load_children(conn, ids), kanban_cfg=cfg, now=now_ts,
    )


def board_explain(conn, tasks: Iterable[Any], *, kanban_cfg: Optional[dict] = None,
                  now: Optional[int] = None) -> dict[str, dict]:
    """``{task_id: explain}`` for a whole board in three queries (see module doc)."""
    rows = list(tasks)
    ids = [_row_get(t, "id", None) for t in rows]
    ids = [i for i in ids if i]
    cfg = _kanban_cfg(kanban_cfg)
    now_ts = int(now if now is not None else time.time())
    task_map = _load_tasks(conn, ids)
    runs = _load_runs(conn, ids)
    children = _load_children(conn, ids)
    return {
        tid: _explain_for(
            tid, tasks=task_map, runs=runs, children=children, kanban_cfg=cfg, now=now_ts,
        )
        for tid in ids
    }
