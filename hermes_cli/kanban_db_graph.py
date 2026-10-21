"""Task graph initialization and atomic decomposition persistence."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Optional

def inherit_creator_origin(
    conn: sqlite3.Connection, task_id: str, creator_task_id: Optional[str], *,
    created_at: int,
) -> None:
    """Copy durable origin inside creation's transaction, never adding dependencies."""
    if not creator_task_id:
        return
    from hermes_cli.kanban_db import _inherit_notify_subs

    conn.execute(
        "UPDATE tasks SET session_id = COALESCE(session_id, "
        "(SELECT session_id FROM tasks WHERE id = ?)) WHERE id = ?",
        (creator_task_id, task_id),
    )
    _inherit_notify_subs(conn, task_id, (creator_task_id,), created_at=created_at)


def initial_task_state(
    conn: sqlite3.Connection, parents: tuple[str, ...], initial_status: str,
    triage: bool, tenant: Optional[str],
) -> tuple[str, Optional[str]]:
    """Resolve state and tenant under the creator's write transaction.

    Parent order breaks ties in this soft namespace; explicit tenant wins.
    Validate parents even for parked tasks so links never dangle.
    """
    rows = {}
    if parents:
        rows = {row["id"]: row for row in conn.execute(
            "SELECT id, status, tenant FROM tasks WHERE id IN "
            "(" + ",".join("?" * len(parents)) + ")", parents,
        )}
        missing = [pid for pid in parents if pid not in rows]
        if missing:
            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
        if tenant is None:
            tenant = next((rows[pid]["tenant"] for pid in parents if rows[pid]["tenant"]), None)
    if initial_status == "blocked":
        return "blocked", tenant
    if triage:
        return "triage", tenant
    if any(row["status"] != "done" for row in rows.values()):
        return "todo", tenant
    return "ready", tenant


def _validate_children_graph(children: list) -> None:
    """DB-free shape check + Kahn's cycle check on the sibling graph (a cycle
    would deadlock every involved child in ``todo`` forever)."""
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(f"child[{idx}].parents[{p}] is not a valid index into children")
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")

    in_deg = [0] * len(children)
    adj: list[list[int]] = [[] for _ in children]
    for i, c in enumerate(children):
        for p in (c.get("parents") or []):
            adj[p].append(i)
            in_deg[i] += 1
    queue = [i for i in range(len(children)) if in_deg[i] == 0]
    seen = 0
    while queue:
        seen += 1
        for nb in adj[queue.pop()]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    if seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")


# ── Fan-out ledger guard ────────────────────────────────────────────────────
#
# `decomposed` events are the durable proof that a card's body was fanned out;
# they are never cleared and never deleted, and each payload carries `body_sha`.
# The guard refuses a fan-out iff the card's CURRENT body digest is already in
# that ledger:
#
#   already-fanned-out   these exact bytes fanned out before -> refuse
#   legacy-digestless    a `decomposed` event predates the digest -> fail closed
#   not-in-triage        the card left triage -> nothing to fan out
#   None                 a body that has never fanned out -> proceed
#
# CONTENT decides, never an event. `specify_triage_task` appends `specified`
# unconditionally — payload NULL when nothing changed (kanban_db.py:5303-5306) —
# and `edit_task` records `edited {"fields": ["body"]}` WITHOUT comparing bytes
# (kanban_db.py:4291-4320). An event-keyed guard is therefore wrong in both
# directions (a no-op specify opened it into a duplicate fan-out of the same
# body; a real `edit --body` left it shut). Both were measured before this
# form was chosen.
#
# Also deliberately NOT done here: clearing the marker so a re-entered triage
# card "recovers". That re-fans the SAME body — measured on the incident shape,
# a 2-child root became 6 board rows with 5 parents and duplicate child titles —
# which on side-effecting children is a duplicated real-world action. The
# recoverable class is a card whose body CHANGED; that one fans out again,
# exactly once per body content, through this same guard.
REFUSAL_ALREADY_FANNED_OUT = "already-fanned-out"
REFUSAL_LEGACY_DIGESTLESS = "legacy-digestless"
REFUSAL_NOT_IN_TRIAGE = "not-in-triage"

#: The refusal codes that mean "the fan-out budget for this card's CURRENT body is
#: spent". A PRE-check — the cheap read before an aux LLM call or before a
#: per-tick decomposer slot is charged — short-circuits ONLY on these. Every other
#: verdict falls through to the caller's own load/report path, because a
#: pre-check is an OPTIMISATION: it may save work, it may never change behaviour.
#: (``not-in-triage`` is exactly that case — a pre-check can see it for a card
#: that moved a millisecond ago, and refusing there would replace the caller's
#: precise message with a guess.)
SPENT_REFUSALS = (REFUSAL_ALREADY_FANNED_OUT, REFUSAL_LEGACY_DIGESTLESS)


def body_sha(body: Optional[str]) -> str:
    """The fan-out ledger's only input: sha256 hex of the task body's bytes."""
    return hashlib.sha256((body or "").encode("utf-8")).hexdigest()


def fanout_ledger(conn: sqlite3.Connection, task_id: str) -> list[tuple[int, Optional[str]]]:
    """``[(event_id, body_sha | None)]`` for every `decomposed` event, oldest first.

    Read-only: the ledger is append-only and no decision rewrites or hides an
    event. A payload that cannot be parsed yields ``None`` for its digest, which
    :func:`_refuse` treats as digest-less (fail closed) rather than as "never
    fanned out".
    """
    out: list[tuple[int, Optional[str]]] = []
    for row in conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = 'decomposed' "
        "ORDER BY id",
        (task_id,),
    ):
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        digest = payload.get("body_sha") if isinstance(payload, dict) else None
        out.append((row["id"], digest if isinstance(digest, str) and digest else None))
    return out


def _refuse(*, status: str, body: Optional[str], ledger: list[tuple[int, Optional[str]]]) -> Optional[str]:
    """The guard's decision over an already-read row + ledger (no I/O)."""
    if status != "triage":
        return REFUSAL_NOT_IN_TRIAGE
    if not ledger:
        return None  # never fanned out: nothing to protect
    if any(digest is None for _, digest in ledger):
        # A pre-digest `decomposed` marker: the fanned-out body is unknowable, so
        # no comparison can be trusted. Fail CLOSED — the operator message must
        # say archive + re-create, because no verb can open a digest that was
        # never recorded. (No backfill: the body is not recoverable.)
        return REFUSAL_LEGACY_DIGESTLESS
    current = body_sha(body)
    if any(digest == current for _, digest in ledger):
        return REFUSAL_ALREADY_FANNED_OUT
    return None


def fanout_refusal(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Would a fan-out of this card duplicate work? ``None`` means "fan out".

    The single guard shared by every layer that can start a fan-out:
    ``kanban_decompose.decompose_task`` (before spending the aux LLM call),
    ``gateway.kanban_watchers_dispatcher.auto_decompose_tick`` (before spending
    one of the ``auto_decompose_per_tick`` slots) and
    :func:`decompose_triage_task`'s own commit (inside the write transaction).

    Advisory for the pre-checks, authoritative at the commit: only the commit
    runs inside the same IMMEDIATE transaction as the write, so a body edited
    between a pre-check and the commit is caught there.
    """
    row = conn.execute(
        "SELECT id, status, body FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if row is None:
        return REFUSAL_NOT_IN_TRIAGE
    return _refuse(status=row["status"], body=row["body"],
                   ledger=fanout_ledger(conn, task_id))


def decompose_triage_task(
    conn: sqlite3.Connection, task_id: str, *, root_assignee: Optional[str], children: list[dict],
    author: Optional[str] = None, auto_promote: bool = True,
    expected_body_sha: Optional[str] = None,
) -> Optional[list[str]]:
    """Fan a triage task out into children and move the root to ``todo``; the root
    waits on every child and wakes (``ready``) when all are done.

    ``children``: dicts of ``title`` (required), ``body``, ``assignee``,
    ``parents`` (indices into this list), optional workspace overrides.
    Returns child ids in input order, or None when the root is missing / not
    in triage, when :func:`fanout_refusal` refuses the fan-out, or when the
    body changed after the children were derived (see ``expected_body_sha``).
    Atomic: malformed entries abort fan-out.

    ``expected_body_sha`` is the digest of the body the CALLER put in the
    decomposer's prompt. When given, the commit re-reads ``tasks.body`` and
    aborts (returns None, writes nothing) if it no longer matches — otherwise a
    mid-flight ``edit --body`` would pin a body the children were never derived
    from, silently blocking that body forever. The ``decomposed`` event records
    the digest of the body that actually committed, which is what
    :func:`fanout_refusal` compares against on the next attempt.
    """
    from hermes_cli.kanban_db import (
        _canonical_assignee, _link, _append_event, _insert_comment,
        write_txn, recompute_ready,
    )

    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    _validate_children_graph(children)

    # ONE txn so the fan-out is atomic; helpers that open their own write_txn
    # (create_task, link_tasks, add_comment) must not be called in here.
    now = int(time.time())
    with write_txn(conn):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path, body "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if root_row is None or root_row["status"] != "triage":
            return None
        # Dependency links alone do not imply lineage: the `decomposed` event is
        # committed with the graph and survives re-triage or unlinking, so the
        # LEDGER of those events is the lineage record. Decide on CONTENT, in the
        # same transaction as the write — never on the `specified`/`edited`
        # events, which fire without a byte changing (see `fanout_refusal`).
        if _refuse(status=root_row["status"], body=root_row["body"],
                   ledger=fanout_ledger(conn, task_id)) is not None:
            return None
        if expected_body_sha is not None and body_sha(root_row["body"]) != expected_body_sha:
            # The body moved after the children were derived: commit nothing so
            # no `decomposed` digest is recorded for a body nobody has children for.
            return None
        child_ids = [
            _insert_decomposed_child(conn, task_id, root_row, child, author, now)
            for child in children
        ]
        # Sibling edges within the decomposed graph.
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id, child_id = child_ids[p_idx], child_ids[idx]
                _link(conn, parent_id, child_id)
                _append_event(conn, child_id, "linked", {"parent": parent_id, "child": child_id})
        # Root waits for the whole graph: link it under EVERY child (simpler
        # than computing leaves; cycle-free since the root is only ever a child).
        for cid in child_ids:
            _link(conn, cid, task_id)
            # G-1: the root edge is audited, symmetric with the sibling edges
            # above — so a later orientation check can tell a designed
            # root-as-child row from an inverted (root-as-parent) one.
            _append_event(
                conn, task_id, "linked",
                {"parent": cid, "child": task_id, "kind": "root_edge"},
            )
        # INVARIANT-LINK post-write assertion: the root must never sit in the
        # parent_id column for a fanout child. ``_assert_root_not_parent``
        # raises ValueError on a wrong-direction row (caught and surfaced by
        # ``kanban_decompose._apply_fanout`` as "DB rejected graph: …"). A
        # wrong-direction row inside our own atomic txn means the writer is
        # broken; healing is a post-hoc, audited repair (``selfheal_root_edges``).
        _assert_root_not_parent(conn, task_id, child_ids)
        # Flip the root triage -> todo, assignee -> orchestrator.
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None:
            sets.append("assignee = ?")
            params.append(root_assignee)
        params.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", tuple(params))
        if author and author.strip():
            _insert_comment(
                conn, task_id, author.strip(),
                "Decomposed into " + ", ".join(child_ids)
                + ". Root will wake when all children complete.",
                now,
            )
        _append_event(
            conn, task_id, "decomposed",
            {
                "child_ids": child_ids,
                "root_assignee": root_assignee,
                # The fan-out ledger: what `fanout_refusal` compares a later
                # body against. Recorded from the row read INSIDE this txn, so
                # it is exactly the body these children were committed for.
                "body_sha": body_sha(root_row["body"]),
            },
        )
    # Outside the txn (own IMMEDIATE txn). ``auto_promote=False`` leaves the
    # children in ``todo`` for manual-review-first workflows.
    if auto_promote:
        recompute_ready(conn)
    return child_ids


def _insert_decomposed_child(
    conn: sqlite3.Connection, root_id: str, root_row: sqlite3.Row, child: dict,
    author: Optional[str], now: int,
) -> str:
    """Insert one decomposed child as ``todo`` (linked under the root later so
    the dispatcher only ever sees a coherent graph); returns its id.

    ``child["phase"]`` (canonical SDLC phase name, see
    :mod:`hermes_cli.kanban_phases`) is persisted into ``current_step_key``
    when set — the card's advisory phase survives into every child instead of
    living only in prose. An unset/unknown phase writes NULL (the column was
    unused before this; plain cards leave it null).

    Workspace: per-child override wins, else inherit the root's kind. Path
    inherits only when kinds match (a 'dir' child must not point at the
    root's worktree) and NEVER for worktrees — siblings dispatch concurrently
    and one shared checkout would put them all on the first sibling's branch
    with no lock; leaving it unset makes dispatch materialize a fresh
    ``<repo>/.worktrees/<child-id>`` per child from the board anchor.
    """
    from hermes_cli.kanban_db import (
        _new_task_id, _canonical_assignee, _append_event,
    )
    from hermes_cli.kanban_phases import normalize_phase

    root_ws_kind = root_row["workspace_kind"] or "scratch"
    child_ws_kind = child.get("workspace_kind") or root_ws_kind
    if child.get("workspace_path"):
        child_ws_path = child.get("workspace_path")
    elif child_ws_kind == "worktree":
        child_ws_path = None
    elif child_ws_kind == root_ws_kind:
        child_ws_path = root_row["workspace_path"]
    else:
        child_ws_path = None
    new_id = _new_task_id()
    body = child.get("body")
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, workspace_kind, "
        " workspace_path, tenant, created_at, created_by, current_step_key) "
        "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?, ?)",
        (
            new_id, child["title"].strip(), body if isinstance(body, str) else None,
            _canonical_assignee(child.get("assignee")), child_ws_kind, child_ws_path,
            root_row["tenant"], now, (author or "decomposer"),
            normalize_phase(child.get("phase")),
        ),
    )
    _append_event(
        conn, new_id, "created", {"by": author or "decomposer", "from_decompose_of": root_id},
    )
    inherit_creator_origin(conn, new_id, root_id, created_at=now)
    return new_id


def _assert_root_not_parent(
    conn: sqlite3.Connection, root_id: str, child_ids: list[str],
) -> None:
    """INVARIANT-LINK guard: refuse a fan-out where any ``(root_id, child_id)``
    row exists (the root wrongly in ``parent_id``).

    Raised inside ``decompose_triage_task``'s atomic write_txn, so a
    wrong-direction row means the writer is broken and the whole fan-out
    aborts (no partial graph). ``ValueError`` is caught and surfaced by
    ``kanban_decompose._apply_fanout`` as "DB rejected graph: …".

    The predicate is the INTERSECTION with the given fanout ``child_ids`` —
    NOT ``parent_id = root_id`` in general, which also matches legitimate
    downstream dependency edges (``kanban_create(parents=[root])``). Here the
    caller passes exactly the fanout children it just wrote, so the check is
    precise.
    """
    if not child_ids:
        return
    placeholders = ",".join("?" * len(child_ids))
    bad = conn.execute(
        f"SELECT child_id FROM task_links "
        f"WHERE parent_id = ? AND child_id IN ({placeholders})",
        (root_id, *child_ids),
    ).fetchall()
    if bad:
        names = ", ".join(r["child_id"] for r in bad)
        raise ValueError(
            f"root {root_id} wrongly present in parent_id for fanout children: {names}"
        )


def _fanout_child_ids_from_audit(
    conn: sqlite3.Connection, root_id: str,
) -> Optional[list[str]]:
    """The authoritative fanout child set for ``root_id`` from its audit trail.

    Returns ``child_ids`` from the newest ``decomposed`` event payload when it
    exists, else ``None`` (a pre-audit root — caller then falls back to the
    link-table walk). Children's own ``created`` payloads carrying
    ``from_decompose_of == root_id`` are an equivalent cross-check.
    """
    from hermes_cli.kanban_db import _json_dict, _row_get

    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'decomposed' "
        "ORDER BY id DESC LIMIT 1", (root_id,),
    ).fetchone()
    payload = _json_dict(_row_get(row, "payload"))
    child_ids = payload.get("child_ids")
    if isinstance(child_ids, list) and child_ids:
        return list(child_ids)
    return None


def selfheal_root_edges(
    conn: sqlite3.Connection, root_id: str, *, author: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    r"""Restore INVARIANT-LINK for a decomposed root that an external actor
    inverted. Deletes any ``(root_id -> c)`` row where ``c`` is a FANOUT child
    of ``root_id``, and ensures ``(c -> root_id)`` rows exist for those
    children. Emits ONE ``link_selfheal`` event. Idempotent — a second call
    returns ``changed=False`` and emits nothing.

    Returns ``{"flipped": [{"parent_id","child_id"}...], "deleted_rows": [...],
             "inserted_rows": [...], "root": root_id, "children": [...],
             "changed": bool, "dry_run": bool}``.

    The delete predicate is the INTERSECTION with the decompose audit's
    ``child_ids`` (``decomposed`` event payload), NOT ``parent_id = root_id``
    — the latter also matches legitimate downstream dependency edges and would
    destroy them (SPEC §1.5.1 / pitfall 55). The link-table walk is the
    fallback for pre-audit roots (no ``decomposed`` event) only.
    """
    from hermes_cli.kanban_db import _link, _append_event, write_txn

    # 1. Fanout child set — audit-first.
    child_ids = _fanout_child_ids_from_audit(conn, root_id)
    if child_ids is None:
        # Pre-audit fallback: enumerate children from rows touching root_id,
        # minus root_id itself. Direction is read from the row, never assumed.
        rows = conn.execute(
            "SELECT parent_id, child_id FROM task_links "
            "WHERE parent_id = ? OR child_id = ?", (root_id, root_id),
        ).fetchall()
        child_ids = sorted({
            r["child_id"] if r["parent_id"] == root_id else r["parent_id"]
            for r in rows if (r["child_id"] != root_id and r["parent_id"] != root_id)
        })

    flipped: list[dict] = []
    deleted_rows: list[dict] = []
    inserted_rows: list[dict] = []
    changed = False

    # 2. Deletion pass — the intersection predicate.
    if child_ids:
        placeholders = ",".join("?" * len(child_ids))
        inverted = conn.execute(
            f"SELECT parent_id, child_id FROM task_links "
            f"WHERE parent_id = ? AND child_id IN ({placeholders})",
            (root_id, *child_ids),
        ).fetchall()
        for r in inverted:
            flipped.append({"parent_id": r["parent_id"], "child_id": r["child_id"]})
            deleted_rows.append({"parent_id": r["parent_id"], "child_id": r["child_id"]})
            changed = True

    # 3. Ensure (c -> root) rows exist for every fanout child.
    existing_root_children = set()
    if child_ids:
        placeholders = ",".join("?" * len(child_ids))
        rows = conn.execute(
            f"SELECT parent_id FROM task_links WHERE child_id = ? "
            f"AND parent_id IN ({placeholders})", (root_id, *child_ids),
        ).fetchall()
        existing_root_children = {r["parent_id"] for r in rows}

    if dry_run:
        for d in deleted_rows:
            d["dry_run"] = True
        # Mirror the apply path exactly: insert (c -> root) for every fanout
        # child that lacks one — including inverted children just flipped,
        # which also need their (c -> root) row restored (0 inserts in the
        # dry-run must EQUAL 0 in the apply, or the forecast is dishonest).
        for c in child_ids or []:
            if c not in existing_root_children:
                inserted_rows.append(
                    {"parent_id": c, "child_id": root_id, "dry_run": True}
                )
        return {
            "flipped": flipped, "deleted_rows": deleted_rows,
            "inserted_rows": inserted_rows, "root": root_id,
            "children": child_ids or [], "changed": changed, "dry_run": True,
        }

    if not changed:
        return {
            "flipped": [], "deleted_rows": [], "inserted_rows": [],
            "root": root_id, "children": child_ids or [],
            "changed": False, "dry_run": False,
        }

    with write_txn(conn):
        for r in inverted:
            conn.execute(
                "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
                (r["parent_id"], r["child_id"]),
            )
        for c in child_ids or []:
            if c not in existing_root_children:
                _link(conn, c, root_id)
                inserted_rows.append({"parent_id": c, "child_id": root_id})
        _append_event(
            conn, root_id, "link_selfheal",
            {
                "reason": (author or "orchestrator"),
                "root": root_id,
                "flipped": flipped,
                "inserted_rows": inserted_rows,
                "deleted_rows": deleted_rows,
                "dry_run": False,
            },
        )
    return {
        "flipped": flipped, "deleted_rows": deleted_rows,
        "inserted_rows": inserted_rows, "root": root_id,
        "children": child_ids or [], "changed": True, "dry_run": False,
    }
