"""M3 four-bucket-heading coverage verifier (SPEC §2.8, option (a)).

``kanban_capability.py:34,258`` promises that every decomposed child body
carries the four bucket headings (``Adoption:`` / ``Docs:`` /
``Research-feedback:`` / ``Gate:``) so a reviewer can see they were considered
rather than omitted. Before this module existed, that promise was unpoliced —
``Gate: none`` × 5 children passed silently. This module closes the gap with a
mechanical, advisory coverage check.

The check is advisory only (logged, never blocks a fan-out), mirroring the
M3 cascade audit pattern in ``kanban_decompose._apply_fanout``: a child missing
a heading is a hygiene flag, not a reason to abort an already-committed
fan-out.
"""

from __future__ import annotations

import re
from typing import Optional

from hermes_cli.kanban_capability import BUCKET_HEADINGS


def audit_bucket_headings(conn, root_id: str) -> list[str]:
    """Return one problem string per decomposed child missing a bucket heading.

    Reads the root's ``decomposed`` event payload ``child_ids`` (the
    authoritative fanout child set) and checks each child's ``body`` for all
    four ``BUCKET_HEADINGS``. The heading check mirrors
    ``append_bucket_headings``'s own regex so the two never disagree.

    Returns ``[]`` when the root has no ``decomposed`` event (pre-audit), the
    event has no ``child_ids``, or every child is complete.
    """
    from hermes_cli.kanban_db import _json_dict, _row_get

    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'decomposed' "
        "ORDER BY id DESC LIMIT 1", (root_id,),
    ).fetchone()
    if row is None:
        return []
    payload = _json_dict(_row_get(row, "payload"))
    child_ids = payload.get("child_ids")
    if not isinstance(child_ids, list) or not child_ids:
        return []

    problems: list[str] = []
    for cid in child_ids:
        child = conn.execute(
            "SELECT id, title, body FROM tasks WHERE id = ?", (cid,),
        ).fetchone()
        if child is None:
            problems.append(f"child {cid} of root {root_id} has no tasks row")
            continue
        body = child["body"] or ""
        existing = {h.lower() for h in BUCKET_HEADINGS if re.search(
            rf"(?im)^\s*(?:#{{1,4}}\s*|\*\*)?\s*{re.escape(h)}\b", body,
        )}
        missing = [h for h in BUCKET_HEADINGS if h.lower() not in existing]
        if missing:
            problems.append(
                f"child {cid} ({child['title'] or '(untitled)'!r}) of root {root_id} "
                f"is missing bucket heading(s): {', '.join(missing)}"
            )
    return problems


def audit_and_log(logger, conn, root_id: str) -> None:
    """Advisory wrapper: run ``audit_bucket_headings`` and log each problem at
    warning level (matching the ``cascade_violations`` pattern). Never raises."""
    try:
        problems = audit_bucket_headings(conn, root_id)
        for problem in problems:
            logger.warning("decompose: M3 bucket-heading audit: %s", problem)
    except Exception:
        logger.exception("decompose: M3 bucket-heading audit failed on task %s", root_id)
