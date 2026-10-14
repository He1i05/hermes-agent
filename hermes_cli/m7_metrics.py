#!/usr/bin/env python3
"""M7 review-lane metrics (UG-2, design §4 C2.5) — read-only over kanban.db.

For every review handoff (``review_requested`` / ``changes_requested`` events),
record whether the non-code buckets the card's body declared were actually
delivered. ``changes_requested`` against a declared bucket is the feedback
signal that routing mis-scoped it (F2.1).

Usage:
  m7_metrics.py [--db PATH] [--root ROOT_TASK_ID] [--limit N] [--json]

Exit codes: 0 = ran, 2 = usage error. Never writes to the DB (ro connection).
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys

BUCKETS = ("Adoption", "Docs", "Research-feedback", "Gate")
DB_DEFAULT = "/home/juozas/.hermes/kanban.db"

_REVIEW_KINDS = ("review_requested", "changes_requested")


def _declared_buckets(body: str) -> dict[str, str]:
    """bucket -> commitment text ('none' when the card declared nothing)."""
    out = {}
    for b in BUCKETS:
        m = re.search(
            rf"(?im)^\s*(?:#{{1,4}}\s*|\*\*)?\s*{re.escape(b)}\b\s*:?\s*(.*)$", body or "",
        )
        text = (m.group(1).strip() if m else "")
        # "none" (the C2.3 placeholder) means the bucket was considered but
        # carries no delivery commitment — not a declared bucket.
        if text.lower().rstrip(".,") == "none":
            text = ""
        out[b] = text
    return out


def _rereview_note(body: str) -> list[str]:
    """Free-text mentions of mis-scoped buckets in the changes_requested body."""
    hits = []
    for b in BUCKETS:
        if re.search(rf"(?i)\b{re.escape(b)}\b", body or ""):
            hits.append(b)
    return hits


def collect(conn: sqlite3.Connection, root: str | None, limit: int) -> dict:
    args: list = []
    where = "kind IN ('review_requested','changes_requested')"
    if root:
        # Review handoffs on the root's descendants (or the root itself).
        where += (
            " AND (task_id = ? OR task_id IN "
            "  (SELECT child_id FROM task_links WHERE parent_id = ?))"
        )
        args += [root, root]
    rows = conn.execute(
        f"""
        SELECT e.id, e.task_id, e.kind, e.created_at, t.title, t.body, t.assignee
          FROM task_events e
          JOIN tasks t ON t.id = e.task_id
         WHERE {where}
         ORDER BY e.created_at DESC
         LIMIT ?
        """,
        args + [limit],
    ).fetchall()
    cards = []
    for r in rows:
        declared = _declared_buckets(r["body"] or "")
        cards.append({
            "task_id": r["task_id"],
            "title": (r["title"] or "")[:80],
            "event": r["kind"],
            "event_at": r["created_at"],
            "declared_buckets": {b: c for b, c in declared.items() if c},
            "empty_buckets": [b for b, c in declared.items() if not c],
        })
    changes = [c for c in cards if c["event"] == "changes_requested"]
    # Rework signal: a changes_requested on a card whose declared buckets were
    # all empty (routing delivered the bucket scope nowhere).
    rework = [c for c in changes if not c["declared_buckets"]]
    verdict = {
        "window": {"root": root, "limit": limit},
        "review_events_total": len(cards),
        "review_requested": len(cards) - len(changes),
        "changes_requested": len(changes),
        "bucket_delivery_rate": (
            round(sum(1 for c in cards if c["declared_buckets"]) / len(cards), 3)
            if cards else None
        ),
        "rework_against_undeclared_scope": len(rework),
        "cards": cards,
    }
    return verdict


def main(argv: list[str]) -> int:
    db = DB_DEFAULT
    root = None
    limit = 200
    argv = argv[1:]
    if "--db" in argv:
        db = argv[argv.index("--db") + 1]
    if "--root" in argv:
        root = argv[argv.index("--root") + 1]
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        print(json.dumps(collect(conn, root, limit), indent=2))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))