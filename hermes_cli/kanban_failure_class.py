"""Kanban failure classification — ONE predicate, TWO consumers.

The ``abandon`` eligibility gate asks "is every open card under this root
infrastructure-killed?", and the board's ``explain`` payload asks the same
question about one card. If those two used different evidence (the gate a text
match, the badge the run metadata), the board could invite an operator to
abandon work it does not understand, or refuse to abandon what the badge calls
``infra``. So both read this module and nothing else.

Vocabulary (the ``failure_class`` contract):

* ``FAILURE_CARD`` — the card's own content/model/protocol caused the death
  (``protocol_violation``, ``work_present``, a terminal provider error, a
  provider quota wall).
* ``FAILURE_INFRA`` — the HOST caused it (gateway restart, owner process gone,
  gateway socket lost, owner refusing the RPC).
* ``FAILURE_UNKNOWN`` (``None``) — nothing to classify. **Fails closed**: an
  unclassified card is never abandonable and the payload says why.

Precedence is CARD FIRST. The card classes are decided by exit code and artifact
evidence before any text is read, so a worker that printed gateway vocabulary in
prose and then died of its own defect must still classify as ``card``.

Dependency-light on purpose: stdlib only, no ``kanban_db`` import, so
``kanban_db`` can import this at module load time.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

FAILURE_CARD = "card"
FAILURE_INFRA = "infrastructure"
FAILURE_UNKNOWN: Optional[str] = None

#: Statuses that satisfy dependency gating (``_parents_satisfied``): history,
#: not blockers. ``abandon`` never writes to these.
TERMINAL_STATUSES = frozenset({"done", "archived"})

#: Outcomes that are not failure bookings at all. ``rate_limited`` is a quota
#: wall the breaker deliberately does not charge, so it is NEUTRAL: a classifier
#: walks past it to the previous real booking.
NEUTRAL_OUTCOMES = frozenset({"rate_limited"})
SUCCESS_OUTCOMES = frozenset({"completed"})

# --- card-first precedence ---------------------------------------------------

#: Durable run-metadata markers the booking sites write (they are decided from
#: exit code / artifact evidence, never from text).
_CARD_METADATA_MARKERS = ("protocol_violation", "work_present", "terminal_provider")

#: Text markers for the same classes, for legacy rows that predate the metadata
#: keys. Both the underscored and the prose forms occur in the tree.
_CARD_TEXT_MARKERS = (
    "protocol_violation",
    "protocol violation",
    "work_present",
    "work present",
    "terminal_provider",
    "terminal provider",
)

#: Provider quota/usage wall, verbatim from ``kanban_db_dispatch._PROVIDER_WALL_RE``.
_PROVIDER_WALL_RE = re.compile(
    r"HTTP 429|session usage limit|out of credits|Non-retryable client error", re.IGNORECASE
)

# --- host loss ---------------------------------------------------------------

#: Printed once per refusal by ``kanban_worker_client`` — the measured proof that
#: a worker died inside the gateway submit path rather than in its own reasoning.
HOST_LOSS_WRAPPER = "Kanban owner refused:"

#: The anchor window: the wrapper must appear in the last N characters of the
#: dying worker's own output. Tight on purpose (0 of 231 crashed runs carry
#: gateway vocabulary without the wrapper).
HOST_LOSS_TAIL_CHARS = 200

#: ``(text token, variant)`` — ordered CAUSE first, CONSEQUENCE last. A dying
#: worker can print the outcome line and a named cause in either order (both
#: measured), so position cannot decide the label; a fixed priority can.
WRAPPED_CLASSES: tuple[tuple[str, str], ...] = (
    ("Gateway connection timed out", "gateway_timeout"),
    ("Gateway connection failed", "gateway_connection_failed"),
    ("Gateway protocol mismatch", "gateway_protocol_mismatch"),
    ("Gateway disconnected", "gateway_disconnected"),
    ("Kanban execution unknown; retained by owner", "execution_unknown"),
)

#: Bare tokens that need no wrapper. ``WorkerPersistenceError('owner_unavailable')``
#: is a ``RuntimeError`` so it escapes the submitter's ``except`` clause.
BARE_TOKENS: tuple[tuple[str, str], ...] = (("owner_unavailable", "owner_unavailable"),)

#: Operator-facing sentence per variant; the board shows this instead of a
#: 500-char failure blob.
OWNER_LOSS_TEXT = {
    "gateway_connection_failed": "the gateway connection failed under this worker — no local fallback",
    "gateway_timeout": "the gateway connection timed out under this worker",
    "gateway_disconnected": "the gateway disconnected under this worker",
    "gateway_protocol_mismatch": "the gateway protocol version disagreed with this worker",
    "execution_unknown": "the gateway lost this worker's execution record and retained the claim",
    "owner_unavailable": "the worker's owning process was unavailable",
    "owner_refused_timeout": "the gateway owner refused the RPC (timed out)",
    "owner_refused_other": "the gateway owner refused the RPC",
}

#: Shared test vector — the classifier PR and this module must agree on this
#: table, so neither can drift silently. ``(text, expected failure_class)``.
ERROR_VECTORS: tuple[tuple[str, Optional[str]], ...] = (
    (
        "pid 3613818 exited with code 1 Worker's last output: "
        "'Kanban owner refused: Gateway connection failed; no local fallback'",
        FAILURE_INFRA,
    ),
    (
        "pid 2005113 exited with code 1 Worker's last output: "
        "'Kanban owner refused: Gateway connection timed out; no local fallback'",
        FAILURE_INFRA,
    ),
    (
        "pid 42 exited with code 1 Worker's last output: 'Kanban owner refused: Gateway disconnected'",
        FAILURE_INFRA,
    ),
    (
        "pid 42 exited with code 1 Worker's last output: "
        "'Kanban owner refused: Kanban execution unknown; retained by owner'",
        FAILURE_INFRA,
    ),
    (
        "pid 42 exited with code 1 Worker's last output: \"WorkerPersistenceError('owner_unavailable')\"",
        FAILURE_INFRA,
    ),
    (
        "pid 42 exited cleanly (rc=0) without a terminal kanban call (kanban_complete, kanban_block) "
        "— protocol violation.",
        FAILURE_CARD,
    ),
    ("booked work_present: the card keeps landing work without filing a handoff", FAILURE_CARD),
    ("terminal_provider: model gpt-x is gone (404)", FAILURE_CARD),
    ("provider wall: HTTP 429 Too Many Requests", FAILURE_CARD),
    # Ambiguous tail and empty text: no evidence, so no classification (fail closed).
    ("pid 42 exited with code 1 Worker's last output: 'Traceback (most recent call last)'", None),
    ("", None),
)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _landed_host_loss_class():
    """The landed predicate from ``kanban_db_dispatch`` (commit 56683f8fe2), or None.

    The classifier that decides ``infrastructure`` at booking time already exists in
    this tree. This module must not carry a SECOND copy of that rule: if it did, the
    badge could say ``infra`` about a run the sweep booked as ``card``. So the
    predicate is imported lazily (dispatch imports ``kanban_db``, which imports this
    module, hence the deferred import) and the constants above are kept only as the
    shared test vector — ``test_kanban_failure_class.py`` asserts the two agree
    label-for-label, so either side drifting turns a test red.
    """
    try:
        from hermes_cli import kanban_db_dispatch as kdd
    except Exception:  # pragma: no cover - import-order only
        return None
    fn = getattr(kdd, "_host_loss_class", None)
    return fn if callable(fn) else None


def classifier_deployed() -> bool:
    """True when the landed host-loss predicate is importable (fail closed when not)."""
    return _landed_host_loss_class() is not None


def host_loss_variant(text: Any) -> Optional[str]:
    """Closed-vocabulary host-loss variant from a worker's dying words, or None.

    Delegates to the LANDED classifier (``kanban_db_dispatch._host_loss_class``),
    which owns the anchor + cause-first label decisions. Absent predicate ⇒ ``None``,
    i.e. unclassified: the read model fails closed rather than guessing.
    """
    fn = _landed_host_loss_class()
    if fn is None:
        return None
    try:
        return fn(_text(text))
    except Exception:  # pragma: no cover - a predicate bug must not 500 the board
        return None


def is_card_class_text(text: Any) -> bool:
    """True when the text carries a CARD-class marker (checked before host loss)."""
    full = _text(text)
    if not full:
        return False
    lowered = full.lower()
    for marker in _CARD_TEXT_MARKERS:
        if marker in lowered:
            return True
    return bool(_PROVIDER_WALL_RE.search(full))


def classify_error_text(text: Any) -> Optional[str]:
    """The predicate: ``card`` | ``infrastructure`` | ``None`` for one error text.

    Card classes outrank host loss (precedence, not position); host loss needs its
    anchor; anything else is ``None`` — no evidence, fails closed.
    """
    if is_card_class_text(text):
        return FAILURE_CARD
    if host_loss_variant(text):
        return FAILURE_INFRA
    return None


def _row_get(row: Any, col: str, default: Any = None) -> Any:
    """``row[col]`` for a ``sqlite3.Row``/dict/object; ``default`` when absent."""
    if row is None:
        return default
    try:
        if hasattr(row, "keys") and col in row.keys():
            return row[col]
    except Exception:
        pass
    if isinstance(row, dict):
        return row.get(col, default)
    return getattr(row, col, default)


def _meta(row: Any) -> dict:
    raw = _row_get(row, "metadata", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def classify_run(run: Any) -> Optional[str]:
    """``failure_class`` of one ``task_runs`` row, or ``None``.

    Order is the contract: card metadata markers → ``infrastructure`` metadata →
    error/worker-output text → ``None``. Successes and neutral quota requeues are
    not bookings and return ``None``.
    """
    if run is None:
        return FAILURE_UNKNOWN
    meta = _meta(run)
    for marker in _CARD_METADATA_MARKERS:
        if meta.get(marker):
            return FAILURE_CARD
    if meta.get("infrastructure"):
        return FAILURE_INFRA
    text = "\n".join(
        part for part in (
            _text(_row_get(run, "error", None)),
            _text(meta.get("worker_output")),
        ) if part
    )
    classified = classify_error_text(text)
    if classified is not None:
        return classified
    if (_row_get(run, "outcome", None) or "") in SUCCESS_OUTCOMES | NEUTRAL_OUTCOMES:
        return FAILURE_UNKNOWN
    return FAILURE_UNKNOWN


def _ended_runs(conn, task_id: str, limit: int = 10) -> list:
    return conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC, id DESC LIMIT ?",
        (task_id, int(limit)),
    ).fetchall()


def latest_booking(conn, task_id: str):
    """The newest ended run that actually booked a failure (``None`` if none).

    Walks past ``rate_limited`` (neutral: the breaker never charges it) and stops
    at the first ``completed`` run — a success means nothing is failing now.
    """
    for row in _ended_runs(conn, task_id):
        outcome = _row_get(row, "outcome", None) or ""
        if outcome in NEUTRAL_OUTCOMES:
            continue
        if outcome in SUCCESS_OUTCOMES:
            return None
        return row
    return None


def blocking_class(conn, task_id: str) -> Optional[str]:
    """``failure_class`` of the newest non-terminal booking of ``task_id``."""
    return classify_run(latest_booking(conn, task_id))


def failure_evidence(conn, task_id: str) -> Optional[dict]:
    """``{run_id, at, error, source, variant}`` for the blocking booking, or None."""
    row = latest_booking(conn, task_id)
    if row is None:
        return None
    meta = _meta(row)
    error = _text(_row_get(row, "error", None)) or _text(meta.get("worker_output"))
    source = None
    if meta.get("infrastructure") or any(meta.get(k) for k in _CARD_METADATA_MARKERS):
        source = "run_metadata"
    elif classify_error_text(error) is not None:
        source = "error_text"
    return {
        "run_id": _row_get(row, "id", None),
        "at": _row_get(row, "ended_at", None),
        "error": error[:500] or None,
        "source": source,
        "variant": host_loss_variant(_text(meta.get("worker_output")) or error)
        if classify_run(row) == FAILURE_INFRA else None,
    }


def is_fan_in_root(conn, task_id: str) -> bool:
    """True when the card is the child side of >= 1 dependency edge.

    A decomposed root waits on its whole fan-out: ``decompose_triage_task`` links
    every child with ``_link(conn, cid, task_id)``, so the children sit in
    ``task_links.parent_id`` and the root in ``task_links.child_id``. **Read the
    subtree as ``WHERE child_id = <root>``, never ``WHERE parent_id = <root>``** —
    the latter walks the root's dependents and would invert the abandon scope.
    """
    return conn.execute(
        "SELECT 1 FROM task_links WHERE child_id = ? LIMIT 1", (task_id,)
    ).fetchone() is not None


def fan_in_children(conn, root_id: str) -> list[str]:
    """Direct fan-out children of ``root_id`` (edge direction: children → root)."""
    return [
        row["parent_id"]
        for row in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
            (root_id,),
        ).fetchall()
    ]


def subtree_ids(conn, root_id: str, *, max_depth: int = 8) -> list[str]:
    """Every descendant of ``root_id`` in the fan-in direction (breadth-first).

    Bounded so a corrupt cyclic graph cannot spin: the walk keeps a ``seen`` set
    and stops at ``max_depth``.
    """
    out: list[str] = []
    seen: set[str] = set()
    frontier = [root_id]
    for _ in range(max(1, max_depth)):
        nxt: list[str] = []
        for node in frontier:
            for child in fan_in_children(conn, node):
                if child in seen or child == root_id:
                    continue
                seen.add(child)
                out.append(child)
                nxt.append(child)
        if not nxt:
            break
        frontier = nxt
    return out


def explain_reason(status: str, failure_class: Optional[str], variant: Optional[str]) -> str:
    """One operator-facing sentence for why a descendant sits where it does."""
    if status == "running":
        return "a worker is executing this card right now"
    if status == "review":
        return "this card is a review handoff — a human decides, not a cascade"
    if failure_class == FAILURE_INFRA:
        return OWNER_LOSS_TEXT.get(variant or "", "the host caused this failure, not the card")
    if failure_class == FAILURE_CARD:
        return "the card's own content or protocol caused this failure"
    return "no classified failure — nothing proves the host (or the card) caused it"


def subtree_classes(conn, root_id: str) -> list[dict]:
    """``[{id, status, failure_class, reason, variant}]`` per descendant of the root."""
    out: list[dict] = []
    for cid in subtree_ids(conn, root_id):
        row = conn.execute(
            "SELECT id, status, title FROM tasks WHERE id = ?", (cid,)
        ).fetchone()
        if row is None:
            continue
        status = row["status"]
        klass = blocking_class(conn, cid)
        evidence = failure_evidence(conn, cid) if klass else None
        variant = (evidence or {}).get("variant")
        out.append({
            "id": cid,
            "title": row["title"],
            "status": status,
            "failure_class": klass,
            "variant": variant,
            "reason": explain_reason(status, klass, variant),
        })
    return out


def nearest_fan_in_root(conn, task_id: str, *, max_depth: int = 8) -> Optional[str]:
    """The nearest card that fans this one in, or ``None`` when the card is a leaf.

    A card that is itself a fan-in root is its own root (Appendix C item 6); a
    leaf inside a bigger tree returns the nearest ancestor that has children.
    Walks UP through dependents (``WHERE parent_id = <card>``), which is the
    inverse of the subtree walk and is bounded the same way.
    """
    seen = {task_id}
    ids = [task_id]
    frontier = [task_id]
    for _ in range(max(1, max_depth)):
        nxt: list[str] = []
        for node in frontier:
            rows = conn.execute(
                "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
                (node,),
            ).fetchall()
            for row in rows:
                dep = row["child_id"]
                if dep in seen:
                    continue
                seen.add(dep)
                ids.append(dep)
                nxt.append(dep)
        if not nxt:
            break
        frontier = nxt
    children: dict[str, list[dict]] = {}
    for cid in ids:
        kids = [
            {"id": row["parent_id"], "status": None}
            for row in conn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
                (cid,),
            ).fetchall()
        ]
        if kids:
            children[cid] = kids
    return nearest_fan_in_root_from(task_id, children=children)


def nearest_fan_in_root_from(task_id: str, *, children: dict[str, list]) -> Optional[str]:
    """In-memory twin of :func:`nearest_fan_in_root` over a preloaded ``children`` map.

    ``children[root]`` = the root's fan-out children (``task_links.child_id = root``).
    The upward walk therefore inverts that map (a card's dependents are the roots
    that list it as a child). One implementation of the rule, two loaders.
    """
    dependents: dict[str, list[str]] = {}
    for root_id, kids in (children or {}).items():
        for kid in kids:
            cid = kid.get("id") if isinstance(kid, dict) else kid
            if cid:
                dependents.setdefault(cid, []).append(root_id)
    if children.get(task_id):
        return task_id
    seen = {task_id}
    frontier = [task_id]
    for _ in range(8):
        nxt: list[str] = []
        for node in frontier:
            for dep in sorted(dependents.get(node, [])):
                if dep in seen:
                    continue
                seen.add(dep)
                if children.get(dep):
                    return dep
                nxt.append(dep)
        if not nxt:
            break
        frontier = nxt
    return None
