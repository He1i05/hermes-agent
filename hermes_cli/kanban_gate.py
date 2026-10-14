"""Structural user-gate propagation for decomposed quests (SPEC Shape B).

This module owns the *review-lane routing* concern that the four-bucket
heading writer (``kanban_capability.append_bucket_headings``) does not: decide
whether a decomposed quest requires a human (user) verdict, which child
carries the gate, and how that gate fires structurally instead of as mutable
prose.

What this module is NOT: ``kanban_capability.py`` owns the four-bucket
*heading* contract (the ``Adoption:``/``Docs:``/``Research-feedback:``/``Gate:``
lines it appends). A real ``Gate: <detail>`` line survives byte-identically —
nothing strips it, so no "preservation" logic belongs here either. The actual
gap is that the gate is never *propagated*: it is re-inferred by each worker's
LLM from the visible prompt (title + first 4000 body chars). This module
closes that by extracting the gate deterministically from the root body/title
and carrying it onto the chosen deliverable child as a machine-readable marker
plus a ``gate_required`` task_event, and by making ``request_review`` default
its reviewer to the reserved ``"user"`` literal while the gate is un-discharged
(see ``required_reviewer``).

``gate_required`` is a PURE function of ``(title, body)`` so the 14/14 corpus +
synthetic fixtures port unchanged into the test suite.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_streams as streams
from hermes_cli.kanban_phases import normalize_phase, PHASE_INDEX

logger = logging.getLogger(__name__)

# Reserved assignee literal for the human review lane. The DB layer stores it
# as-is; the dispatcher parks a review card on this assignee as
# `skipped_nonspawnable` (never spawned), and kanban-user-gate-notifier.py
# keys its query on `status='review' AND assignee='user'`.
USER_GATE_ASSIGNEE = "user"

# ---- Extraction regexes, grounded on the real boards measured in SPEC §2.2 ----
_GATE_BUCKET_RE = re.compile(r"(?im)^\s*(?:#{1,4}\s*|\*\*)?\s*gate\s*:\s*(?P<v>[^\n]*)")
_GATE_PROSE_RE = re.compile(r"(?i)\buser[- ]?gate(?:d)?\b")
_GATE_VERDICT_RE = re.compile(
    r"(?i)\b(?:needs?|requires?|awaiting|pending)\b[^.\n]{0,40}"
    r"\b(?:user|human|operator)\b[^.\n]{0,20}\b(?:verdict|approval|decision|sign[- ]?off)\b"
)
_GATE_NONE_RE = re.compile(r"(?i)\bno\s+user\s+gate\b")
# A 'user-gate' hit is a DECLARATION only when it is not surrounded by topic
# words. Discriminates t_21c99170's "Class B report, user-gated" (declaration)
# from t_289e99bb's "preservation logic for USER-GATE lines" (topic mention).
_MENTION_WORDS = re.compile(
    r"(?i)\b(lines?|prose|text|marker|preserv\w*|propagat\w*|"
    r"drop\w*|absent|regex|string)\b"
)
_MENTION_WINDOW = 48

_DELIVERABLE_RE = re.compile(r"(?i)\b(compile|synthes|report|verdict|summary|recommend)\w*")

# The structural marker line appended to the deliverable child. Never
# overwritten when a real non-`none` gate already exists.
_GATE_MARKER_RE = re.compile(r"(?im)^Gate:\s*reviewer=user\b")

_ORCHESTRATION_ONLY_RE = re.compile(r"(?im)^\s*orchestration[- ]only\s*:\s*true\s*$")

#: Event kind for the structural-gate census (W-9, plan §2 S-1 / wiki S-1). A
#: new free-text ``task_events.kind`` VALUE — never a new table or column.
GATE_CENSUS_KIND = "gate_stamp_census"

#: The census ``channel`` literal naming the stamp channel this event audits.
#: The wiki's stamp-propagation table enumerates five channels; this is channel
#: 1, ``propagate_root_gate`` — the only one that is single-recipient BY
#: CONSTRUCTION (1/N on every gated root), which is exactly what the census
#: exists to make visible instead of silent.
GATE_CENSUS_CHANNEL = "structural-user-gate"


def parse_orchestration_only(body: Optional[str]) -> bool:
    """``True`` when the root body carries the frontmatter opt-in line
    ``orchestration-only: true`` (SPEC D-C1). Deterministic, auditable, no
    extra LLM call — cannot misfire on a quest that did not ask for it."""
    return bool(body) and bool(_ORCHESTRATION_ONLY_RE.search(body))


def _mention_filtered(hits: re.Match, body: str) -> bool:
    """True when a positive-marker match is NOT surrounded (within
    ``_MENTION_WINDOW`` chars) by a ``_MENTION_WORDS`` token."""
    lo = max(0, hits.start() - _MENTION_WINDOW)
    hi = min(len(body), hits.end() + _MENTION_WINDOW)
    return not _MENTION_WORDS.search(body[lo:hi])


def gate_required(root_title: str, root_body: Optional[str]) -> tuple[bool, str]:
    """Deterministic, ordered decision rule.

    Returns ``(bool, reason)``. ``True`` means a human (user) verdict is
    required on this quest. Prototype-verified 14/14 (5/5 real corpus + 9/9
    synthetic edge cases — the rule was corrected after a first-cut rule
    produced two false positives on t_289e99bb/t_60636fc3, both kept as
    regression fixtures).

    Order:
    1. explicit ``Gate:`` bucket line whose value is not ``none``.
    2. all bucket lines ``none`` -> the explicit declaration wins, UNLESS the
       title is positive (human-authored, `bucket-none-title-override`).
    3. negative guard: ``no user gate`` anywhere -> False.
    4. positive marker in the TITLE (mention-filtered).
    5. positive marker in the BODY (mention-filtered).
    6. else False.
    """
    title = root_title or ""
    body = root_body or ""

    bucket_values = [m.group("v").strip() for m in _GATE_BUCKET_RE.finditer(body)]
    non_none = [v for v in bucket_values if v.lower() != "none"]
    if non_none:
        return True, "explicit-Gate-bucket"

    def _title_positive() -> bool:
        for m in (_GATE_PROSE_RE.finditer(title), _GATE_VERDICT_RE.finditer(title)):
            for mm in m:
                if _mention_filtered(mm, title):
                    return True
        return False

    def _body_positive() -> bool:
        if _GATE_NONE_RE.search(body):
            return False
        for m in (_GATE_PROSE_RE.finditer(body), _GATE_VERDICT_RE.finditer(body)):
            for mm in m:
                if _mention_filtered(mm, body):
                    return True
        return False

    if bucket_values and all(v.lower() == "none" for v in bucket_values):
        if _title_positive():
            return True, "bucket-none-title-override"
        return False, "bucket-none"

    if _GATE_NONE_RE.search(body):
        return False, "negative-guard"

    if _title_positive():
        return True, "title-positive"

    if _body_positive():
        return True, "body-positive"

    return False, "default"


def _select_deliverable_child(children: list[dict], child_ids: list[str]) -> Optional[str]:
    """Back-compat wrapper: the chosen child id (or ``None``)."""
    return select_deliverable(children, child_ids).child_id


@dataclass(frozen=True)
class Selection:
    """The deliverable selection AND how it was reached (W-10, card t_82eafe14).

    ``step`` is the ordered rule that decided the recipient:

    * ``"declared"`` — a child DECLARED itself the deliverable (step 0);
    * ``"title"``    — a child title matched the legacy ``_DELIVERABLE_RE``;
    * ``"phase"``    — the LAST child at the maximum canonical phase index
      (the strict ``>`` never fired for equal phases, so the FIRST max-phase
      child won — measured WRONG on ``t_6ae16538``, where the actual gate
      recipient is the SECOND of two ``research`` children);
    * ``"last-child"`` — no titled/phase-classified child, last in the list;
    * ``"none"``     — no children at all.

    ``fallback`` is True for every step except ``"declared"``: the selector
    fell through to a heuristic, which is exactly the fact the census (W-9)
    must name rather than let a silent choice pass (plan §2 W-10 AC4).
    """

    child_id: Optional[str]
    step: str
    reason: str
    fallback: bool
    declared_marker: Optional[str] = None


#: A child body declaring itself the deliverable: ``deliverable: true`` or
#: ``delivers: <root-id>`` (plan §2 W-10). The literals and the parser live in
#: ``kanban_streams`` — the W-8 contract module — so the prompt contract, the
#: reply parser and this selector key off ONE source of truth and cannot drift.
DELIVERABLE_TRUE_KEY = streams.DELIVERABLE_KEY
DELIVERS_KEY = streams.DELIVERS_KEY
declared_deliverable = streams.declared_deliverable


def select_deliverable(
    children: list[dict], child_ids: list[str], *, root_id: Optional[str] = None,
) -> Selection:
    """Deterministic, ordered deliverable selection with the decision recorded.

    Step 0 (W-10, the cure for the 1-of-N wrong-recipient defect): a child that
    DECLARES itself the deliverable wins. When ``root_id`` is given, a
    ``delivers: <root_id>`` line for a DIFFERENT root is not this root's
    deliverable; a bare ``deliverable: true`` always applies.

    Then the legacy heuristic, unchanged in shape but with the tie-break
    corrected to LAST-wins (``>=``) as the docstring always claimed:
    (i) child whose title matches ``_DELIVERABLE_RE``; (ii) else the LAST child
    at the maximum canonical phase index; (iii) else the last child.
    """
    if not child_ids:
        return Selection(None, "none", "no children", True)

    # Step 0 — the declared deliverable marker.
    for cid, child in zip(child_ids, children):
        marker = declared_deliverable(child.get("body"))
        if marker is None:
            continue
        if marker != "true" and root_id is not None and marker != root_id:
            # Declared for another root — not this root's deliverable.
            continue
        reason = (
            f"child {cid} declares {DELIVERABLE_TRUE_KEY}: true"
            if marker == "true"
            else f"child {cid} declares {DELIVERS_KEY}: {marker}"
        )
        return Selection(cid, "declared", reason, False, marker)

    # (i) legacy title heuristic.
    for cid, child in zip(child_ids, children):
        title = child.get("title") or ""
        if _DELIVERABLE_RE.search(title):
            return Selection(cid, "title", f"title matches {_DELIVERABLE_RE.pattern}", True)

    # (ii) LAST child at the maximum canonical phase index.
    best: Optional[str] = None
    best_idx = -1
    for cid, child in zip(child_ids, children):
        phase = normalize_phase(child.get("phase"))
        if phase is None or phase not in PHASE_INDEX:
            continue
        idx = PHASE_INDEX[phase]
        # Last-wins: the docstring always said "LAST canonical phase"; the
        # strict ``>`` made the FIRST max-phase child win instead.
        if idx >= best_idx:
            best_idx = idx
            best = cid
    if best is not None:
        return Selection(best, "phase", f"phase-order fallback (max index {best_idx})", True)

    # (iii) last child.
    return Selection(child_ids[-1], "last-child", "no titled/phase-classified child", True)


def record_gate(conn, child_id: str, detail: str, root_id: str) -> None:
    """Emit a ``gate_required`` task_event on the deliverable child (SPEC B4).
    Payload mirrors the other decompose audit events: ``root``, ``detail``,
    ``deliverable``. Runs inside the caller's write txn; ``_append_event`` is
    imported lazily to mirror ``kanban_capability``'s pattern."""
    from hermes_cli.kanban_db import _append_event

    _append_event(
        conn, child_id, "gate_required",
        {"root": root_id, "detail": detail, "deliverable": True},
    )


def census_payload(
    root_id: str, recipient: str, non_recipients: list[str], reason: str,
    *, channel: str = GATE_CENSUS_CHANNEL, step: str = "",
) -> dict:
    """The ``gate_stamp_census`` event payload (plan §2 W-9 / wiki S-1).

    Fixed key set: ``{channel, recipient, non_recipients[], reason}``, plus the
    selector ``step`` so a census can be told apart from a *silent* choice and
    from the other stamp channels. ``non_recipients`` is the count-complete
    complement of the recipient over the emitted children — every child that
    did NOT receive the stamp is named, which is the whole point (the old
    behaviour named nobody and the 1-of-N went unremarked).
    """
    return {
        "channel": channel,
        "root": root_id,
        "recipient": recipient,
        "non_recipients": list(non_recipients),
        "reason": reason,
        "step": step,
    }


def record_gate_census(
    conn, root_id: str, recipient: str, child_ids: list[str], reason: str,
    *, step: str = "", channel: str = GATE_CENSUS_CHANNEL,
) -> dict:
    """Emit the ``gate_stamp_census`` event on the ROOT, inside the caller's
    write txn (plan §2 W-9: "inside the same ``kb.write_txn`` as
    ``record_gate``"). Returns the payload written (or the existing one).

    The census is filed on the ROOT, not the recipient: it is a fact about the
    fan-out, and the root is the card every downstream reader already loads
    (``kanban_coverage.fanout_child_ids`` reads the root's events).

    IDEMPOTENT: exactly ONE census event per gated root (plan §2 W-9 AC2). A
    re-propagation whose facts are unchanged is a no-op — a duplicate row would
    only inflate the event count and make "exactly one" unassertable. A
    re-propagation that lands on a DIFFERENT recipient (or a different child
    set) is a changed fact and IS appended, so the census never hides a
    re-stamp.
    """
    from hermes_cli.kanban_db import _append_event

    non_recipients = [c for c in child_ids if c != recipient]
    existing = latest_gate_census(conn, root_id)
    if (
        existing is not None
        and existing.get("channel") == channel
        and existing.get("recipient") == recipient
        and list(existing.get("non_recipients") or []) == non_recipients
    ):
        return existing

    payload = census_payload(
        root_id, recipient, non_recipients, reason, channel=channel, step=step,
    )
    _append_event(conn, root_id, GATE_CENSUS_KIND, payload)
    return payload


def latest_gate_census(conn, root_id: str) -> Optional[dict]:
    """The newest ``gate_stamp_census`` payload for ``root_id`` (or ``None``).

    Read-only helper so a detector (the W-9 test suite, the release watchdog's
    operator views) can assert the census without re-deriving it. Never writes.
    """
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (root_id, GATE_CENSUS_KIND),
    ).fetchone()
    if row is None or not row["payload"]:
        return None
    try:
        payload = json.loads(row["payload"])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _append_gate_marker(body: Optional[str], detail: str) -> str:
    """Idempotent append of the structural marker line (SPEC §2.5 step 4).
    Appends ``Gate: reviewer=user — <detail>`` if absent; never overwrites an
    existing marker (detection mirrors ``append_bucket_headings``)."""
    base = (body or "").rstrip()
    marker = f"Gate: reviewer={USER_GATE_ASSIGNEE} — {detail}"
    if base and _GATE_MARKER_RE.search(base):
        # An existing structural marker is authoritative — do not duplicate.
        return base
    # Do not clobber an existing human-authored bucket `Gate:` line either.
    if base and re.search(r"(?im)^Gate\s*:\s*(?!reviewer=user)\S", base):
        return base + "\n" + marker
    return (base + "\n\n" + marker) if base else marker


def propagate_root_gate(
    root_id: str, child_ids: list[str], children: list[dict],
    author: Optional[str] = None,
) -> Optional[str]:
    """Extract the root's gate and carry it onto the chosen deliverable child.

    Returns the chosen child id, or ``None`` when the root does not require a
    user verdict (no event, no body change — byte-identical fan-out for
    un-gated quests; the decomposer's existing ``Gate: none`` bucket lines are
    left alone).

    Steps (SPEC §2.5):
    1. Load the FULL root title+body (``kb.get_task``, not the prompt slice).
    2. ``gate_required`` -> None when False.
    3. Deliverable child selection (deterministic; see
       ``select_deliverable``, W-10) -> emitted in the event as
       ``deliverable_child``, together with the step that decided it so a
       fallback choice is never silent (the W-9 census reads this).
    4. Append the structural marker line to that child's body.
    5. Inside ONE ``kb.write_txn``: ``record_gate`` -> one ``gate_required``
       event on the recipient; ``record_gate_census`` (W-9) -> one
       ``gate_stamp_census`` event on the ROOT naming the recipient and every
       non-recipient; comment the child naming the exact call to make.
    """
    author = author or "decomposer"
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, root_id)
        if task is None:
            return None
        required, _reason = gate_required(task.title or "", task.body)
        if not required:
            return None

        # W-10: `root_id` is passed so a `delivers: <other-root>` declaration
        # cannot steal this root's gate.
        selection = select_deliverable(children, child_ids, root_id=root_id)
        deliverable = selection.child_id
        if deliverable is None:
            return None

        detail = (task.title or root_id).strip() or root_id
        # Find the child dict for the deliverable id.
        child = next(
            (c for c, cid in zip(children, child_ids) if cid == deliverable), None,
        )
        new_body = None
        if child is not None:
            new_body = _append_gate_marker(child.get("body"), detail)
            child["body"] = new_body

        # W-9 (plan §2 S-1): the stamp is single-recipient BY CONSTRUCTION —
        # one UPDATE ... WHERE id=?. That is the 1-of-N defect. The census names
        # the recipient AND every non-recipient so the leak is a recorded fact
        # rather than a silent consequence of the UPDATE's shape.
        census_reason = (
            f"step={selection.step}"
            + ("" if not selection.fallback else " (FALLBACK heuristic — no child declared "
               f"`{DELIVERABLE_TRUE_KEY}:`/`{DELIVERS_KEY}: <root-id>`)")
            + f": {selection.reason}"
        )

        # Durable writes for the structural gate: persist the marker line into
        # the child's body (so it reaches every worker, not just the local
        # child dict), emit the gate_required event, emit the gate_stamp_census
        # on the root, and comment the child — all inside ONE txn.
        with kb.write_txn(conn):
            if new_body is not None:
                conn.execute(
                    "UPDATE tasks SET body = ? WHERE id = ?", (new_body, deliverable),
                )
            record_gate(conn, deliverable, detail, root_id)
            record_gate_census(
                conn, root_id, deliverable, list(child_ids), census_reason,
                step=selection.step,
            )
            from hermes_cli.kanban_db import _insert_comment
            import time
            _insert_comment(
                conn, deliverable, author,
                f"Root {root_id} requires a user verdict. When the deliverable is "
                f"ready, request review with a human: "
                f"`kanban_request_review(reviewer=\"{USER_GATE_ASSIGNEE}\")`.",
                int(time.time()),
            )
        if selection.fallback:
            # Never a silent choice: a heuristic-decided recipient is logged
            # with the step that picked it (plan §2 W-10 AC4). W-9 turns this
            # into a `gate_stamp_census` event naming recipient + fallback.
            logger.warning(
                "kanban_gate: root %s gate recipient %s chosen by FALLBACK step=%s (%s) "
                "— no child declared `%s:`/`%s:`",
                root_id, deliverable, selection.step, selection.reason,
                DELIVERABLE_TRUE_KEY, DELIVERS_KEY,
            )
        logger.info(
            "kanban_gate: root %s requires a user gate; stamped deliverable child %s (step=%s)",
            root_id, deliverable, selection.step,
        )
        return deliverable


def required_reviewer(conn, task_id: str) -> Optional[str]:
    """``USER_GATE_ASSIGNEE`` when the newest ``gate_required`` event for the
    card is un-discharged, else ``None``. Discharged = a LATER
    ``review_requested`` event whose payload ``reviewer == USER_GATE_ASSIGNEE``.
    Called by ``kanban_db.request_review`` to default an unspecified reviewer
    to ``"user"`` — the gate fires without the worker having to remember."""
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'gate_required' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if row is None or not row["payload"]:
        return None
    # Discharged = a LATER review_requested whose payload reviewer == "user".
    # Match the exact JSON serialization of the reviewer key so a reviewer
    # named e.g. "user-script" (substring) cannot falsely discharge the gate.
    discharge = conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? "
        "AND kind = 'review_requested' AND payload LIKE ? ORDER BY id DESC LIMIT 1",
        (task_id, f'%"reviewer": "{USER_GATE_ASSIGNEE}"%'),
    ).fetchone()
    return None if discharge else USER_GATE_ASSIGNEE


def propagate_deferral(
    conn, root_id: str, child_ids: list[str], author: Optional[str] = None,
) -> None:
    """Stamp each consumed child of a graph-deferred root (SPEC D-C4). The
    ``scheduled`` reason does NOT propagate to children (SKILL pitfall 49), so
    the decompose-time scheduling must append a comment on each child telling
    it the root is scheduled orchestration-only with
    ``reactivate_on=children_done`` and that the chain completes normally.

    Advisory: a failure here must never fail the fan-out (caller wraps in
    try/except). Runs inside a single write txn on the caller's connection.
    """
    from hermes_cli.kanban_db import _insert_comment
    import time

    author = author or "decomposer"
    note = (
        f"Root {root_id} is scheduled orchestration-only with "
        f"reactivate_on=children_done; the chain completes normally."
    )
    with kb.write_txn(conn):
        for cid in child_ids:
            _insert_comment(conn, cid, author, note, int(time.time()))
