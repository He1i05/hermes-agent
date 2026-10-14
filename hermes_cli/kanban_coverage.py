"""Post-decompose coverage receipt (W-7, quest t_bec82259 / card t_6a91f797).

What this module is
-------------------
The DETECTOR half of the decomposer-completeness fix. W-8
(``hermes_cli/kanban_streams.py``) established the CONVENTION — a root body
declares its streams, every emitted child declares which it ``covers:`` or
folds in, and a declared stream with neither declaration is a coverage gap.
This module turns that convention into a human-readable RECEIPT: the mapping
stream -> child plus the fold-in accounting, commented on the root card so a
reviewer (and the synthesis child) can see, after the fact, exactly which
declared streams the fan-out did and did not account for.

It is deliberately NOT a gate. Per rule C-4 the fan-out has already committed
by the time any check can run (``kanban_decompose._apply_fanout`` wraps every
post-commit audit in ``try/except`` precisely so a post-commit failure cannot
abort a committed graph). This module writes a comment and (optionally) an
event; it never changes a card's status, bodies, or links.

The two oracles, and why only one is implemented
------------------------------------------------
Measured read-only against the live board (2026-09-30, code pin ``26b000cc0f``,
both roots ``done``, children resolved by :func:`fanout_child_ids` — the union
of the newest ``decomposed`` event's ids and the live ``task_links`` children):

    root        body  declared  children  declaration oracle  naive diff
    t_26d8eb94  8234  S1..S6    7         GAPS = S1..S6 (6)   gaps = 0
    t_bec82259  8501  R1..R5    11        GAPS = R1..R5 (5)   gaps = 0

The naive body-vs-children keyword diff FALSE-PASSES both roots (6/6 and 5/5
truly-missing streams read as COVERED) because every stream heading shares
generic tokens with the children that do exist ("pipeline", "audit", ...). It
is rejected as a mechanism and kept only as a permanent negative fixture in the
test suite, so nobody can "simplify" this module back into a diff. The
declaration oracle is the only one that scores 0 false-passes.

The full-body read, and why the prompt slice is a trap
------------------------------------------------------
Rule C-2: the receipt reads the FULL root body, never the decomposer's prompt
slice. On ``t_bec82259`` the slice (``body[:3999]``) declares ZERO of the five
streams — a receipt built from it would report "0 declared, 0 gaps" and pass
VACUOUSLY. ``read_root_body`` therefore takes a task id and reads the row; and
every receipt states the cut position plus how many declared streams fell
inside vs beyond it, so a *compliance* failure (W-8's problem) is
distinguishable from a *context-boundary* failure (W-7's).

Scope boundary — what this module does NOT do
---------------------------------------------
  * Emit the ``gate_stamp_census`` event — that lives in
    ``kanban_gate.propagate_root_gate`` (W-9, ``t_78427b7f``), inside the same
    ``kb.write_txn`` as ``record_gate``, because the census is a fact about the
    gate stamp and must commit with it. This module only reports the fan-out
    coverage mapping.
  * Emit the ``coverage_gap`` synthesis-gate precondition (rule C-4 — a
    synthesis child reporting "complete" while the root carries an unaddressed
    gap). This module only reports the fan-out-time mapping; the synthesis-time
    precondition is owned by the review lane.
  * Parse the convention. Tag parsing, the ``covers:``/``Fold-in:`` vocabulary,
    the prompt limit and the gap oracle all live in ``kanban_streams`` — this
    module imports them, so the two can never drift.

Wiring (W-9, landed)
--------------------
``kanban_decompose._apply_fanout`` calls :func:`emit_receipt` from its own
advisory ``try/except`` after the graph commits, passing the exact
``child_ids`` the fan-out emitted. Rollback is removing that call site: the
emitted comment/event are inert data and nothing else imports this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Optional

from hermes_cli import kanban_streams as streams

#: The receipt's banner. Emitted once per receipt, at the top.
RECEIPT_HEADING = "## Stream coverage receipt (W-7)"

#: Emitted (loudly) when the receipt finds at least one declared stream with no
#: mapping. Asserted verbatim by the test suite — do not reword.
LOUD_FAIL = "LOUD FAIL"

#: The per-gap line prefix. One line per unaccounted stream, so the gap count is
#: countable from the receipt text alone (``text.count(GAP_LINE_PREFIX)`` ==
#: number of gaps) — the property the plan's W-7 AC1/AC2 assert on.
GAP_LINE_PREFIX = "COVERAGE GAP"

#: Emitted instead of a gap table for a root that declares no streams at all
#: (rule C-1: absence is the signal; a gap is never invented). Mirrors
#: ``kanban_streams.NO_STREAMS_DECLARED``.
NO_STREAMS_LINE = streams.NO_STREAMS_DECLARED

#: The comment author the receipt is filed under. Not a profile name — the
#: receipt is machine-written, and the board should read it as such.
RECEIPT_AUTHOR = "coverage-receipt"

#: Event kind for the structured half. A new free-text ``task_events.kind``
#: VALUE (never a new table/column) — see the plan's §0.4 no-schema-surgery
#: constraint. ``emit_receipt`` writes it; W-9's call site invokes that.
RECEIPT_EVENT_KIND = "decomposition_receipt"

#: Disposition labels, in the order the table renders them.
DISPOSITION_COVERED = "COVERED"
DISPOSITION_FOLDED_IN = "FOLDED IN"
DISPOSITION_GAP = "GAP"


@dataclass(frozen=True)
class CoverageGap:
    """One declared stream no child accounted for (``covers:``/``Fold-in:``).

    ``seen_by_llm`` is the *context-boundary* fact, not a compliance judgement:
    a gap that was never in front of the decomposer cannot be blamed on the
    model (plan §7.2).
    """

    tag: str
    offset: int
    seen_by_llm: bool


@dataclass(frozen=True)
class StreamMapping:
    """One row of the receipt: what happened to a single declared stream."""

    tag: str
    offset: int
    seen_by_llm: bool
    disposition: str
    child_id: Optional[str]
    evidence: str


@dataclass
class FanoutReceipt:
    """The whole post-decompose receipt for one root.

    ``mappings`` is the stream -> child table (one row per declared stream, in
    document order); ``gaps`` is the subset with no mapping. ``cut`` is the char
    offset the decomposer's prompt actually showed (``limit - 1`` when the body
    was clamped, the whole length when it fit) — the same boundary
    ``kanban_streams.coverage_appendix`` states. The structured fields back
    :meth:`payload`, which is the ``decomposition_receipt`` task_event payload
    W-9 emits alongside the comment.
    """

    root_id: str
    body_len: int
    cut: int
    declared_total: int
    children_total: int
    mappings: list[StreamMapping] = field(default_factory=list)
    declaring_children: int = 0

    @property
    def gaps(self) -> list[CoverageGap]:
        return [
            CoverageGap(m.tag, m.offset, m.seen_by_llm)
            for m in self.mappings
            if m.disposition == DISPOSITION_GAP
        ]

    @property
    def covered_total(self) -> int:
        return self.declared_total - len(self.gaps)

    @property
    def has_gaps(self) -> bool:
        return bool(self.gaps)

    @property
    def declared_none(self) -> bool:
        return self.declared_total == 0

    @property
    def truncated(self) -> bool:
        """True when the decomposer's prompt body was actually clamped."""
        return self.body_len > self.cut

    @property
    def beyond_cut_tags(self) -> list[str]:
        return [m.tag for m in self.mappings if not m.seen_by_llm]

    @property
    def inside_cut_tags(self) -> list[str]:
        return [m.tag for m in self.mappings if m.seen_by_llm]

    def payload(self) -> dict:
        """The ``decomposition_receipt`` event payload (plan §Metadata Injection).

        Shape is the documented one: ``{root_id, declared_streams[],
        children_emitted[], gaps[], outside_slice[]}``, plus the mapping rows so
        a reader does not have to re-parse the comment markdown.
        """
        return {
            "root_id": self.root_id,
            "declared_streams": [m.tag for m in self.mappings],
            "children_emitted": self.children_total,
            "gaps": [g.tag for g in self.gaps],
            "outside_slice": self.beyond_cut_tags,
            "covered": self.covered_total,
            "declaring_children": self.declaring_children,
            "body_len": self.body_len,
            "cut": self.cut,
            "mappings": [
                {
                    "tag": m.tag,
                    "offset": m.offset,
                    "seen_by_llm": m.seen_by_llm,
                    "disposition": m.disposition,
                    "child_id": m.child_id,
                }
                for m in self.mappings
            ],
        }

    def text(self) -> str:
        """The comment body: banner, counts, the per-stream table, LOUD FAIL."""
        head = (
            f"{RECEIPT_HEADING} — "
            + (
                f"no streams declared (0 gaps)"
                if self.declared_none
                else f"{self.covered_total}/{self.declared_total} covered, "
                     f"{len(self.gaps)} GAP{'S' if len(self.gaps) != 1 else ''}"
            )
        )
        out: list[str] = [head, ""]
        out.append(
            f"Root: {self.root_id or '(unknown)'} · declared "
            f"{self.declared_total} stream(s) · {self.children_total} child(ren) emitted · "
            f"{self.declaring_children} child(ren) declare coverage"
        )

        if self.truncated:
            out.append(
                f"Prompt slice: CUT at char {self.cut} of {self.body_len} — "
                f"{len(self.inside_cut_tags)} declared stream(s) inside the slice, "
                f"{len(self.beyond_cut_tags)} BEYOND it "
                f"({', '.join(self.beyond_cut_tags) or 'none'})."
            )
        else:
            out.append(
                f"Prompt slice: the body ({self.body_len} char(s)) fits the prompt "
                f"budget — every stream was visible to the decomposer."
            )

        if self.declared_none:
            out.append("")
            out.append(
                f"{NO_STREAMS_LINE}: this root declares no `{streams.STREAMS_FIELD}:` "
                f"line and no `## STREAMS` section, so no coverage can be declared "
                f"for it. This is NOT a failure — the receipt invents no gaps."
            )
            return "\n".join(out)

        out.append("")
        out.append("| stream | @char | seen by LLM | disposition | evidence |")
        out.append("|---|---|---|---|---|")
        for m in self.mappings:
            seen = "yes" if m.seen_by_llm else "**NO — beyond the cut**"
            if m.disposition == DISPOSITION_GAP:
                disposition = f"**{DISPOSITION_GAP}**"
            elif m.child_id:
                disposition = f"{m.disposition} by `{m.child_id}`"
            else:
                disposition = m.disposition
            out.append(
                f"| {m.tag} | {m.offset} | {seen} | {disposition} | {m.evidence} |"
            )

        out.append("")
        if self.has_gaps:
            tags = ", ".join(g.tag for g in self.gaps)
            out.append(
                f"**{LOUD_FAIL} — {len(self.gaps)} declared stream(s) have no "
                f"mapping: {tags}**"
            )
            out.append("")
            for g in self.gaps:
                seen = "seen by the decomposer" if g.seen_by_llm else (
                    "BEYOND the prompt slice (never seen by the decomposer)"
                )
                out.append(
                    f"- {GAP_LINE_PREFIX}: {g.tag} @char {g.offset} — {seen}"
                )
            out.append("")
            if self.beyond_cut_tags:
                out.append(
                    "Boundary note: "
                    f"{len(self.beyond_cut_tags)} of the {len(self.gaps)} gap(s) "
                    f"({', '.join(self.beyond_cut_tags)}) lie BEYOND the prompt "
                    "slice, so they may be a CONTEXT-BOUNDARY failure (the "
                    "decomposer never saw them) rather than a compliance failure "
                    "(W-8's contract was ignored). Both classes are named here so "
                    "the reviewer can tell which."
                )
            out.append("")
            out.append(
                "A gap here is advisory at fan-out (the graph has already "
                "committed) and is a **precondition on the synthesis child**: a "
                "synthesis reporting \"complete\" while this root carries an "
                "unaddressed gap fails review. Re-file the missing coverage with "
                f"`{streams.COVERS_KEY}: <tag>`, or record an intentional descope "
                f"with `{streams.FOLDIN_KEY}: <tag> — <why>`."
            )
        else:
            out.append(
                f"Every declared stream is accounted for ({self.declared_total}/"
                f"{self.declared_total}). No gaps."
            )
        return "\n".join(out)


# --- building ----------------------------------------------------------------


def prompt_visible_end(body_len: int, *, limit: int) -> int:
    """The char offset the decomposer's prompt actually showed.

    Mirrors ``kanban_streams.prompt_body`` exactly: the clamp fires only when
    the body exceeds ``limit``, and ``coverage_appendix`` then reports
    ``CUT at char {limit - 1}``; a body that fits is returned whole. Streams are
    "inside the slice" iff ``offset < prompt_visible_end(...)`` — the same
    comparison ``coverage_appendix`` makes, so the receipt and the prompt agree
    on which side of the cut a stream fell on.
    """
    return min(body_len, limit - 1) if body_len > limit else body_len


def _coverage_index(children: Iterable[object]) -> tuple[dict, dict]:
    """``(covers_index, foldin_index)`` -> ``{tag: [(child_id, declaration)]}``.

    ``children`` may be child BODIES (str) or child dicts carrying ``body`` +
    ``id``. Order is preserved so the evidence column names the first child that
    made the declaration, matching ``kanban_streams.missing_coverage`` (which is
    order-insensitive — any child accounting for a tag closes it).
    """
    covers: dict[str, list[tuple[str, str]]] = {}
    foldins: dict[str, list[tuple[str, str]]] = {}
    for idx, child in enumerate(children):
        if isinstance(child, dict):
            body = child.get("body") or ""
            cid = child.get("id") or f"child[{idx}]"
        else:
            body = child if isinstance(child, str) else ""
            cid = f"child[{idx}]"
        for tag in streams.covers_tags(body):
            covers.setdefault(tag, []).append((cid, f"`{streams.COVERS_KEY}: {tag}`"))
        for tag in streams.foldin_streams(body):
            foldins.setdefault(tag, []).append((cid, f"`{streams.FOLDIN_KEY}: {tag}`"))
    return covers, foldins


def fanout_receipt(
    root_body: object, children: Iterable[object], *, root_id: str = "",
    limit: Optional[int] = None,
) -> FanoutReceipt:
    """Build the :class:`FanoutReceipt` for a root body + its emitted children.

    ``children`` takes child bodies or child dicts. ``root_body`` MUST be the
    FULL body (rule C-2) — ``build_receipt``/``read_root_body`` are the call
    shapes that guarantee it; passing a prompt slice yields a vacuous "0
    declared" receipt, which is exactly why the slice is never used.

    Declaration-only: a stream is COVERED iff some child wrote ``covers: <tag>``
    and FOLDED IN iff some child wrote ``Fold-in: <tag>``; otherwise it is a
    GAP. No keyword/text inference anywhere in this function.
    """
    text = root_body if isinstance(root_body, str) else ""
    declared = streams.declared_streams(text)
    child_list = list(children)
    covers, foldins = _coverage_index(child_list)
    declaring = sum(
        1 for c in child_list
        if streams.accounted_tags(c.get("body") if isinstance(c, dict) else c)
    )

    budget = limit if limit is not None else _prompt_body_limit()
    visible_end = prompt_visible_end(len(text), limit=budget)

    mappings: list[StreamMapping] = []
    for s in declared:
        seen = s.offset < visible_end
        if s.tag in covers:
            cid, evidence = covers[s.tag][0]
            disposition, row_cid = DISPOSITION_COVERED, cid
        elif s.tag in foldins:
            cid, evidence = foldins[s.tag][0]
            disposition, row_cid = DISPOSITION_FOLDED_IN, cid
        else:
            disposition, row_cid = DISPOSITION_GAP, None
            evidence = "no child declares it"
        mappings.append(
            StreamMapping(
                tag=s.tag, offset=s.offset, seen_by_llm=seen,
                disposition=disposition, child_id=row_cid, evidence=evidence,
            )
        )

    return FanoutReceipt(
        root_id=root_id, body_len=len(text), cut=visible_end,
        declared_total=len(declared), children_total=len(child_list),
        mappings=mappings, declaring_children=declaring,
    )


def _prompt_body_limit() -> int:
    """The prompt body budget, read from the module that owns it (no second
    drifting literal; see ``kanban_specify.PROMPT_BODY_LIMIT``)."""
    from hermes_cli import kanban_specify as specify

    return int(specify.PROMPT_BODY_LIMIT)


def build_receipt(root_body: object, children: Iterable[object], *,
                  root_id: str = "", limit: Optional[int] = None) -> str:
    """The receipt comment text for a root body + its children (plan §2 W-7)."""
    return fanout_receipt(
        root_body, children, root_id=root_id, limit=limit,
    ).text()


# --- board I/O (full-body read + comment; no status/body writes) --------------


def read_root_body(conn, root_id: str) -> Optional[str]:
    """The root's FULL body, or ``None`` when the row is gone (rule C-2).

    Deliberately the same source ``kanban_gate.propagate_root_gate`` uses
    (``kb.get_task``), NOT the prompt slice.
    """
    from hermes_cli import kanban_db as kb

    task = kb.get_task(conn, root_id)
    return None if task is None else (task.body or "")


def fanout_child_ids(conn, root_id: str) -> list[str]:
    """The ids the root's fan-out emitted, read from the board.

    Unions the newest ``decomposed`` event payload (``child_ids`` — the
    immutable record of what the decomposer emitted) with the LIVE
    ``task_links`` children, event ids first, de-duplicated. The union matters
    on the measured roots: ``t_26d8eb94``'s event lists 4 ids while the board
    also carries 3 later children the receipt must judge (``S3``/``S4``/``S5``),
    and ``t_bec82259``'s event lists 10 while the live board shows 1. Judging
    only one of the two sets would misreport coverage — reading only the links
    would drop declarations the event's children made, and reading only the
    event would miss the 3 S-children that actually carry the audit verdicts.

    Read-only; no fabricated ids.
    """
    from hermes_cli import kanban_db as kb

    ids: list[str] = []
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'decomposed' "
        "ORDER BY id DESC LIMIT 1", (root_id,),
    ).fetchone()
    if row is not None and row["payload"]:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            event_ids = payload.get("child_ids")
            if isinstance(event_ids, list):
                ids.extend(str(c) for c in event_ids if c)
    for cid in kb.child_ids(conn, root_id):
        if cid not in ids:
            ids.append(cid)
    return ids


def load_children(conn, child_ids: Iterable[str]) -> list[dict]:
    """``[{id, body}]`` for each id that still has a row (read-only).

    A child row that has since been deleted is dropped rather than fabricated:
    its declaration cannot be read, and the honest consequence is that any
    stream only it accounted for reports as a gap.
    """
    from hermes_cli import kanban_db as kb

    out: list[dict] = []
    for cid in child_ids:
        task = kb.get_task(conn, cid)
        if task is not None:
            out.append({"id": task.id, "body": task.body or ""})
    return out


def append_receipt_comment(conn, root_id: str, text: str, *,
                           author: str = RECEIPT_AUTHOR) -> int:
    """File the receipt as a comment on the ROOT (rule C-3).

    Uses ``kb.add_comment``, which opens its own write txn with
    ``allow_nested=True`` — so this is safe both standalone and under W-9's
    existing advisory block. Adds a comment (+ its ``commented`` event) and
    nothing else: no task row is read-modify-written.
    """
    from hermes_cli import kanban_db as kb

    return kb.add_comment(conn, root_id, author, text)


def record_receipt_event(conn, receipt: FanoutReceipt, *, author: str = RECEIPT_AUTHOR) -> None:
    """Emit the structured ``decomposition_receipt`` event on the root.

    A new ``task_events.kind`` VALUE only (no schema change). W-9 owns the
    call; this helper exists so the payload shape has a single definition.
    """
    from hermes_cli.kanban_db import _append_event

    payload = receipt.payload()
    payload["author"] = author
    _append_event(conn, receipt.root_id, RECEIPT_EVENT_KIND, payload)


@dataclass
class ReceiptOutcome:
    """What a one-shot receipt run did — returned so W-9 can log it."""

    root_id: str
    ok: bool
    declared_total: int = 0
    gaps: int = 0
    children_total: int = 0
    comment_id: int = 0
    reason: str = ""


def emit_receipt(
    root_id: str, *, child_ids: Optional[Iterable[str]] = None,
    author: str = RECEIPT_AUTHOR, conn=None,
) -> ReceiptOutcome:
    """One-shot: read the FULL body, build the receipt, comment it on the root.

    Never raises on a missing root / empty fan-out — it returns an
    ``ok=False`` outcome with a reason, matching the advisory contract the
    caller (W-9) relies on. Opening a connection is the caller's job unless
    ``conn`` is supplied.

    This function writes a comment and an event and NOTHING else. It must stay
    that way: it runs post-commit, on a graph that is already durable.
    """
    from hermes_cli import kanban_db_connect as kbc

    def _run(c) -> ReceiptOutcome:
        body = read_root_body(c, root_id)
        if body is None:
            return ReceiptOutcome(root_id, False, reason="root task not found")
        ids = list(child_ids) if child_ids is not None else fanout_child_ids(c, root_id)
        children = load_children(c, ids)
        receipt = fanout_receipt(body, children, root_id=root_id)
        comment_id = append_receipt_comment(c, root_id, receipt.text(), author=author)
        try:
            record_receipt_event(c, receipt, author=author)
        except Exception:
            # The comment is the deliverable; the event is a structured mirror.
            # A failure on the mirror must not lose the receipt.
            pass
        return ReceiptOutcome(
            root_id, True, declared_total=receipt.declared_total,
            gaps=len(receipt.gaps), children_total=receipt.children_total,
            comment_id=comment_id,
        )

    try:
        if conn is not None:
            return _run(conn)
        with kbc.connect_closing() as c:
            return _run(c)
    except Exception as exc:  # advisory: never raise into a committed fan-out
        return ReceiptOutcome(root_id, False, reason=f"{type(exc).__name__}: {exc}")


__all__ = [
    "CoverageGap",
    "DISPOSITION_COVERED",
    "DISPOSITION_FOLDED_IN",
    "DISPOSITION_GAP",
    "FanoutReceipt",
    "GAP_LINE_PREFIX",
    "LOUD_FAIL",
    "NO_STREAMS_LINE",
    "RECEIPT_AUTHOR",
    "RECEIPT_EVENT_KIND",
    "RECEIPT_HEADING",
    "ReceiptOutcome",
    "StreamMapping",
    "append_receipt_comment",
    "build_receipt",
    "emit_receipt",
    "fanout_child_ids",
    "fanout_receipt",
    "load_children",
    "prompt_visible_end",
    "read_root_body",
    "record_receipt_event",
]
