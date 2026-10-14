"""Review findings must reach the implementer on a card whose thread is LARGE.

Regression cover for the t_9642c749 defect: on card ``t_599b5666`` the comment
thread grew to ~109 KB, which pushed the ``kanban_show`` payload past the
tool-result spillover threshold (100,000 chars). The agent then replaced the
whole payload with a file pointer, only the first 1,500 chars survived inline,
and rounds 4, 5 and 6 of the implementer ran with ZERO write-tool calls —
re-reporting rounds 1-3's work as their own, because the reviewer's findings had
stopped arriving. Measured from the incident's own sessions:

    round 1-3  inline payload   -> 14 / 11 / 10 write calls
    round 4-6  spilled payload  -> 0 / 0 / 0 write calls

Three contracts are pinned here:

1. The verdict rides the HEAD of ``kanban_show``'s JSON, small enough to survive
   the spillover preview (the preview keeps the FIRST
   ``DEFAULT_PREVIEW_SIZE_CHARS`` of an oversized result).
2. The same verdict is rendered at the top of ``build_worker_context`` — the
   string the worker reads as `kanban_show`'s ``worker_context`` field (both
   channels ride the same payload; the head position is what makes them safe) —
   and is tail-capped, not lost.
3. Consecutive request-changes rounds whose handoffs declare no new revision
   auto-escalate to a human instead of being re-dispatched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

VERDICT = (
    "REQUIRED (HARD) — the doubled leading pipe on "
    "wiki/system/j6-composite-exit.md:92-94 is still present: 8 pipes in a "
    "7-pipe table. MD5(disk)==MD5(git show HEAD:) and "
    "`git log 26774d7..HEAD -- wiki/` is EMPTY."
)
# Enough per-round bulk to push a multi-round card past the 100,000-char
# spillover threshold — the whole point of the regression.
BULK = (
    "Re-measured this round: HEAD is still 26774d7 (the round-3 commit), the "
    "working tree is clean, and all four round-3 REQUIRED items remain open. "
    "The reviewer re-read each named page end-to-end, re-ran wiki-validate.py "
    "--pages j6-composite-exit d-lane-metrics-audit, and diffed every table "
    "row against the implementation handoffs in parents t_3ab78e44 and "
    "t_787a3ad0. Evidence: md5(disk) == md5(git show HEAD:<path>) for all "
    "three named files; `git log 26774d7..HEAD -- wiki/` returns EMPTY.\n"
)
LARGE_THREAD_ROUNDS = 9


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Isolated board + a claimed implementer task."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return str(home)


def _round(conn, tid: str, n: int, *, commit: str | None, reason: str = VERDICT,
           comment_filler: str = "", metadata: dict | None = None) -> None:
    """One review cycle: implementer handoff -> reviewer comment -> request_changes."""
    run_id = kb.get_task(conn, tid).current_run_id
    if metadata is None:
        metadata = {"changed_files": ["wiki/system/j6-composite-exit.md"]}
        if commit:
            metadata["commit"] = commit
    assert kb.request_review(
        conn, tid, reviewer="reviewer", expected_run_id=run_id, metadata=metadata,
        summary=f"Round-{n}: addressed findings (commit {commit or 'none'})",
    )
    kb.add_comment(conn, tid, "reviewer", f"Round {n}: {reason}\n{comment_filler}")
    review = kb.claim_review_task(conn, tid, claimer=f"reviewer:{n}")
    assert review is not None
    ok, detail = kb.request_changes(
        conn, tid, reason=f"Round {n}: {reason}\n{comment_filler}",
        expected_run_id=review.current_run_id,
    )
    assert ok is True, detail


def _rework_claim(conn, tid: str, n: int) -> bool:
    """Claim the card as the implementer again; False once it left the loop."""
    if kb.get_task(conn, tid).status in ("blocked", "triage"):
        return False
    kb.claim_task(conn, tid, claimer=f"builder:{n}")
    return True


def _last_event(conn, tid: str, kind: str):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind][-1]


# ---------------------------------------------------------------------------
# 1. The verdict is the FIRST key of kanban_show and survives spillover
# ---------------------------------------------------------------------------

def test_verdict_rides_the_head_of_an_oversized_kanban_show(scratch: str) -> None:
    """A payload past the spillover threshold keeps only its head inline, so the
    verdict must be the first key AND fit inside the preview budget."""
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="large thread", assignee="builder",
            body="Land the wiki pages.\n" * 40,
        )
        kb.claim_task(conn, tid, claimer="builder:0")
        for n in range(1, LARGE_THREAD_ROUNDS + 1):
            _round(conn, tid, n, commit=f"abc{n}", comment_filler=BULK * 6)
            if not _rework_claim(conn, tid, n):
                break

        from tools.tool_result_storage import generate_preview
        from tools.budget_config import DEFAULT_PREVIEW_SIZE_CHARS

        raw = kt._handle_show({"task_id": tid})
        assert len(raw) > 100_000, (
            f"fixture must cross the 100k spillover threshold, got {len(raw):,}"
        )
        data = json.loads(raw)
        assert next(iter(data)) == "review_directive"
        directive = data["review_directive"]
        assert VERDICT in directive
        # The preview keeps the leading DEFAULT_PREVIEW_SIZE_CHARS characters of
        # an oversized payload. Anything past that is replaced by a pointer, so
        # the verdict has to land inside it.
        preview, _has_more = generate_preview(raw, DEFAULT_PREVIEW_SIZE_CHARS)
        assert json.dumps(VERDICT)[1:-1] in preview
        assert len(json.dumps(directive)) < DEFAULT_PREVIEW_SIZE_CHARS


def test_verdict_survives_maybe_persist_tool_result_end_to_end(scratch: str) -> None:
    """Drive the REAL persistence layer, not a size comparison: the delivered
    string must still carry the finding after spillover."""
    from tools import kanban_tools as kt
    from tools.tool_result_storage import maybe_persist_tool_result

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="spill me", assignee="builder",
            body="Land the wiki pages.\n" * 40,
        )
        kb.claim_task(conn, tid, claimer="builder:0")
        for n in range(1, LARGE_THREAD_ROUNDS + 1):
            _round(conn, tid, n, commit=f"abc{n}", comment_filler=BULK * 6)
            if not _rework_claim(conn, tid, n):
                break

        raw = kt._handle_show({"task_id": tid})
        delivered = maybe_persist_tool_result(raw, "kanban_show", "call_regression")

    assert "<persisted-output>" in delivered, (
        "fixture did not cross the spillover threshold — the regression would "
        "not be exercised"
    )
    # json.dumps escapes the verdict's non-ASCII punctuation; the preview is a
    # raw slice of that JSON, so compare against the escaped form.
    assert json.dumps(VERDICT)[1:-1] in delivered, (
        "the reviewer's finding did not survive spillover — this is exactly the "
        "defect that made request_changes unable to converge on t_599b5666"
    )


# A verdict written entirely in CJK: json.dumps escapes each character to
# ``\uXXXX`` (6 chars), so the serialized value is ~4.4x its char length. This is
# the shape that exposed the char-vs-escaped budget mismatch (t_9642c749 review
# F3): the old bound measured 1,200 unescaped chars, which escaped to ~5,200 and
# pushed the receipt pointer past the end of the 1.5 KB preview.
CJK_VERDICT = (
    "必须修复：wiki/system/j6-composite-exit.md 中 J6 表格的重复竖线；"
    "队列数量应为 17 而不是 18；Stored R 单元格需要从交易日志重新计算；"
    "另外 log.md 的复述也要同步修正。 " * 14
)


def test_non_ascii_verdict_keeps_its_receipt_pointer_inside_the_preview(
    scratch: str,
) -> None:
    """The bound must apply to the ESCAPED length, because the preview slices the
    serialized payload — not the string the reader would see.

    A CJK verdict of ~1,580 chars escapes past the whole preview under a
    char-only bound, cutting the verdict mid-sentence and dropping the receipt
    pointer: the one pointer to the untruncated text (t_9642c749 review F3,
    reproduced on the real ``_handle_show`` + ``maybe_persist_tool_result``
    path)."""
    from tools import kanban_tools as kt
    from tools.budget_config import DEFAULT_PREVIEW_SIZE_CHARS
    from tools.tool_result_storage import maybe_persist_tool_result

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="cjk verdict", assignee="builder",
            body="Land the wiki pages.\n" * 40,
        )
        kb.claim_task(conn, tid, claimer="builder:0")
        # Rounds 1..N-1 grow the thread past the spillover threshold; the LAST
        # handoff carries the CJK verdict, because ``latest_review_directive``
        # returns the newest un-addressed one.
        for n in range(1, LARGE_THREAD_ROUNDS):
            _round(conn, tid, n, commit=f"abc{n}", comment_filler=BULK * 6)
            assert _rework_claim(conn, tid, n)
        _round(conn, tid, LARGE_THREAD_ROUNDS, commit="abclast",
               reason=CJK_VERDICT, comment_filler=BULK * 6)

        raw = kt._handle_show({"task_id": tid})
        delivered = maybe_persist_tool_result(raw, "kanban_show", "call_cjk")

    assert "<persisted-output>" in delivered, (
        f"fixture did not spill (raw {len(raw):,} chars) — the regression would "
        f"not be exercised"
    )
    # The preview is the first DEFAULT_PREVIEW_SIZE_CHARS of the raw JSON, so
    # this is literally what survives inline.
    preview = raw[:DEFAULT_PREVIEW_SIZE_CHARS]
    assert "review_receipts/" in preview, (
        "the receipt pointer — the only pointer to the untruncated verdict — "
        "fell outside the spillover preview for a non-ASCII verdict"
    )
    assert json.dumps(CJK_VERDICT[:24])[1:-1] in preview, (
        "the head of the verdict did not reach the worker inline"
    )
    directive = json.loads(raw)["review_directive"]
    assert json.dumps(directive)[1:-1] in raw, "directive is not a JSON value of the payload"
    assert len(json.dumps(directive)) - 2 <= kb._SHOW_REVIEW_VERDICT_ESCAPED_CHARS, (
        "the emitted directive exceeds its escaped budget"
    )


def test_ascii_verdict_is_not_shrunk_by_the_escaped_bound(scratch: str) -> None:
    """The escaped bound must not truncate ordinary ASCII verdicts more than the
    char bound already did: for ASCII the two budgets are the same size."""
    huge = VERDICT + "\n" + ("padding detail line\n" * 3000)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ascii huge", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        _round(conn, tid, 1, commit="abc1", reason=huge)
        directive = kb.latest_review_directive(conn, tid)

    ascii_block = kb.review_directive_block(
        directive, escaped_limit=kb._SHOW_REVIEW_VERDICT_ESCAPED_CHARS,
    )
    ctx_block = kb.review_directive_block(directive)
    assert len(ascii_block) <= kb._SHOW_REVIEW_VERDICT_ESCAPED_CHARS
    # The worker_context channel keeps its far larger allowance.
    assert len(ascii_block) < len(ctx_block)
    assert "review_receipts/" in ascii_block
    assert "chars omitted" in ascii_block


# ---------------------------------------------------------------------------
# 2. build_worker_context renders the verdict at the top, bounded
# ---------------------------------------------------------------------------

def test_worker_context_leads_with_the_verdict(scratch: str) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="ctx", assignee="builder", body="Land the wiki pages.",
        )
        kb.claim_task(conn, tid, claimer="builder:0")
        _round(conn, tid, 1, commit="abc1")
        ctx = kb.build_worker_context(conn, tid)

    directive_at = ctx.index("## Requested changes (latest review verdict)")
    body_at = ctx.index("## Body")
    thread_at = ctx.index("## Comment thread")
    assert directive_at < body_at < thread_at, (
        "the directive must precede the body and the comment thread: an "
        "over-threshold payload only keeps its head"
    )
    assert VERDICT in ctx
    assert kb.review_receipt_path(tid, 1).exists()


def test_worker_context_directive_is_bounded_on_a_huge_verdict(scratch: str) -> None:
    """A pathological verdict is truncated with a visible marker and a receipt
    path — never dropped, never unbounded."""
    huge = VERDICT + "\n" + ("padding detail line\n" * 3000)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="huge verdict", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        _round(conn, tid, 1, commit="abc1", reason=huge)
        directive = kb.latest_review_directive(conn, tid)

    assert directive is not None
    block = kb.review_directive_block(directive)
    assert len(block) < kb._CTX_MAX_REVIEW_DIRECTIVE_BYTES + 2000
    assert "chars omitted" in block
    assert directive["receipt_path"] and "round-1.md" in directive["receipt_path"]


def test_answered_verdict_is_not_replayed_as_a_directive(scratch: str) -> None:
    """A verdict the implementer already handed back is history, not a directive."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="answered", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        _round(conn, tid, 1, commit="abc1")
        assert kb.latest_review_directive(conn, tid) is not None

        # The implementer claims the rework and hands it back.
        kb.claim_task(conn, tid, claimer="builder:1")
        run_id = kb.get_task(conn, tid).current_run_id
        kb.request_review(conn, tid, reviewer="reviewer", expected_run_id=run_id,
                          metadata={"commit": "abc2"}, summary="rework done")
        assert kb.latest_review_directive(conn, tid) is None, (
            "a verdict already answered by a newer handoff must not be pinned to "
            "the top of the context forever"
        )


def test_directive_is_absent_without_a_review_verdict(scratch: str) -> None:
    """No review history -> no directive, for both delivery channels."""
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="fresh", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        ctx = kb.build_worker_context(conn, tid)
    assert "## Requested changes" not in ctx
    data = json.loads(kt._handle_show({"task_id": tid}))
    assert data["review_directive"] is None


# ---------------------------------------------------------------------------
# 3. Non-convergence auto-escalates instead of re-dispatching
# ---------------------------------------------------------------------------

def test_consecutive_no_revision_rounds_escalate_to_a_human(scratch: str) -> None:
    """The incident shape: rounds 1-3 land revisions, rounds 4+ do not. The
    third consecutive no-revision rework round must leave the dispatch loop."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="non converging", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        # Rounds 1-2 land real revisions (the card CAN converge).
        _round(conn, tid, 1, commit="abc1")
        assert _rework_claim(conn, tid, 1)
        _round(conn, tid, 2, commit="abc2")
        assert kb.get_task(conn, tid).status == "ready"
        assert _rework_claim(conn, tid, 2)

        # From here the tree never moves — exactly what the reviewer measured.
        _round(conn, tid, 3, commit=None)
        assert kb.get_task(conn, tid).status == "ready", (
            "one no-revision round is a normal review cycle, not a stall"
        )
        assert _rework_claim(conn, tid, 3)
        _round(conn, tid, 4, commit=None)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked", (
            "two consecutive no-revision rework rounds must escalate to a human "
            "instead of being re-dispatched a third time"
        )
        assert task.block_kind == "needs_input"
        events = {e.kind: e for e in kb.list_events(conn, tid)}
        assert "review_not_converging" in events
        payload = events["review_not_converging"].payload
        assert payload["no_revision_streak"] >= 2
        # The verdict that triggered escalation is still delivered.
        assert events["changes_requested"].payload["escalated"] is True
        assert kb.latest_review_directive(conn, tid)["reason"].strip()


def test_escalation_does_not_spend_the_block_loop_breaker(scratch: str) -> None:
    """Escalation is a review outcome, not a block loop: the reviewer must stay
    able to request changes, and repeated review must never trip the breaker."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="breaker", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        _round(conn, tid, 1, commit="abc1")
        _rework_claim(conn, tid, 1)
        _round(conn, tid, 2, commit=None)
        _rework_claim(conn, tid, 2)
        _round(conn, tid, 3, commit=None)

        row = conn.execute(
            "SELECT block_recurrences, block_kind FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert (row["block_recurrences"] or 0) == 0
        assert "block_loop_detected" not in {e.kind for e in kb.list_events(conn, tid)}


def test_escalated_card_resumes_in_the_implementer_lane(scratch: str) -> None:
    """An operator unblock must NOT land the card in ``review`` while the
    IMPLEMENTER still holds it: the dispatcher claims review rows with
    ``row["assignee"]``, so that spawns the implementer as its own reviewer
    (t_9642c749 review F1 — self-approval on the terminal human-decision path).
    It resumes in the implementer's own lane instead, carrying the decision."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="resume", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        _round(conn, tid, 1, commit="abc1")
        _rework_claim(conn, tid, 1)
        _round(conn, tid, 2, commit=None)
        _rework_claim(conn, tid, 2)
        _round(conn, tid, 3, commit=None)
        assert kb.get_task(conn, tid).status == "blocked"

        # The escalation no longer asks to resume in the reviewer's lane, and
        # the legacy fallback key must not say ``review`` either: it would
        # override ``resume_status`` and restore the self-review lane.
        blocked = _last_event(conn, tid, "blocked")
        assert blocked.payload["resume_status"] == "ready"
        assert blocked.payload.get("source_status") != "review"

        assert kb.unblock_task(conn, tid) is True
        resumed = kb.get_task(conn, tid)
        assert resumed.status == "ready"
        assert resumed.assignee == "builder"


def test_unblock_never_spawns_the_implementer_into_a_review_run(
    scratch: str, all_assignees_spawnable
) -> None:
    """The regression F1 asks for: assert the PROFILE THE DISPATCHER SPAWNS,
    driven through the real dispatch path — not the landed status.

    Two shapes of ``resume_status='review'``:

    * a genuine reviewer-owned block resumes in ``review`` with the reviewer
      spawned (the ordinary path must keep working);
    * a block whose card is still assigned to the implementer of record must
      not produce a review-lane run for that implementer.
    """
    from hermes_cli import kanban_db_dispatch as kbd

    def _tick() -> list[str]:
        seen: list[str] = []

        def _record(task, workspace, board=None):
            seen.append(task.assignee)
            return None

        with kbc.connect() as conn:
            kbd.dispatch_once(conn, spawn_fn=_record, dry_run=False, board=None)
        return seen

    with kbc.connect() as conn:
        # (1) Ordinary path: the reviewer holds the card -> review lane, reviewer spawned.
        ordinary = kb.create_task(conn, title="ordinary review", assignee="builder")
        kb.claim_task(conn, ordinary, claimer="builder:0")
        run_id = kb.get_task(conn, ordinary).current_run_id
        kb.request_review(conn, ordinary, reviewer="reviewer", expected_run_id=run_id,
                          metadata={"commit": "abc1"}, summary="ready")
        review = kb.claim_review_task(conn, ordinary, claimer="reviewer:1")
        assert kb.block_task(conn, ordinary, reason="needs a human",
                             kind="needs_input", expected_run_id=review.current_run_id)
        assert kb.unblock_task(conn, ordinary) is True
        assert kb.get_task(conn, ordinary).status == "review"
        assert kb.get_task(conn, ordinary).assignee == "reviewer"
        assert _tick() == ["reviewer"], "the reviewer of record must be the one spawned"

        # (2) F1 shape: implementer is still the assignee -> no review-lane run.
        f1 = kb.create_task(conn, title="f1", assignee="builder")
        kb.claim_task(conn, f1, claimer="builder:0")
        _round(conn, f1, 1, commit="abc1")
        _rework_claim(conn, f1, 1)
        _round(conn, f1, 2, commit=None)
        _rework_claim(conn, f1, 2)
        _round(conn, f1, 3, commit=None)
        assert kb.get_task(conn, f1).status == "blocked"
        # Force the legacy hint: an operator unblocks a row whose newest
        # lifecycle event still asks for the review lane.
        with kb.write_txn(conn):
            kb._append_event(conn, f1, "blocked",
                             {"reason": "legacy", "kind": "needs_input",
                              "resume_status": "review", "source_status": "review",
                              "origin": "review_not_converging"})
        assert kb.unblock_task(conn, f1) is True
        landed = kb.get_task(conn, f1)
        assert landed.status == "ready", (
            "unblock spawned the implementer into a review-lane run"
        )
        spawned = _tick()
        assert "builder" in spawned
        with kbc.connect() as probe:
            payload = probe.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='claimed' "
                "ORDER BY id DESC LIMIT 1", (f1,)).fetchone()["payload"]
        assert "review" not in str(payload), (
            "the implementer was given a reviewer-owned run on its own card"
        )


def test_commitless_workflow_never_escalates_on_this_signal(scratch: str) -> None:
    """A card whose handoffs never carry a commit (docs/analysis work) must not
    be escalated by a signal it cannot satisfy."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="docs card", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        for n in range(1, 6):
            _round(conn, tid, n, commit=None)
            assert kb.get_task(conn, tid).status == "ready", (
                f"escalated at round {n} on a card that never declared a commit"
            )
            assert _rework_claim(conn, tid, n)

        state = kb.review_convergence_state(conn, tid)
        assert state["ever_declared_revision"] is False
        assert state["escalate"] is False
        assert state["no_revision_streak"] == 5


@pytest.mark.parametrize(
    "metadata, expected",
    [
        # Accepted — every shape a worker actually writes.
        ({"commit": "abc1"}, True),
        ({"commits": "abc1"}, True),
        ({"commits": ["abc1", "abc2"]}, True),
        ({"commits": ("abc1",)}, True),
        ({"merged": True}, True),
        ({"head": "abc1"}, True),
        # t_9642c749 review F2: the dict shapes, incl. this card's own round-2
        # handoff (``commits: {fix: ..., doc_note: ...}``).
        ({"commits": {"fix": "abc1", "doc_note": "abc2"}}, True),
        ({"commit": {"sha": "abc1"}}, True),
        ({"merged": {"pr": 7}}, True),
        ({"commits": [{"fix": "abc1"}]}, True),
        ({"commits": {"fix": {"sha": "abc1"}}}, True),
        ({"head": 12345}, True),
        ({"commits": 17}, True),
        # Rejected — no revision is named.
        ({}, False),
        ({"commit": None}, False),
        ({"commits": {}}, False),
        ({"commits": []}, False),
        ({"commits": [{}]}, False),
        ({"commits": {"fix": None}}, False),
        ({"commit": ""}, False),
        ({"commit": "   "}, False),
        ({"merged": False}, False),
        ({"head": 0}, False),
        ({"commit": 0}, False),
        ({"changed_files": ["a.py"]}, False),
    ],
)
def test_revision_value_shapes_are_read_not_guessed(
    metadata: dict, expected: bool
) -> None:
    """The revision predicate must accept every shape that NAMES a revision.

    A false negative here is not cosmetic: it is the DB-visible proxy for "the
    tree did not move", so reading a declared revision as absent escalates a
    card that is progressing (t_9642c749 review F2 — the shipped predicate
    accepted str/bool/list but not a dict, while ``commits`` is a plural key and
    a per-artifact mapping is its natural shape)."""
    assert kb._handoff_declares_revision(metadata) is expected


def test_revision_value_shapes_never_recurse_forever() -> None:
    """A self-referential structure must not hang the dispatcher: the board's
    metadata column is JSON (acyclic), but the predicate is a public-ish helper
    that a caller can hand anything."""
    cyclic: dict = {"fix": "abc1"}
    cyclic["self"] = cyclic
    assert kb._handoff_declares_revision({"commits": cyclic}) is True
    cycle_only: dict = {}
    cycle_only["self"] = cycle_only
    assert kb._handoff_declares_revision({"commits": cycle_only}) is False


def test_dict_shaped_commits_do_not_escalate_a_progressing_card(scratch: str) -> None:
    """End-to-end F2: a card that declares its revisions as a per-artifact
    mapping every round must keep being re-dispatched, not escalated."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="dict commits", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        for n in range(1, 4):
            _round(conn, tid, n, commit=None, metadata={
                "changed_files": ["hermes_cli/kanban_db.py"],
                "commits": {f"artifact_{n}": f"sha{n}", "doc": f"sha{n}b"},
            })
            state = kb.review_convergence_state(conn, tid)
            assert state["no_revision_streak"] == 0, (
                f"round {n} declared its revisions as a per-artifact mapping and "
                f"was read as no-new-revision (streak {state['no_revision_streak']})"
            )
            assert state["escalate"] is False
            assert kb.get_task(conn, tid).status == "ready", (
                f"escalated at round {n} on a card that declared a real commit"
            )
            assert _rework_claim(conn, tid, n)


def test_review_receipt_is_written_per_round_with_the_full_verdict(scratch: str) -> None:
    """The receipt is the size-independent fallback: full text, stable path."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="receipts", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")
        _round(conn, tid, 1, commit="abc1")
        _round(conn, tid, 2, commit="abc2", reason=VERDICT + " SECOND ROUND DETAIL")

        first = kb.review_receipt_path(tid, 1)
        second = kb.review_receipt_path(tid, 2)
        assert first.exists() and second.exists()
        assert VERDICT in first.read_text(encoding="utf-8")
        assert "SECOND ROUND DETAIL" in second.read_text(encoding="utf-8")
        assert "round 2" in second.read_text(encoding="utf-8")


def test_review_receipt_write_failure_never_fails_the_transition(
    scratch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filesystem failure is advisory: the verdict must still be recorded."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="fs fail", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder:0")

        def _boom(*a, **kw):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(kb.Path, "write_text", _boom)
        _round(conn, tid, 1, commit="abc1")

        event = [e for e in kb.list_events(conn, tid) if e.kind == "changes_requested"][-1]
        assert event.payload["reason"].strip()
        assert event.payload["round"] == 1
        # The declared path must not outlive a failed write: a dangling receipt
        # pointer would send the implementer to a file that does not exist.
        assert event.payload["receipt_path"] is None
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.latest_review_directive(conn, tid)["receipt_path"] is None
