"""Every DOCUMENTED audit-event name is pinned against the MEASURED one (t_6c8342e0).

Round-3 review Finding 1: round 3 introduced ``<stem>_displaced_not_owner`` for
the pin-less operator lane *specifically* so the audit trail would not claim a
``--force`` that never happened — and then left nine source sites naming
``<stem>_forced_not_owner`` for that same lane, an event the lane never emits.
It is the round-2 Finding 2 class ("a comment claims an audit event that does not
happen"), one instance fixed and the class surviving at nine more; round-1
Finding 4 and round-2 Finding 2 were the adjacent citation class. The T6 group in
``test_kanban_live_run_close_guard.py`` pins the EMITTED name. Nothing pinned the
CLAIMED one, which is why the class kept recurring — this file does.

Three halves, and all three are needed:

  * MEASURED (``test_t8_measured_lane_matrix_is_verbatim``, driven by ``_drive``) —
    drive each (verb, block kind, caller shape, force) cell against a hermetic
    card whose live run belongs to ANOTHER process, and read back the
    ``*_not_owner`` kind that actually landed in ``task_events``. ALL SIX run-end
    verb stems are driven — ``completion | changes | block | schedule | reclaim |
    archive`` — because comment 911 on the wiki-landing card hands the wiki-keeper
    a 6-row table; a matrix that measured two of the six could not support that
    handoff (round-4 review Finding 3). The matrix is asserted verbatim, so a
    change to an emitted name has to be stated here rather than discovered by a
    reader of the comments.
  * CLAIMED (``test_t8_documented_names_match_the_measured_event``) — for every
    source site that names an event for a caller lane it describes, assert the
    name MEASURED for that lane appears in the site's own text, and that the
    site's anchor markers still exist. Renaming an emitted event without updating
    the site that documents it fails; moving or deleting the site fails loudly
    instead of drifting.
  * COVERED (``test_t8_every_literal_in_the_documented_sources_is_inside_a_site``)
    — the union of the declared segments contains EVERY literal
    ``<verb>_<kind>_not_owner`` mention in the four documented source files. That
    is what makes the handoff's "cite the comments freely" a measured statement
    instead of a hope (round-4 review Finding 3), and it is the specific gap that
    let 17 source lines name an event no segment covered.

Lane vocabulary — what a site declares it documents:

  * ``pinless_lands`` — the pin-less operator lane's event where the lane LANDS:
    ``reclaim``/``archive`` (the recovery verbs) and ``block`` + kind
    ``"dependency"`` (the T5 STOP park). Emits ``<stem>_displaced_not_owner``.
  * ``pinless_refused`` — the pin-less lane where it does NOT land: every other
    verb/kind is refused from a pin-less shell, so the refusal IS that lane's
    event. Emits ``<stem>_refused_not_owner``.
  * ``refused`` — the worker-pinned refusal, ``<stem>_refused_not_owner``.
  * ``override`` — the ``force=True`` override, ``<stem>_forced_not_owner``: the
    only lane the ``forced`` name is actually about.
  * ``pinless_generic`` — the ``<stem>_`` placeholder form of ``pinless_lands``,
    for the module-level policy blocks that state the rule for the whole verb
    family rather than for one verb.

The pin-less recovery row is the load-bearing one: ``--force`` is INERT there
(the lane already lands), so a doc that presents ``<stem>_forced_not_owner`` as
what a pin-less ``hermes kanban reclaim``/``archive`` records is a false claim
about both the event and the flag (round-3 Finding 1). The ``forced`` name
belongs to the worker-pinned lane, which is refused without the flag.

Round-4 review Finding 2 was this file's own docstring naming a driver symbol that
exists nowhere in the repo (the real driver is ``_drive``) — the same class as the
defect it was built to stop, inside the file built to stop it.
``test_t8_self_check_no_dangling_symbols`` now pins that class: every
double-backtick ``_<token>`` in this module's text must resolve to a real callable
or module attribute.

Hermetic-only: the ``kanban_home`` fixture builds a temp board under a per-test
HERMES_HOME (never ``~/.hermes/kanban.db``). Run via ``scripts/run_tests.sh``.
"""

from __future__ import annotations

import ast
import os
import re
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Pids that are never this process: the card's live run must belong to someone
#: else for the "not owner" lanes to be reachable at all.
FOREIGN_PID = os.getpid() + 1

#: The four source files whose prose documents an audit-event name. Kept in one
#: place because both the coverage half and the claimed half reason over them.
DOCUMENTED_SOURCES: tuple[str, ...] = (
    "hermes_cli/kanban_db.py",
    "hermes_cli/kanban.py",
    "hermes_cli/kanban_parser.py",
    "plugins/kanban/dashboard/plugin_api.py",
)

#: A literal audit-event name for one of the six run-end verb stems. The
#: ``<stem>_`` placeholder form deliberately does NOT match: it is policy prose
#: for the whole family, and the claimed half asserts it separately.
_EVENT_LITERAL = re.compile(
    r"(completion|changes|block|schedule|reclaim|archive)_(refused|forced|displaced)_not_owner"
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Temp board + stripped worker pins (mirrors the live-run-close-guard suite)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(kb, "_OPENED_RUNS", {}, raising=False)
    return db_path


def _raw(sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(os.environ["HERMES_KANBAN_DB"]))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _audit_kinds(tid: str) -> list[str]:
    """Every audit-family kind on *tid*, in write order."""
    return [
        row[0] for row in _raw(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,),
        )
        if row[0].endswith("_not_owner")
    ]


def _displacement_card(title: str) -> tuple[str, int]:
    """A card whose live run belongs to ANOTHER process — the shape every
    "not owner" lane is about.

    The ledger entry is dropped afterwards: in production the DISPATCHER opens
    the run, so the caller holds no opened-run proof of ownership. Without that,
    the in-process claim would make this test's caller the owner and every
    measurement below would take the owner lane instead.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=title, assignee="builder")
        kb.claim_task(conn, tid, claimer="host:7777")
        live = kb._current_run_id(conn, tid)
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (FOREIGN_PID, tid))
        conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (FOREIGN_PID, live))
        board = kb._board_identity(conn)
    kb._OPENED_RUNS.pop((board, tid), None)
    return tid, live


def _assert_run_untouched(tid: str, live: int) -> None:
    assert _raw("SELECT status FROM tasks WHERE id=?", (tid,))[0][0] == "running"
    assert _raw("SELECT current_run_id FROM tasks WHERE id=?", (tid,))[0][0] == live
    ended, outcome = _raw(
        "SELECT ended_at, outcome FROM task_runs WHERE id=?", (live,),
    )[0]
    assert ended is None, f"the foreign live run was ENDED ({outcome!r})"


def _call_verb(conn, verb: str, tid: str, *, kind=None, force: bool):
    """Invoke one run-end verb; return (ok, extra) with ``ok`` a plain bool.

    Every terminal verb is driven through its real API — no verb is asserted by
    name alone, so "refused" here means the call returned a refusal, not that a
    string matched. ``request_changes`` returns ``(ok, reason)`` and is folded to
    its first element; the reason is only useful when the call was admitted.
    """
    if verb == "reclaim":
        return bool(kb.reclaim_task(
            conn, tid, reason="x", force=force, signal_fn=lambda *a: None,
        )), None
    if verb == "archive":
        return bool(kb.archive_task(
            conn, tid, force=force, signal_fn=lambda *a: None,
        )), None
    if verb == "block":
        return bool(kb.block_task(
            conn, tid, reason="x", kind=kind, force=force,
        )), None
    if verb == "schedule":
        return bool(kb.schedule_task(conn, tid, reason="x", force=force)), None
    if verb == "complete":
        return bool(kb.complete_task(conn, tid, summary="x", force=force)), None
    if verb == "request_changes":
        ok, extra = kb.request_changes(conn, tid, reason="x", force=force)
        return bool(ok), extra
    raise AssertionError(f"unhandled verb {verb!r}")


def _drive(
    verb: str, *, shape: str, force: bool, monkeypatch, kind: str | None = None,
) -> str:
    """Drive one (verb, block kind, caller shape, force) cell; return the audit kind.

    ``shape`` is ``"pinless"`` (the operator shell: invariant I3, no worker pin —
    exactly what ``hermes kanban <verb>`` from a shell is) or ``"worker"`` (pinned
    to this card via ``HERMES_KANBAN_TASK``, so it must be refused without
    ``force``).

    Exactly one audit event must land: a cell that emitted two would make the
    matrix's single expected name meaningless, and ``_audit_kinds`` would return
    the pair, so the equality assertion below would pass on neither.
    """
    tid, _live = _displacement_card(f"t8 {verb} {kind} {shape} force={force}")
    if shape == "worker":
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "0")
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    try:
        with kbc.connect() as conn:
            _call_verb(conn, verb, tid, kind=kind, force=force)
    finally:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    kinds = _audit_kinds(tid)
    assert kinds, f"{verb}/{kind}/{shape}/force={force} left no audit event at all"
    assert len(kinds) == 1, f"{verb}/{kind}/{shape}/force={force} emitted {kinds}"
    return kinds[0]


#: ``block`` is the only verb whose operator admission is scoped on the block
#: kind (``_operator_park``), so it needs two cells per (shape, force) rather
#: than one. ``needs_input`` is the representative "any other kind".
#: The six run-end verbs, by the documented EVENT STEM their sites use. The API
#: verb differs from the stem for two of them (``complete`` -> ``completion``,
#: ``request_changes`` -> ``changes``), which is exactly the kind of naming
#: mismatch a gate should make explicit rather than paper over.
_STEM_TO_API_VERB: dict[str, str] = {
    "completion": "complete",
    "changes": "request_changes",
    "block": "block",
    "schedule": "schedule",
    "reclaim": "reclaim",
    "archive": "archive",
}

#: ``(stem, block kind)`` rows the matrix must cover — the declared shape of the
#: matrix, asserted against the built cells below so a row can neither be dropped
#: from the dict nor appear in it without being declared here.
_MATRIX_ROWS: tuple[tuple[str, str | None], ...] = (
    ("completion", None),
    ("changes", None),
    ("block", "dependency"),
    ("block", "needs_input"),
    ("schedule", None),
    ("reclaim", None),
    ("archive", None),
)

#: The four (caller shape, force) combinations each row is driven in.
_MATRIX_SHAPES: tuple[tuple[str, bool], ...] = (
    ("pinless", False),
    ("pinless", True),
    ("worker", False),
    ("worker", True),
)


def _cell_key(stem: str, kind: str | None, shape: str, force: bool) -> str:
    """The matrix key for one cell: the event stem is the unit of documentation."""
    return f"{stem}|{kind}|{shape}|{'force' if force else 'plain'}"


def _cells() -> dict[str, str]:
    """The measured matrix, verbatim — all six event stems, all four shapes.

    Round-3 review Finding 1's own table, with the two corrections it demanded
    stated as data instead of prose; round-4 review Finding 3 then required the
    remaining four stems, because comment 911 on the wiki-landing card hands the
    wiki-keeper a six-row table and calls it "pinned by tests". These constants
    are the pin. 28 cells = 7 (stem, kind) rows x 4 (shape, force) combinations.
    """
    cells: dict[str, str] = {}

    # completion / changes — STRICT: no pin-less operator lane at all, so the
    # pin-less caller is REFUSED and the refusal is its event.
    for stem in ("completion", "changes"):
        for shape in ("pinless", "worker"):
            cells[_cell_key(stem, None, shape, False)] = f"{stem}_refused_not_owner"
            cells[_cell_key(stem, None, shape, True)] = f"{stem}_forced_not_owner"
    # block + kind="dependency" — the T5 STOP park. Pin-less LANDS (displaced),
    # `--force` is INERT for a pin-less caller; the worker-pinned caller is
    # refused without the flag and forced with it.
    cells[_cell_key("block", "dependency", "pinless", False)] = "block_displaced_not_owner"
    cells[_cell_key("block", "dependency", "pinless", True)] = "block_displaced_not_owner"
    cells[_cell_key("block", "dependency", "worker", False)] = "block_refused_not_owner"
    cells[_cell_key("block", "dependency", "worker", True)] = "block_forced_not_owner"
    # block + any OTHER kind — refused from every shape without `--force`.
    for shape in ("pinless", "worker"):
        cells[_cell_key("block", "needs_input", shape, False)] = "block_refused_not_owner"
        cells[_cell_key("block", "needs_input", shape, True)] = "block_forced_not_owner"
    # schedule — refused from every shape without `--force` (the runbook never
    # schedules; the documented STOP is a dependency block).
    for shape in ("pinless", "worker"):
        cells[_cell_key("schedule", None, shape, False)] = "schedule_refused_not_owner"
        cells[_cell_key("schedule", None, shape, True)] = "schedule_forced_not_owner"
    # reclaim / archive — the operator RECOVERY lane: LANDS pin-less, and the
    # honest name for that is `displaced`. `--force` changes NOTHING here.
    for stem in ("reclaim", "archive"):
        cells[_cell_key(stem, None, "pinless", False)] = f"{stem}_displaced_not_owner"
        cells[_cell_key(stem, None, "pinless", True)] = f"{stem}_displaced_not_owner"
        cells[_cell_key(stem, None, "worker", False)] = f"{stem}_refused_not_owner"
        cells[_cell_key(stem, None, "worker", True)] = f"{stem}_forced_not_owner"

    # The built matrix must have exactly the DECLARED shape: one cell per
    # (row, caller shape, force), no more and no fewer. Without this, a row could
    # be dropped from the dict (or silently double-written) while _MATRIX_ROWS
    # kept claiming coverage — which is round-4 review Finding 1's shape.
    declared = {
        _cell_key(stem, kind, shape, force)
        for stem, kind in _MATRIX_ROWS
        for shape, force in _MATRIX_SHAPES
    }
    # An explicit raise, not an ``assert``: ``python -O`` strips asserts, and a
    # guard the interpreter can remove is not a guard (round-3 Finding 1's shape).
    if set(cells) != declared:
        raise AssertionError(
            "the matrix does not match _MATRIX_ROWS x _MATRIX_SHAPES: "
            f"undeclared={sorted(set(cells) - declared)}, "
            f"missing={sorted(declared - set(cells))}"
        )
    return cells


MEASURED_LANES: dict[str, str] = _cells()


def _iter_cells():
    """Yield ``(api_verb, shape, force, kind, key, expected)`` for every cell.

    The matrix is keyed by string, so the key IS the test vector — decoding it
    here rather than keeping a parallel list means a cell cannot be added to the
    dict and forgotten by the driver.
    """
    for key, expected in MEASURED_LANES.items():
        stem, kind, shape, flag = key.split("|")
        yield (
            _STEM_TO_API_VERB[stem],
            shape,
            flag == "force",
            (None if kind == "None" else kind),
            key,
            expected,
        )



def _measured_for(stem: str, lane: str) -> str:
    """The documented-name expectation for one (event stem, lane) row.

    Every expectation is DERIVED from :data:`MEASURED_LANES`, so renaming an
    emitted event cannot be absorbed by editing this function — it fails the
    measured half first, and the doc sites with it. *stem* is the event stem
    (``completion``, ``changes``, ...), not the API verb name: the sites name
    events, so the stem is the right unit here.

    The two "pin-less" lanes are per-stem facts rather than a free choice, which
    is the point: a site may only declare the pin-less lane that verb actually
    has. ``completion``/``changes``/``schedule`` have no pin-less landing lane at
    all (they are refused), so asking for ``pinless_lands`` on them is a
    programming error in this file, and it raises rather than silently returning
    some other verb's name.
    """
    landing = {"reclaim": None, "archive": None, "block": "dependency"}
    refused = {
        "completion": None, "changes": None, "schedule": None,
        "block": "needs_input",
    }
    if lane == "pinless_lands":
        if stem not in landing:
            raise AssertionError(
                f"{stem!r} has no pin-less landing lane — it is refused from a "
                "pin-less shell, so its pin-less lane is 'pinless_refused'"
            )
        return MEASURED_LANES[_cell_key(stem, landing[stem], "pinless", False)]
    if lane == "pinless_refused":
        if stem not in refused:
            raise AssertionError(
                f"{stem!r} HAS a pin-less landing lane — its pin-less lane is "
                "'pinless_lands', not 'pinless_refused'"
            )
        return MEASURED_LANES[_cell_key(stem, refused[stem], "pinless", False)]
    if lane == "refused":
        return MEASURED_LANES[_cell_key(
            stem, "needs_input" if stem == "block" else None, "worker", False,
        )]
    if lane == "override":
        return MEASURED_LANES[_cell_key(
            stem, "needs_input" if stem == "block" else None, "worker", True,
        )]
    if lane == "pinless_generic":
        literal = _measured_for(stem, "pinless_lands")
        return "<stem>" + literal[len(stem):]
    raise AssertionError(f"unknown lane {lane!r}")


#: The pin-less lane each event stem actually has, in the gate's vocabulary. Used
#: by the cross-verb sanity test so it asks each stem the question that stem has
#: an answer to (declaring the wrong one raises in ``_measured_for``).
PINLESS_LANE_FOR_VERB: dict[str, str] = {
    "completion": "pinless_refused",
    "changes": "pinless_refused",
    "block": "pinless_lands",
    "schedule": "pinless_refused",
    "reclaim": "pinless_lands",
    "archive": "pinless_lands",
}


def test_t8_measured_lane_matrix_is_verbatim(kanban_home, monkeypatch):
    """The measured half: pin every cell of the matrix above.

    A rename of an emitted kind, or a change to which lane ``force`` selects, has
    to be stated in ``MEASURED_LANES`` — and the CLAIMED half below then fails
    until the site that documents that lane is updated with it. That coupling is
    the whole point of this file (round-3 Finding 1 said nothing pinned the
    claimed name against the measured one).
    """
    assert len(MEASURED_LANES) == 28, "six verb stems x four shapes is 28 cells"
    for verb, shape, force, kind, key, expected in _iter_cells():
        got = _drive(
            verb, shape=shape, force=force, monkeypatch=monkeypatch, kind=kind,
        )
        assert got == expected, (
            f"{key}: measured {got!r}, MEASURED_LANES says {expected!r}"
        )


def test_t8_pinless_force_is_inert_exactly_where_measurement_says_it_is(
    kanban_home, monkeypatch,
):
    """The direct, measurement-derived form of round-3 Finding 1.

    For every (stem, block kind) row, compare the pin-less cell WITHOUT the flag
    to the pin-less cell WITH it:

    * equal events -> the flag is INERT for that caller, and the event must NOT be
      a ``forced`` name (nothing was overridden: the lane already landed, or was
      already refused). This is round-3 Finding 1's own correction, and for
      ``reclaim``/``archive`` it is the load-bearing claim the wiki page rests on;
    * differing events -> ``--force`` genuinely selects the override, and the
      event must be the ``forced`` name for that stem.

    Stated as a measurement rather than a hard-coded list so the test cannot go
    stale if a verb's lane moves — the matrix and the sites both follow it.
    """
    rows = sorted({(stem, kind) for stem, kind, _shape, _flag in
                   (k.split("|") for k in MEASURED_LANES)})
    assert rows, "matrix is empty"
    for stem, kind in rows:
        block_kind = None if kind == "None" else kind
        plain = _drive(
            _STEM_TO_API_VERB[stem], shape="pinless", force=False,
            monkeypatch=monkeypatch, kind=block_kind,
        )
        forced = _drive(
            _STEM_TO_API_VERB[stem], shape="pinless", force=True,
            monkeypatch=monkeypatch, kind=block_kind,
        )
        if plain == forced:
            assert not plain.endswith("_forced_not_owner"), (
                f"pin-less `{stem}` (kind={kind}) emits {plain!r} both with and "
                "without --force, so nothing was overridden — reporting it as a "
                "`forced` event is round-3 Finding 1 exactly"
            )
        else:
            assert forced.endswith("_forced_not_owner"), (
                f"pin-less `{stem}` (kind={kind}): --force selects {forced!r} "
                f"over {plain!r}, so the override event must be the `forced` name"
            )


def test_t8_a_refused_caller_leaves_the_foreign_run_alone(kanban_home, monkeypatch):
    """The refusal cells above must actually refuse — measured, not just named.

    Without this, ``MEASURED_LANES`` could record ``*_refused_not_owner`` for a
    call that also displaced the run (a refusal that is not a refusal). Driven
    for every verb, not just the two recovery ones.
    """
    for verb, shape, force, kind, _key, expected in _iter_cells():
        if not expected.endswith("_refused_not_owner"):
            continue
        tid, live = _displacement_card(f"t8 refusal {verb} {kind} {shape}")
        if shape == "worker":
            monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
            monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "0")
        else:
            monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
            monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
        try:
            with kbc.connect() as conn:
                ok, _extra = _call_verb(conn, verb, tid, kind=kind, force=force)
        finally:
            monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
            monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
        assert ok is False, f"{verb}/{kind}/{shape}: the caller was NOT refused"
        _assert_run_untouched(tid, live)
        assert _audit_kinds(tid) == [expected], f"{verb}/{kind}/{shape}"


# ── CLAIMED: every documented name, pinned against the measurement ──────────
#
# One row per source site that names an audit event for a caller lane. The row
# carries the markers that delimit the site's own prose (so a moved or renamed
# site fails loudly) and the (verb, lane) pairs whose measured kind must be
# named there. The lane vocabulary is defined in the module docstring.


class Site:
    """One source site that documents an audit-event name.

    ``lanes`` is a flat ``(verb, lane, ...)`` list and MUST have even length —
    the assertion loop consumes it in pairs, and a trailing odd element would
    otherwise be silently dropped, which is exactly round-4 review Finding 1
    (``HELP_SITES`` declared three lanes per site and the paired loop checked
    two). ``lane`` is one of the five names in the module docstring.
    """

    __slots__ = ("label", "path", "start", "end", "lanes")

    def __init__(self, label: str, path: str, start: str, end: str, lanes: tuple):
        # An explicit raise, not an ``assert``: this module is the gate, and a
        # guard that ``python -O`` can strip would be a claim the file does not
        # actually enforce — the round-3-Finding-1 pattern one level up.
        if len(lanes) % 2 != 0:
            raise ValueError(
                f"{label}: lanes must be (verb, lane) pairs — got {len(lanes)} "
                "elements, so the trailing one would never be asserted"
            )
        self.label, self.path, self.start, self.end, self.lanes = (
            label, path, start, end, lanes,
        )


SITES: tuple[Site, ...] = (
    Site(
        "kanban_db.reclaim_task docstring",
        "hermes_cli/kanban_db.py",
        "def reclaim_task(",
        "# Run-end invariant (t_6c8342e0 Finding 2): a worker-sponsored caller",
        ("reclaim", "pinless_lands", "reclaim", "override"),
    ),
    Site(
        "kanban_db.reassign_task docstring",
        "hermes_cli/kanban_db.py",
        "def reassign_task(",
        "if reclaim_first:",
        ("reclaim", "pinless_lands", "reclaim", "override"),
    ),
    Site(
        "kanban_db._OPERATOR_AUDITED_VERBS policy block",
        "hermes_cli/kanban_db.py",
        "#: Recovery verbs a pin-less operator keeps UNCONDITIONALLY",
        "_OPERATOR_AUDITED_VERBS = frozenset",
        ("reclaim", "pinless_generic", "archive", "pinless_generic"),
    ),
    Site(
        "kanban_db._run_end_allowed docstring",
        "hermes_cli/kanban_db.py",
        "def _run_end_allowed(",
        "ok, cas_run = _live_run_ownership_guard(conn, task_id, expected_run_id)\n    if ok:\n        return True, cas_run",
        ("reclaim", "pinless_generic", "archive", "pinless_generic"),
    ),
    Site(
        "kanban_db.reclaim_task call site (owner/displaced branch)",
        "hermes_cli/kanban_db.py",
        "# Operator recovery (CLI ``reclaim``, dashboard button, reassign)",
        # Hand-merge (t_788c8b56): upstream added ``worker_started_at`` to this
        # SELECT inside the range, so the anchor follows the merged source.
        'row = conn.execute(\n        "SELECT status, claim_lock, worker_pid, worker_started_at FROM tasks WHERE id = ?"',
        ("reclaim", "pinless_lands"),
    ),
    Site(
        "kanban_db.archive_task docstring",
        "hermes_cli/kanban_db.py",
        "def archive_task(",
        "# Run-end invariant (t_6c8342e0 Finding 2). Before the txn so a refusal\n    # leaves the card byte-identical; re-checked inside, where the CAS enforces it.",
        ("archive", "pinless_lands", "archive", "refused", "archive", "override"),
    ),
    Site(
        "kanban_db.archive_task operator-lane comment",
        "hermes_cli/kanban_db.py",
        "# Operator recovery/cleanup taking a live unowned run: audited as",
        '        _operator_displaced_live_run(conn, task_id, expected_run_id, verb="archive")',
        ("archive", "pinless_lands"),
    ),
    Site(
        "kanban_db.complete_task force paragraph",
        "hermes_cli/kanban_db.py",
        "``force=True`` is the explicit human override of the force-close invariant",
        "inside the graceful-park contract's zero-hard-kills rule).",
        ("completion", "override"),
    ),
    Site(
        "kanban_db.completion_refusal_detail message",
        "hermes_cli/kanban_db.py",
        'f"task {task_id} is running under a live run (run {live_run_id}) owned by "',
        "def complete_task(",
        ("completion", "override"),
    ),
    Site(
        "kanban_db._refuse_completion_not_owner docstring",
        "hermes_cli/kanban_db.py",
        "def _refuse_completion_not_owner(",
        '    _audit_run_end_refusal(\n        conn, task_id, "complete", live_run_id, expected_run_id, attempted,\n    )',
        ("completion", "refused"),
    ),
    Site(
        "kanban_db._record_forced_completion docstring",
        "hermes_cli/kanban_db.py",
        "def _record_forced_completion(",
        '        conn, task_id, "complete", run_id, expected_run_id, attempted,',
        ("completion", "override"),
    ),
    Site(
        "kanban_db.block_task docstring",
        "hermes_cli/kanban_db.py",
        "def block_task(",
        "    # Run-end invariant (t_6c8342e0 Finding 2). Checked BEFORE the txn so a",
        ("block", "pinless_lands", "block", "override"),
    ),
    Site(
        "kanban_db.schedule_task docstring",
        "hermes_cli/kanban_db.py",
        "def schedule_task(",
        "    # Run-end invariant (t_6c8342e0 Finding 2). Before the txn so a refusal leaves\n    # the card byte-identical; the in-txn CAS below enforces it.",
        ("schedule", "refused", "schedule", "override"),
    ),
    Site(
        "kanban_db.request_changes docstring",
        "hermes_cli/kanban_db.py",
        "def request_changes(",
        "    # Run-end invariant (t_6c8342e0 Finding 2): the verdict may only be issued by",
        ("changes", "override"),
    ),
    Site(
        "kanban._cmd_reclaim",
        "hermes_cli/kanban.py",
        "def _cmd_reclaim(",
        "def _cmd_reassign(",
        ("reclaim", "pinless_lands", "reclaim", "override"),
    ),
    Site(
        "kanban._cmd_reassign",
        "hermes_cli/kanban.py",
        "def _cmd_reassign(",
        "def _rows_by_task(",
        ("reclaim", "pinless_lands"),
    ),
    Site(
        "kanban._cmd_archive",
        "hermes_cli/kanban.py",
        "def _cmd_archive(",
        "def _cmd_stats(",
        ("archive", "pinless_lands", "archive", "override"),
    ),
    Site(
        "dashboard.terminate_run_endpoint",
        "plugins/kanban/dashboard/plugin_api.py",
        "def terminate_run_endpoint(",
        "# --- Recovery actions",
        ("reclaim", "pinless_lands", "reclaim", "override"),
    ),
    Site(
        "dashboard.reassign_task_endpoint",
        "plugins/kanban/dashboard/plugin_api.py",
        "def reassign_task_endpoint(",
        "# Estimate: rough token/complexity read via the auxiliary model.",
        ("reclaim", "pinless_lands"),
    ),
    Site(
        "dashboard.done verb comment",
        "plugins/kanban/dashboard/plugin_api.py",
        "# A forced close is audited as ``completion_forced_not_owner`` (t_6c8342e0).",
        '    "done": lambda conn, tid, p: kanban_db.complete_task(',
        ("completion", "override"),
    ),
    Site(
        "dashboard.archive verb comment",
        "plugins/kanban/dashboard/plugin_api.py",
        "# ``archive_displaced_not_owner`` (t_6c8342e0 round-2 Finding 1/2).",
        "        ok = kanban_db.archive_task(conn, task_id, force=True)",
        ("archive", "pinless_lands"),
    ),
)

#: The CLI ``--force`` help strings. Two of them describe the FLAG for a verb
#: whose pin-less lane LANDS (``reclaim``/``archive``), so they must name both
#: lanes: the worker-pinned one the flag actually selects, and the pin-less
#: recovery lane that lands (and is recorded) with or without it. The rest
#: describe a flag whose pin-less caller is refused, so they name the override
#: lane only — and the negative form below is what keeps that honest.
HELP_SITES: tuple[Site, ...] = (
    Site(
        "parser reclaim --force help",
        "hermes_cli/kanban_parser.py",
        '_cmd("reclaim", [_TASK_ID, _RECLAIM_REASON,',
        '], help="Release an active worker claim on a running task")',
        ("reclaim", "pinless_lands", "reclaim", "override"),
    ),
    Site(
        "parser archive --force help",
        "hermes_cli/kanban_parser.py",
        '_cmd("archive", [',
        '], help="Archive one or more tasks")',
        ("archive", "pinless_lands", "archive", "override"),
    ),
    Site(
        "parser complete --force help",
        "hermes_cli/kanban_parser.py",
        # Hand-merge (t_788c8b56): retargeted to the MERGED help text — the replay
        # kept our wording here (it also names the measured event, which is what
        # this Site asserts), so the anchor follows the merged source.
        '_arg("--force", action="store_true",\n             help="Override the live-claim guard: complete a running, claimed task "',
        '], help="Mark one or more tasks done"),',
        ("completion", "override"),
    ),
    Site(
        "parser block --force help",
        "hermes_cli/kanban_parser.py",
        '_arg("--force", action="store_true",\n             help="Override the live-run ownership guard: end a live run that "',
        '], help="Mark one or more tasks blocked"),',
        ("block", "override"),
    ),
    Site(
        "parser schedule --force help",
        "hermes_cli/kanban_parser.py",
        '_arg("--force", action="store_true",\n             help="Override the live-run ownership guard: park a task whose live "',
        '], help="Park one or more tasks in Scheduled (waiting on time, not human input)"),',
        ("schedule", "override"),
    ),
    Site(
        "parser changes --force help",
        "hermes_cli/kanban_parser.py",
        '_arg("--force", action="store_true",\n             help="Override the live-run ownership guard: issue the verdict even "',
        '         help="Reviewer verdict: return the active review run to its implementer"),',
        ("changes", "override"),
    ),
)

#: Every site whose own prose describes a caller lane — i.e. everything the
#: positive AND negative halves reason over. ``SITES`` + ``HELP_SITES``; the
#: negative half used to iterate ``SITES`` alone, which is why the two
#: ``--force`` help strings escaped it (round-4 review Finding 1).
ALL_SITES: tuple[Site, ...] = SITES + HELP_SITES


def _site_text(site: Site) -> str:
    """The site's own source segment, delimited by its declared markers.

    Both markers must be present and unique in the file, so a renamed, moved or
    deleted site is a loud failure rather than a silently narrower window.
    """
    source = (REPO_ROOT / site.path).read_text(encoding="utf-8")
    assert source.count(site.start) == 1, (
        f"{site.label}: start marker is not unique in {site.path}: {site.start!r}"
    )
    assert source.count(site.end) == 1, (
        f"{site.label}: end marker is not unique in {site.path}: {site.end!r}"
    )
    start = source.index(site.start)
    end = source.index(site.end, start)
    assert end > start, f"{site.label}: markers are inverted"
    return source[start:end]


def _lanes_of(site: Site) -> list[tuple[str, str]]:
    """``site.lanes`` as (verb, lane) pairs, the trailing element included."""
    return [
        (site.lanes[i], site.lanes[i + 1])
        for i in range(0, len(site.lanes), 2)
    ]


def test_t8_documented_names_match_the_measured_event():
    """One assertion per documented name: the site must name what MEASUREMENT says.

    This is the pin round-3 Finding 1 asked for ("the existing T6 group pins the
    emitted name; nothing pins the claimed one"). It is deliberately
    source-inspecting: the defect was the docs, so the docs are what is asserted.
    """
    checked = 0
    for site in ALL_SITES:
        text = _site_text(site)
        for verb, lane in _lanes_of(site):
            expected = _measured_for(verb, lane)
            assert expected in text, (
                f"{site.label} does not name {expected!r} — MEASURED for the "
                f"{lane} lane of `{verb}`. Documented-name drift is exactly "
                f"round-3 Finding 1. Site text:\n{text}"
            )
            checked += 1
    # The declared pair count must equal the asserted count: a lane that is
    # declared but never reached is round-4 review Finding 1.
    declared = sum(len(site.lanes) // 2 for site in ALL_SITES)
    assert checked == declared == sum(
        len(_lanes_of(s)) for s in ALL_SITES
    ), f"asserted {checked} of {declared} declared (verb, lane) pairs"


def test_t8_no_site_claims_the_forced_name_for_the_pinless_lane():
    """The negative form, so the class cannot return by *adding* a wrong name.

    Applied to EVERY site that documents a caller lane — ``SITES`` and
    ``HELP_SITES`` alike (round-4 review Finding 1: the two ``--force`` help
    strings were excluded from the old iteration over ``SITES`` alone).

    Three rules, and each is derived from what was MEASURED rather than from
    taste:

    1. A site that declares a pin-less lane must NAME that lane's measured event,
       so a reader is never left with the forced name as the only candidate.
    2. A site that names ``<verb>_forced_not_owner`` for a RECOVERY verb must also
       name ``<verb>_displaced_not_owner`` — because for those verbs ``--force``
       is INERT for a pin-less shell (both cells measure ``displaced``,
       asserted below), so presenting the forced name as what a pin-less
       ``hermes kanban reclaim``/``archive`` records is round-3 Finding 1 exactly.
       For every other verb ``--force`` genuinely selects the forced name from
       either caller shape, so naming it alone is accurate prose — that is why
       the ``complete``/``changes``/``schedule`` help strings are not in
       ``HELP_SITES``' reclaim/archive shape.
    3. A site cannot name a ``<verb>_displaced_not_owner`` literal without
       declaring ``(verb, "pinless_lands")`` — naming the operator lane's event is
       declaring it.

    Deliberately NOT an ordering assertion: ``archive_task``'s docstring
    describes the worker-pinned lane before the operator lane, which is correct
    prose — the defect is a site that offers *only* the forced name for a
    pin-less caller.
    """
    # Recovery verbs, as the production policy block names them, and the reason
    # the rule below applies to them alone.
    recovery = {"reclaim", "archive"}
    for verb in recovery:
        plain = MEASURED_LANES[_cell_key(verb, None, "pinless", False)]
        forced = MEASURED_LANES[_cell_key(verb, None, "pinless", True)]
        assert plain == forced == f"{verb}_displaced_not_owner", (
            f"{verb}: '--force is inert for a pin-less recovery caller' is the "
            f"premise of rule 2, and measurement now says {plain!r} / {forced!r}"
        )

    for site in ALL_SITES:
        text = _site_text(site)
        lanes = _lanes_of(site)
        declared = set(lanes)
        # rule 1
        for verb, lane in lanes:
            if lane in ("pinless_lands", "pinless_refused"):
                expected = _measured_for(verb, lane)
                assert expected in text, (
                    f"{site.label} declares the {lane} lane for `{verb}` but "
                    f"never names {expected!r}"
                )
        # rule 2 — scan the TEXT for recovery verbs' forced names, not just the
        # declared lanes: a site that names the forced name for a recovery verb
        # and declares no lane at all is in worse shape than one that declares
        # the wrong lane, and iterating `lanes` alone would skip it.
        for match in _EVENT_LITERAL.finditer(text):
            forced_name = match.group(0)
            if not forced_name.endswith("_forced_not_owner"):
                continue
            forced_verb = forced_name[: forced_name.index("_forced_not_owner")]
            if forced_verb not in recovery:
                continue
            assert (forced_verb, "pinless_lands") in declared, (
                f"{site.label} offers {forced_name!r} and does not declare "
                f"({forced_verb!r}, 'pinless_lands') — for a recovery verb the "
                "flag is inert pin-less, so naming the forced name without the "
                "displaced one is round-3 Finding 1 exactly"
            )
        # rule 3
        for match in _EVENT_LITERAL.finditer(text):
            name = match.group(0)
            if not name.endswith("_displaced_not_owner"):
                continue
            displacer = name[: name.index("_displaced_not_owner")]
            assert (displacer, "pinless_lands") in declared, (
                f"{site.label} names {name!r} but does not declare "
                f"({displacer!r}, 'pinless_lands')"
            )


def test_t8_every_literal_in_the_documented_sources_is_inside_a_site():
    """COVERAGE: no ``*_not_owner`` literal in the sources escapes the gate.

    Round-4 review Finding 3: 17 source lines named a ``*_not_owner`` event while
    sitting inside NO gate segment (completion x6, block x3, schedule x3,
    changes x2, archive x3 — measured with the reviewer's own
    ``rv4/gate_coverage3.py``). Comment 911 on the wiki-landing card tells the
    wiki-keeper the six-row table is "pinned by tests, not prose"; that claim is
    only true if the segments cover every literal, which is what this asserts.

    The declared markers are re-derived from this module's own source (AST) so a
    new ``Site`` automatically joins the coverage set, and a site whose markers no
    longer resolve fails here rather than silently narrowing.
    """
    covered: list[tuple[str, int, int, str]] = []
    for node in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Site"):
            continue
        args = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if len(args) < 4 or not isinstance(args[1], str):
            continue
        label, path, start, end = args[0], args[1], args[2], args[3]
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert source.count(start) == 1 and source.count(end) == 1, (
            f"{label}: markers do not resolve uniquely in {path} — the coverage "
            "set is narrower than it claims"
        )
        begin = source.index(start)
        finish = source.index(end, begin)
        covered.append((path, begin, finish, label))

    uncovered: list[str] = []
    total = 0
    for path in DOCUMENTED_SOURCES:
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        offset = 0
        for linenum, line in enumerate(source.splitlines(), 1):
            for match in _EVENT_LITERAL.finditer(line):
                total += 1
                where = offset + match.start()
                if not any(
                    cpath == path and begin <= where < finish
                    for cpath, begin, finish, _label in covered
                ):
                    uncovered.append(f"{path}:{linenum} [{match.group(0)}]")
            offset += len(line) + 1  # +1 for the newline splitlines() removed
    assert total > 0, "the literal scanner found nothing — the regex or paths rotted"
    assert not uncovered, (
        f"{len(uncovered)} `*_not_owner` literals in the documented sources sit "
        "inside NO declared Site segment, so a claim about them is unpinned "
        "(round-4 review Finding 3):\n  " + "\n  ".join(uncovered)
    )


def test_t8_self_check_no_dangling_symbols():
    """No ``_<token>`` citation in this file's prose may fail to resolve.

    Round-4 review Finding 2: the gate's own docstring cited a driver symbol that
    exists nowhere in the repo (the real driver is ``_drive``) — the same "the
    citation cannot support the claim" class the file exists to stop, inside the
    file itself. This pins it: every double-backtick ``_<token>`` in this module's
    text must be a name defined here, or an attribute of ``kanban_db`` /
    ``kanban_db_connect``.

    There is deliberately NO allowlist. A metasyntactic placeholder is written in
    a form the scanner's character class cannot match (``_<token>``), so prose
    can stay general without opening a hole a dangling citation could be parked
    in — an allowlist would itself be the round-3-Finding-1 pattern, a claim the
    check does not actually make.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    tokens = set(re.findall(r"``(_[A-Za-z][A-Za-z0-9_]*)``", source))
    assert tokens, "no private tokens found — the scanner rotted"
    defined = globals()
    unresolvable = [
        token for token in sorted(tokens)
        if token not in defined
        and not any(hasattr(mod, token) for mod in (kb, kbc))
    ]
    assert not unresolvable, (
        "this file's own prose names symbols that do not exist (round-4 review "
        f"Finding 2): {unresolvable}"
    )


def test_t8_the_pinless_names_are_distinct_from_the_forced_names():
    """Cross-verb sanity over the whole family.

    Every verb must have a pin-less event, a refusal and an override, and the
    pin-less and forced names must differ — otherwise the assertions above could
    be satisfied by an accidental prefix match. The pin-less expectation is taken
    from :data:`PINLESS_LANE_FOR_VERB`, so each verb is asked the question it has
    an answer to.
    """
    assert set(PINLESS_LANE_FOR_VERB) == {
        "completion", "changes", "block", "schedule", "reclaim", "archive",
    }
    for verb, lane in PINLESS_LANE_FOR_VERB.items():
        pinless = _measured_for(verb, lane)
        refused = _measured_for(verb, "refused")
        forced = _measured_for(verb, "override")
        stem = "changes" if verb == "changes" else (
            "completion" if verb == "completion" else verb
        )
        assert pinless != forced, verb
        assert pinless.endswith("_not_owner"), verb
        assert refused == f"{stem}_refused_not_owner", (verb, refused)
        assert forced == f"{stem}_forced_not_owner", (verb, forced)
    # The three landing lanes are `displaced` (not `forced`), which is round-3
    # Finding 1's own correction stated as data.
    assert _measured_for("reclaim", "pinless_lands") == "reclaim_displaced_not_owner"
    assert _measured_for("archive", "pinless_lands") == "archive_displaced_not_owner"
    assert _measured_for("block", "pinless_lands") == "block_displaced_not_owner"
    assert _measured_for("reclaim", "pinless_generic") == "<stem>_displaced_not_owner"
    # The three refused pin-less lanes are `refused`, and asking for a landing
    # lane on them is an error in this file rather than a quiet wrong answer.
    for verb in ("completion", "changes", "schedule"):
        assert _measured_for(verb, "pinless_refused") == f"{verb}_refused_not_owner"
        with pytest.raises(AssertionError):
            _measured_for(verb, "pinless_lands")
