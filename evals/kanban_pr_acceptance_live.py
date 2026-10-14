"""Local HTTP/lifecycle probe; no external GitHub writes or inference.
Run: python evals/kanban_pr_acceptance_live.py ROOT [FIXTURE_ROOT]

Safety (2026-09-13 leak hardening)
----------------------------------
This probe writes a real SQLite kanban DB. The temp ``HERMES_HOME`` its own
fixture monkeypatches in is NOT isolation: kanban path resolution gives
``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_HOME`` precedence over ``HERMES_HOME``,
and a dispatched kanban worker session exports
``HERMES_KANBAN_DB=<real root>/kanban.db`` by default. Under that ambient pin
every ``create_task`` / ``complete_task`` below landed on the operator's live
board (reproduced and recorded: rc=0, 6 rows written, no complaint). Nothing
else covers a directly-launched eval: the in-code ``kanban_guard`` fails OPEN
when ``PYTEST_CURRENT_TEST`` / ``PYTEST_VERSION`` / ``HERMES_TEST_ISOLATION``
are all absent, which is the normal state of a worker shell.

So the CLI entrypoint refuses, non-zero, unless the **effective resolved kanban
DB path** -- production's own ``hermes_cli.kanban_db.kanban_db_path()``, resolved
by the shared predicate in ``evals/kanban_child_guard.py`` -- is provably
outside the real ``~/.hermes``.

Entry shape: one guard call at the top of ``main()``, before the first
``connect()``. The check lives in ``guard_refusal()`` rather than at module
scope so importing this module stays inert (pure definitions, no refusal armed,
no board touched) -- the module is loadable as a library. It uses the shared
``refusal_reason`` rather than ``require_isolated_child`` because the shared
``--child`` refusal text ("this child ... must only run under the parent
wrapper's temp HERMES_HOME") describes a re-exec wrapper this file does not
have: same predicate, honest message.

The predicate is the resolved path, so ``HERMES_KANBAN_DB``,
``HERMES_KANBAN_HOME``, a hand-edited ``<root>/kanban/current`` and any future
path pin are all covered by construction. Refusal is fail-closed.
"""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile


def guard_refusal(root: Path):
    """Why this probe must be refused, or ``None`` when it is provably isolated.

    Delegates to the one shared definition of "the effective resolved DB path"
    (``evals/kanban_child_guard.py``), which imports production's own
    ``kanban_db_path()`` from *root* and refuses on that resolved path -- never
    on env-var names, which is what makes the ambient ``HERMES_KANBAN_DB``
    worker pin, ``HERMES_KANBAN_HOME``, a hand-edited ``<root>/kanban/current``
    and any future pin covered by construction.

    Appends the evals dir to ``sys.path`` instead of inserting at position 0:
    the parent of this file contains directory names that shadow real packages
    (``gateway``, ``providers``, ``memory``, ...), so front-insertion would let
    an unrelated ``import <name>`` resolve to a sibling eval.
    """
    evals_dir = str(Path(__file__).resolve().parent)
    if evals_dir not in sys.path:
        sys.path.append(evals_dir)
    from kanban_child_guard import refusal_reason

    return refusal_reason(root)


def main() -> int:
    root = Path(sys.argv[1]).resolve()

    # ── Refuse before the first board access ───────────────────────────────
    reason = guard_refusal(root)
    if reason is not None:
        sys.exit(
            f"refusing: {reason}; this probe writes a real kanban DB, so it must "
            f"run with HERMES_HOME and every kanban path pin (HERMES_KANBAN_DB / "
            f"HERMES_KANBAN_HOME / HERMES_KANBAN_BOARD) pointing at a scratch "
            f"root outside the real ~/.hermes"
        )

    sys.path.insert(0, str(root))
    os.environ.pop('HERMES_DELEGATED_CHILD_CONTEXT', None)
    import pytest
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    def create(conn, title):
        if 'completion_contract' in inspect.signature(kb.create_task).parameters:
            return kb.create_task(conn, title=title, completion_contract='acme/repo')
        tid = kb.create_task(conn, title=title)
        if 'completion_contract' not in {r[1] for r in conn.execute('PRAGMA table_info(tasks)')}:
            conn.execute('ALTER TABLE tasks ADD COLUMN completion_contract TEXT')
        conn.execute('UPDATE tasks SET completion_contract=? WHERE id=?', ('acme/repo', tid))
        conn.commit()
        return tid

    fixture_root = Path(sys.argv[2]) if len(sys.argv) > 2 else root
    spec = importlib.util.spec_from_file_location('http_fixture', fixture_root / 'tests/hermes_cli/test_kanban_pr_acceptance.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    patch = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory(prefix='hermes-pr-live-') as home:
        setup = fixture.github.__wrapped__(Path(home), patch)
        state = next(setup)
        reports = []
        try:
            with connect() as conn:
                for outcome in ('failure', 'cancelled', 'timed_out', 'success'):
                    state.update(conclusion=outcome)
                    tid = create(conn, 'PR acceptance live')
                    accepted = kb.complete_task(conn, tid, summary='PR acceptance probe',
                                                   metadata={'published_pr': 'https://github.com/acme/repo/pull/7'})
                    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC", (tid,)).fetchone()
                    receipt = json.loads(row[0]) if row else None
                    reports.append({'case': outcome, 'accepted': accepted, 'status': kb.get_task(conn, tid).status, 'receipt': receipt})
                for outcome in (('success', 'failure') if 'completion_contract' in inspect.signature(kb.create_task).parameters else ()):
                    tid = kb.create_task(conn, title='Concurrent ownership', completion_contract='acme/repo')
                    owner = kb.claim_task(conn, tid)
                    run_id = owner.current_run_id
                    def reclaim():
                        with connect() as other:
                            kb.block_task(other, tid, reason='Reassigned during acceptance')
                            kb.unblock_task(other, tid)
                            state['replacement'] = kb.claim_task(other, tid).current_run_id
                    state.update(conclusion=outcome, race=reclaim)
                    accepted = kb.complete_task(conn, tid, expected_run_id=run_id, summary='PR acceptance probe',
                                                   metadata={'published_pr': 'https://github.com/acme/repo/pull/7'})
                    reports.append({'case': 'CAS-'+outcome, 'accepted': accepted, 'run_id': kb.get_task(conn, tid).current_run_id,
                        'receipts': conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0]})
                    del state['race']
            print(json.dumps({'reports': reports, 'requests': state['requests']}, indent=2))
        finally:
            try:
                next(setup)
            except StopIteration:
                pass
            patch.undo()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
