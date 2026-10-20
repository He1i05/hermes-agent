"""Task claims, not caller-supplied launch policy, authorize Kanban execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


# Nine tests, six of which boot a real gateway daemon, dispatcher and worker. Measured on the
# 4-core Pi: 251.1 s for the five probes sequentially at load 1.2-5.0; 797.5 s for the whole file
# with three gateway files in flight at load 9.2-15.2; and 1217.9 s for one attempt (2098.6 s with
# its runner retry) in the review's loaded run. Under the flat 300 s cap the file was SIGKILL'd
# before it collected and the sweep reported "no tests ran" — the miscount this declaration
# removes. A ceiling, not a cost: a green probe returns as soon as its predicate holds, and the
# probe's own chain (boot + 5 × wait + tail, fixtures/probe_budgets.owner_parent_cap = 5280 s at
# the current defaults: four lifecycle waits plus one arrival wait for the bounded start retry) is
# what bounds a real hang. This floor must stay above that per-param cap or the runner kills the
# file while the probe still has ceilings ahead of it — 5400 sits above it and is 2.6x the worst
# attempt+retry measured (2098.6 s). The sixth daemon test (the injected dead start, 48 s measured)
# runs the same probe chain, so it needs no ceiling of its own.
FILE_TIMEOUT_MIN_SECONDS = 5400


def test_claim_freezes_task_policy_and_rejects_forgery(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'pyproject.toml').write_text('[project]\nname = "task"\n')  # a code worktree
    from contextlib import closing
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned acceptance', body='Goal criteria', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['owned-skill'], goal_mode=True,
            goal_max_turns=3, model_override='loop-model', provider_override='custom')
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    from gateway.session_kanban import build_kanban_policy
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal', 'file']},
              'agent': {'coding_context': 'focus'}}  # the worktree must not collapse a worker into the coding posture
    policy, _secrets = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    assert policy.source == 'kanban' and policy.cwd == str(workspace)
    assert policy.model == task.model_override and policy.provider == task.provider_override
    assert {'kanban', 'terminal', 'file'} <= set(policy.toolsets)
    assert context['task_id'] == task.id and context['run_id'] == task.current_run_id
    assert context['board'] == 'owned' and context['claim_lock'] == task.claim_lock
    assert context['skills'] == ['owned-skill'] and context['goal_mode'] and context['goal_max_turns'] == 3
    assert context['accept_hooks'] is True and 'Goal criteria' in context['goal_text']
    for changed in ({'run_id': task.current_run_id + 1}, {'board': 'foreign'}, {'claim_lock': 'forged'}, {'cwd': '/tmp'}):
        with pytest.raises(RuntimeStoreError):
            build_kanban_policy(connection, params | changed, config)
    connection.native_owner = False
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        build_kanban_policy(connection, params, config)
    from gateway.session_policy import build_policy
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        build_policy({'source': 'kanban', 'cwd': str(workspace)}, config)


@pytest.mark.parametrize('mode', ['complete', 'timeout', 'rate_limit', 'billing', 'crash'])
def test_real_dispatcher_lifecycle(tmp_path, mode):
    receipt = run_owner_probe(tmp_path, mode)
    if mode != 'complete':
        assert receipt['task_status'] == 'ready'
        return
    # A green run may not hide a start retry: the first attempt is the one that must arrive.
    assert receipt['start_retries'] == 0, receipt
    assert receipt['task_status'] == 'done' and receipt['source'] == 'kanban'
    assert receipt['tools'] and receipt['task_context'] and receipt['skill_context'] and receipt['hook_effect']
    assert receipt['admissions'] == 1 and receipt['retry_same_session']
    assert receipt['goal_continuation'] and receipt['stable_prefix']
    print(json.dumps(receipt))


def test_one_dead_worker_start_costs_one_redispatch(tmp_path):
    """A worker that dies before its first model call must cost a re-dispatch, not a red file.

    The probe kills the first worker client the dispatcher spawns (a real kill through
    KANBAN_PROBE_FAULT_START_EXITS=1), so the spawn itself succeeds and the client is dead
    before it posts to the loopback model. That is the shape that used to surface either as
    'model never reached' at the probe's arrival ceiling or — in timeout mode — as an all-empty
    DispatchResult, because the interposer raised *inside* dispatch.dispatch_once and the
    dispatcher swallowed it as a spawn_failed attempt (t_6c3e6f7f, reproduced from primary
    evidence).

    What has to hold, and is asserted here: the product books the dead attempt as a crash and
    re-queues it, the probe re-dispatches exactly once, and the surviving attempt still satisfies
    every assertion of mode=complete. The retry is not a way to skip assertions — it is one
    ordinary next dispatcher tick, and the receipt names it.
    """
    receipt = run_owner_probe(tmp_path, 'complete', KANBAN_PROBE_FAULT_START_EXITS='1')
    assert receipt['start_retries'] == 1, receipt
    failed = receipt['start_failures'][0]
    assert failed['cause'] == 'early-client-exit', failed
    assert failed['model_requests'] == 0, failed['model_requests']
    assert failed['client_rc'] and failed['client_rc'][0] is not None, failed['client_rc']
    assert failed['task']['worker_alive'] is False, failed['task']
    # The product booked it: the next tick's reclaim reported the crash and re-spawned the card.
    recovered = receipt['start_recovery'][0]
    assert receipt['task_id'] in recovered['crashed'], recovered
    assert recovered['spawned'] == [receipt['task_id']], recovered
    # The surviving attempt carries the whole acceptance, unchanged.
    assert receipt['task_status'] == 'done' and receipt['source'] == 'kanban'
    assert receipt['tools'] and receipt['task_context'] and receipt['skill_context'] and receipt['hook_effect']
    assert receipt['admissions'] == 1 and receipt['retry_same_session']
    assert receipt['goal_continuation'] and receipt['stable_prefix']
    assert receipt['client_exit'] == ['clean_exit', 0], receipt
    print(json.dumps(receipt))


def run_owner_probe(tmp_path, mode, **overrides):
    """Run kanban_owner_probe.py under its own ceiling chain, return its receipt.

    The probe boots a real daemon out of a borrowing data root and then drives one full
    dispatch lifecycle. Ceilings, not costs — a green run returns as soon as its predicate
    holds. The parent cap is derived from the probe's OWN ceiling chain so it outlasts it:
    boot (local_recovery_probe.daemon) + the four budget() waits in kanban_owner_probe.py
    (release, the arrival, the timeout-mode managed-worker exit, the post-release wait_for) +
    the lifecycle tail. Both numbers come from fixtures/probe_budgets.py, which the probe
    imports too, so the cap cannot drift under the chain. Deriving it from boot alone was
    wrong: at boot + 120 s (540 s) it fired on mode=billing while the probe still had 1200 s of
    ceilings ahead of it (3600 s sweep at c6b00422ea, t_97673f26 round 3). The budgets are
    forwarded explicitly because the probe env is a whitelist (an inherited value could not
    reach it); ``overrides`` are forwarded the same way for the probe's own test-only knobs.
    """
    repo = Path(__file__).resolve().parents[2]
    from tests.gateway.fixtures.local_recovery_probe import child_env
    from tests.gateway.fixtures.probe_budgets import BOOT_DEADLINE_DEFAULT, PROBE_WAIT_DEFAULT, owner_parent_cap
    env = child_env()
    boot = int(os.environ.get('KANBAN_RECOVERY_BOOT_DEADLINE', str(BOOT_DEADLINE_DEFAULT)))
    wait = int(os.environ.get('KANBAN_PROBE_WAIT_BUDGET', str(PROBE_WAIT_DEFAULT)))
    env.update(HOME=str(tmp_path / 'home'), USERPROFILE=str(tmp_path / 'home'), HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=str(repo), KANBAN_PROBE_MODE=mode,
               KANBAN_RECOVERY_BOOT_DEADLINE=str(boot),
               KANBAN_RECOVERY_HANDSHAKE_TIMEOUT=os.environ.get('KANBAN_RECOVERY_HANDSHAKE_TIMEOUT', '60'),
               KANBAN_PROBE_WAIT_BUDGET=str(wait))
    env.update({key: str(value) for key, value in overrides.items()})
    timeout = int(os.environ.get('KANBAN_PROBE_TIMEOUT', str(owner_parent_cap(boot, wait))))
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_owner_probe.py')],
        env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt))
    return receipt


def test_late_launcher_cannot_stamp_replacement_claim(tmp_path, monkeypatch):
    from contextlib import closing
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='Owned race', assignee='default')
        old = kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason='replace claim', expected_run_id=old.current_run_id)
        kb.unblock_task(conn, tid)
        new = kb.claim_task(conn, tid)
        dispatch._set_worker_pid(conn, tid, 424242, run_id=old.current_run_id, claim_lock=old.claim_lock)
        assert kb.get_task(conn, tid).worker_pid is None
        assert conn.execute('SELECT worker_pid FROM task_runs WHERE id=?', (new.current_run_id,)).fetchone()[0] is None


def test_claim_reclaimed_before_admission_marker_is_refused(tmp_path, monkeypatch):
    """A claim that stops being current between policy/session creation and the marker
    transaction is refused: no owner_admitted row, no resume, no submit."""
    import asyncio
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned race', assignee='default', workspace_kind='dir', workspace_path=str(workspace))
        kb.recompute_ready(conn)
        old = kb.claim_task(conn, task_id)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    calls = []
    def create_local_session(authority, actor, params, *, trusted_policy=None, trusted_secrets=None):
        # Another process reclaims the task while the owner session is being created.
        with closing(connect(board='owned')) as conn:
            kb.block_task(conn, task_id, reason='replace claim', expected_run_id=old.current_run_id)
            kb.unblock_task(conn, task_id)
            kb.claim_task(conn, task_id)
        return SimpleNamespace(session_id='owner-session', profile_id=str(tmp_path))
    monkeypatch.setattr('gateway.session_local.create_local_session', create_local_session)
    async def submit(actor, submission):
        calls.append(('submit', submission))
    async def resume(ref, params):
        calls.append(('resume', ref))
    db = SimpleNamespace(get_session=lambda sid: None, db_path=str(tmp_path / 'state.db'))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, submit=submit)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    params = dict(board='owned', task_id=task_id, run_id=old.current_run_id, claim_lock=old.claim_lock)
    from gateway.session_kanban import run_task
    with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    with closing(connect(board='owned')) as conn:
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='owner_admitted'", (task_id,)).fetchone()[0] == 0
