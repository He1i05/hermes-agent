"""Actual dispatcher argv, ordinary daemon, loopback tool round, disposable board."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from local_recovery_probe import daemon, websocket, rpc
from probe_budgets import PROBE_WAIT_DEFAULT


# Test-only fault injection: kills the first N worker clients the dispatcher spawns, right
# after their Popen returns, so the bounded start retry below has a real transient failure to
# recover from (tests/gateway/test_session_kanban.py drives it with 1). Zero — the default —
# injects nothing.
FAULT_START_EXITS_ENV = 'KANBAN_PROBE_FAULT_START_EXITS'

# One re-dispatch, never a retry loop. A worker that dies before its first model call costs
# exactly one crash the *product* books and re-queues (the booking ``mode=crash`` asserts), so
# the probe tolerates exactly that one and lets the surviving attempt carry every assertion.
# A second death is not a start flake — it fails with its cause, as before.
MAX_START_ATTEMPTS = 2


def fault_start_exits():
    return max(0, int(os.environ.get(FAULT_START_EXITS_ENV, '0') or 0))


def budget(default=PROBE_WAIT_DEFAULT):
    """This probe's arrival/discovery ceiling. Ceiling, not cost: a green run returns as soon
    as its predicate holds. The value and its measurement live in probe_budgets.py, which the
    calling test also imports, so the test's outer cap cannot drift under this chain."""
    return max(1, int(os.environ.get('KANBAN_PROBE_WAIT_BUDGET', default)))


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        if messages and not self.server.blocked.is_set():
            self.server.blocked.set()
            assert self.server.release.wait(budget())
        if messages and self.server.mode in {'rate_limit', 'billing'}:
            payload = json.dumps({'error': {'message': 'Rate limit exceeded' if self.server.mode == 'rate_limit' else 'Insufficient quota',
                'type': 'rate_limit_error' if self.server.mode == 'rate_limit' else 'insufficient_quota',
                'code': 'rate_limit_exceeded' if self.server.mode == 'rate_limit' else 'insufficient_quota'}}).encode()
            self.send_response(429)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        has_result = any(m['role'] == 'tool' for m in messages)
        message = {'role': 'assistant', 'content': 'OWNED_COMPLETE'}
        finish = 'stop'
        if messages and not body.get('tools'):
            message['content'] = json.dumps({'verdict': 'done', 'reason': 'Owned acceptance satisfied'})
        elif messages and (not has_result or messages[-1]['role'] == 'user'):
            continuation = has_result
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': 'owned-complete',
                'type': 'function', 'function': {'name': 'kanban_complete' if continuation else 'kanban_comment', 'arguments': json.dumps({'summary': 'Owned probe complete'} if continuation else {'body': 'Owned first turn'})}}]}
            finish = 'tool_calls'
        payload = json.dumps({'id': 'owned', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'owned', 'choices': [{'index': 0, 'delta': message, 'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    root = Path(__file__).resolve().parents[3]
    home = Path(os.environ['HERMES_HOME'])
    home.mkdir(parents=True, exist_ok=True)
    Path(os.environ['HOME']).mkdir(parents=True, exist_ok=True)
    os.environ['HERMES_KANBAN_HOME'] = str(home)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.mode = os.environ.get('KANBAN_PROBE_MODE', 'complete')
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    workspace = home / 'workspace'
    workspace.mkdir(exist_ok=True)
    skill = home / 'skills' / 'owned-skill'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: owned-skill\ndescription: Owned fixture\n---\nKANBAN_SKILL_SENTINEL\n', encoding='utf-8')
    hook = home / 'hook.py'
    hook.write_text('from pathlib import Path\nPath(' + repr(str(home / 'hook-effect')) + ').write_text("accepted")\n', encoding='utf-8')
    cfg = {'gateway': {'multiplex_profiles': False}, 'model': {'provider': 'custom', 'default': 'loop-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': ['terminal', 'file']},
        'terminal': {'backend': 'local'}, 'agent': {'api_max_retries': 1}, 'kanban': {'dispatch_in_gateway': False},
        'hooks': {'post_tool_call': [{'command': f'{sys.executable} {hook}'}]}}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli import kanban_db_dispatch as dispatch
    # Pin the installed launcher to this checkout for base/fixed argv comparisons.
    dispatch._resolve_hermes_argv = lambda: [sys.executable, '-m', 'hermes_cli.main']
    os.environ['HERMES_KANBAN_CRASH_GRACE_SECONDS'] = '0'
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='KANBAN_TASK_SENTINEL', body='Acceptance: finish owned card', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['owned-skill'],
            goal_mode=peer.mode == 'complete', goal_max_turns=2,
            max_runtime_seconds=3600 if peer.mode == 'timeout' else None)

    def diagnostics(cause, started, tick=None):
        """Attribute an expired ceiling, a short-circuited start or an empty dispatch tick.

        A sweep log keeps the last 30 lines of a failed file and pytest deletes this probe's
        temp root, so a failed predicate has to explain itself here: was the spawned client
        alive or already dead (rc), did the board book the run (status/claim/expiry/run/pid),
        what do the task's own event and run rows say, what did the worker log, and how loaded
        was the host. Without it 'model never reached' was unattributable from the sweep that
        hit it (t_238f2d3f round 1).

        ``tick`` carries the DispatchResult of a tick that returned no spawn. Its skip buckets
        cannot explain that shape: a spawn that raises inside ``dispatch_once`` is swallowed by
        ``_dispatch_lane_task`` as a ``spawn_failed`` run with the claim released, so the caller
        sees ``spawned=[]`` with every skip list empty and ``skipped_locked=False`` — identical
        to "the ready lane was empty" (reproduced from primary evidence in t_6c3e6f7f: the row
        was ``ready`` with ``claim_lock`` NULL, ``failures=1``, event ``spawn_failed``, and the
        probe's own exception text in ``last_failure_error``).
        """
        def tail(path, lines=8):
            try:
                return '\n'.join(path.read_text(errors='replace').splitlines()[-lines:])
            except OSError:
                return '<no log>'
        try:
            row = kb.get_task(conn, tid)
            if row is None:
                raise LookupError('task row missing')
            task_state = {'status': row.status, 'claim_lock': row.claim_lock,
                          'claim_expires': row.claim_expires, 'run_id': row.current_run_id,
                          'worker_pid': row.worker_pid, 'failures': row.consecutive_failures,
                          'error': (row.last_failure_error or '')[:300],
                          'worker_alive': kb._pid_alive(row.worker_pid) if row.worker_pid else None}
        except Exception as exc:  # a diagnostic must never mask the failure it explains
            task_state = {'error': repr(exc)}
        try:
            # Newest first: the interesting rows are the last events before the predicate gave up.
            events = [{'kind': event[0], 'payload': (event[1] or '')[:300]} for event in conn.execute(
                'SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 8', (tid,))]
            runs = [{'id': run[0], 'outcome': run[1]} for run in conn.execute(
                'SELECT id, outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 3', (tid,))]
        except Exception as exc:
            events = runs = [{'error': repr(exc)}]
        logs = {path.name: tail(path) for path in sorted(kb.worker_logs_dir(board='owned').glob('*.log'))}
        for path in (home / 'restart.log', home / 'first.log'):
            if path.exists():
                logs[path.name] = tail(path)
        state = {'cause': cause, 'mode': peer.mode, 'elapsed_s': round(time.monotonic() - started, 1),
                 'budget_s': budget(), 'client_rc': [p.poll() for p in clients],
                 'model_requests': len(peer.requests), 'task': task_state,
                 'events': events, 'runs': runs,
                 'load1': [round(os.getloadavg()[0], 2)] if hasattr(os, 'getloadavg') else None,
                 'logs': logs}
        if tick is not None:
            state['tick'] = {'spawned': tick.spawned, 'reclaimed': tick.reclaimed,
                             'promoted': tick.promoted, 'crashed': tick.crashed,
                             'respawn_guarded': tick.respawn_guarded, 'auto_blocked': tick.auto_blocked,
                             'rate_limited': tick.rate_limited, 'skipped_locked': tick.skipped_locked,
                             'memory_pressure': tick.memory_pressure}
        return state

    env = dict(os.environ)
    receipt, clients = {'task_id': tid}, []
    original_popen = subprocess.Popen

    def client_alive():
        return bool(clients) and clients[-1].poll() is None

    def wait_arrival():
        """Did the worker's first model call land? False as soon as the spawned client is dead.

        Waiting out the ceiling for a client that has already exited cannot succeed — the only
        thing it buys is minutes of sweep time before the same assertion fires. Measured on the
        loaded Pi: a worker whose ws dial timed out exits rc=1 at ~12 s, and the probe then sat
        on its full 300 s (1800 s in the measurement run) before reporting 'model never
        reached' (t_238f2d3f round 2).
        """
        until = time.monotonic() + budget()
        while time.monotonic() < until:
            if peer.blocked.is_set():
                return True
            if not client_alive():
                return False
            time.sleep(.05)
        return False

    def arrival_cause():
        return 'arrival' if client_alive() else 'early-client-exit'
    fault_exits = fault_start_exits()
    def spawn(*args, **kwargs):
        nonlocal fault_exits
        child = original_popen(*args, **kwargs)
        if 'hermes_cli.kanban_worker_client' in args[0]:
            clients.append(child)
            if fault_exits > 0:
                # Test-only injected transient: the client dies before its first model call, so
                # the product books the attempt as a crash and re-queues it — the same booking
                # mode=crash asserts. Nothing in the product is bypassed; the recovery below is
                # an ordinary next dispatcher tick.
                fault_exits -= 1
                child.kill()
        return child
    def wait_for(predicate, what='predicate'):
        started = time.monotonic()
        deadline = started + budget()
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.05)
        raise AssertionError(('deadline', what, diagnostics('deadline', started)))
    try:
        with daemon(root, home, env, barrier=False) as (_proc, desc):
            subprocess.Popen = spawn
            try:
                with closing(connect(board='owned')) as conn:
                    dispatched = time.monotonic()
                    retries = 0
                    while True:
                        spawned_at = time.monotonic()
                        launched = dispatch.dispatch_once(conn, board='owned', max_spawn=1)
                        if retries:
                            # What the product booked for the attempt that died: the reclaim
                            # phase names the crash and the same tick re-spawns the card.
                            receipt.setdefault('start_recovery', []).append(
                                {'crashed': launched.crashed, 'reclaimed': launched.reclaimed,
                                 'rate_limited': launched.rate_limited,
                                 'reconciled_orphans': launched.reconciled_orphans,
                                 'spawned': [s[0] for s in launched.spawned]})
                        if [s[0] for s in launched.spawned] != [tid]:
                            raise AssertionError(('dispatcher did not spawn the owned card',
                                                  diagnostics('no-spawn', spawned_at, launched)))
                        task = kb.get_task(conn, tid)
                        params = dict(task_id=tid, board='owned', run_id=task.current_run_id, claim_lock=task.claim_lock)
                        if wait_arrival():
                            break
                        cause = arrival_cause()
                        if cause != 'early-client-exit' or retries >= MAX_START_ATTEMPTS - 1:
                            raise AssertionError(('model never reached', diagnostics(cause, spawned_at)))
                        # Bounded recovery for a worker that died before its first model call. The
                        # product has already booked that attempt a crash and re-queued it (the
                        # booking mode=crash asserts), so the next tick is an ordinary dispatch —
                        # no assertion is bypassed; the surviving attempt carries them all, and
                        # the retry is recorded in the receipt so it can never be silent.
                        retries += 1
                        receipt.setdefault('start_failures', []).append(diagnostics(cause, spawned_at))
                    receipt['arrival_seconds'] = round(time.monotonic() - dispatched, 1)
                    receipt['start_retries'] = retries
                    active = kb.get_task(conn, tid)
                    attempt_client = clients[-1]
                    if peer.mode == 'timeout':
                        # The spawned client only submits the turn: the daemon binds the interpreter
                        # that runs it, and that pid (worker_bound) is the attempt's real worker,
                        # while the dispatcher's spawn stays the run's kill handle.
                        with closing(connect(board='owned')) as snapshot:
                            bound = snapshot.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? "
                                                     "AND kind='worker_bound' ORDER BY id DESC LIMIT 1",
                                                     (tid, params['run_id'])).fetchone()
                        receipt['managed_pid'] = json.loads(bound[0])['pid'] if bound else None
                        receipt['managed_arrival_seconds'] = receipt['arrival_seconds']
                    receipt.update(client_pid=attempt_client.pid, stored_pid=active.worker_pid, run_id=task.current_run_id)
                    if peer.mode == 'timeout':
                        # The spawned worker stays the run's handle; the owner's interpreter dies with it.
                        receipt['pid_preserved'] = active.worker_pid == attempt_client.pid != receipt['managed_pid']
                        # Deterministically elapse this owned attempt; keep real timeout/kill logic.
                        conn.execute('UPDATE task_runs SET started_at=? WHERE id=?', (int(time.time()) - 3601, task.current_run_id))
                        conn.commit()
                        swept = dispatch.dispatch_once(conn, board='owned', max_spawn=0)
                        receipt['timed_out'] = swept.timed_out
                        wait_for(lambda: not kb._pid_alive(receipt['managed_pid']), 'managed worker exit')
                        receipt['managed_dead'] = True
                        assert receipt['pid_preserved'] and receipt['managed_dead'], receipt
                    elif peer.mode == 'crash':
                        import signal
                        # Windows has no SIGKILL; os.kill there is TerminateProcess for any signal.
                        os.kill(active.worker_pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
                    peer.release.set()
                    if peer.mode != 'timeout':
                        wait_for(lambda: not kb._pid_alive(active.worker_pid), 'worker exit')
                    wait_for(lambda: not kb._pid_alive(attempt_client.pid), 'client exit')
                    dispatch.reap_worker_zombies()
                    receipt['client_exit'] = dispatch._classify_worker_exit(attempt_client.pid)
                    dispatch._recent_worker_exits.clear()
                    swept = dispatch.dispatch_once(conn, board='owned', max_spawn=0)
                    task = kb.get_task(conn, tid)
                    run = dict(conn.execute('SELECT * FROM task_runs WHERE id=?', (params['run_id'],)).fetchone())
                    from gateway.session_kanban import worker_exit_code
                    assert worker_exit_code(kb.kanban_db_path(board='owned'), params | {'claim_lock': 'foreign'}) == 1
                    receipt.update(task_status=task.status, failures=task.consecutive_failures, run_outcome=run['outcome'],
                        run_metadata=kb._json_dict(run['metadata']), rate_limited=swept.rate_limited,
                        cooldown=dispatch.check_respawn_guard(conn, tid))
                    if peer.mode in {'rate_limit', 'billing'}:
                        # consecutive_failures counts crashes only (a quota exit is booked without
                        # one), so the bounded start retry's crashes are exactly what this run
                        # spent: an exact equality, zero in the normal path.
                        assert task.status == 'ready' and task.consecutive_failures == retries and run['outcome'] == 'rate_limited', receipt
                        assert receipt['client_exit'] == ('rate_limited', kb.KANBAN_RATE_LIMIT_EXIT_CODE), receipt
                        assert receipt['cooldown'] == 'rate_limit_cooldown', receipt
                    elif peer.mode == 'crash':
                        # Killing the spawned worker ends the attempt: one crash, re-queued, as before the owner.
                        assert task.status == 'ready' and task.consecutive_failures == 1 + retries and run['outcome'] == 'crashed', receipt
                    elif peer.mode == 'timeout':
                        assert task.status == 'ready' and task.consecutive_failures == 1 + retries and run['outcome'] == 'timed_out', receipt
            finally:
                subprocess.Popen = original_popen
            if peer.mode != 'complete':
                return
            assert receipt['client_exit'] == ('clean_exit', 0), receipt
            async def retry():
                async with websocket(home, desc) as ws:
                    first = await rpc(ws, 'kanban.run', **params)
                    second = await rpc(ws, 'kanban.run', **params)
                    assert 'result' in first and 'result' in second, (first, second)
                    return first['result']['session_id'], first['result']['session_id'] == second['result']['session_id']
            sid, same = asyncio.run(retry())
            import sqlite3
            with closing(sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
                receipt['source'] = db.execute('SELECT source FROM sessions WHERE id=?', (sid,)).fetchone()[0]
                receipt['admissions'] = db.execute('SELECT count(*) FROM session_admissions').fetchone()[0]
            with closing(connect(board='owned')) as conn:
                receipt['task_status'] = kb.get_task(conn, tid).status
            wire = json.dumps(peer.requests)
            turns = [r for r in peer.requests if r.get('tools')]
            prefixes = [json.dumps([m for m in r['messages'] if m['role'] in {'system', 'developer'}]) for r in turns]
            receipt.update(goal_continuation=len(turns) >= 4, stable_prefix=len(set(prefixes)) == 1,
                retry_same_session=same, tools='kanban_complete' in wire,
                task_context='KANBAN_TASK_SENTINEL' in wire, skill_context='KANBAN_SKILL_SENTINEL' in wire,
                hook_effect=(home / 'hook-effect').exists())
    finally:
        peer.release.set()
        for child in clients:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        peer.shutdown()
        peer.server_close()
        (home / 'model-requests.json').write_text(json.dumps(peer.requests, indent=2), encoding='utf-8')
        (home / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
