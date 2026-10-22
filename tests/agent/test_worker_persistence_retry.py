"""Regression tests for the bounded owner-miss retry (root cause t_ae2c07f0).

The failure being pinned: one transient RPC miss against a saturated owner used to set a
sticky `self.failure`, after which every later `_apply` re-raised it and the worker's turn
died - discarding work that had already landed. Retry is now bounded and transient-only;
a contract violation still fails closed on the first attempt.
"""
import pytest

from agent.runtime_session_store import (
    PERSIST_RETRY_DELAYS,
    RuntimeSessionStore,
    WorkerPersistenceError,
)

SCOPE = {'epoch': 1, 'session_id': 'sess-1', 'execution_id': 'exec-1', 'generation': 1}


def _store(tmp_path, rpc):
    return RuntimeSessionStore(rpc, SCOPE, tmp_path / 'outbox')


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr('agent.runtime_session_store.time.sleep', lambda _s: None)


def test_transient_owner_miss_is_retried_then_succeeds(tmp_path):
    calls = {'n': 0}

    def rpc(method, **params):
        calls['n'] += 1
        if calls['n'] <= 2:
            raise WorkerPersistenceError('owner_unavailable')
        return {'sequence': params.get('sequence'), 'committed': True}

    store = _store(tmp_path, rpc)
    store._apply('turn.cleanup', {'target': SCOPE['session_id']})

    assert calls['n'] == 3, 'two transient misses must be absorbed by the retry, not the caller'
    assert store.journal['pending'] == []
    assert store.failure is None


def test_retry_budget_is_bounded_and_then_goes_sticky(tmp_path):
    calls = {'n': 0}

    def rpc(method, **params):
        calls['n'] += 1
        raise WorkerPersistenceError('owner_unavailable')

    store = _store(tmp_path, rpc)
    with pytest.raises(WorkerPersistenceError):
        store._apply('turn.cleanup', {'target': SCOPE['session_id']})

    assert calls['n'] == len(PERSIST_RETRY_DELAYS) + 1, 'one attempt plus the bounded retries'
    assert store.failure == 'owner_unavailable'
    # The entry stays journaled: a later attempt can still replay it exactly once.
    assert len(store.journal['pending']) == 1


def test_contract_violation_does_not_retry(tmp_path):
    calls = {'n': 0}

    def rpc(method, **params):
        calls['n'] += 1
        raise WorkerPersistenceError('permission_denied')

    store = _store(tmp_path, rpc)
    with pytest.raises(WorkerPersistenceError):
        store._apply('turn.cleanup', {'target': SCOPE['session_id']})

    assert calls['n'] == 1, 'a permission failure is not a transient owner miss'
    assert store.failure == 'permission_denied'


def test_socket_error_is_transient(tmp_path):
    calls = {'n': 0}

    def rpc(method, **params):
        calls['n'] += 1
        if calls['n'] == 1:
            raise ConnectionResetError('peer went away')
        return {'sequence': params.get('sequence'), 'committed': True}

    store = _store(tmp_path, rpc)
    store._apply('turn.cleanup', {'target': SCOPE['session_id']})
    assert calls['n'] == 2


def test_sticky_transient_failure_is_recovered_on_next_apply(tmp_path):
    """A prior attempt that exhausted its budget must not poison the rest of the worker's life."""
    calls = {'n': 0}
    state = {'fail': True}

    def rpc(method, **params):
        calls['n'] += 1
        if state['fail']:
            raise WorkerPersistenceError('owner_unavailable')
        return {'sequence': params.get('sequence'), 'committed': True}

    store = _store(tmp_path, rpc)
    with pytest.raises(WorkerPersistenceError):
        store._apply('turn.cleanup', {'target': SCOPE['session_id']})
    assert store.failure == 'owner_unavailable'

    state['fail'] = False
    store._apply('token.flush', {'counts': [1, 2, 3]})

    assert store.failure is None
    assert store.journal['pending'] == []
