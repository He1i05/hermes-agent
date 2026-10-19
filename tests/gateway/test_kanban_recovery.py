"""Real dispatch preserves custom board identity and unknown attempts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


# The probe traces the transport client's opens with strace (Linux-only tool, Linux openat
# flags); without it the probe cannot prove the client never opens state.db writable.
@pytest.mark.platforms('linux')
@pytest.mark.skipif(shutil.which('strace') is None, reason='requires strace on PATH')
@pytest.mark.parametrize('mode', ['custom', 'cross_profile', 'restart'])
def test_owner_board_recovery(tmp_path, mode):
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    # Both budgets are ceilings, not costs: a green run returns as soon as the predicate holds.
    # They are forwarded explicitly because the probe env is a whitelist, and a borrowing data
    # root's per-process dependency prep on a slow host outlasts the old hardcoded values
    # (t_97673f26 item 2).
    boot = os.environ.get('KANBAN_RECOVERY_BOOT_DEADLINE', '420')
    env.update(HOME=str(tmp_path / 'user'), HERMES_HOME=str(tmp_path / 'state'),
               PYTHONPATH=str(repo), KANBAN_RECOVERY_MODE=mode,
               KANBAN_RECOVERY_BOOT_DEADLINE=boot,
               KANBAN_RECOVERY_WAIT_BUDGET=os.environ.get('KANBAN_RECOVERY_WAIT_BUDGET', boot))
    result = subprocess.run([sys.executable, str(repo / 'tests/gateway/fixtures/kanban_recovery_probe.py')],
                            cwd=repo, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True,
                            timeout=int(boot),
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    print(json.dumps(json.loads((tmp_path / 'state/receipt.json').read_text())))
