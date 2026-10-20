"""Real dispatch preserves custom board identity and unknown attempts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


# The runner's flat 300 s per-file cap cannot fit three real daemon + worker recoveries, and
# `test_durations.json` — the only other source of a larger bound — is gitignored, so a fresh
# checkout (and every CI runner) has nothing to scale from. Measured on the 4-core Pi:
# 234.6 s green solo after the fixture's lazy-install opt-out, 1117-1284 s under a 4-way sweep.
# A ceiling, not a cost: the probe returns as soon as its predicate holds, and the probe's own
# KANBAN_RECOVERY_BOOT_DEADLINE/KANBAN_PROBE_TIMEOUT ceilings are what bound a real hang.
FILE_TIMEOUT_MIN_SECONDS = 1500


# The probe traces the transport client's opens with strace (Linux-only tool, Linux openat
# flags); without it the probe cannot prove the client never opens state.db writable.
@pytest.mark.platforms('linux')
@pytest.mark.skipif(shutil.which('strace') is None, reason='requires strace on PATH')
@pytest.mark.parametrize('mode', ['custom', 'cross_profile', 'restart'])
def test_owner_board_recovery(tmp_path, mode):
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    # Same allowlist boundary as child_env(): without the suite's lazy-install switch the
    # probe's worker client re-syncs this borrowing data root before it ever reaches the
    # board (see tests/gateway/fixtures/local_recovery_probe.py:child_env, t_238f2d3f).
    env['HERMES_DISABLE_LAZY_INSTALLS'] = '1'
    # Both budgets are ceilings, not costs: a green run returns as soon as the predicate holds.
    # They are forwarded explicitly because the probe env is a whitelist, and a borrowing data
    # root's per-process dependency prep on a slow host outlasts the old hardcoded values
    # (t_97673f26 item 2).
    boot = int(os.environ.get('KANBAN_RECOVERY_BOOT_DEADLINE', '420'))
    wait = int(os.environ.get('KANBAN_RECOVERY_WAIT_BUDGET', str(boot)))
    env.update(HOME=str(tmp_path / 'user'), HERMES_HOME=str(tmp_path / 'state'),
               PYTHONPATH=str(repo), KANBAN_RECOVERY_MODE=mode,
               KANBAN_RECOVERY_BOOT_DEADLINE=str(boot),
               KANBAN_RECOVERY_WAIT_BUDGET=str(wait))
    # The parent's cap must outlast the probe's own two ceilings (boot + first wait) plus the
    # lifecycle, or a loaded host reports TimeoutExpired where the probe would have asserted:
    # in the 145-file sweep [custom] was killed at the old cap == boot exactly (420 s) while
    # its siblings passed in the same run (t_97673f26 item 2).
    timeout = int(os.environ.get('KANBAN_PROBE_TIMEOUT', str(boot + wait + 360)))
    result = subprocess.run([sys.executable, str(repo / 'tests/gateway/fixtures/kanban_recovery_probe.py')],
                            cwd=repo, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True,
                            timeout=timeout,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    print(json.dumps(json.loads((tmp_path / 'state/receipt.json').read_text())))
