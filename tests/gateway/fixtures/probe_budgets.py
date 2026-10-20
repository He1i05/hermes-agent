"""Slow-host ceilings shared by the real-dispatcher gateway probes and their calling tests.

One number, one derivation. The probes run as scripts (their fixtures dir on ``sys.path``) and
the tests import this module as ``tests.gateway.fixtures.probe_budgets``; keeping it import-free
is what lets both read the same constants. Every value here is a CEILING, not a cost: a green
run returns as soon as its predicate holds. The justification for each number lives next to it
and must come from a measured run, never from a wish — a ceiling below the load band this host
produces turns a slow host into a red file, and a caller cap below the probe's own chain turns
an assertion into a kill that a sweep log reports as a hang (t_97673f26 round 3).
"""

# Owner/recovery boot deadline: one gateway daemon warming its first request path from a
# borrowing data root. Measured green boots on the 4-core Pi: 32-96 s; the value is the old
# hardcoded one, kept because no boot has ever been the expired ceiling in this series.
BOOT_DEADLINE_DEFAULT = 420

# Arrival/discovery ceiling for every wait_for() predicate in either probe (the worker's first
# model call, the client's exit, the admission going terminal). Measured green arrivals on this
# 4-core Pi, 2026-10-20, from the probes' own receipts:
#   sequential, 5 owner params in one process at load 1.2-5.0: 26.4 28.0 29.6 30.1 30.1 30.3 s
#   three gateway files in flight at load 9.2-15.2:        56.3 70.4 89.7 109.9 113.1 113.1 s
#   recovery discovery, same concurrent run:               74.3 95.7 106.8 s
# 900 s is 8x the slowest arrival ever measured here, i.e. outside the load band this host
# produces rather than inside it. The failures at the old 300 s default were not dilation: the
# spawned client had already exited rc=1 on its loopback ws dial (see CONNECT_CEILING_DEFAULT),
# and the probe then sat out the whole budget. Raising the ceiling is only correct together with
# the fast-fail in kanban_owner_probe.wait_arrival().
PROBE_WAIT_DEFAULT = 900

# Tail a caller adds to the cap it puts on a probe subprocess: spawn, the lifecycle assertions
# that follow the arrival, and teardown.
LIFECYCLE_TAIL_SECONDS = 360

# Ceiling for the worker client's loopback websocket dial, which the product exposes as
# HERMES_GATEWAY_CONNECT_TIMEOUT (hermes_cli/gateway_client.py). Production dials with 10 s;
# measured on this 4-core Pi at load 10-11 with three gateway files in flight, the dial took
# 12-13 s, so the worker exited rc=1 "Gateway connection failed; no local fallback" before its
# first model call — reported by both probes as a start failure. Ceiling, not cost: a green dial
# returns in ~1 s.
CONNECT_CEILING_DEFAULT = 120


def owner_parent_cap(boot, wait):
    """Outer cap a caller must put on kanban_owner_probe.py.

    The probe's own chain is the boot deadline plus four ``budget()`` waits (release, timeout
    mode's managed arrival, the post-release wait_for, and the arrival itself) plus the
    lifecycle tail. Deriving the cap from the boot alone is what fired at 540 s inside a chain
    that still had 1200 s of waits ahead of it.
    """
    return boot + 4 * wait + LIFECYCLE_TAIL_SECONDS


# Ceiling chains for kanban_recovery_probe.py, one entry per mode as (boots, waits). Counting
# only the slowest mode under-derives the cap for the others: ``restart`` boots the owner twice
# (the first attempt is killed, the second is the recovery) and needs three waits (discovery,
# clients exited after the second boot, clients exited at the end), while ``custom`` and
# ``cross_profile`` boot once and need four (discovery, admission terminal, task done, clients
# exited). Deriving the cap from ``restart`` alone put those two modes' parent
# subprocess.run() 120 s inside their own chain, so it killed the probe before its last wait
# could assert (t_238f2d3f round 2).
RECOVERY_CHAINS = {'custom': (1, 4), 'cross_profile': (1, 4), 'restart': (2, 3)}


def recovery_parent_cap(boot, wait):
    """Outer cap a caller must put on kanban_recovery_probe.py.

    The cap is the largest of the per-mode chains in ``RECOVERY_CHAINS`` plus the lifecycle
    tail, so it outlasts whichever mode is running. At the current defaults that is
    ``custom``/``cross_profile``: 420 + 4 * 900 + 360 = 4380 s, against ``restart``'s
    2 * 420 + 3 * 900 + 360 = 3900 s.
    """
    return max(boots * boot + waits * wait
               for boots, waits in RECOVERY_CHAINS.values()) + LIFECYCLE_TAIL_SECONDS

