"""Duration-aware per-file timeout scaling in scripts/run_tests_parallel.py.

The flat --file-timeout cap (default 300s) falsely SIGKILL'd
known-slow large-collection files under CI load, then the automatic
retry passed — manufacturing FLAKY reports for healthy files
(tests/test_hermes_state.py, 2026-08-18 on main). The scaler gives a
file max(flat_cap, 3 × last observed duration) and never lowers the cap.
Only first-attempt-clean durations feed the cache, so a hang can never
compound its own bound.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scaler_never_lowers_the_flat_cap() -> None:
    mod = _load_runner()
    uncached = REPO_ROOT / "tests" / "test_example.py"
    fast = REPO_ROOT / "tests" / "test_fast.py"
    zero = REPO_ROOT / "tests" / "test_zero.py"
    durations = {
        mod._format_file(fast, REPO_ROOT): 4.2,
        mod._format_file(zero, REPO_ROOT): 0.0,
    }
    assert mod._effective_file_timeout(uncached, REPO_ROOT, 300.0, None) == 300.0
    assert mod._effective_file_timeout(uncached, REPO_ROOT, 300.0, durations) == 300.0
    assert mod._effective_file_timeout(fast, REPO_ROOT, 300.0, durations) == 300.0
    assert mod._effective_file_timeout(zero, REPO_ROOT, 300.0, durations) == 300.0


def test_slow_file_gets_proportional_headroom() -> None:
    mod = _load_runner()
    f = REPO_ROOT / "tests" / "test_hermes_state.py"
    durations = {mod._format_file(f, REPO_ROOT): 205.0}
    # 205s last run → 615s bound: a load-dilated healthy run survives,
    # a genuine hang is still killed.
    assert mod._effective_file_timeout(f, REPO_ROOT, 300.0, durations) == 615.0


def test_only_first_attempt_clean_durations_feed_the_cache() -> None:
    """A timed-out or retried file must not raise its own future bound."""
    mod = _load_runner()
    clean = REPO_ROOT / "tests" / "test_clean.py"
    hung = REPO_ROOT / "tests" / "test_hung.py"
    flaky = REPO_ROOT / "tests" / "test_flaky.py"
    file_times = [(clean, 12.0), (hung, 300.4), (flaky, 250.0)]
    failures = [(hung, "(300s exceeded; process tree SIGKILL'd)", {})]
    flaky_results = [(flaky, "⚠ FLAKY: failed on attempt 1, passed on retry")]

    kept = mod._clean_pass_durations(file_times, failures, flaky_results)

    assert kept == [(clean, 12.0)]


def test_declared_floor_covers_a_cost_the_cache_cannot_supply(tmp_path) -> None:
    """A structurally slow file declares its own ceiling.

    Real-daemon files need 15+ minutes even with every fast path taken, and the duration
    cache that would scale their bound is gitignored — a fresh checkout (and every CI
    runner) has no entry, so the flat 300s cap SIGKILL'd them before collection and the
    summary reported "no tests ran" (t_238f2d3f). The declaration travels with the file.
    """
    mod = _load_runner()
    slow = tmp_path / "test_slow_gateway.py"
    slow.write_text(
        '"""Real daemons."""\n'
        "FILE_TIMEOUT_MIN_SECONDS = 2700  # measured 900s clean; 3x for load dilation\n"
        "\n\ndef test_x():\n    assert True\n"
    )
    floors = mod._declared_timeout_floors([slow])
    assert floors == {slow: 2700.0}

    # No cache entry (fresh checkout / CI) → the declaration alone is the bound.
    assert mod._effective_file_timeout(slow, REPO_ROOT, 300.0, None, floors) == 2700.0
    assert mod._effective_file_timeout(slow, REPO_ROOT, 300.0, {}, floors) == 2700.0

    # A larger cached duration still wins; a smaller one cannot lower the floor.
    assert mod._effective_file_timeout(
        slow, REPO_ROOT, 300.0, {mod._format_file(slow, REPO_ROOT): 1200.0}, floors
    ) == 3600.0
    assert mod._effective_file_timeout(
        slow, REPO_ROOT, 300.0, {mod._format_file(slow, REPO_ROOT): 200.0}, floors
    ) == 2700.0

    # An explicit --file-timeout above the declaration also wins (max, never min).
    assert mod._effective_file_timeout(slow, REPO_ROOT, 4000.0, None, floors) == 4000.0


def test_missing_declaration_and_unreadable_file_add_no_floor(tmp_path) -> None:
    mod = _load_runner()
    plain = tmp_path / "test_plain.py"
    plain.write_text("def test_x():\n    assert True\n")
    assert mod._declared_timeout_floors([plain]) == {}
    assert mod._declared_timeout_floors([tmp_path / "absent.py"]) == {}
    assert mod._effective_file_timeout(plain, REPO_ROOT, 300.0, None, {}) == 300.0


def test_filtered_runs_do_not_poison_the_duration_cache() -> None:
    """A `-k`/`-m`/nodeid run measures a subset, so its per-file wall is not a duration.

    Measured failure (t_238f2d3f): `-k timeout` on a 5-probe file cached 55.7s, so the next
    full sweep sized that file at max(300, 3 × 55.7) = 300s, SIGKILL'd it before it
    collected, and reported "no tests ran".
    """
    mod = _load_runner()
    assert mod._narrows_per_file(["-k", "timeout"])
    assert mod._narrows_per_file(["-k=timeout"])
    assert mod._narrows_per_file(["-m", "not slow"])
    assert mod._narrows_per_file(["--deselect", "tests/x.py::test_y"])
    assert mod._narrows_per_file(["--deselect=tests/x.py::test_y"])
    assert mod._narrows_per_file(["tests/gateway/test_x.py::test_y"])
    # Whole-file runs (paths, verbosity, output flags) keep caching durations.
    assert not mod._narrows_per_file([])
    assert not mod._narrows_per_file(["-v", "--tb=long"])
    assert not mod._narrows_per_file(["tests/gateway/test_x.py"])
