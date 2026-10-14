"""A successful run resolves the job's open incidents; the same error afterwards re-opens them.

The ledger only ever grew: a one-off failure (drift skip, provider outage) stayed ``detected``/``alerted``
after the job had been green for weeks, so ``hermes cron incidents`` listed 32 "open" incidents on an
install whose every job was healthy. ``resolved`` is auto and re-openable; ``closed`` is the operator's
ack and stays silent.

t_49fc646c regression lock: the resolve is gated on a REAL successful run — a silent success (a
no_agent script's ``{"wakeAgent": false}`` bail, empty output, a no-change monitor tick) must NOT
resolve open incidents. Grounded masking path: backup-snapshot.sh's PC-off bail exited 0 nightly
("not mounted → Skipping backup"), counted as "job recovered", and closed the open incident for the
real failures — 5 nightly incidents ended ``resolved`` with ``acked_at=NULL``. The signature must
also survive run-varying tokens: the rsync stderr embeds the dated output dir
(``hermes-backup/2026-09-28/`` vs ``2026-10-08/``), which minted ONE incident per night instead of
deduping into one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron import incidents
import cron.scheduler as sched


def _point_db(monkeypatch, tmp_path):
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return incidents


def test_successful_run_resolves_open_incidents_and_a_repeat_reopens_them(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "provider 503")
    sched._mark_incident_alerted(inc_id)
    other_id, _ = inc.upsert_incident("job-2", "provider 503")

    content, *_ = sched._compose_run_delivery(
        {"id": "job-1", "name": "j1"}, success=True, error=None, final_response="all good", output_file=None)

    assert content == "all good"
    assert inc.get_incident(inc_id)["state"] == "resolved"
    assert inc.get_incident(inc_id)["closed_at"]
    assert inc.get_incident(other_id)["state"] == "detected"  # another job's incident is untouched
    assert inc.count_incidents("resolved") == 1

    same_id, reopened = inc.upsert_incident("job-1", "provider 503")
    assert same_id == inc_id and reopened is True
    assert inc.get_incident(inc_id)["state"] == "detected"
    assert inc.get_incident(inc_id)["closed_at"] is None


def test_operator_ack_survives_a_successful_run(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "known flaky")
    inc.ack_incident(inc_id)

    assert inc.close_incidents_for_recovered_job("job-1") == 0
    assert inc.get_incident(inc_id)["state"] == "closed"
    _, reopened = inc.upsert_incident("job-1", "known flaky")
    assert reopened is False and inc.get_incident(inc_id)["state"] == "closed"


# ── t_49fc646c: a silent success must NOT auto-resolve open incidents ──────


def test_silent_wake_gate_skip_does_not_resolve_open_incidents(monkeypatch, tmp_path):
    """A no_agent script's wakeAgent=false bail (PC-off skip night) is a SILENT run, not a
    recovery: the open incident for the real failure must stay open. Grounded masking path —
    5 nightly incidents auto-resolved by bail nights (acked_at=NULL), t_49fc646c."""
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("cdf245", "Script exited with code 1: rsync Permission denied")
    sched._mark_incident_alerted(inc_id)

    sched._compose_run_delivery(
        {"id": "cdf245", "name": "layer2-backup-windows"},
        success=True, error=None, final_response=sched.SILENT_MARKER, output_file=None)

    row = inc.get_incident(inc_id)
    assert row["state"] == "alerted", "a skip night must not resolve the open incident"
    assert row["closed_at"] is None


def test_empty_response_success_does_not_resolve_open_incidents(monkeypatch, tmp_path):
    """Empty-output silent success path (``Status: silent (empty output)``) — same gate."""
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "boom")

    sched._compose_run_delivery(
        {"id": "job-1", "name": "j1"}, success=True, error=None, final_response="", output_file=None)

    assert inc.get_incident(inc_id)["state"] in ("detected", "alerted")
    assert inc.get_incident(inc_id)["closed_at"] is None


def test_bracketless_silence_variants_do_not_resolve_open_incidents(monkeypatch, tmp_path):
    """The silence matcher recognises bare SILENT/NO_REPLY lines, not just [SILENT] — all of
    them must block the auto-resolve."""
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "boom")

    sched._compose_run_delivery(
        {"id": "job-1", "name": "j1"}, success=True, error=None, final_response="NO_REPLY",
        output_file=None)

    assert inc.get_incident(inc_id)["state"] != "resolved"


# ── t_49fc646c: the signature must survive run-varying tokens in the error ──


def test_signature_dedups_rsync_dated_dir_across_nights(monkeypatch, tmp_path):
    """Grounded failure shape: identical nightly rsync stderr except the dated backup dir.
    Pre-fix this minted ONE incident per night (6 rows, one per 2026-09-28..10-08); same
    failure must dedup into ONE incident, refreshed each night."""
    inc = _point_db(monkeypatch, tmp_path)
    nightly = (
        'Script exited with code 1\nstderr:\nrsync: [generator] recv_generator: mkdir '
        '"/mnt/mainpc/hermes-backup/{date}/hermes/kanban/snapshots/t_6ae16538/db" failed: '
        "Permission denied (13)\n*** Skipping any contents from this failed directory ***\n"
        "rsync: [receiver] mkstemp \"/mnt/mainpc/hermes-backup/{date}/hermes/kanban/snapshots/"
        't_6ae16538/.MANIFEST.json.{rand}" failed: Permission denied (13)\n'
        "rsync error: some files/attrs were not transferred (code 23) [sender=3.5.0]"
    )

    id1, is_new = inc.upsert_incident(
        "cdf245a5abd0", nightly.format(date="2026-09-28", rand="a4imyc"))
    id2, is_new2 = inc.upsert_incident(
        "cdf245a5abd0", nightly.format(date="2026-10-08", rand="96vsjp"))

    assert id1 == id2, "same failure with a different run date must dedup into one incident"
    assert is_new is True and is_new2 is False
    assert inc.count_incidents() == 1
    assert inc.get_incident(id1)["state"] == "detected", "refresh keeps an open state open"


def test_signature_still_distinguishes_genuinely_different_errors(monkeypatch, tmp_path):
    """The volatile-token scrub must not collapse distinct failures: a Permission-denied and a
    vanished-source failure differ outside the scrubbed tokens."""
    inc = _point_db(monkeypatch, tmp_path)
    id1, _ = inc.upsert_incident("job-1", 'rsync: mkdir "/backup/2026-10-08/x" failed: Permission denied (13)')
    id2, is_new = inc.upsert_incident("job-1", 'rsync: symlink stats "/backup/2026-10-09/y" failed: No such file (2)')

    assert id1 != id2 and is_new is True
    assert inc.count_incidents() == 2