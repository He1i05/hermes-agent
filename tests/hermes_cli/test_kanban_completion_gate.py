"""M13 completion-evidence gate — shadow-only build (``t_c06dd31a``).

Specification: ``~/.hermes/data/m13-gate/M13_decision_packet.md`` §6 (flag shape +
ladder) and ``M13_policy_spec.md`` §1–§2 (refusal class R2, exemptions E1–E4).

The gate must never refuse in shadow; every exempt class must be explicit; the
``off`` default must do no work; and the flag must be live per call (the activation
step is a one-command rollback only if the key is hot).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_completion_gate as gate
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

_FALSE_SHA = "0123456789abcdef0123456789abcdef01234567"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    gate.reset_mode_cache()
    yield root
    gate.reset_mode_cache()


@pytest.fixture
def conn(home):
    with kbc.connect() as c:
        yield c


def _set_mode(home: Path, mode: str) -> None:
    """Write ``kanban.completion_gate`` to this home's config and drop the caches."""
    (home / "config.yaml").write_text(
        f"kanban:\n  completion_gate: {mode}\n", encoding="utf-8"
    )
    gate.reset_mode_cache()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A real git repo: ``main`` at one commit, plus a side commit not on ``main``."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "one")
    head = _git(root, "rev-parse", "HEAD")
    # A commit that exists but is NOT an ancestor of main.
    _git(root, "checkout", "-q", "-b", "side")
    (root / "b.txt").write_text("two\n", encoding="utf-8")
    _git(root, "add", "b.txt")
    _git(root, "commit", "-q", "-m", "two")
    side = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "main")
    return {"path": root, "head": head, "side": side}


def _task(conn, workspace_path=None) -> str:
    tid = kb.create_task(
        conn, title="do the thing", assignee="coder", workspace_kind="scratch",
        workspace_path=str(workspace_path) if workspace_path else None,
    )
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


def _gate_events(conn, tid) -> list[dict]:
    return [
        e.payload or {}
        for e in kb.list_events(conn, tid)
        if e.kind in (gate.EVENT_SHADOW, gate.EVENT_ERROR)
    ]


def _only_gate_event(conn, tid) -> dict:
    events = _gate_events(conn, tid)
    assert len(events) == 1, f"expected exactly one gate event, got {events}"
    return events[0]


# --------------------------------------------------------------------------- #
# flag shape
# --------------------------------------------------------------------------- #
def test_default_is_off_and_off_records_nothing(home, conn):
    assert gate.configured_mode() == "off"
    tid = _task(conn)
    assert kb.complete_task(conn, tid, summary="claim", metadata={"commit": _FALSE_SHA}) is True
    assert _gate_events(conn, tid) == []


def test_flag_values_and_unknown_values(home):
    for value in ("off", "shadow", "enforce"):
        assert gate.resolve_mode(value) == value
    for junk in ("refuse", "warn", "", None, True, 1, " SHADOW "):
        expected = "shadow" if junk == " SHADOW " else "off"
        assert gate.resolve_mode(junk) == expected


def test_flag_is_hot_per_call_no_restart(home, conn, repo):
    """The activation (and the one-key rollback) must not need a restart."""
    assert gate.configured_mode() == "off"
    _set_mode(home, "shadow")
    assert gate.configured_mode() == "shadow"
    tid = _task(conn, repo["path"])
    assert kb.complete_task(conn, tid, summary="claim", metadata={"commit": _FALSE_SHA}) is True
    assert _only_gate_event(conn, tid)["would_refuse"] is True
    # rollback is the same one command, in the other direction
    _set_mode(home, "off")
    tid2 = _task(conn, repo["path"])
    assert kb.complete_task(conn, tid2, summary="claim", metadata={"commit": _FALSE_SHA}) is True
    assert _gate_events(conn, tid2) == []


# --------------------------------------------------------------------------- #
# shadow never refuses
# --------------------------------------------------------------------------- #
def test_shadow_records_would_refuse_and_completes_unchanged(home, conn, repo):
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(
        conn, tid,
        summary="landed the fix",
        metadata={"commit": _FALSE_SHA, "changed_files": ["a.txt"]},
    ) is True
    task = kb.get_task(conn, tid)
    assert task.status == "done"
    assert task.result is None or "refus" not in (task.result or "").lower()
    payload = _only_gate_event(conn, tid)
    assert payload["mode"] == "shadow"
    assert payload["evaluated"] is True
    assert payload["would_refuse"] is True
    assert payload["refusal_class"] == gate.REFUSAL_CLASS
    assert payload["exempt_class"] is None
    assert payload["elapsed_ms"] >= 0.0


def test_shadow_records_allow_when_commit_is_an_ancestor(home, conn, repo):
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(
        conn, tid, summary="landed it", metadata={"commit": repo["head"]},
    ) is True
    payload = _only_gate_event(conn, tid)
    assert payload["would_refuse"] is False
    assert payload["resolved_sha"] == repo["head"]
    assert "resolves" in payload["reason"]


def test_shadow_refuses_commit_that_is_not_an_ancestor(home, conn, repo):
    """Ancestor, never equality: a real commit outside HEAD's history is unverified."""
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(
        conn, tid, summary="landed it", metadata={"commit": repo["side"]},
    ) is True
    payload = _only_gate_event(conn, tid)
    assert payload["would_refuse"] is True
    assert payload["resolved_sha"] is None


def test_shadow_refuses_changed_files_with_no_commit(home, conn, repo):
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(
        conn, tid, summary="touched files",
        metadata={"changed_files": ["hermes_cli/kanban_db.py"]},
    ) is True
    payload = _only_gate_event(conn, tid)
    assert payload["would_refuse"] is True
    assert "cites no commit" in payload["reason"]


def test_shadow_allows_a_completion_with_no_change_claim(home, conn, repo):
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(conn, tid, summary="read the code, nothing to change") is True
    payload = _only_gate_event(conn, tid)
    assert payload["would_refuse"] is False
    assert payload["claims_code_change"] is False


def test_shadow_finds_a_commit_cited_only_in_prose(home, conn, repo):
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(
        conn, tid, summary=f"committed as {repo['head']} and pushed",
    ) is True
    payload = _only_gate_event(conn, tid)
    assert payload["resolved_sha"] == repo["head"]
    assert payload["would_refuse"] is False


# --------------------------------------------------------------------------- #
# exempt classes (each explicit, each tested)
# --------------------------------------------------------------------------- #
def test_exempt_review_approval(home, conn, repo):
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.request_review(conn, tid, summary="please look") is True
    assert kb.get_task(conn, tid).status == "review"
    assert kb.complete_task(conn, tid, metadata={"commit": _FALSE_SHA}) is True
    payload = _only_gate_event(conn, tid)
    assert payload["exempt_class"] == gate.EXEMPT_REVIEW_APPROVAL
    assert payload["would_refuse"] is False


def test_exempt_no_implementable_artifact_scratch_workspace(home, conn, tmp_path):
    """A research / decision-packet card: its workspace is not a git tree."""
    _set_mode(home, "shadow")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    tid = _task(conn, scratch)
    assert kb.complete_task(
        conn, tid,
        summary="wrote REPORT_foo.md",
        metadata={"changed_files": ["REPORT_foo.md"], "commit": _FALSE_SHA},
    ) is True
    payload = _only_gate_event(conn, tid)
    assert payload["exempt_class"] == gate.EXEMPT_NO_ARTIFACT
    assert payload["would_refuse"] is False


def test_exempt_declared_read_only(home, conn, repo):
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(
        conn, tid, summary="audited it",
        metadata={"read_only": True, "changed_files": ["REPORT.md"]},
    ) is True
    payload = _only_gate_event(conn, tid)
    assert payload["exempt_class"] == gate.EXEMPT_NO_ARTIFACT
    assert payload["would_refuse"] is False


def test_exempt_adoption_evidence_waiver(home, conn, repo):
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(
        conn, tid,
        summary="shipped it",
        metadata={
            "commit": _FALSE_SHA,
            "adoption_evidence": {"waiver": "no-adoption-risk"},
        },
    ) is True
    payload = _only_gate_event(conn, tid)
    assert payload["exempt_class"] == gate.EXEMPT_ADOPTION_WAIVER
    assert payload["would_refuse"] is False


def test_exempt_adoption_probe_absent_fail_open_literal(home, conn, repo):
    _set_mode(home, "shadow")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(
        conn, tid, summary="shipped it",
        metadata={"commit": _FALSE_SHA, "adoption_evidence": "probe_absent_fail_open"},
    ) is True
    assert _only_gate_event(conn, tid)["exempt_class"] == gate.EXEMPT_ADOPTION_WAIVER


def test_exempt_classes_are_exactly_the_three_the_card_names():
    assert set(gate.EXEMPT_CLASSES) == {
        "review_approval", "adoption_evidence_waiver", "no_implementable_artifact",
    }


# --------------------------------------------------------------------------- #
# enforce is reserved, not implemented
# --------------------------------------------------------------------------- #
def test_enforce_value_is_accepted_but_refuses_nothing(home, conn, repo):
    _set_mode(home, "enforce")
    tid = _task(conn, repo["path"])
    assert kb.complete_task(
        conn, tid, summary="claim", metadata={"commit": _FALSE_SHA},
    ) is True
    assert kb.get_task(conn, tid).status == "done"
    payload = _only_gate_event(conn, tid)
    assert payload["mode"] == "enforce"
    assert payload["enforce_ignored"] is True
    assert payload["would_refuse"] is True  # recorded, not acted on


def test_no_refusal_class_raises_anywhere_in_the_gate_module():
    """The build ships no refusal path: the module defines no error type and the
    only ``raise``-free hook is asserted by the shadow tests above."""
    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert "class GateRefusal" not in source
    assert "would_block" not in source


# --------------------------------------------------------------------------- #
# failure containment + cost guards
# --------------------------------------------------------------------------- #
def test_gate_internal_error_never_blocks_the_completion(home, conn, repo, monkeypatch):
    _set_mode(home, "shadow")

    def boom(**kwargs):
        raise RuntimeError("synthetic gate bug")

    monkeypatch.setattr(gate, "evaluate_completion", boom)
    tid = _task(conn, repo["path"])
    assert kb.complete_task(conn, tid, summary="landed it") is True
    assert kb.get_task(conn, tid).status == "done"
    events = _gate_events(conn, tid)
    assert events and events[0]["error"].startswith("RuntimeError")


def test_evidence_scan_is_capped(home):
    decision = gate.evaluate_completion(
        prior_status="running",
        workspace_path=None,
        summary="x" * 10_000 + " commit " + _FALSE_SHA + " y" * 200_000,
        metadata={"changed_files": ["a.py"]},
        mode="shadow",
    )
    assert len(decision.sha_candidates) <= gate._MAX_SHA_CANDIDATES
    # archived as "no git tree" — the point is the scan terminated cheaply
    assert decision.elapsed_ms < 5_000


def test_payload_names_the_card_scoped_repo(home, tmp_path, repo):
    """A scratch card that names the tree it worked in is still evaluated."""
    decision = gate.evaluate_completion(
        prior_status="running",
        workspace_path=str(tmp_path / "elsewhere"),
        summary=f"committed as {repo['head']} in {repo['path']}",
        mode="shadow",
    )
    assert decision.would_refuse is False
    assert decision.resolved_sha == repo["head"]
    assert any(d.startswith("payload:") for d in decision.resolution_domains)


def test_backticked_checksum_is_not_treated_as_a_cited_commit(home, monkeypatch):
    """A 32-hex md5 in backticks is a checksum, not a commit: no manufactured refusal."""
    decision = gate.evaluate_completion(
        prior_status="running",
        workspace_path=None,
        summary="packet verified: `6c44bd055e3d54e93b2e1a1763ba7a8f` (14591 B)",
        metadata={"read_only": True},
        mode="shadow",
    )
    assert decision.sha_candidates == []
    assert decision.claims_code_change is False


def test_home_and_hermes_home_are_never_resolution_domains(home, tmp_path, monkeypatch):
    """``~`` and ``~/.hermes`` are the agent's own trees, not a card's repo."""
    assert "payload" not in "".join(
        gate.evaluate_completion(
            prior_status="running",
            workspace_path=None,
            summary=f"edited {Path.home()}/x/y.txt",
            mode="shadow",
        ).resolution_domains
    )
