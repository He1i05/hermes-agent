"""Credential-free phase-cascade probe for the Kanban decomposer (T1).

Runs the REAL decomposer (real prompt, real DB persistence) against a stub
aux LLM that returns a hand-built cascade, in an isolated temp HERMES_HOME, then
asserts the two behavioral properties the phase design depends on:

  1. every emitted child's phase is persisted to ``current_step_key`` and drawn
     from the closed enum;
  2. the emitted chain is a contiguous, parent-linked subsequence of the
     canonical chain (skip-non-applicable preserved).

Usage: python evals/kanban_phase_cascade.py --repo PATH

Safety: the child writes to a real SQLite kanban DB. The parent wrapper
re-execs the child with a temp HERMES_HOME; invoking ``--child`` directly is
refused unless BOTH hold: ``HERMES_HOME`` is set and points OUTSIDE the real
``~/.hermes``, AND the effective resolved kanban DB path
(``hermes_cli.kanban_db.kanban_db_path()``) also lands outside it. The second
condition is load-bearing: ``HERMES_KANBAN_DB`` (exported into every kanban
worker session) wins over ``HERMES_HOME``, so the first guard iteration — which
checked ``HERMES_HOME`` only — passed while the write still hit the live board.
See ``evals/kanban_child_guard.py`` (2026-09-12/13 leak hardening).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CASCADES = {
    # Full 7-phase quest with adoption/docs support cards (unphased).
    "full": [
        {"title": "ground the question", "phase": "grounding", "parents": []},
        {"title": "review prior art", "phase": "review", "parents": [0]},
        {"title": "research the space", "phase": "research", "parents": [1]},
        {"title": "design the solution", "phase": "system-design", "parents": [2]},
        {"title": "write the plan", "phase": "engineering-planning", "parents": [3]},
        {"title": "implement it", "phase": "implementation", "parents": [4]},
        {"title": "test it", "phase": "testing", "parents": [5]},
        {"title": "adoption + docs", "phase": None, "parents": [4]},
    ],
    # Skip-non-applicable: pure implementation work.
    "skipped": [
        {"title": "implement it", "phase": "implementation", "parents": []},
        {"title": "test it", "phase": "testing", "parents": [0]},
    ],
    # Single-card skip: an unphased one-off chore.
    "single": [
        {"title": "rotate the credential", "phase": None, "parents": []},
    ],
}


def _require_isolated_child(repo: Path) -> None:
    """Refuse a DB-writing child run that is not provably isolated.

    The isolation contract is: the parent creates a temp dir and re-execs
    ``--child`` with HERMES_HOME pointing into it. A bare ``--child``
    invocation (or one whose HERMES_HOME is the real ~/.hermes) would
    otherwise hit the live kanban DB — refuse loudly instead.

    Delegated to ``evals/kanban_child_guard.py``, which refuses on the
    **effective resolved DB path** rather than on env var names: an isolated
    HERMES_HOME with an inherited ``HERMES_KANBAN_DB`` (the standard env of
    every kanban worker session) resolves to the live board and is refused.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kanban_child_guard import require_isolated_child

    require_isolated_child(repo)


def _child(spec: dict) -> dict:
    return {"title": spec["title"], "body": f"phase: {spec['phase']}", "assignee": "default",
            "parents": spec["parents"], "phase": spec["phase"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if not args.child:
        with tempfile.TemporaryDirectory(prefix="kanban-phase-") as home:
            env = {"HOME": home, "HERMES_HOME": home + "/hermes", "PATH": os.defpath,
                   "LANG": "C.UTF-8", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1"}
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--repo", str(args.repo), "--child"],
                cwd=args.repo, env=env, check=False,
            ).returncode
    _require_isolated_child(args.repo)

    sys.path.insert(0, str(args.repo))
    from unittest.mock import MagicMock, patch

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_decompose as decomp
    from hermes_cli import kanban_phases as ph

    from types import SimpleNamespace

    fake_profiles = [
        SimpleNamespace(name=n, is_default=(n == "default"), description=f"desc {n}",
                        description_auto=False, model="m", provider="p", skill_count=1)
        for n in ("default",)
    ]

    def _fake_response(content: str):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        return resp

    report: dict = {"repo": str(args.repo), "chain": list(ph.PHASE_CHAIN)}
    kb.init_db()
    for name, specs in CASCADES.items():
        captured: dict = {}

        def _stub(**kwargs):
            captured.update(kwargs)
            return _fake_response(json.dumps({
                "fanout": True,
                "rationale": name,
                "tasks": [_child(s) for s in specs],
            }))

        with kbc.connect_closing() as conn:
            tid = kb.create_task(conn, title=f"probe {name}", triage=True)
        with patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles), \
                patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x == "default"), \
                patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
                patch("agent.auxiliary_client.call_llm", side_effect=_stub):
            outcome = decomp.decompose_task(tid, author="probe")

        assert outcome.ok, (name, outcome.reason)
        with kbc.connect_closing() as conn:
            rows = [kb.get_task(conn, cid) for cid in outcome.child_ids]
            # Chain integrity over the PERSISTED column, not the emitted field.
            links = conn.execute(
                "SELECT parent_id, child_id FROM task_links WHERE child_id IN ("
                + ",".join("?" * len(outcome.child_ids)) + ")",
                outcome.child_ids,
            ).fetchall()
        by_id = {r.id: r for r in rows}
        persisted = [r.current_step_key for r in rows]
        entry_by_id = {cid: {"phase": by_id[cid].current_step_key,
                             "parents": []} for cid in outcome.child_ids}
        # Rebuild the entry list in emitted order, mapping DB linkage back onto
        # the sibling indices the audit understands.
        order = {cid: idx for idx, cid in enumerate(outcome.child_ids)}
        for link in links:
            if link["parent_id"] in order and link["child_id"] in order:
                entry_by_id[link["child_id"]]["parents"].append(order[link["parent_id"]])
        entries = [entry_by_id[cid] for cid in outcome.child_ids]

        user_prompt = captured.get("messages", [{}, {"content": ""}])[1].get("content", "")
        report[name] = {
            "persisted_phases": persisted,
            "enum_clean": all(p is None or p in ph.PHASE_SET for p in persisted),
            "cascade_violations": ph.cascade_violations(entries),
            "prompt_carried_chain": ph.CHAIN_ARROW in user_prompt,
            "expected": [s["phase"] for s in specs],
        }
        print(json.dumps(report[name], sort_keys=True), flush=True)

    assert report["full"]["enum_clean"], report
    assert report["full"]["cascade_violations"] == [], report
    assert report["full"]["persisted_phases"] == [
        "grounding", "review", "research", "system-design",
        "engineering-planning", "implementation", "testing", None,
    ], report
    assert report["skipped"]["persisted_phases"] == ["implementation", "testing"], report
    assert report["skipped"]["cascade_violations"] == [], report
    assert report["single"]["persisted_phases"] == [None], report
    assert all(r["prompt_carried_chain"] for r in report.values() if isinstance(r, dict)), report
    print(json.dumps({"ok": True, "report": report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
