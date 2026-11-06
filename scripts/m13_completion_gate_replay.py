#!/usr/bin/env python3
"""Replay the M13 shadow completion gate over a live board — READ ONLY.

Answers one question for the 14-day shadow evaluation (`t_c06dd31a` acceptance):
*what would the gate have recorded over the last N days, and how often would it have
said `would_refuse`?*

Run it before enabling shadow and again at the end of the window. The pre-registered
promotion thresholds (`M13_decision_packet.md` §6) are printed with the result so the
verdict is never computed by hand from a re-tuned rule:

    >=150 evaluations, >=25 would_refuse hits, >=26/30 precision on a 30-card
    hand audit, exactly 0 exempt-class hits, <=10 s median added latency,
    <=66 crashes/11d

The DB is opened ``mode=ro``: this script cannot mutate the board.

Two evidence scopes are reported, because they differ by an order of magnitude:

``workspace``
    Literal M13 wording: the cited sha must resolve into the card's **own workspace
    tree**. A managed ``scratch`` card's workspace is normally not a git tree, and
    ``_cleanup_workspace`` deletes it at completion, so this scope has almost no
    coverage.
``card``
    The scoped default the gate ships: workspace tree, plus a repo path the worker
    declared in completion metadata, plus an absolute path named in the completion
    payload. Still card-scoped — never a host-wide repo search, which is what makes a
    single-HEAD rule refuse the wrong card on a host with 13 ``hermes_cli`` checkouts.

Usage:
    python3 scripts/m13_completion_gate_replay.py [--days 11] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hermes_cli import kanban_completion_gate as gate

#: Pre-registered promotion thresholds, verbatim from the M13 packet §6.
PROMOTION_THRESHOLDS = {
    "evaluations": 150,
    "would_refuse_hits": 25,
    "audit_precision": "26/30",
    "exempt_class_hits": 0,
    "median_added_latency_s": 10,
    "crashes_per_11d": 66,
}


def _board_path() -> str:
    explicit = os.environ.get("HERMES_KANBAN_DB")
    if explicit:
        return explicit
    home = os.environ.get("HERMES_KANBAN_HOME") or os.path.join(
        os.path.expanduser("~"), ".hermes"
    )
    return os.path.join(home, "kanban.db")


def _latest_run(conn, task_id: str) -> tuple[str | None, str | None]:
    row = conn.execute(
        "SELECT summary, metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return None, None
    return row["summary"], row["metadata"]


def _was_review_approval(conn, task_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'review_requested' LIMIT 1",
        (task_id,),
    ).fetchone() is not None


def _load(conn, days: int) -> list[dict]:
    now = int(time.time())
    since = now - days * 86400
    rows = conn.execute(
        """
        SELECT id, assignee, title, result, completed_at, workspace_kind, workspace_path
          FROM tasks
         WHERE status = 'done' AND completed_at >= ?
         ORDER BY completed_at
        """,
        (since,),
    ).fetchall()
    cards = []
    for row in rows:
        summary, raw_metadata = _latest_run(conn, row["id"])
        try:
            metadata = json.loads(raw_metadata) if raw_metadata else {}
        except Exception:
            metadata = {}
        cards.append(
            {
                "id": row["id"],
                "assignee": row["assignee"],
                "title": row["title"],
                "result": row["result"],
                "summary": summary,
                "metadata": metadata if isinstance(metadata, dict) else {},
                "workspace_path": row["workspace_path"],
                "workspace_kind": row["workspace_kind"],
                "completed_at": row["completed_at"],
                "review_approval": _was_review_approval(conn, row["id"]),
            }
        )
    return cards, since, now


def _replay(cards: list[dict], scope: str) -> dict:
    decisions = []
    for card in cards:
        decision = gate.evaluate_completion(
            prior_status="review" if card["review_approval"] else "running",
            workspace_path=card["workspace_path"],
            result=card["result"],
            summary=card["summary"],
            metadata=card["metadata"],
            mode="shadow",
            domain_scope=scope,
        )
        decisions.append((card, decision))
    return decisions


def _summarise(
    cards: list[dict], decisions: list[tuple[dict, gate.GateDecision]], days: int
) -> dict:
    exempt = Counter()
    reasons = Counter()
    latencies = []
    refusals = []
    errors = 0
    for card, decision in decisions:
        latencies.append(decision.elapsed_ms)
        reasons[decision.reason] += 1
        if decision.error:
            errors += 1
        if decision.exempt_class:
            exempt[decision.exempt_class] += 1
        if decision.would_refuse:
            refusals.append(
                {
                    "task_id": card["id"],
                    "assignee": card["assignee"],
                    "exempt_class": decision.exempt_class,
                    "sha_candidates": decision.sha_candidates,
                    "domains": decision.resolution_domains,
                    "reason": decision.reason,
                    "title": (card["title"] or "")[:80],
                }
            )
    n = len(decisions)
    hits = len(refusals)
    return {
        "cards": n,
        "evaluated": sum(1 for _, d in decisions if d.evaluated),
        "errors": errors,
        "exempt": dict(exempt),
        "exempt_total": sum(exempt.values()),
        "allowed": n - sum(exempt.values()) - hits,
        "would_refuse": hits,
        "would_refuse_share": round(hits / n, 4) if n else 0.0,
        "per_week": round(hits / (days / 7), 2) if n and days else 0.0,
        "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": (
            round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2)
            if latencies
            else 0.0
        ),
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        "reasons": dict(reasons.most_common(12)),
        "refusals": refusals,
    }


def _host_attribution(cards: list[dict], repos: list[str]) -> dict:
    """DIAGNOSTIC ONLY — what a *host-wide* attribution would have said.

    The gate never does this (§7: a single-HEAD rule refuses the wrong cards on a host
    with 13 ``hermes_cli`` checkouts). It is reported so the replay can state the size
    of the gap R2 is aimed at, separately from what the shipped rule can attribute per
    card. Repos are named explicitly on the command line; nothing is discovered.
    """
    import subprocess

    def git(repo: str, *args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["git", "-C", repo, *args], capture_output=True, text=True, timeout=8, check=False
            )
        except Exception:
            class _Failed:
                returncode = 1
            return _Failed()  # type: ignore[return-value]

    considered = resolved = 0
    unresolved: list[str] = []
    for card in cards:
        text = gate._capped_text(card["summary"], card["result"])
        shas = gate._dedup_capped(
            gate._metadata_shas(card["metadata"]) + gate._text_shas(text),
            gate._MAX_SHA_CANDIDATES,
        )
        if not shas:
            continue
        considered += 1
        hit = False
        for sha in shas:
            for repo in repos:
                if git(repo, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}").returncode != 0:
                    continue
                if git(repo, "merge-base", "--is-ancestor", sha, "HEAD").returncode == 0:
                    hit = True
                    break
            if hit:
                break
        if hit:
            resolved += 1
        else:
            unresolved.append(card["id"])
    return {
        "cards_with_sha_claims": considered,
        "resolved_in_a_named_host_repo": resolved,
        "unresolved": unresolved,
        "repos": repos,
    }


def _print_table(days: int, since: int, now: int, card_scoped: dict, workspace_only: dict) -> None:
    window = f"{days}d  [{time.strftime('%Y-%m-%d %H:%M', time.localtime(since))} .. " \
             f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}]"
    print(f"M13 completion gate — shadow replay, window {window}")
    print(f"board: {_board_path()} (read-only)")
    print()
    header = f"{'scope':<12}{'N':>5}{'eval':>6}{'exempt':>8}{'allow':>7}{'would_ref':>11}{'share':>8}{'median ms':>11}"
    print(header)
    print("-" * len(header))
    for label, result in (("card", card_scoped), ("workspace", workspace_only)):
        print(
            f"{label:<12}{result['cards']:>5}{result['evaluated']:>6}"
            f"{result['exempt_total']:>8}{result['allowed']:>7}"
            f"{result['would_refuse']:>11}{result['would_refuse_share'] * 100:>7.1f}%"
            f"{result['median_latency_ms']:>11.2f}"
        )
    print()
    print("exempt-class breakdown (card scope):")
    for klass, count in sorted(card_scoped["exempt"].items()):
        print(f"  {klass:<28}{count:>5}")
    print()
    print("would_refuse cards (card scope):")
    if not card_scoped["refusals"]:
        print("  (none)")
    for item in card_scoped["refusals"]:
        print(
            f"  {item['task_id']:<14}{item['assignee'] or '-':<12}"
            f"shas={item['sha_candidates'] or '-'}  {item['reason'][:60]}"
        )
    print()
    print("pre-registered promotion thresholds (M13 packet §6):")
    for key, value in PROMOTION_THRESHOLDS.items():
        print(f"  {key:<28}{value}")
    print()
    print(f"gate errors: {card_scoped['errors']} (an error is recorded as a completion_gate_error "
          "event and never blocks the completion)")


def _print_host_diagnostic(diag: dict) -> None:
    print()
    print("DIAGNOSTIC (NOT SHIPPED) — host-wide sha attribution against named repos:")
    for repo in diag["repos"]:
        print(f"  repo: {repo}")
    print(f"  cards citing a sha:      {diag['cards_with_sha_claims']}")
    print(f"  sha resolves somewhere:  {diag['resolved_in_a_named_host_repo']}")
    print(f"  sha resolves nowhere:    {len(diag['unresolved'])}  {diag['unresolved']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=11)
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument(
        "--host-repo", dest="host_repos", action="append", default=[],
        metavar="PATH",
        help="diagnostic only: also report whether each cited sha resolves in this repo",
    )
    args = parser.parse_args(argv)

    board = _board_path()
    if not os.path.exists(board):
        print(f"board not found: {board}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(f"file:{board}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cards, since, now = _load(conn, args.days)
    finally:
        conn.close()

    card_scoped = _summarise(cards, _replay(cards, "card"), args.days)
    workspace_only = _summarise(cards, _replay(cards, "workspace"), args.days)
    _print_table(args.days, since, now, card_scoped, workspace_only)

    diag = None
    if args.host_repos:
        diag = _host_attribution(cards, [os.path.abspath(p) for p in args.host_repos])
        _print_host_diagnostic(diag)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "window_days": args.days,
                    "since": since,
                    "until": now,
                    "board": board,
                    "card_scope": card_scoped,
                    "workspace_scope": workspace_only,
                    "host_attribution_diagnostic": diag,
                    "promotion_thresholds": PROMOTION_THRESHOLDS,
                },
                handle,
                indent=2,
            )
        print(f"\njson -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
