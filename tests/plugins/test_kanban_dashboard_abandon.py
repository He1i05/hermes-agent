"""Dashboard plugin: the explain payload and the ``POST /tasks/{id}/abandon`` verb.

The board's whole job here is to tell the operator WHY a card sits where it sits and
what they can do about it, so these tests assert the API surface the UI reads:
``explain`` on every card and on the drawer, and an abandon endpoint whose refusal
names the safeguard instead of returning a bare 409.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_graph as kbg

OWNER_LOSS = (
    "pid 9 exited with code 1 Worker's last output: "
    "'Kanban owner refused: Gateway connection failed; no local fallback'"
)
REASON = "host killed the whole frontier on 2026-10-24"


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_abandon_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _conn():
    from hermes_cli import kanban_db_connect as kbc

    return kbc.connect_closing()


def _book(conn, task_id, *, error=None, metadata=None, status="blocked", outcome="crashed"):
    now = int(time.time())
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, error, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (task_id, outcome, now - 30, now, outcome, error, json.dumps(metadata or {})),
    )
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()


def _tree(conn, *, infra=2, card=0):
    root = kb.create_task(conn, title="root quest", triage=True)
    specs = ([{"title": f"infra {i}", "assignee": "builder"} for i in range(infra)]
             + [{"title": f"card {i}", "assignee": "builder"} for i in range(card)])
    kids = kbg.decompose_triage_task(conn, root, root_assignee="builder", children=specs)
    infra_ids, card_ids = kids[:infra], kids[infra:]
    for cid in infra_ids:
        _book(conn, cid, error=OWNER_LOSS, metadata={"worker_output": OWNER_LOSS})
    for cid in card_ids:
        _book(conn, cid, error="protocol violation: no handoff", metadata={"protocol_violation": True})
    return root, infra_ids, card_ids


def _card(client, task_id):
    board = client.get("/api/plugins/kanban/board").json()
    for column in board["columns"]:
        for card in column["tasks"]:
            if card["id"] == task_id:
                return card
    raise AssertionError(f"{task_id} not on the board")


def test_board_cards_carry_the_explain_payload(env):
    client = env
    with _conn() as conn:
        root, infra_ids, _ = _tree(conn, infra=2)

    card = _card(client, root)
    explain = card["explain"]
    assert explain["failure_class"] is None
    assert explain["abandon"]["eligible"] is True
    assert sorted(explain["abandon"]["scope"]) == sorted(infra_ids)
    assert explain["abandon"]["command"].startswith(f"hermes kanban abandon {root}")
    assert explain["retry_budget"]["limit"] >= 1
    assert explain["cooldown"]["kind"] is None
    assert explain["classifier"]["deployed"] is True

    child = _card(client, infra_ids[0])
    assert child["explain"]["failure_class"] == "infrastructure"
    assert child["explain"]["owner_loss"]["variant"] == "gateway_connection_failed"
    assert child["explain"]["abandon"]["root"] == root


def test_task_endpoint_carries_the_explain_payload_and_diagnostics(env):
    client = env
    with _conn() as conn:
        _, infra_ids, _ = _tree(conn, infra=1)
    payload = client.get(f"/api/plugins/kanban/tasks/{infra_ids[0]}").json()
    explain = payload["task"]["explain"]
    assert explain["failure_class"] == "infrastructure"
    assert explain["failure_evidence"]["source"] == "error_text"
    kinds = {d["kind"] for d in payload["task"].get("diagnostics", [])}
    assert "infrastructure_parked" in kinds


def test_abandon_endpoint_archives_the_infra_frontier(env):
    client = env
    with _conn() as conn:
        root, infra_ids, _ = _tree(conn, infra=2)
    response = client.post(f"/api/plugins/kanban/tasks/{root}/abandon", json={"reason": REASON})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert sorted(body["archived"]) == sorted([root, *infra_ids])
    with _conn() as conn:
        assert kb.get_task(conn, root).status == "archived"
        for cid in infra_ids:
            task = kb.get_task(conn, cid)
            assert task.status == "archived"
            assert task.abandon_reason == REASON


def test_abandon_endpoint_refuses_with_the_safeguard_named(env):
    client = env
    with _conn() as conn:
        root, infra_ids, card_ids = _tree(conn, infra=1, card=1)
    response = client.post(f"/api/plugins/kanban/tasks/{root}/abandon", json={"reason": REASON})
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["blocked_by"] == [f"card_class_descendant:{card_ids[0]}"]
    assert detail["abandon"]["eligible"] is False
    with _conn() as conn:
        assert kb.get_task(conn, root).status == "todo"
        assert kb.get_task(conn, infra_ids[0]).status == "blocked"


def test_abandon_endpoint_dry_run_writes_nothing(env):
    client = env
    with _conn() as conn:
        root, _infra_ids, _ = _tree(conn, infra=1)
    response = client.post(f"/api/plugins/kanban/tasks/{root}/abandon",
                           json={"reason": REASON, "dry_run": True})
    assert response.status_code == 200, response.text
    assert response.json()["dry_run"] is True
    with _conn() as conn:
        assert kb.get_task(conn, root).status == "todo"


def test_abandon_endpoint_requires_a_reason(env):
    client = env
    with _conn() as conn:
        root, _, _ = _tree(conn, infra=1)
    response = client.post(f"/api/plugins/kanban/tasks/{root}/abandon", json={"reason": " "})
    assert response.status_code == 400, response.text


def test_open_parent_refusal_names_the_real_root_and_the_abandon_verb(env):
    """The 409 an operator hits on a fan-in root must name the escape hatch."""
    client = env
    with _conn() as conn:
        root, _infra_ids, _ = _tree(conn, infra=2)
    response = client.patch(f"/api/plugins/kanban/tasks/{root}", json={"status": "done"})
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert "unsatisfied parent dependencies" in detail
    assert f"hermes kanban abandon {root}" in detail
