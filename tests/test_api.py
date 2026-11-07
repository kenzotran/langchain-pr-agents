"""REST API tests (FastAPI TestClient, offline models, temp storage)."""
import pytest
from fastapi.testclient import TestClient

from pr_agents import api, pipeline as P


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OUTPUT", tmp_path)
    for var in ("GOOGLE_API_KEY", "DATABASE_URL", "PAGE_ACCESS_TOKEN", "SHEET_CSV_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VERIFY_TOKEN", "test-token")
    api._state.clear()
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "X-Request-ID" in r.headers


def test_meta_webhook_verification(client):
    ok = client.get("/messenger/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "test-token", "hub.challenge": "42"})
    bad = client.get("/messenger/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "42"})
    assert ok.status_code == 200 and ok.text == "42" and bad.status_code == 403


def test_messenger_message_gets_grounded_reply(client):
    payload = {"object": "page", "entry": [{"messaging": [
        {"sender": {"id": "u1"}, "message": {"text": "Do I need a computer science degree?"}},
        {"sender": {"id": "u1"}, "delivery": {"mids": ["x"]}}]}]}          # receipts are ignored
    r = client.post("/messenger/webhook", json=payload)
    replies = r.json()["replies"]
    assert r.status_code == 200 and len(replies) == 1
    assert "any major" in replies[0]["reply"] and replies[0]["sent"] is False   # dry run without a page token


def test_webhook_returns_200_even_if_agent_crashes(client, monkeypatch):
    monkeypatch.setattr(api.M, "reply", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    r = client.post("/messenger/webhook", json={"entry": [{"messaging": [{"sender": {"id": "u"}, "message": {"text": "hi"}}]}]})
    assert r.status_code == 200 and r.json()["replies"][0]["reply"] == api.FALLBACK_REPLY


def test_post_flow_from_sheet_row_then_approve(client):
    r = client.post("/posts", json={"row_id": "1"})
    body = r.json()
    assert r.status_code == 200 and body["awaiting_approval"] and body["draft"]["poster_qa"]["passed"]
    done = client.post(f"/posts/{body['thread_id']}/decision", json={"decision": "approve"}).json()
    assert done["status"].startswith("published") and len(done["channel_posts"]) == 3
    again = client.post(f"/posts/{body['thread_id']}/decision", json={"decision": "approve"})
    assert again.status_code == 409                                    # cannot publish twice


def test_post_validation_errors(client):
    assert client.post("/posts", json={"row_id": "999"}).status_code == 404
    assert client.post("/posts", json={"topic": "x"}).status_code == 422
    assert client.post("/posts/abc/decision", json={"decision": "maybe"}).status_code == 422
