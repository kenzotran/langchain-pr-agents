"""REST API (FastAPI) for both agents.

    GET  /health                       liveness check (also used by Docker healthcheck)
    GET  /messenger/webhook            Meta verification handshake (hub.challenge)
    POST /messenger/webhook            incoming Messenger messages -> agent reply -> Send API
    POST /posts                        start a content run (topic+category or a sheet row_id); returns the draft awaiting approval
    POST /posts/{thread_id}/decision   approve / reject -> resumes the paused graph and publishes

Run:  uvicorn pr_agents.api:app --port 8000
"""
from __future__ import annotations

import logging
import os
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from langgraph.types import Command
from pydantic import BaseModel

from pr_agents import messenger_agent as M
from pr_agents import pipeline as P

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("pr_agents.api")
app = FastAPI(title="langchain-pr-agents", version="1.0")
GRAPH_URL = "https://graph.facebook.com/v21.0/me/messages"
FALLBACK_REPLY = "Sorry, I can't answer right now. A recruiter will follow up with you soon."
_state: dict = {}


def pipeline_app():
    if "pipeline" not in _state:
        _state["pipeline"] = P.build_pipeline(checkpointer=M.make_checkpointer(P.OUTPUT / "pipeline_state.db"))
    return _state["pipeline"]


def messenger_app():
    if "messenger" not in _state:
        _state["messenger"] = M.build_agent()
    return _state["messenger"]


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0, rid = time.perf_counter(), uuid.uuid4().hex[:8]
    response = await call_next(request)
    log.info("request id=%s %s %s -> %s in %dms", rid, request.method, request.url.path,
             response.status_code, (time.perf_counter() - t0) * 1000)
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/health")
def health():
    return {"status": "ok", "llm": "gemini" if os.getenv("GOOGLE_API_KEY") else "offline-fake",
            "storage": "postgres" if os.getenv("DATABASE_URL") else "local"}


# ---------------------------------------------------------------- Messenger
@app.get("/messenger/webhook", response_class=PlainTextResponse)
def verify(mode: str = Query(alias="hub.mode"), token: str = Query(alias="hub.verify_token"),
           challenge: str = Query(alias="hub.challenge")):
    if mode == "subscribe" and token and token == os.getenv("VERIFY_TOKEN"):
        return challenge
    raise HTTPException(status_code=403, detail="verification failed")


def send_to_messenger(recipient: str, text: str) -> bool:
    token = os.getenv("PAGE_ACCESS_TOKEN")
    if not token:
        return False                                  # dry run: no token configured
    r = httpx.post(GRAPH_URL, params={"access_token": token}, timeout=15,
                   json={"recipient": {"id": recipient}, "messaging_type": "RESPONSE", "message": {"text": text[:2000]}})
    r.raise_for_status()
    return True


@app.post("/messenger/webhook")
async def incoming(request: Request):
    body = await request.json()
    replies = []
    for entry in body.get("entry", []):
        for event in entry.get("messaging", []):
            sender, text = event.get("sender", {}).get("id"), event.get("message", {}).get("text")
            if not sender or not text:                # delivery receipts, stickers, etc.
                continue
            try:
                answer = M.reply(messenger_app(), sender, text)
            except Exception:                         # never crash the webhook: Meta retries on non-200
                log.exception("agent failed for sender=%s", sender)
                answer = FALLBACK_REPLY
            try:
                sent = send_to_messenger(sender, answer)
            except httpx.HTTPError:
                log.exception("Send API failed for sender=%s", sender)
                sent = False
            replies.append({"sender": sender, "reply": answer, "sent": sent})
    return {"status": "ok", "replies": replies}       # always 200 so Meta does not re-deliver


# ---------------------------------------------------------------- content pipeline
class PostRequest(BaseModel):
    topic: str | None = None
    category: str | None = None
    row_id: str | None = None
    poster_candidates: list[str] = []


class Decision(BaseModel):
    decision: str                                     # "approve" or "reject"


def _view(thread_id: str, state: dict) -> dict:
    waiting = "__interrupt__" in state
    return {"thread_id": thread_id, "awaiting_approval": waiting,
            "draft": state["__interrupt__"][0].value if waiting else None,
            "status": state.get("status"), "attempts": state.get("attempts"), "channel_posts": state.get("channel_posts")}


@app.post("/posts")
def start_post(req: PostRequest):
    try:
        inp = P.input_from_row(req.row_id) if req.row_id else {"topic": req.topic, "category": req.category,
                                                              "poster_candidates": req.poster_candidates}
    except StopIteration:
        raise HTTPException(status_code=404, detail=f"row {req.row_id} not found in the content sheet")
    if not inp.get("topic") or not inp.get("category"):
        raise HTTPException(status_code=422, detail="give topic + category, or a row_id")
    thread_id = uuid.uuid4().hex
    try:
        state = pipeline_app().invoke(inp, {"configurable": {"thread_id": thread_id}})
    except Exception as e:
        log.exception("pipeline failed")
        raise HTTPException(status_code=502, detail=f"pipeline error: {e}")
    return _view(thread_id, state)


@app.post("/posts/{thread_id}/decision")
def decide(thread_id: str, body: Decision):
    if body.decision not in ("approve", "reject"):
        raise HTTPException(status_code=422, detail="decision must be 'approve' or 'reject'")
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshot = pipeline_app().get_state(cfg)
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="nothing is waiting for approval on this thread")
    state = pipeline_app().invoke(Command(resume=body.decision), cfg)
    return _view(thread_id, state)
