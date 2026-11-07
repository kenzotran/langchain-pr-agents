"""24/7 Messenger-style support agent: agentic RAG + tool calling + persistent memory (LangGraph).

    user message ─► agent (LLM decides) ──tool call──► tools: search_faq (RAG) | get_open_positions (live data)
                       ▲                                   │
                       └──────────── tool results ◄────────┘
                       │ final answer
                       ▼
    memory: every turn is checkpointed per user (thread_id = Messenger sender id)
            SQLite file by default, PostgreSQL when DATABASE_URL is set -> survives restarts.
    knowledge: FAQ embeddings in a vector store, PostgreSQL + pgvector when DATABASE_URL is set.

The agent decides per message WHETHER to retrieve and WHICH tool to use (that is what makes it
"agentic" RAG). The step budget (recursion_limit) stops tool loops.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from pr_agents.observability import callbacks
from pr_agents.pipeline import BACKUP_DEFAULT, DATA, OUTPUT, PRIMARY_DEFAULT, _gemini, keyword_rank, make_vector_store

HISTORY_WINDOW = 20      # messages sent to the model each turn (full history stays in the checkpoint)
STEP_BUDGET = 8          # max graph steps per user message (prevents infinite tool loops)
SYSTEM = SystemMessage(
    "You are FPT Software's recruitment assistant on Messenger. Answer in the user's language, briefly and kindly. "
    "For questions about the program, ALWAYS call search_faq; for current openings or deadlines, call get_open_positions. "
    "Answer ONLY from tool results or the conversation. If the answer is not there, say you are not sure and that a "
    "recruiter will follow up. Never invent dates, salaries or benefits.")


def _faq_docs() -> list[Document]:
    with open(DATA / "faq.csv", encoding="utf-8", newline="") as f:
        return [Document(page_content=f"Q: {r['question']}\nA: {r['answer']}") for r in csv.DictReader(f)]


def _faq_store():
    """FAQ embeddings in PostgreSQL + pgvector when DATABASE_URL is set, otherwise in memory."""
    return make_vector_store(_faq_docs(), "messenger_faq")


_STORE = None


@tool
def search_faq(query: str) -> str:
    """Search the recruitment FAQ knowledge base. Use for any question about the program, test, training or offices."""
    global _STORE
    _STORE = _STORE or _faq_store()
    hits = _STORE.similarity_search(query, k=3) + _faq_docs()         # vector hits + lexical re-rank (hybrid)
    best = keyword_rank(query, list({d.page_content: d for d in hits}.values()), k=2)
    return "\n\n".join(d.page_content for d in best)


@tool
def get_open_positions(city: str = "") -> str:
    """Get CURRENT open positions and application deadlines (live data). Optionally filter by city."""
    jobs = json.loads((DATA / "open_positions.json").read_text(encoding="utf-8"))
    jobs = [j for j in jobs if city.lower() in j["city"].lower()] if city else jobs
    return json.dumps(jobs, ensure_ascii=False) if jobs else "No open positions found."


TOOLS = [search_faq, get_open_positions]


class _ScriptedToolModel(GenericFakeChatModel):
    """Offline stand-in for a tool-calling chat model (returns pre-written AIMessages)."""
    def bind_tools(self, tools, **kwargs):
        return self


class _OfflineRagModel(GenericFakeChatModel):
    """Offline demo brain: always searches the FAQ first, then answers with the best FAQ answer."""
    messages: object = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        last = messages[-1]
        if isinstance(last, HumanMessage):
            msg = AIMessage(content="", tool_calls=[{"name": "search_faq", "args": {"query": last.content}, "id": "offline-1"}])
        else:
            best = str(last.content).split("\n\n")[0]
            msg = AIMessage(content=best.split("A: ", 1)[-1] if "A: " in best else "I'm not sure. A recruiter will follow up with you.")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def make_model(script=None):
    if script is not None:
        return _ScriptedToolModel(messages=iter(script))
    if not os.getenv("GOOGLE_API_KEY"):
        return _OfflineRagModel()
    primary = _gemini(os.getenv("PRIMARY_MODEL", PRIMARY_DEFAULT)).bind_tools(TOOLS)
    backup = _gemini(os.getenv("BACKUP_MODEL", BACKUP_DEFAULT)).bind_tools(TOOLS)
    return primary.with_fallbacks([backup])


def make_checkpointer(path: Path | None = None):
    """PostgreSQL when DATABASE_URL is set (pip install langgraph-checkpoint-postgres), else a local SQLite file."""
    url = os.getenv("DATABASE_URL")
    if url:
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        # long-lived connection for a server process (from_conn_string is a context manager meant for scripts)
        conn = psycopg.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row)
        saver = PostgresSaver(conn)
        saver.setup()                                   # creates checkpoint tables on first run
        return saver
    from langgraph.checkpoint.sqlite import SqliteSaver
    path = path or OUTPUT / "messenger_memory.db"
    path.parent.mkdir(exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def _window(messages):
    recent = messages[-HISTORY_WINDOW:]
    while recent and not isinstance(recent[0], HumanMessage):   # never start mid tool-call
        recent = recent[1:]
    return recent


def build_agent(model=None, checkpointer=None):
    model = model or make_model()

    def agent(state: MessagesState):
        return {"messages": [model.invoke([SYSTEM, *_window(state["messages"])])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)            # tool call -> tools, else -> END
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or make_checkpointer())


def reply(app, sender_id: str, text: str) -> str:
    cfg = {"configurable": {"thread_id": f"messenger:{sender_id}"}, "recursion_limit": STEP_BUDGET,
           "callbacks": callbacks(OUTPUT / "traces.jsonl")}
    out = app.invoke({"messages": [HumanMessage(text)]}, cfg)
    return out["messages"][-1].content
