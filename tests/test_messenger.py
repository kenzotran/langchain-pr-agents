"""Messenger agent tests: tool routing, grounding, persistent memory across restarts, step budget."""
import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from pr_agents import messenger_agent as M


def call(name, args, cid="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)


def test_agent_calls_faq_tool_then_answers_from_it(tmp_path):
    model = M.make_model([call("search_faq", {"query": "Do I need a computer science degree?"}),
                          AIMessage(content="No, any major can apply if you pass the entry test.")])
    app = M.build_agent(model, M.make_checkpointer(tmp_path / "m.db"))
    out = app.invoke({"messages": [("user", "Do I need a CS degree?")]}, {"configurable": {"thread_id": "u1"}})
    tool_msg = [m for m in out["messages"] if m.type == "tool"][0]
    assert "any major" in tool_msg.content and out["messages"][-1].content.startswith("No")


def test_live_data_tool_filters_by_city():
    hcm = M.get_open_positions.invoke({"city": "Ho Chi Minh"})
    assert "Fresher AI Engineer" in hcm and "Hanoi" not in hcm


def test_faq_retrieval_returns_the_relevant_answer():
    assert "entry test covers" in M.search_faq.invoke({"query": "what does the entry test cover"})


def test_memory_persists_across_restarts(tmp_path):
    db = tmp_path / "memory.db"
    app1 = M.build_agent(M.make_model([AIMessage(content="Nice to meet you, Lan!")]), M.make_checkpointer(db))
    M.reply(app1, "user42", "Hi, my name is Lan")
    app2 = M.build_agent(M.make_model([AIMessage(content="ok")]), M.make_checkpointer(db))   # simulated restart
    history = app2.get_state({"configurable": {"thread_id": "messenger:user42"}}).values["messages"]
    assert history[0].content == "Hi, my name is Lan" and len(history) == 2


def test_users_do_not_share_memory(tmp_path):
    app = M.build_agent(M.make_model([AIMessage(content="a"), AIMessage(content="b")]), M.make_checkpointer(tmp_path / "m.db"))
    M.reply(app, "alice", "I am Alice")
    M.reply(app, "bob", "I am Bob")
    bob = app.get_state({"configurable": {"thread_id": "messenger:bob"}}).values["messages"]
    assert all("Alice" not in m.content for m in bob)


def test_step_budget_stops_endless_tool_loops(tmp_path):
    loop = [call("search_faq", {"query": "x"}, cid=f"c{i}") for i in range(20)]
    app = M.build_agent(M.make_model(loop), M.make_checkpointer(tmp_path / "m.db"))
    with pytest.raises(GraphRecursionError):
        M.reply(app, "u", "loop please")
