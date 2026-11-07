"""Offline tests for the 3-agent content pipeline: scripted fake models, no API key, no network."""
import json
import uuid

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from pr_agents import pipeline as P
from pr_agents.observability import TraceLogger

GOOD, BAD = P._GOOD, P._BAD
PASS = json.dumps({"score": 9, "passed": True, "issues": []})
FAIL = json.dumps({"score": 3, "passed": False, "issues": ["unsupported claim"]})
POSTER_OK = json.dumps({"passed": True, "issues": []})
POSTER_BAD = json.dumps({"passed": False, "issues": ["logo missing"]})
PLAN = json.dumps(P._PLAN)
POSTERS = ["data/posters/template_a.png", "data/posters/template_b.png"]


def fakes(writer, judge, vision=(POSTER_OK,), publisher=(PLAN,)):
    return {"writer": list(writer), "judge": list(judge), "vision": list(vision), "publisher": list(publisher)}


def run(f, category="fresher_ai", resume=None, posters=None, tracer=None):
    app = P.build_pipeline(fakes=f, tracer=tracer)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = app.invoke({"topic": "Fresher AI program is open", "category": category, "poster_candidates": posters or []}, cfg)
    if resume is not None and "__interrupt__" in state:
        state = app.invoke(Command(resume=resume), cfg)
    return state


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OUTPUT", tmp_path)
    for var in ("GOOGLE_API_KEY", "DATABASE_URL", "SHEET_CSV_URL"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_self_correction_then_approval_publishes_to_every_channel(offline):
    s = run(fakes([json.dumps(BAD), json.dumps(GOOD)], [FAIL, PASS]), resume="approve")
    assert s["attempts"] == 2 and s["status"].startswith("published (dry run) to 3 channels")
    log = json.loads((offline / "post_log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert {p["channel"] for p in log["posts"]} == set(P.CHANNELS)
    assert all("https://forms.example/fresher-ai" in p["text"] and "#FPTSoftware" in p["text"] for p in log["posts"])


def test_rules_catch_hallucinated_numbers_even_if_judge_passes():
    s = run(fakes([json.dumps(BAD)] * 3, [PASS] * 3))
    assert s["attempts"] == P.MAX_ATTEMPTS and s["status"].startswith("stopped: still failing")
    assert any("hallucination" in i for i in s["judge"]["issues"])


def test_invalid_json_is_retried_with_error_feedback():
    s = run(fakes(["not json at all", json.dumps(GOOD)], [PASS]), resume="approve")
    assert s["attempts"] == 2 and s["status"].startswith("published")


def test_missing_form_stops_before_any_llm_call():
    s = run(fakes([], [], [], []), category="campus_talk")
    assert s["status"] == "stopped: no registration form for this category" and "caption" not in s


def test_reviewer_reject_means_not_published(offline):
    s = run(fakes([json.dumps(GOOD)], [PASS]), resume="reject")
    assert s["status"].startswith("not published") and not (offline / "post_log.jsonl").exists()


def test_graph_pauses_for_human_before_publishing():
    s = run(fakes([json.dumps(GOOD)], [PASS]))
    assert "__interrupt__" in s and "status" not in s


def test_vision_auto_remediation_switches_to_next_template():
    s = run(fakes([json.dumps(GOOD)], [PASS], vision=[POSTER_BAD, POSTER_OK]), resume="approve", posters=POSTERS)
    qa = s["poster_qa"]
    assert qa["passed"] and qa["poster"].endswith("template_b.png") and [t["passed"] for t in qa["tried"]] == [False, True]
    assert all(p["poster"].endswith("template_b.png") for p in s["channel_posts"])


def test_all_posters_failing_blocks_publish():
    s = run(fakes([json.dumps(GOOD)], [PASS], vision=[POSTER_BAD, POSTER_BAD]), resume="approve", posters=POSTERS)
    assert not s["poster_qa"]["passed"] and s["status"].startswith("not published")


def test_publisher_version_that_breaks_rules_falls_back_to_approved_caption():
    bad_plan = json.dumps({"posts": [{"channel": "linkedin", "text": "Apply now, 999 seats! #AI"}]})
    s = run(fakes([json.dumps(GOOD)], [PASS], publisher=[bad_plan]), resume="approve")
    linkedin = next(p for p in s["channel_posts"] if p["channel"] == "linkedin")
    assert "999" not in linkedin["text"] and GOOD["call_to_action"] in linkedin["text"]


def test_failover_uses_backup_when_primary_errors():
    def boom(_):
        raise RuntimeError("503 model overloaded")
    backup = P.FakeListChatModel(responses=["backup answered"])
    assert RunnableLambda(boom).with_fallbacks([backup]).invoke("hi").content == "backup answered"


def test_knowledge_includes_sheet_rows_and_docs():
    sources = {d.metadata["source"] for d in P.load_facts()}
    assert {"brand_facts.md", "fresher_ai_brief.md", "content_timeline"} <= sources
    assert P.keyword_rank("Fresher AI program training LangGraph", P.load_facts(), k=1)[0].metadata["section"].startswith("Fresher AI")


def test_input_from_sheet_row():
    inp = P.input_from_row("1")
    assert inp["category"] == "fresher_ai" and len(inp["poster_candidates"]) == 2


def test_every_agent_step_is_traced(offline):
    tracer = TraceLogger(offline / "traces.jsonl")
    run(fakes([json.dumps(GOOD)], [PASS]), resume="approve", tracer=tracer)
    runs = {json.loads(l)["run"] for l in (offline / "traces.jsonl").read_text(encoding="utf-8").splitlines()}
    assert {"writer_agent", "judge_agent", "publisher_agent", "llm"} <= runs


def test_writer_uses_imported_n8n_system_prompt(tmp_path, monkeypatch):
    (tmp_path / "group_writer.md").write_text("# House rules {{ $json.x }}\nViết ngắn gọn.", encoding="utf-8")
    monkeypatch.setattr(P, "PROMPTS", tmp_path)
    msgs = P.writer_prompt().format_messages(format_instructions="JSON", topic="t", form_url="u", context="c", feedback="none")
    assert msgs[0].content.startswith("# House rules {{ $json.x }}") and msgs[0].content.endswith("JSON")


def test_defaults_match_n8n_models():
    assert (P.PRIMARY_DEFAULT, P.BACKUP_DEFAULT) == ("gemini-3.5-flash", "gemini-3.5-flash-lite")
