"""Multi-agent PR content pipeline built with LangChain + LangGraph.

Three LLM agents: WRITER (generate), JUDGE (evaluate) and PUBLISHER (adapt per channel), plus a
multimodal Gemini Vision QA step with auto-remediation and a human approval gate.

    content row / topic
      ─► map_form ─► retrieve (RAG over Sheets + Docs + brand facts) ─► write ─► check (rules + LLM judge)
            │ no form: STOP                      ▲                                   │
                                                 └──── feedback, max 3 attempts ─────┘ (else → needs_human)
                                                                                     │ pass
      vision_qa: Gemini Vision checks each poster candidate; a failed poster is auto-remediated by
                 switching to the next template candidate and re-checking (max = number of candidates)
            ─► human_review (LangGraph interrupt: approve / reject)
            ─► publish: PUBLISHER agent rewrites the approved post for each channel, rules re-check each
                        version, then posts (dry run by default) and logs

Every LLM call goes through a primary model with an automatic fallback (`.with_fallbacks`), and every
call is traced (latency, tokens, errors) by `observability.TraceLogger`. Without GOOGLE_API_KEY the
pipeline runs on scripted fake models, so tests and demos work offline.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from pr_agents.observability import TraceLogger

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
MAX_ATTEMPTS = 3          # self-correction budget before a human must step in
MAX_CAPTION_CHARS = 1200
REQUIRED_HASHTAG = "#FPTSoftware"
CHANNELS = ("facebook_group", "facebook_page", "linkedin")


# ---------------------------------------------------------------- schemas (structured output)
class Caption(BaseModel):
    title: str = Field(description="Hook line, max 12 words")
    body: str = Field(description="Post body grounded ONLY in the provided facts")
    call_to_action: str = Field(description="One sentence telling readers what to do, must include the form link")
    hashtags: list[str] = Field(description="3-6 hashtags, must include #FPTSoftware")


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall quality 1-10")
    passed: bool = Field(description="True only if on-brand, factual vs context, and clear")
    issues: list[str] = Field(default_factory=list, description="Concrete problems to fix")


class PosterVerdict(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)


class ChannelPost(BaseModel):
    channel: Literal["facebook_group", "facebook_page", "linkedin"]
    text: str = Field(description="Ready-to-post text for this channel, including the form link and hashtags")


class ChannelPlan(BaseModel):
    posts: list[ChannelPost]


# ---------------------------------------------------------------- models with failover
def _gemini(model: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0.4, timeout=60, max_retries=2)


def make_llm(role: str, fakes: dict[str, list[str]] | None = None):
    """Primary model + fallback model. Real Gemini when GOOGLE_API_KEY is set, scripted fakes otherwise."""
    if fakes is not None or not os.getenv("GOOGLE_API_KEY"):
        script = (fakes or DEMO_SCRIPT)[role]
        return FakeListChatModel(responses=script)
    primary = _gemini(os.getenv("PRIMARY_MODEL", PRIMARY_DEFAULT))
    backup = _gemini(os.getenv("BACKUP_MODEL", BACKUP_DEFAULT))
    return primary.with_fallbacks([backup])     # graceful degradation: 429/503 on primary -> backup


# Same models as the n8n production agents (primary Gemini 3.5 Flash, backup Flash-Lite); override in .env.
PRIMARY_DEFAULT, BACKUP_DEFAULT = "gemini-3.5-flash", "gemini-3.5-flash-lite"
PROMPTS = ROOT / "prompts"


def house_prompt(role: str) -> str | None:
    """System prompt imported from the n8n agent (tools/import_n8n_prompts.py), braces escaped for the template."""
    path = PROMPTS / f"{role}.md"
    return path.read_text(encoding="utf-8").replace("{", "{{").replace("}", "}}") if path.exists() else None


# ---------------------------------------------------------------- knowledge sources (RAG)
def _read_csv(source: str) -> list[dict]:
    """A local CSV file or a Google Sheet published/exported as CSV (…/export?format=csv)."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as r:
            text = r.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def sheet_source() -> str:
    return os.getenv("SHEET_CSV_URL") or str(DATA / "content_timeline.csv")


def load_rows() -> list[dict]:
    return _read_csv(sheet_source())


def load_facts() -> list[Document]:
    """Knowledge base = brand facts + every doc in data/docs/ + approved rows of the content sheet."""
    docs = []
    for path in [DATA / "brand_facts.md", *sorted((DATA / "docs").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for chunk in re.split(r"\n(?=## )", text):
            chunk = chunk.strip()
            if chunk.startswith("## "):
                docs.append(Document(page_content=chunk, metadata={"source": path.name, "section": chunk.splitlines()[0][3:]}))
    for row in load_rows():
        if row.get("status", "").lower() in ("approved", "published"):
            docs.append(Document(page_content=f"## {row['topic']}\n{row['description']}",
                                 metadata={"source": "content_timeline", "section": row["topic"], "row_id": row["row_id"]}))
    return docs


def make_embeddings():
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=os.getenv("EMBED_MODEL", "models/gemini-embedding-001"))
    return DeterministicFakeEmbedding(size=256)   # offline: deterministic vectors, keyword re-rank below


def make_vector_store(docs: list[Document], collection: str):
    """PostgreSQL + pgvector when DATABASE_URL is set, otherwise in memory."""
    url = os.getenv("DATABASE_URL")
    if url:
        from langchain_postgres import PGVector
        store = PGVector(embeddings=make_embeddings(), collection_name=collection,
                         connection=url.replace("postgresql://", "postgresql+psycopg://", 1), pre_delete_collection=True)
        store.add_documents(docs)
        return store
    return InMemoryVectorStore.from_documents(docs, make_embeddings())


def keyword_rank(topic: str, docs: list[Document], k: int = 3) -> list[Document]:
    """Cheap lexical scorer (BM25-style intuition) used to re-rank / back up vector search."""
    words = set(re.findall(r"\w+", topic.lower()))
    return sorted(docs, key=lambda d: -len(words & set(re.findall(r"\w+", d.page_content.lower()))))[:k]


# ---------------------------------------------------------------- deterministic guardrails
NUM = re.compile(r"\d[\d.,]*\d|\d")


def rule_check(caption: Caption, context: str, form_url: str) -> list[str]:
    return text_rules(" ".join([caption.title, caption.body, caption.call_to_action]), caption.hashtags,
                      context, form_url, cta=caption.call_to_action)


def text_rules(full: str, hashtags: list[str], context: str, form_url: str, cta: str | None = None) -> list[str]:
    issues = []
    if len(full) > MAX_CAPTION_CHARS:
        issues.append(f"too long: {len(full)} > {MAX_CAPTION_CHARS} characters")
    if form_url not in (cta if cta is not None else full):
        issues.append("must contain the registration form link")
    if REQUIRED_HASHTAG not in hashtags:
        issues.append(f"hashtags must include {REQUIRED_HASHTAG}")
    invented = set(NUM.findall(full.replace(form_url, ""))) - set(NUM.findall(context))
    if invented:
        issues.append(f"numbers not found in the facts (possible hallucination): {sorted(invented)}")
    return issues


# ---------------------------------------------------------------- prompts
GENERIC_WRITER_SYSTEM = ("You are the social-media writer for FPT Software's recruitment & training team. "
                         "Write for Vietnamese IT students: friendly, energetic, clear. Use ONLY the facts provided; "
                         "never invent numbers, dates or benefits.")


def writer_prompt(role: str = "group_writer") -> ChatPromptTemplate:
    """The n8n agent's own system prompt when imported, else a generic one; JSON schema instructions always appended."""
    system = house_prompt(role) or GENERIC_WRITER_SYSTEM
    return ChatPromptTemplate.from_messages([
        ("system", system + "\n\n{format_instructions}"),
        ("human", "Topic: {topic}\nRegistration form: {form_url}\n\nFacts:\n{context}\n\n"
                  "Reviewer feedback from the previous attempt (fix all of it): {feedback}"),
    ])


WRITER_PROMPT = writer_prompt()
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a strict brand editor acting as an LLM judge. Score the draft against this rubric: "
               "1) every claim is supported by the facts, 2) tone fits Gen Z IT students, 3) clear call to action. "
               "Fail it if any claim is unsupported. {format_instructions}"),
    ("human", "Facts:\n{context}\n\nDraft:\n{draft}"),
])
PUBLISHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the publishing agent. Adapt an APPROVED post for each channel without adding any new facts: "
               "facebook_group = casual and short with emojis; facebook_page = official brand tone; "
               "linkedin = professional, career-focused. Every version must keep the registration link and #FPTSoftware. "
               "{format_instructions}"),
    ("human", "Channels: {channels}\nApproved post (JSON):\n{caption}"),
])
VISION_PROMPT = ("You are a QA reviewer for a recruitment poster. Check: the FPT logo is present, text is readable, "
                 "the poster matches this topic: '{topic}', and nothing is inappropriate. Reply as JSON: "
                 '{{"passed": true/false, "issues": [..]}}')


# ---------------------------------------------------------------- graph state
class PostState(TypedDict, total=False):
    topic: str
    category: str
    poster_candidates: list[str]      # template renders to try, best first
    form_url: str
    context: str
    caption: dict
    feedback: str
    attempts: int
    judge: dict
    poster_qa: dict
    decision: str
    channel_posts: list[dict]
    status: str


def build_pipeline(fakes: dict[str, list[str]] | None = None, checkpointer=None, tracer: TraceLogger | None = None):
    tracer = tracer or TraceLogger(OUTPUT / "traces.jsonl")
    cfg = {"callbacks": [tracer]}
    llms = {r: make_llm(r, fakes) for r in ("writer", "judge", "vision", "publisher")}
    cap_parser = PydanticOutputParser(pydantic_object=Caption)
    judge_parser = PydanticOutputParser(pydantic_object=JudgeVerdict)
    plan_parser = PydanticOutputParser(pydantic_object=ChannelPlan)
    writer = (writer_prompt().partial(format_instructions=cap_parser.get_format_instructions()) | llms["writer"] | cap_parser).with_config(run_name="writer_agent", **cfg)
    judge = (JUDGE_PROMPT.partial(format_instructions=judge_parser.get_format_instructions()) | llms["judge"] | judge_parser).with_config(run_name="judge_agent", **cfg)
    publisher = (PUBLISHER_PROMPT.partial(format_instructions=plan_parser.get_format_instructions()) | llms["publisher"] | plan_parser).with_config(run_name="publisher_agent", **cfg)
    vision_llm = llms["vision"].with_config(run_name="vision_qa", **cfg)
    knowledge = load_facts()
    retriever = make_vector_store(knowledge, "pr_knowledge").as_retriever(search_kwargs={"k": 4})
    forms = json.loads((DATA / "forms.json").read_text(encoding="utf-8"))

    def map_form(s: PostState) -> PostState:
        url = forms.get(s["category"])
        return {"form_url": url, "attempts": 0, "feedback": "none"} if url else {"status": "stopped: no registration form for this category"}

    def retrieve(s: PostState) -> PostState:
        docs = keyword_rank(s["topic"], retriever.invoke(s["topic"]) + knowledge, k=4)   # hybrid: vector + lexical
        unique = list({d.page_content: d for d in docs}.values())
        return {"context": "\n\n".join(d.page_content for d in unique)}

    def write(s: PostState) -> PostState:
        try:
            cap = writer.invoke({"topic": s["topic"], "form_url": s["form_url"], "context": s["context"], "feedback": s["feedback"]})
            return {"caption": cap.model_dump(), "attempts": s["attempts"] + 1}
        except Exception as e:                      # bad JSON / schema mismatch -> retry with the error as feedback
            return {"caption": {}, "attempts": s["attempts"] + 1, "feedback": f"Output was not valid JSON for the schema: {e}"}

    def check(s: PostState) -> PostState:
        if not s.get("caption"):
            return {"judge": {"passed": False, "score": 1, "issues": [s["feedback"]]}}
        cap = Caption(**s["caption"])
        issues = rule_check(cap, s["context"], s["form_url"])
        verdict = judge.invoke({"context": s["context"], "draft": cap.model_dump_json()})
        passed = verdict.passed and not issues
        all_issues = issues + verdict.issues
        return {"judge": {"passed": passed, "score": verdict.score, "issues": all_issues},
                "feedback": "; ".join(all_issues) or "none"}

    def after_check(s: PostState) -> Literal["vision_qa", "write", "needs_human"]:
        if s["judge"]["passed"]:
            return "vision_qa"
        return "write" if s["attempts"] < MAX_ATTEMPTS else "needs_human"

    def needs_human(s: PostState) -> PostState:
        return {"status": f"stopped: still failing after {s['attempts']} attempts, sent to a human editor"}

    def vision_qa(s: PostState) -> PostState:
        candidates = s.get("poster_candidates") or []
        if not candidates:
            return {"poster_qa": {"passed": True, "poster": None, "tried": [], "issues": ["no poster attached: text-only post"]}}
        tried = []
        for path in candidates:                     # auto-remediation: failed poster -> next template, re-check
            b64 = base64.b64encode(Path(path).read_bytes()).decode()
            msg = HumanMessage(content=[{"type": "text", "text": VISION_PROMPT.format(topic=s["topic"])},
                                        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}])
            raw = vision_llm.invoke([msg]).content
            verdict = PosterVerdict.model_validate_json(re.search(r"\{.*\}", raw, re.S).group(0))
            tried.append({"poster": path, "passed": verdict.passed, "issues": verdict.issues})
            if verdict.passed:
                return {"poster_qa": {"passed": True, "poster": path, "tried": tried, "issues": []}}
        return {"poster_qa": {"passed": False, "poster": None, "tried": tried,
                              "issues": [i for t in tried for i in t["issues"]]}}

    def human_review(s: PostState) -> PostState:
        # Pause the graph. A person sees the draft and replies "approve" or "reject" (human-in-the-loop).
        decision = interrupt({"caption": s["caption"], "judge": s["judge"], "poster_qa": s["poster_qa"]})
        return {"decision": str(decision).strip().lower()}

    def after_review(s: PostState) -> Literal["publish", "rejected"]:
        return "publish" if s["decision"] == "approve" and s["poster_qa"]["passed"] else "rejected"

    def rejected(s: PostState) -> PostState:
        return {"status": "not published: rejected by reviewer or poster failed QA"}

    def publish(s: PostState) -> PostState:
        cap = Caption(**s["caption"])
        try:
            plan = publisher.invoke({"channels": ", ".join(CHANNELS), "caption": cap.model_dump_json()})
            versions = {p.channel: p.text for p in plan.posts}
        except Exception:                           # publisher failed -> fall back to the approved caption everywhere
            versions = {}
        posts = []
        for ch in CHANNELS:
            text = versions.get(ch)
            if not text or text_rules(text, re.findall(r"#\w+", text), s["context"], s["form_url"]):
                text = f"{cap.title}\n\n{cap.body}\n\n{cap.call_to_action}\n{' '.join(cap.hashtags)}"   # safe fallback
            posts.append({"channel": ch, "text": text, "poster": s["poster_qa"].get("poster")})
        OUTPUT.mkdir(exist_ok=True)
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "topic": s["topic"],
                  "category": s["category"], "attempts": s["attempts"], "score": s["judge"]["score"], "posts": posts}
        with open(OUTPUT / "post_log.jsonl", "a", encoding="utf-8") as f:   # dry run: log instead of calling channel APIs
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"channel_posts": posts, "status": f"published (dry run) to {len(posts)} channels and logged"}

    g = StateGraph(PostState)
    for name, fn in [("map_form", map_form), ("retrieve", retrieve), ("write", write), ("check", check),
                     ("needs_human", needs_human), ("vision_qa", vision_qa), ("human_review", human_review),
                     ("rejected", rejected), ("publish", publish)]:
        g.add_node(name, fn)
    g.add_edge(START, "map_form")
    g.add_conditional_edges("map_form", lambda s: "retrieve" if s.get("form_url") else END, ["retrieve", END])
    g.add_edge("retrieve", "write")
    g.add_edge("write", "check")
    g.add_conditional_edges("check", after_check, ["vision_qa", "write", "needs_human"])
    g.add_edge("vision_qa", "human_review")
    g.add_conditional_edges("human_review", after_review, ["publish", "rejected"])
    for end_node in ("needs_human", "rejected", "publish"):
        g.add_edge(end_node, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def input_from_row(row_id: str) -> PostState:
    """Turn one content-timeline row (Google Sheet) into pipeline input."""
    row = next(r for r in load_rows() if r["row_id"] == str(row_id))
    posters = [p for p in row.get("poster_candidates", "").split(";") if p]
    return {"topic": row["topic"], "category": row["category"], "poster_candidates": posters}


# ---------------------------------------------------------------- offline demo script
_GOOD = {"title": "Fresher AI Engineer: start your AI career at FPT Software",
         "body": "Join the Fresher AI program at FPT Software Academy: hands-on training in RAG, LLMOps and AI agents with LangGraph, then real projects for global clients.",
         "call_to_action": "Register now: https://forms.example/fresher-ai",
         "hashtags": ["#FPTSoftware", "#FresherAI", "#AIEngineer"]}
_BAD = {**_GOOD, "body": "Join 500 freshers this year and earn a 100% salary bonus!", "hashtags": ["#AI"]}
_PLAN = {"posts": [
    {"channel": "facebook_group", "text": "🚀 Fresher AI is open! Learn RAG, LLMOps and LangGraph agents at FPT Software Academy. Register: https://forms.example/fresher-ai #FPTSoftware #FresherAI"},
    {"channel": "facebook_page", "text": "FPT Software Academy opens the Fresher AI program: training in RAG, LLMOps and AI agents, then real projects for global clients. Register: https://forms.example/fresher-ai #FPTSoftware"},
    {"channel": "linkedin", "text": "Start your AI engineering career: FPT Software Academy's Fresher AI program covers RAG, LLMOps and agentic AI with LangGraph. Apply: https://forms.example/fresher-ai #FPTSoftware #AIEngineer"}]}
DEMO_SCRIPT = {
    "writer": [json.dumps(_BAD), json.dumps(_GOOD)],
    "judge": [json.dumps({"score": 3, "passed": False, "issues": ["claims about 500 freshers and salary bonus are not in the facts"]}),
              json.dumps({"score": 9, "passed": True, "issues": []})],
    "vision": [json.dumps({"passed": False, "issues": ["logo is cropped"]}), json.dumps({"passed": True, "issues": []})],
    "publisher": [json.dumps(_PLAN)],
}
