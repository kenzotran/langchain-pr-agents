# langchain-pr-agents

Two tested LangChain + LangGraph systems: a **3-agent PR content platform** and a **24/7 Messenger agent**,
deployed with Docker, a REST API and PostgreSQL + pgvector. They implement in Python code the concepts behind my
n8n automations at FPT Software (Recruitment & Learning), whose production versions run on n8n's LangChain-based AI nodes.

| CV capability | Where it is implemented | Proof |
|---|---|---|
| Multi-agent platform: 3 LLM agents **generate, evaluate, publish** | `pipeline.py`: `writer_agent`, `judge_agent` (LLM-as-a-judge), `publisher_agent` (rewrites the approved post for Facebook group, Facebook page and LinkedIn) | `test_self_correction_then_approval_publishes_to_every_channel` |
| RAG grounding over **Google Sheets, Docs** & knowledge sources | `load_facts()`: content-timeline sheet (local CSV or `SHEET_CSV_URL` = a Google Sheet exported as CSV) + `data/docs/*.md` (Docs exports) + brand facts; hybrid vector + keyword retrieval | `test_knowledge_includes_sheet_rows_and_docs` |
| LLM-as-a-judge **evals** | judge in the loop + `evals/run_evals.py` on a labelled golden set (precision / recall / accuracy report) | `evals/report.md` |
| Schema-enforced structured outputs | Pydantic schemas + `PydanticOutputParser` for every agent; invalid JSON becomes retry feedback | `test_invalid_json_is_retried_with_error_feedback` |
| ReAct reasoning | Messenger agent loop: think → call tool → observe → answer (`ToolNode` + `tools_condition`) | `test_agent_calls_faq_tool_then_answers_from_it` |
| Self-correcting loops | `check → write` with feedback, max 3, then a human | `test_rules_catch_hallucinated_numbers_even_if_judge_passes` |
| Multi-model failover | `primary.with_fallbacks([backup])` on every LLM | `test_failover_uses_backup_when_primary_errors` |
| Gemini Vision QA with **auto-remediation** + human sign-off | `vision_qa` checks each poster template; a failed one is replaced by the next and re-checked; LangGraph `interrupt()` for approval | `test_vision_auto_remediation_switches_to_next_template` |
| 24/7 Messenger agent, **agentic RAG**, tool calling to live data | `messenger_agent.py`: agent chooses `search_faq` (RAG) or `get_open_positions` (live data) | `test_messenger_message_gets_grounded_reply` |
| Persistent memory in a **PostgreSQL vector database** | LangGraph `PostgresSaver` checkpoints per user + FAQ/knowledge embeddings in **pgvector** (`langchain-postgres`) | verified in Docker: memory survives a container restart |
| **Docker**, **REST APIs**, error handling, **observability** | `api.py` (FastAPI), `Dockerfile`, `docker-compose.yml` (API + pgvector), webhook never crashes, request IDs + access logs, `observability.py` traces every agent step and LLM call (latency, tokens, errors) to `output/traces.jsonl`, plus Langfuse if keys are set | `test_webhook_returns_200_even_if_agent_crashes`, `test_every_agent_step_is_traced` |

## Same models and prompts as the n8n production agents

Defaults are the n8n agents' models: **Gemini 3.5 Flash** (primary) with **Gemini 3.5 Flash-Lite** fallback.
`tools/import_n8n_prompts.py <workflow.json>` copies the system prompts of the three n8n AI Agent nodes
(group caption, page caption, tip post) into `prompts/`, and the writer agent then uses that exact house prompt
(+ the JSON schema instructions). `prompts/` is git-ignored because it contains internal brand rules.

## Architecture

```
Content platform (pipeline.py)
sheet row / topic ─► map_form ─► retrieve (Sheets + Docs + facts, pgvector) ─► WRITER ─► rules + JUDGE ─┐
        │ no form: stop                                   ▲                                          │
                                                          └──── feedback (max 3) ◄── fail ───────────┤ pass
                    PUBLISHER ◄─ approve ◄─ human_review (interrupt) ◄─ Vision QA ⟲ next template ◄──┘
                    (group / page / LinkedIn versions, rules re-checked) ─► post (dry run) + log

Messenger agent (messenger_agent.py)
webhook ─► agent ⇄ tools {search_faq (pgvector RAG), get_open_positions (live)} ─► reply ─► Send API
           memory: PostgresSaver checkpoint per sender (SQLite locally)
```

## Run

```bash
# local, offline (no keys needed: scripted fake models + deterministic embeddings)
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q                  # 25 tests
.venv/Scripts/python evals/run_evals.py            # guardrail evaluation report
.venv/Scripts/python run.py "Fresher AI program is open" --category fresher_ai --auto-approve

# full stack: API + PostgreSQL/pgvector
docker compose up -d --build
curl localhost:8000/health
curl -X POST localhost:8000/posts -H "content-type: application/json" -d '{"row_id": "1"}'
curl -X POST localhost:8000/posts/<thread_id>/decision -H "content-type: application/json" -d '{"decision": "approve"}'
```

Real models: copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (Gemini). Messenger: set `VERIFY_TOKEN` and
`PAGE_ACCESS_TOKEN`. Never commit `.env`.

## Evaluation result (offline, rules only)

10 labelled captions: accuracy 90 %, precision 100 %, recall 83 %. The miss is "Guaranteed job offer for everyone",
an unsupported claim with no numbers in it, which rules cannot catch. That's exactly why the LLM judge sits on top.
Run with a Gemini key to score rules + judge together.

## Limits

Publishing is a dry run (no Facebook/LinkedIn post). Sample data only. Offline fakes test the *logic*;
model quality needs the eval set run with a real key.
