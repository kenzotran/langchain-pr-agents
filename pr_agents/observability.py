"""Observability: trace every agent step and LLM call (latency, tokens, errors) to JSONL, plus Langfuse if configured.

Each line in output/traces.jsonl, e.g.:
    {"time": "2026-11-07T10:00:00+00:00", "run": "writer_agent", "event": "step_end", "latency_ms": 812, "error": null}
    {"time": "2026-11-07T10:00:00+00:00", "run": "llm", "event": "llm_end", "latency_ms": 790,
     "input_tokens": 950, "output_tokens": 180, "error": null}
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger("pr_agents")
NAMED_STEPS = {"writer_agent", "judge_agent", "publisher_agent", "vision_qa"}


class TraceLogger(BaseCallbackHandler):
    def __init__(self, path: Path):
        self.path, self._start, self._lock = Path(path), {}, Lock()

    def _write(self, record: dict) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("trace %s", record)

    def _done(self, run_id, event: str, **extra) -> None:
        t0, name = self._start.pop(run_id, (time.perf_counter(), "llm"))
        self._write({"run": name, "event": event, "latency_ms": round((time.perf_counter() - t0) * 1000), **extra})

    # named agent steps (set with .with_config(run_name=...))
    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        if kwargs.get("name") in NAMED_STEPS:
            self._start[run_id] = (time.perf_counter(), kwargs["name"])

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        if run_id in self._start:
            self._done(run_id, "step_end", error=None)

    def on_chain_error(self, error, *, run_id, **kwargs):
        if run_id in self._start:
            self._done(run_id, "step_error", error=str(error)[:300])

    # raw model calls (tokens + latency)
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start[run_id] = (time.perf_counter(), "llm")

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start[run_id] = (time.perf_counter(), "llm")

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = {}
        for gens in response.generations:
            for g in gens:
                meta = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                usage = {"input_tokens": meta.get("input_tokens"), "output_tokens": meta.get("output_tokens")}
        self._done(run_id, "llm_end", **usage, error=None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._done(run_id, "llm_error", error=str(error)[:300])


def callbacks(path: Path) -> list:
    """Trace file always; Langfuse too when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set."""
    handlers = [TraceLogger(path)]
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.langchain import CallbackHandler   # pip install langfuse
            handlers.append(CallbackHandler())
        except ImportError:
            log.warning("LANGFUSE keys set but the langfuse package is not installed")
    return handlers
