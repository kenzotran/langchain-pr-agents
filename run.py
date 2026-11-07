"""Run the pipeline end to end.

    python run.py "Fresher AI program is open" --category fresher_ai [--poster path.png] [--auto-approve]

Without GOOGLE_API_KEY it uses scripted fake models (offline demo). With a key it uses Gemini + fallback.
"""
import argparse
import json
import uuid

from langgraph.types import Command

from pr_agents.pipeline import build_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("--category", default="fresher_ai")
    ap.add_argument("--poster")
    ap.add_argument("--auto-approve", action="store_true", help="skip the prompt (demo only)")
    a = ap.parse_args()

    app = build_pipeline()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = app.invoke({"topic": a.topic, "category": a.category, **({"poster_path": a.poster} if a.poster else {})}, cfg)

    if "__interrupt__" in state:                          # graph paused for the human reviewer
        draft = state["__interrupt__"][0].value
        print(json.dumps(draft, indent=2, ensure_ascii=False))
        answer = "approve" if a.auto_approve else input("\nType 'approve' to publish or 'reject': ")
        state = app.invoke(Command(resume=answer), cfg)   # resume from the checkpoint
    print("\nattempts:", state.get("attempts"), "| status:", state.get("status"))


if __name__ == "__main__":
    main()
