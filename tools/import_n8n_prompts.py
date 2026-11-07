"""Import the system prompts of the n8n AI Agent nodes into prompts/ so the LangChain agents use the same instructions.

    python tools/import_n8n_prompts.py ../PRAutomation/audit/2026-10-28/workflow_live_ops11f.json

Writes prompts/<role>.md (git-ignored: these are internal brand rules, keep them out of public repos).
"""
import json
import sys
from pathlib import Path

ROLES = {"B7 Viet caption (AI)": "group_writer", "B7P Viet caption Page (AI)": "page_writer", "TP1 Viet bai TIP (AI)": "tip_writer"}
OUT = Path(__file__).resolve().parent.parent / "prompts"


def main(path: str) -> dict:
    wf = json.loads(Path(path).read_text(encoding="utf-8"))
    wf = wf.get("data", wf)
    OUT.mkdir(exist_ok=True)
    written = {}
    for node in wf["nodes"]:
        role = ROLES.get(node["name"])
        prompt = node.get("parameters", {}).get("options", {}).get("systemMessage", "")
        if role and prompt:
            (OUT / f"{role}.md").write_text(prompt, encoding="utf-8")
            written[role] = len(prompt)
    print(written)
    return written


if __name__ == "__main__":
    main(sys.argv[1])
