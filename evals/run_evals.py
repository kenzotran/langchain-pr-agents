"""Evaluate the caption guardrails on a labelled golden set.

    python evals/run_evals.py            # rules only (offline)
    GOOGLE_API_KEY=... python evals/run_evals.py   # rules + Gemini LLM judge

Each golden item: {"id", "caption": {title, body, call_to_action, hashtags}, "expected_pass": bool, "note"}.
"Positive" = a caption that SHOULD be blocked, so recall = share of bad captions we caught.
Writes evals/report.md.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pr_agents import pipeline as P  # noqa: E402

HERE = Path(__file__).resolve().parent
FORM = "https://forms.example/fresher-ai"


def main():
    golden = [json.loads(l) for l in (HERE / "golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    context = "\n\n".join(d.page_content for d in P.load_facts())
    judge = None
    if os.getenv("GOOGLE_API_KEY"):
        parser = P.PydanticOutputParser(pydantic_object=P.JudgeVerdict)
        judge = P.JUDGE_PROMPT.partial(format_instructions=parser.get_format_instructions()) | P.make_llm("judge") | parser
    rows, tp, fp, fn, tn = [], 0, 0, 0, 0
    for g in golden:
        cap = P.Caption(**g["caption"])
        issues = P.rule_check(cap, context, FORM)
        judged = judge.invoke({"context": context, "draft": cap.model_dump_json()}) if judge else None
        predicted_pass = not issues and (judged.passed if judged else True)
        blocked, should_block = not predicted_pass, not g["expected_pass"]
        tp += blocked and should_block; fp += blocked and not should_block
        fn += (not blocked) and should_block; tn += (not blocked) and not should_block
        rows.append((g["id"], g["expected_pass"], predicted_pass, "; ".join(issues + (judged.issues if judged else [])) or "-", g["note"]))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    accuracy = (tp + tn) / len(golden)
    mode = "rules + LLM judge" if judge else "rules only (offline)"
    lines = [f"# Guardrail evaluation ({mode})", "",
             f"Items: {len(golden)} · accuracy **{accuracy:.0%}** · precision **{precision:.0%}** · recall **{recall:.0%}**", "",
             f"Confusion (positive = should be blocked): TP {tp} · FP {fp} · FN {fn} · TN {tn}", "",
             "| id | expected pass | predicted pass | issues | note |", "|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows]
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:5]))
    return accuracy


if __name__ == "__main__":
    main()
