#!/usr/bin/env python3
"""Regenerates answers.md by running the 7 published questions through the
same code path as answer.py (imports answer_question directly, doesn't
shell out, so it's fast and gives identical results)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Force output streams to UTF-8 to prevent charmap errors on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data_loader import load_colleges
from src.retriever import Retriever
from src.llm import LLMClient
from src.logger import log_query
from src.cost_tracker import load_and_print_summary
from answer import answer_question, LOG_PATH

QUESTIONS = [
    "I scored 78% and have a budget of \u20b91.5 lakh/year \u2014 which engineering colleges can I consider?",
    "Which colleges offer an MBA, and what do they cost?",
    "List the government colleges that have hostel facilities.",
    "What's the average placement package at North Ridge Institute of Technology?",
    "Does Ganga Valley University offer a PhD in Physics?",
    "Which colleges offer scholarships for students from low-income families?",
    "Which college is best for me? I have \u20b91 lakh per semester.",
]


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    if key.strip() == "GEMINI_API_KEY":
                        os.environ["GEMINI_API_KEY"] = val.strip().strip("'\"")
                        break

    df = load_colleges()
    retriever = Retriever(df)
    llm = LLMClient()

    lines = ["# answers.md\n", "Verbatim output of `answer.py` for the 7 published questions.\n"]
    for q in QUESTIONS:
        parsed, qc = answer_question(q, retriever, llm)
        log_query(LOG_PATH, q, parsed, qc)
        output = {
            "answer": parsed.get("answer", ""),
            "citations": parsed.get("citations", []),
            "answered": parsed.get("answered", False),
            "reason_if_unanswered": parsed.get("reason_if_unanswered"),
        }
        lines.append(f"## Q: {q}\n")
        lines.append("```json")
        lines.append(json.dumps(output, indent=2, ensure_ascii=False))
        lines.append("```\n")

    Path("answers.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote answers.md")

    # Print aggregated cost metrics across all runs
    load_and_print_summary(LOG_PATH)


if __name__ == "__main__":
    main()
