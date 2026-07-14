#!/usr/bin/env python3
"""Runs evals/cases.json through the real pipeline and prints a pass rate.
Checks are intentionally loose (substring / citation-set based) since exact
LLM wording varies run to run -- what must NOT vary is: whether it answered,
whether it cited the right college_id(s), and whether specific grounding
facts appear (or specific wrong facts are absent)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force output streams to UTF-8 to prevent charmap errors on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data_loader import load_colleges
from src.retriever import Retriever
from src.llm import LLMClient
from answer import answer_question, log_query, LOG_PATH
from src.cost_tracker import load_and_print_summary


def check_case(case: dict, parsed: dict) -> tuple[bool, list[str]]:
    failures = []
    answered = parsed.get("answered", False)
    citations = set(parsed.get("citations", []))
    answer_text = (parsed.get("answer") or "").lower()

    if "expected_answered" in case and answered != case["expected_answered"]:
        failures.append(f"answered={answered}, expected={case['expected_answered']}")

    if "expected_citations" in case:
        expected = set(case["expected_citations"])
        if citations != expected:
            failures.append(f"citations={sorted(citations)}, expected={sorted(expected)}")

    if "expected_citations_any_of" in case:
        if not citations & set(case["expected_citations_any_of"]):
            failures.append(f"citations={sorted(citations)} matched none of {case['expected_citations_any_of']}")

    for s in case.get("required_substrings", []):
        if s.lower() not in answer_text:
            failures.append(f"missing required substring: '{s}'")

    for s in case.get("forbidden_substrings", []):
        if s.lower() in answer_text:
            failures.append(f"contains forbidden substring: '{s}'")

    return (len(failures) == 0), failures


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    if key.strip() == "GEMINI_API_KEY":
                        os.environ["GEMINI_API_KEY"] = val.strip().strip("'\"")
                        break

    cases = json.loads((Path(__file__).parent / "cases.json").read_text())
    df = load_colleges()
    retriever = Retriever(df)
    llm = LLMClient()

    passed = 0
    for case in cases:
        parsed, qc = answer_question(case["question"], retriever, llm)
        log_query(case["question"], parsed, qc)
        ok, failures = check_case(case, parsed)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']}: {case['question']}")
        if not ok:
            for f in failures:
                print(f"    - {f}")
            print(f"    got: {json.dumps(parsed, ensure_ascii=False)}")
        if ok:
            passed += 1

    total = len(cases)
    print(f"\n{passed}/{total} passed ({passed/total*100:.0f}%)")

    # Print aggregated cost metrics across all runs
    load_and_print_summary(LOG_PATH)


if __name__ == "__main__":
    main()
