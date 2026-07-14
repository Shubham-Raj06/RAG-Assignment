#!/usr/bin/env python3
"""Required interface:

    python answer.py "Which colleges offer an MBA, and what do they cost?"

Prints exactly one JSON object to stdout. Everything else (cost/latency log)
goes to logs/queries.jsonl so stdout stays machine-parseable for grading.
"""
import json
import os
import re as _re
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
from src.prompts import SYSTEM_PROMPT, USER_TEMPLATE, build_context_block
from src.llm import LLMClient, parse_json_answer
from src.cost_tracker import QueryCost

LOG_PATH = Path(__file__).resolve().parent / "logs" / "queries.jsonl"


def answer_question(question: str, retriever: Retriever, llm: LLMClient, top_k: int = 5) -> dict:
    candidates = retriever.retrieve(question, top_k=top_k)

    if candidates.empty:
        # a real structured constraint (budget/cutoff/type/hostel) matched
        # nothing -- this is a true negative, don't call the LLM to guess
        return {
            "answer": "No college in our dataset matches that constraint.",
            "citations": [],
            "answered": False,
            "reason_if_unanswered": "Structured filter (budget/cutoff/type/hostel) matched zero records.",
        }, None

    context = build_context_block(candidates)
    user_msg = USER_TEMPLATE.format(context=context, question=question)
    result = llm.generate(SYSTEM_PROMPT, user_msg)
    parsed = parse_json_answer(result.text)

    # never trust the model's citations blindly -- only allow ids that were
    # actually in the retrieved context (guards against hallucinated ids)
    valid_ids = set(candidates["college_id"])
    parsed["citations"] = list(dict.fromkeys([c for c in parsed.get("citations", []) if c in valid_ids]))
    if not parsed.get("answered", True):
        parsed["citations"] = []

    qc = QueryCost(
        question=question,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=result.latency_s,
        model=result.model,
    )
    return parsed, qc


def scrub_pii(text: str) -> str:
    """Scrubs sensitive personal identifiers (emails, phone numbers, family income metrics)
    from query text and responses before writing to log files for privacy compliance."""
    if not text:
        return text
    # Mask email patterns
    t = _re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_MASKED]", text)
    # Mask Indian / international standard phone formats (10 digits)
    t = _re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_MASKED]", t)
    # Mask family income metrics (preserve prefix "income" or "family income", remove "is " or "Rs ")
    t = _re.sub(
        r"(\b(?:my\s+)?(?:family\s+)?income\s*)(?:is\s*)?(?:rs\.?\s*)?\d+(?:\.\d+)?\s*(?:lakhs?|l)?\b",
        r"\1[INCOME_MASKED]",
        t,
        flags=_re.IGNORECASE
    )
    # Also catch plain standalone Rupees values of large numbers
    t = _re.sub(r"\b(?:rs\.?\s*)(\d{5,7})\b", "[INCOME_MASKED]", t, flags=_re.IGNORECASE)
    return t

sanitize_pii = scrub_pii



def log_query(question: str, parsed: dict, qc):
    LOG_PATH.parent.mkdir(exist_ok=True)
    sanitized_q = sanitize_pii(question)
    sanitized_ans = sanitize_pii(parsed.get("answer", ""))
    
    # Avoid mutating original output dict
    logged_parsed = parsed.copy()
    logged_parsed["answer"] = sanitized_ans
    
    entry = {"question": sanitized_q, "result": logged_parsed}
    if qc is not None:
        entry["cost"] = {
            "input_tokens": qc.input_tokens,
            "output_tokens": qc.output_tokens,
            "latency_s": qc.latency_s,
            "model": qc.model,
            "cost_inr": round(qc.cost_inr, 4),
        }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")



def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "answer": "",
            "citations": [],
            "answered": False,
            "reason_if_unanswered": "No question provided. Usage: python answer.py \"<question>\"",
        }))
        sys.exit(1)

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

    if not os.environ.get("GEMINI_API_KEY"):
        print(json.dumps({
            "answer": "",
            "citations": [],
            "answered": False,
            "reason_if_unanswered": "GEMINI_API_KEY environment variable is not set. "
                                    "Set it before running: export GEMINI_API_KEY=your-key-here",
        }))
        sys.exit(1)


    question = sys.argv[1]
    df = load_colleges()
    retriever = Retriever(df)
    llm = LLMClient()

    parsed, qc = answer_question(question, retriever, llm)
    log_query(question, parsed, qc)

    # stdout: ONLY the JSON object, per required interface
    output = {
        "answer": parsed.get("answer", ""),
        "citations": parsed.get("citations", []),
        "answered": parsed.get("answered", False),
        "reason_if_unanswered": parsed.get("reason_if_unanswered"),
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
