"""Structured logging and PII-scrubbing utilities.

Log entries are written to logs/queries.jsonl in JSON Lines format.
Sensitive fields (email, phone, family income, Aadhaar) are masked before
anything is written to disk.  stdout is never touched here — it stays
reserved for the machine-parseable JSON answer.
"""
import json
import re
import sys
import time
from pathlib import Path


class StructuredLogger:
    """Emits JSON-structured log lines to stderr, keeping stdout clean."""

    def __init__(self, name: str):
        self.name = name

    def _log(self, level: str, message: str, **kwargs):
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "logger": self.name,
            "level": level,
            "message": message,
        }
        if kwargs:
            payload.update(kwargs)
        # Emit logs to stderr to keep stdout perfectly clean for CLI parsing
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)


# ── PII scrubbing ────────────────────────────────────────────────────────────

def scrub_pii(text: str) -> str:
    """Masks sensitive personal identifiers before writing to log files.

    Patterns scrubbed:
      - Email addresses
      - Indian / international phone numbers (10-digit)
      - Aadhaar numbers (12-digit, India national ID)
      - Family income figures (numeric values near income-related keywords)
      - Large standalone Rupee amounts (5-7 digit values prefixed by Rs.)
    """
    if not text:
        return text
    # Email addresses
    t = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_MASKED]", text)
    # Indian / international phone numbers (10-digit groups)
    t = re.sub(
        r"\b(?:\+?[0-9]{1,3}[.\-\s]?)?\(?[0-9]{3}\)?[.\-\s]?[0-9]{3}[.\-\s]?[0-9]{4}\b",
        "[PHONE_MASKED]",
        t,
    )
    # Aadhaar numbers: 12-digit sequences (optionally space-separated in groups of 4)
    t = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[AADHAAR_MASKED]", t)
    # Family income metrics
    t = re.sub(
        r"(\b(?:my\s+)?(?:family\s+)?income\s*)(?:is\s*)?(?:rs\.?\s*)?\d+(?:\.\d+)?\s*(?:lakhs?|l)?\b",
        r"\1[INCOME_MASKED]",
        t,
        flags=re.IGNORECASE,
    )
    # Standalone large Rupee amounts (5-7 digits prefixed by Rs.)
    t = re.sub(r"\b(?:rs\.?\s*)(\d{5,7})\b", "[INCOME_MASKED]", t, flags=re.IGNORECASE)
    return t


# Alias kept for backward compatibility with code that imported sanitize_pii
sanitize_pii = scrub_pii


# ── Query log writer ─────────────────────────────────────────────────────────

def log_query(log_path: Path, question: str, parsed: dict, qc) -> None:
    """Appends one JSONL record to log_path after masking PII.

    Args:
        log_path: Absolute path to the .jsonl file.
        question: The raw question string from the user.
        parsed:   The answer dict returned by answer_question().
        qc:       A QueryCost instance, or None if the LLM call was skipped.
    """
    log_path.parent.mkdir(exist_ok=True)
    sanitized_q = scrub_pii(question)
    sanitized_ans = scrub_pii(parsed.get("answer", ""))

    # Avoid mutating the original output dict
    logged_parsed = parsed.copy()
    logged_parsed["answer"] = sanitized_ans

    entry: dict = {"question": sanitized_q, "result": logged_parsed}
    if qc is not None:
        entry["cost"] = {
            "input_tokens": qc.input_tokens,
            "output_tokens": qc.output_tokens,
            "latency_s": qc.latency_s,
            "model": qc.model,
            "cost_inr": round(qc.cost_inr, 4),
        }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
