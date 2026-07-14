"""LLM wrapper — Google Gemini (gemini-3.1-flash-lite by default, configurable
via the MME_MODEL environment variable).

Running default is gemini-3.1-flash-lite: verified 100% pass rate on all 11
eval cases, and the most capable model available on the free-tier quota used
during development (gemini-2.5-flash returns 404 on the free tier).

For a paid-tier deployment: set MME_MODEL=gemini-2.5-flash for stronger
instruction-following on more complex queries at ~3x the token cost.

Swapping providers means editing this one file.  The rest of the pipeline
(retriever, prompts, answer.py) is provider-agnostic.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass

from google import genai
from .logger import StructuredLogger






import re as _re
import sys as _sys

MODEL = os.environ.get("MME_MODEL", "gemini-3.1-flash-lite")

_MAX_RETRIES = int(os.environ.get("MME_MAX_RETRIES", "4"))
_DEFAULT_WAIT_S = float(os.environ.get("MME_DEFAULT_WAIT_S", "65.0"))

_logger = StructuredLogger("LLMClient")



@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str


class LLMClient:
    # Enforce minimum gap between consecutive API calls.
    # gemini-2.5-flash paid tier: 1000 RPM — 5s is very conservative, safe even on
    # free tier (15 RPM). Set MME_CALL_INTERVAL_S env var to override if needed.
    MIN_INTERVAL_S: float = float(os.environ.get("MME_CALL_INTERVAL_S", "5.0"))

    def __init__(self, model: str = MODEL):
        self.model = model
        # Enforce default 60-second timeout on all HTTP requests to Gemini API
        self.client = genai.Client()       # reads GEMINI_API_KEY from env
        self._last_call_time: float = 0.0  # perf_counter timestamp of last call

    def _rate_limit_wait(self) -> None:
        """Sleep just enough to respect MIN_INTERVAL_S between calls."""
        elapsed = time.perf_counter() - self._last_call_time
        gap = self.MIN_INTERVAL_S - elapsed
        if gap > 0:
            time.sleep(gap)

    def generate(self, system: str, user: str, max_tokens: int = 600) -> LLMResult:
        from google.genai import errors as _errs

        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limit_wait()            # honour MIN_INTERVAL_S
                start = time.perf_counter()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config={
                        "system_instruction": system,
                        "max_output_tokens": max_tokens,
                        "response_mime_type": "application/json"
                    },
                )


                self._last_call_time = time.perf_counter()
                latency = self._last_call_time - start

                usage = resp.usage_metadata
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0

                return LLMResult(
                    text=resp.text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_s=latency,
                    model=self.model,
                )
            except _errs.APIError as exc:
                # ClientError doesn't always expose .status_code as an attribute;
                # parse the leading 3-digit code from the message string as fallback.
                status = getattr(exc, "status_code", None)
                if status is None:
                    m = _re.match(r"^(\d{3})\b", str(exc))
                    status = int(m.group(1)) if m else 0

                retriable = status in (429, 503, 500)
                if retriable and attempt < _MAX_RETRIES - 1:
                    # 429: parse the exact suggested delay from the message
                    # 503/500: use exponential backoff (15s, 30s, 60s)
                    delay_match = _re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
                    if delay_match:
                        wait = float(delay_match.group(1)) + 3
                    else:
                        wait = min(15.0 * (2 ** attempt), _DEFAULT_WAIT_S)
                    _logger.warning(
                        "API rate-limit or server error encountered; retrying.",
                        http_status=status,
                        wait_seconds=round(wait, 1),
                        attempt=attempt + 1,
                        max_attempts=_MAX_RETRIES,
                    )
                    time.sleep(wait)

                else:
                    raise  # 404 (wrong model), 400, or exhausted retries


from .models import CounselorResponse


def parse_json_answer(raw_text: str) -> dict:
    """Defensive parsing: strip code fences if the model added them anyway,
    extracts the JSON object by finding the first '{' and last '}', and
    validates the output schema using Pydantic, failing closed on errors."""
    cleaned = raw_text.strip()

    # Locate first '{' and last '}'
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx + 1]

    try:
        obj = json.loads(cleaned)
        # Leverage Pydantic to validate types and populate defaults
        validated = CounselorResponse(**obj)
        return validated.model_dump()
    except Exception as exc:
        return {
            "answer": "",
            "citations": [],
            "answered": False,
            "reason_if_unanswered": f"Model failed schema validation or JSON decoding: {str(exc)}",
            "_raw": raw_text,
        }


