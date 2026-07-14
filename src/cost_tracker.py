"""Cost instrumentation. Deliberately simple: token counts come straight from
the provider's own usage field in the API response (no re-tokenizing, no
estimating), latency from time.perf_counter(). This is exactly what Part D
asks for -- measured numbers, not intentions.

Pricing table is the one thing that needs manual updating if the provider
changes prices; kept in one place for that reason.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

# Pricing per 1M tokens — Google AI Studio pricing (as of July 2026).
MODEL_PRICING = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-flash-latest": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}
USD_TO_INR = 87.0  # approximate; stated in README Part D


@dataclass
class QueryCost:
    question: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str

    @property
    def cost_usd(self) -> float:
        # Default to 3.1-flash-lite if model not found in table
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["gemini-3.1-flash-lite"])
        return (self.input_tokens / 1_000_000) * pricing["input"] + \
               (self.output_tokens / 1_000_000) * pricing["output"]

    @property
    def cost_inr(self) -> float:
        return self.cost_usd * USD_TO_INR


class CostLog:
    def __init__(self):
        self.entries: list[QueryCost] = []

    def add(self, qc: QueryCost):
        self.entries.append(qc)

    def summary(self) -> dict:
        if not self.entries:
            return {}
        
        # Avoid contamination: only summarize entries matching the latest active model
        active_model = self.entries[-1].model
        model_entries = [e for e in self.entries if e.model == active_model]
        
        n = len(model_entries)
        avg_in = sum(e.input_tokens for e in model_entries) / n
        avg_out = sum(e.output_tokens for e in model_entries) / n
        avg_latency = sum(e.latency_s for e in model_entries) / n
        avg_cost_inr = sum(e.cost_inr for e in model_entries) / n
        return {
            "queries_measured": n,
            "avg_input_tokens": round(avg_in, 1),
            "avg_output_tokens": round(avg_out, 1),
            "avg_latency_s": round(avg_latency, 3),
            "model": active_model,
            "avg_cost_inr_per_query": round(avg_cost_inr, 4),
            "cost_inr_per_1000_queries": round(avg_cost_inr * 1000, 2),
        }

    def save(self, path: Path):
        payload = {
            "summary": self.summary(),
            "entries": [asdict(e) for e in self.entries],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_and_print_summary(log_path: Path) -> dict | None:
    """Reads queries.jsonl, extracts all query costs, and prints/returns the summary."""
    if not log_path.exists():
        print("No log file found.")
        return None
    log = CostLog()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                cost_data = data.get("cost")
                if cost_data:
                    qc = QueryCost(
                        question=data["question"],
                        input_tokens=cost_data["input_tokens"],
                        output_tokens=cost_data["output_tokens"],
                        latency_s=cost_data["latency_s"],
                        model=cost_data["model"],
                    )
                    log.add(qc)
            except Exception:
                pass
    if log.entries:
        s = log.summary()
        print("\n=== COST & LATENCY SUMMARY ===")
        print(f"Queries measured: {s['queries_measured']}")
        print(f"Avg input tokens / query: {s['avg_input_tokens']}")
        print(f"Avg output tokens / query: {s['avg_output_tokens']}")
        print(f"Avg latency / query: {s['avg_latency_s']} s")
        print(f"Model used: {s['model']}")
        print(f"Avg cost / query: INR {s['avg_cost_inr_per_query']}")
        print(f"Cost per 1,000 queries: INR {s['cost_inr_per_1000_queries']}")
        print("==============================\n")
        return s
    else:
        print("No cost entries found in log file.")
        return None
