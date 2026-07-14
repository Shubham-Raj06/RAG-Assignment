"""Stateless parser functions for extracting hard constraints from a query string.

Each function returns a typed value or None (no match).  None means the
constraint is absent from the query -- the caller decides whether to apply
a filter or fall through to TF-IDF ranking.

Keeping parsers separate from the Retriever class makes them trivially
unit-testable (no DataFrame, no TF-IDF matrix, no network) and easy to
extend with a translation pre-pass for Hindi / multilingual queries.
"""
from __future__ import annotations
import re


def parse_budget_limit(query: str) -> float | None:
    """Extracts a budget ceiling from the query and converts it to annual fees (INR).
    Handles 'lakh', 'lakhs', 'L', 'k', and plain numbers (like 90,000 or 150000).
    Per-semester budgets are doubled to align with the dataset's annual fee structure."""
    q = query.lower()
    # Normalize numbers by removing commas inside digits (e.g. 1,50,000 -> 150000)
    q = re.sub(r'(?<=\d),(?=\d)', '', q)

    # 1. Match lakh/lakhs/L pattern
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|l)\b", q)
    if lakh_match:
        val = float(lakh_match.group(1)) * 100000
        if any(term in q for term in ["semester", "sem", "half-yearly", "six month"]):
            val *= 2
        return val

    # 2. Match k pattern (thousands, e.g. 90k, 150k)
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", q)
    if k_match:
        val = float(k_match.group(1)) * 1000
        if any(term in q for term in ["semester", "sem", "half-yearly", "six month"]):
            val *= 2
        return val

    # 3. Match plain numbers representing monetary amounts (>= 5000)
    # This prevents matching years (e.g. 2004) or cutoffs (e.g. 78) as budgets
    numbers = re.findall(r"\b\d+\b", q)
    for num_str in numbers:
        val = float(num_str)
        if val >= 5000:
            if any(term in q for term in ["semester", "sem", "half-yearly", "six month"]):
                val *= 2
            return val

    return None


def parse_cutoff_score(query: str) -> float | None:
    """Extracts a cutoff percentage score from the query.
    Looks for percentages (e.g. '78%') or numbers associated with scoring terms."""
    q = query.lower()

    # 1. Match numbers followed by % or percent or percentage
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|percentage)\b", q)
    if pct_match:
        return float(pct_match.group(1))

    # 2. Match numbers preceded by scoring/cutoff keywords
    prefix_match = re.search(r"(?:scored?|score\s*of|marks?|cutoff\s*of|aggregate|got|have)\s*(\d+(?:\.\d+)?)\b", q)
    if prefix_match:
        val = float(prefix_match.group(1))
        if 30 <= val <= 100:
            return val

    # 3. Fallback: find any number between 30 and 100 if the query contains scoring-related words
    if any(word in q for word in ["score", "scor", "marks", "cutoff", "percent", "percentage", "aggregate"]):
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", q)
        for num_str in numbers:
            val = float(num_str)
            if 30 <= val <= 100:
                # Skip if it looks like a budget abbreviation (e.g. 90k)
                idx = q.find(num_str)
                if idx != -1:
                    after = q[idx + len(num_str):].strip()
                    if after.startswith(('k', 'l')):
                        continue
                return val

    return None


def parse_placement_floor(query: str) -> float | None:
    """Extracts a minimum placement package (LPA) constraint from the query.
    Looks for expressions like 'placement above 5 lakhs', 'placements at least 6 LPA',
    'placement package of 5L', 'min placement 5', etc."""
    q = query.lower()

    # We must see "placement" or "placements" in the query
    idx = q.find("placement")
    if idx == -1:
        return None

    # Segment of query after the word "placement"
    after_str = q[idx + len("placement"):]
    # Match numbers after the word placement, optionally with comparison words or units
    match_after = re.search(
        r"(?:above|over|greater\s*than|at\s*least|min|minimum|of|package|average|avg|\s)\s*(\d+(?:\.\d+)?)\s*(?:lakhs?|lpa|l)?\b",
        after_str
    )
    if match_after:
        return float(match_after.group(1))

    # Segment of query before the word "placement"
    before_str = q[:idx].strip()
    # Match numbers right before the word placement (e.g. "5 LPA placement")
    match_before = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lpa|l)?$", before_str)
    if match_before:
        return float(match_before.group(1))

    return None
