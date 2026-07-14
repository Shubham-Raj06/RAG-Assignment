"""Unit tests for the retrieval parsing functions.

These test the pure-function parser layer in isolation — no LLM calls,
no network I/O, runs in milliseconds.  This is the layer most likely
to have edge-case regressions: a regex change that fixes one case can
silently break another.

Run with:
    pytest tests/test_parsers.py -v
"""
import sys
from pathlib import Path

import pytest

# Allow importing from parent package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import parse_budget_limit, parse_cutoff_score, parse_placement_floor


# ──────────────────────────────────────────────────────────────
# parse_budget_limit
# ──────────────────────────────────────────────────────────────

class TestParseBudgetLimit:
    """Table-driven tests for the budget parser."""

    @pytest.mark.parametrize("query, expected", [
        # Lakh variants
        ("fees under 1.5 lakh",                150_000),
        ("budget of 2 lakhs",                  200_000),
        ("I can afford 1L per year",            100_000),
        ("college under 2.5 lakhs",             250_000),
        # k (thousands)
        ("budget 90k",                          90_000),
        ("under 150k",                          150_000),
        # Plain numbers (>= 5000)
        ("fee below 75000",                     75_000),
        ("college costing 1,50,000",            150_000),  # comma-separated
        # Semester doubling
        ("1 lakh per semester",                 200_000),  # 1L × 2
        ("50k per sem",                         100_000),  # 50k × 2
        ("budget 75000 per semester",           150_000),  # 75k × 2
        # Zero/no match cases
        ("which college is best for me",        None),
        ("I scored 78 percent",                 None),     # 78 < 5000
        ("founded in 2004",                     None),     # 2004 < 5000
    ])
    def test_parse_budget_limit(self, query, expected):
        result = parse_budget_limit(query)
        assert result == expected, f"Query: {query!r} → got {result}, expected {expected}"

    def test_semester_detection_half_yearly(self):
        assert parse_budget_limit("1 lakh half-yearly") == 200_000

    def test_semester_detection_six_month(self):
        assert parse_budget_limit("budget 80000 six month instalment") == 160_000

    def test_large_plain_number(self):
        # 250000 as plain number, annual
        assert parse_budget_limit("fee is 250000") == 250_000

    def test_small_number_ignored(self):
        # Numbers below 5000 should not be treated as budget
        assert parse_budget_limit("I want a college with 4 courses") is None


# ──────────────────────────────────────────────────────────────
# parse_cutoff_score
# ──────────────────────────────────────────────────────────────

class TestParseCutoffScore:
    """Table-driven tests for the cutoff percentage parser."""

    @pytest.mark.parametrize("query, expected", [
        # Percent symbol
        ("I scored 78%",                        78.0),
        ("aggregate of 85.5 percent",           85.5),
        ("85 percentage in class 12",           85.0),
        # Scored/marks keywords
        ("I scored 72 in boards",               72.0),
        ("got 65 marks in class 12",            65.0),
        ("cutoff of 80",                        80.0),
        # Fallback with scoring words
        ("score 55",                            55.0),
        ("aggregate 90",                        90.0),
        # Values outside 30–100 should not match as cutoffs
        ("I scored 25",                         None),   # below 30
        ("I scored 105",                        None),   # above 100
        # No scoring context
        ("which MBA colleges are there",        None),
        ("fees under 1.5 lakh",                 None),
    ])
    def test_parse_cutoff_score(self, query, expected):
        result = parse_cutoff_score(query)
        assert result == expected, f"Query: {query!r} → got {result}, expected {expected}"

    def test_budget_number_not_parsed_as_cutoff(self):
        # "90k" in a budget query should NOT be interpreted as cutoff 90
        result = parse_cutoff_score("budget 90k, scored 75")
        assert result == 75.0, f"Expected 75.0, got {result}"

    def test_decimal_cutoff(self):
        assert parse_cutoff_score("I got 82.5 percent") == 82.5

    def test_cutoff_keyword(self):
        assert parse_cutoff_score("cutoff of 70 for engineering") == 70.0


# ──────────────────────────────────────────────────────────────
# parse_placement_floor
# ──────────────────────────────────────────────────────────────

class TestParsePlacementFloor:
    """Table-driven tests for the placement floor parser."""

    @pytest.mark.parametrize("query, expected", [
        # LPA suffix
        ("placement above 5 LPA",               5.0),
        ("placements at least 6 lpa",           6.0),
        ("placement over 4.5 lpa",              4.5),
        # Lakh suffix
        ("placement above 5 lakhs",             5.0),
        ("placements greater than 7 lakh",      7.0),
        # L suffix
        ("placement of 6L",                     6.0),
        # Min/minimum
        ("placement minimum 8",                 8.0),
        ("min placement 5",                     5.0),
        # No placement constraint in query
        ("engineering college with hostel",     None),
        ("fees under 1.5 lakh",                 None),
        ("which MBA is cheapest",               None),
    ])
    def test_parse_placement_floor(self, query, expected):
        result = parse_placement_floor(query)
        assert result == expected, f"Query: {query!r} → got {result}, expected {expected}"

    def test_combined_budget_and_placement(self):
        # Both budget and placement in one query; placement parser only extracts placement
        result = parse_placement_floor("MBA under 1.8 lakh with placement above 5 lakhs")
        assert result == 5.0

    def test_decimal_placement(self):
        assert parse_placement_floor("placement above 6.5 lpa") == 6.5


# ──────────────────────────────────────────────────────────────
# PII Scrubbing and Pydantic Validation Tests
# ──────────────────────────────────────────────────────────────

def test_scrub_pii():
    from answer import scrub_pii
    # Email masking
    assert scrub_pii("my email is admin.counselor@college.org") == "my email is [EMAIL_MASKED]"
    # Phone number masking
    assert scrub_pii("contact me at 9876543210 immediately") == "contact me at [PHONE_MASKED] immediately"
    # Income metrics masking
    assert scrub_pii("my family income is 3.5 lakhs") == "my family income [INCOME_MASKED]"
    assert scrub_pii("income Rs 500000") == "income [INCOME_MASKED]"
    # Mixed sentence
    mixed = "Contact testing@mme.com or 1234567890. Family income is Rs 400000."
    assert scrub_pii(mixed) == "Contact [EMAIL_MASKED] or [PHONE_MASKED]. Family income [INCOME_MASKED]."


def test_pydantic_schema_validation():
    from src.models import CounselorResponse
    from pydantic import ValidationError

    # Valid schema structure
    valid_data = {
        "answer": "Himalayan College of Engineering (C003) is within your budget.",
        "citations": ["C003"],
        "answered": True,
        "reason_if_unanswered": None,
    }
    model = CounselorResponse.model_validate(valid_data)
    assert model.answered is True
    assert model.citations == ["C003"]

    # Missing mandatory field 'answered'
    invalid_data = {
        "answer": "Some answer",
        "citations": [],
    }
    with pytest.raises(ValidationError):
        CounselorResponse.model_validate(invalid_data)

    # Invalid type for 'citations'
    invalid_types = {
        "answer": "Some answer",
        "citations": "C003",  # should be list, not string
        "answered": True,
    }
    with pytest.raises(ValidationError):
        CounselorResponse.model_validate(invalid_types)
