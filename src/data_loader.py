"""Loads the college dataset and builds a per-college text document used for
retrieval (structured fields flattened into readable sentences + the free-text
`about` field). Keeping this in one place means the retriever and the LLM
context builder always see the exact same representation of a row.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_colleges.csv"


def load_colleges(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize booleans / strings once, at load time, not scattered everywhere
    df["hostel_available"] = df["hostel_available"].astype(str).str.strip().str.lower() == "yes"
    df["courses_offered_list"] = df["courses_offered"].apply(
        lambda s: [c.strip() for c in str(s).split(";") if c.strip()]
    )
    return df


def row_to_document(row: pd.Series) -> str:
    """Flatten one college row into a single text blob for TF-IDF indexing.
    Structured fields are spelled out in words (not just raw numbers) so the
    vectorizer's vocabulary actually contains terms students search with."""
    placement = (
        "placement not reported / not applicable"
        if row["avg_placement_lpa"] == 0
        else f"average placement {row['avg_placement_lpa']} lakhs per annum"
    )
    hostel = "hostel available" if row["hostel_available"] else "no hostel"
    return (
        f"{row['name']} ({row['college_id']}) in {row['city']}, {row['state']}. "
        f"Type: {row['type']}. Courses offered: {', '.join(row['courses_offered_list'])}. "
        f"Annual fees: Rs {row['annual_fees_inr']} per year. "
        f"Last year cutoff: {row['last_year_cutoff_pct']} percent aggregate (hard minimum). "
        f"Total seats: {row['total_seats']}. {hostel}. NAAC grade: {row['naac_grade']}. "
        f"{placement}. Established {row['established_year']}. {row['about']}"
    )


def build_corpus(df: pd.DataFrame) -> list[str]:
    return [row_to_document(r) for _, r in df.iterrows()]
