"""Retrieval layer.

Design decision (documented in README): with 15 rows, a neural embedding
call per query buys nothing but latency and cost. We use:

  1. Metadata filtering (pandas) for anything the question states as a hard
     constraint: fee ceiling, minimum cutoff, college type, hostel, course
     keyword, or an explicitly named college. This is exact, not fuzzy, and
     is what actually prevents wrong-fee/wrong-cutoff answers.
  2. TF-IDF cosine similarity over the flattened per-college document (see
     data_loader.row_to_document) to rank the remaining rows for anything
     that isn't a hard filter -- especially free-text questions answerable
     only from `about` (scholarships, "best for me", etc).

If filtering leaves nothing, we say so (this is what powers a graceful
"I don't know" instead of the model quietly hallucinating a college).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import build_corpus
from .parsers import parse_budget_limit, parse_cutoff_score, parse_placement_floor

# Re-export parsers so any code that imports them from src.retriever
# continues to work without changes (backward compatibility).
__all__ = [
    "BaseRetriever",
    "Retriever",
    "parse_budget_limit",
    "parse_cutoff_score",
    "parse_placement_floor",
]


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Contract for all retrieval implementations."""
        pass



class Retriever(BaseRetriever):
    def __init__(self, df: pd.DataFrame, vectorizer: TfidfVectorizer | None = None):
        self.df = df.reset_index(drop=True)
        self.corpus = build_corpus(self.df)
        self.vectorizer = vectorizer or TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        if vectorizer is None:
            self.doc_matrix = self.vectorizer.fit_transform(self.corpus)
        else:
            self.doc_matrix = self.vectorizer.transform(self.corpus)


    # ---------- metadata filtering ----------

    def filter_by_named_college(self, query: str) -> pd.DataFrame:
        """Substring and multi-word match against college names.
        Ensures name-specific queries only load that college, while preventing single-token
        matches (like 'Ganga') from conflating C002 and C014."""
        q = query.lower()
        q_clean = re.sub(r'[^\w\s]', ' ', q)
        q_words = set(q_clean.split())
        
        stops = {"of", "and", "in", "at", "the", "institute", "college", "university", "school", "technology", "sciences", "polytechnic", "management"}
        hits = []
        
        for _, row in self.df.iterrows():
            name = row["name"].lower()
            name_clean = re.sub(r'[^\w\s]', ' ', name)
            words = name_clean.split()
            
            # Check 1: Multi-word/bigram match (excluding stopword-only bigrams)
            bigram_match = False
            for i in range(len(words) - 1):
                bigram = " ".join(words[i:i + 2])
                bigram_words = words[i:i + 2]
                if all(w in stops for w in bigram_words):
                    continue
                if bigram in q_clean:
                    bigram_match = True
                    break
            
            if bigram_match:
                hits.append(row)
                continue
                
            # Check 2: Require at least 2 distinctive words of the name to be present in query
            distinctive = [w for w in words if w not in stops]
            matched_distinctive = [w for w in distinctive if w in q_words]
            if len(matched_distinctive) >= 2:
                hits.append(row)
                continue
                
            # Check 3: Query contains the full exact name
            if name_clean in q_clean:
                hits.append(row)
                continue

        if hits:
            return pd.DataFrame(hits)
        return pd.DataFrame(columns=self.df.columns)

    def _apply_type_filter(self, df: pd.DataFrame, q: str) -> tuple[pd.DataFrame, bool]:
        if "government" in q:
            return df[df["type"] == "Government"], True
        elif "deemed" in q:
            return df[df["type"] == "Deemed"], True
        elif "private" in q:
            return df[df["type"] == "Private"], True
        return df, False

    def _apply_hostel_filter(self, df: pd.DataFrame, q: str) -> tuple[pd.DataFrame, bool]:
        hostel_keywords = ["with hostel", "have hostel", "has hostel", "hostel facility", "hostel facilities", "hostel available", "hostels available", "having hostel", "accommodation"]
        if "hostel" in q:
            if any(kw in q for kw in hostel_keywords) or re.search(r"\b(?:have|has|with|having|available)\s+hostels?\b", q):
                return df[df["hostel_available"]], True
        return df, False

    def _apply_course_filter(self, df: pd.DataFrame, q: str) -> tuple[pd.DataFrame, bool]:
        course_keywords = {
            "engineering": "B.Tech|Diploma CSE|Diploma ME|Diploma Civil",
            "mba": "MBA",
            "b.tech": "B.Tech",
            "diploma": "Diploma|D.Pharm",
            "law": "LLB|LLM|BA-LLB",
            "medical": "MBBS|BDS",
            "design": "B.Des|M.Des|B.F.A",
            "hotel": "BHM|Hospitality|Culinary",
            "pharmacy": "B.Pharm|D.Pharm|M.Pharm",
            "commerce": "B.Com|M.Com|CA-Foundation",
            "business": "BBA|MBA|PGDM",
            "management": "BBA|MBA|PGDM",
            "media": "BJMC|BA-Film|MA-Mass Comm",
            "nursing": "Nursing",
        }
        matched_tokens = []
        for kw, token in course_keywords.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", q):
                matched_tokens.append(token)
        if matched_tokens:
            combined_pattern = "|".join(matched_tokens)
            return df[df["courses_offered"].str.contains(combined_pattern, case=False, na=False)], True
        return df, False

    def _apply_budget_filter(self, df: pd.DataFrame, query: str) -> tuple[pd.DataFrame, bool]:
        budget_limit = parse_budget_limit(query)
        if budget_limit is not None:
            return df[df["annual_fees_inr"] <= budget_limit], True
        return df, False

    def _apply_cutoff_filter(self, df: pd.DataFrame, query: str) -> tuple[pd.DataFrame, bool]:
        cutoff_score = parse_cutoff_score(query)
        if cutoff_score is not None:
            return df[df["last_year_cutoff_pct"] <= cutoff_score], True
        return df, False

    def _apply_placement_filter(self, df: pd.DataFrame, query: str) -> tuple[pd.DataFrame, bool]:
        placement_floor = parse_placement_floor(query)
        if placement_floor is not None:
            return df[df["avg_placement_lpa"] >= placement_floor], True
        return df, False

    def filter_by_structured_constraints(self, query: str) -> tuple[pd.DataFrame, bool]:
        """Returns (filtered_df, any_constraint_applied).
        Ensures if a constraint is applied and matches zero rows, it propagates
        as an empty result (true negative) instead of falling back to no filter."""
        df = self.df
        q = query.lower()
        applied = False

        # Apply each constraint pre-filter in sequence
        for filter_name, filter_fn in [
            ("type", self._apply_type_filter),
            ("hostel", self._apply_hostel_filter),
            ("course", self._apply_course_filter),
            ("budget", self._apply_budget_filter),
            ("cutoff", self._apply_cutoff_filter),
            ("placement", self._apply_placement_filter),
        ]:
            # Budget, cutoff and placement floor checks require original query casing
            arg = query if filter_name in ("budget", "cutoff", "placement") else q
            df, is_applied = filter_fn(df, arg)
            if is_applied:
                applied = True

        return df, applied


    # ---------- semantic ranking ----------

    def semantic_rank(self, query: str, candidates: pd.DataFrame | None = None, top_k: int = 5):
        pool = self.df if candidates is None or candidates.empty else candidates
        idx = pool.index.tolist()
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix[idx]).flatten()
        ranked = sorted(zip(idx, sims), key=lambda x: x[1], reverse=True)
        top_idx = [i for i, score in ranked[:top_k]]
        return self.df.loc[top_idx]

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Retrieve the top matching candidates.
        An empty result is a valid true negative when a filter eliminates all records.

        Existential expansion: for queries that genuinely ask about ALL colleges
        (any, every, all, each, or multi-word phrases like 'does any', 'are there any'),
        we expand top_k to the full dataset to prevent retrieval-bound false negatives.

        Note: 'which' and 'list' are intentionally excluded from this pattern — they
        appear in almost every comparison query ('which is cheaper', 'list options for me')
        and would triple inference cost and context size unnecessarily.  True exhaustive
        questions use 'any' / 'all' / 'every' semantics."""
        q = query.lower()
        # Pattern deliberately excludes 'which' and 'list' — see docstring above.
        existential_pattern = (
            r"\b(any|every|all|each)\b"
            r"|\b(does\s+any|do\s+any|is\s+there\s+any|are\s+there\s+any)\b"
        )
        if re.search(existential_pattern, q):
            top_k = max(top_k, len(self.df))

        named = self.filter_by_named_college(query)
        if not named.empty:
            return self.semantic_rank(query, named, top_k=top_k)

        constrained, applied = self.filter_by_structured_constraints(query)
        if applied:
            if constrained.empty:
                return constrained  # true negative -- propagate emptiness
            return self.semantic_rank(query, constrained, top_k=top_k)

        return self.semantic_rank(query, None, top_k=top_k)

