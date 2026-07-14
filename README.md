# Make My Education — College RAG Prototype

Answers natural-language questions about the 15-college sample dataset, grounded in the data and cited by `college_id`, through a single CLI entry point: `answer.py`.

##  Quick Start (60 Seconds)

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your API key and run a query:**
   ```bash
   # Windows (PowerShell):
   $env:GEMINI_API_KEY="your-key-here"
   python answer.py "Which colleges offer an MBA, and what do they cost?"

   # Windows (CMD):
   set GEMINI_API_KEY=your-key-here
   python answer.py "Which colleges offer an MBA, and what do they cost?"

   # Linux/macOS:
   export GEMINI_API_KEY="your-key-here"
   python answer.py "Which colleges offer an MBA, and what do they cost?"
   ```

## Running it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your-key-here          # Windows: set GEMINI_API_KEY=...
python answer.py "Which colleges offer an MBA, and what do they cost?"
```

Regenerate `answers.md` for the seven published questions:
```bash
python run_all.py
```

Run the eval suite (11 cases, one per data trap called out in `DATA_DICTIONARY.md`):
```bash
python evals/run_evals.py
```

Run the unit tests for the parsing layer (48 cases, no network or API key needed):
```bash
pytest tests/test_parsers.py -v
```

Swap the model at runtime if you want:
```bash
MME_MODEL=gemini-2.5-flash-lite python answer.py "..."   # cheaper, weaker instruction-following
```

## How a question actually flows through the system

```
question
   │
   ▼
Retriever (src/retriever.py)
   ├─ 1. named-college match (multi-word bigram match + 2-distinctive-word
   │      fallback — stops "Ganga" alone from matching both C002 and C014)
   ├─ 2. structured filter (pandas: budget/cutoff/type/hostel/course/placement)
   │      — a filter that matches zero rows is a true negative and stays
   │        empty; it never silently widens back out
   └─ 3. TF-IDF cosine ranking over whatever survives the filters (or the
         whole dataset, if no hard filter applied) → top-k rows returned
   │
   ▼
Context builder (src/prompts.py)
   turns the top-k rows into a labelled block, one paragraph per college_id
   │
   ▼
LLM (src/llm.py) — gemini-3.1-flash-lite via the google-genai SDK
   system prompt encodes all 11 grounding rules, JSON-only output enforced
   │
   ▼
answer.py
   ├─ citation re-validation: any college_id the model cites that wasn't
   │    actually in the retrieved context gets dropped
   ├─ logs question + result + real token counts/latency/cost to
   │    logs/queries.jsonl (never to stdout)
   └─ stdout gets exactly the required JSON object, nothing else
```

## Design choices, and why

**Model.** Running default is `gemini-3.1-flash-lite`, overridable via `MME_MODEL`. During development, `gemini-2.5-flash` returned a flat 404 on the free-tier quota I was using — flash-lite was the most capable model actually reachable. It clears all 11 eval cases at 100%, and it's the cheaper of the two, so "cheapest model that passes the eval" won out over reflexively reaching for the bigger one. For a paid deployment, setting `MME_MODEL=gemini-2.5-flash` buys stronger instruction-following at roughly **1.4×** the token cost, using this same query mix — worked out in Part D below. `gemini-2.0-flash` isn't even an option anymore; Google shut it down on June 1, 2026. Smaller note: this uses `google-genai`, not the older `google-generativeai` package, which Google deprecated in November 2025.

**Retrieval.** No embedding calls, on purpose. At 15 rows, TF-IDF plus cosine similarity ranks candidates about as well as a neural embedding would, costs nothing, adds no latency, and is fully inspectable — I can print the vocabulary and see exactly why a document ranked where it did. That stops being true well before 10,000 colleges; see Scaling below for what changes then.

**Filtering happens before ranking, not instead of it.** Budget, cutoff, college type, hostel availability, course category, and placement floor are hard constraints — get one of these wrong and you've told a student they can afford a college they can't, or that they're eligible for one they're not. Those are checked with plain pandas comparisons, never left to the LLM or to approximate similarity. TF-IDF only decides ordering *within* whatever survives the hard filters, and if a structured filter matches nothing, that's answered `false` immediately, without spending a token on a call the model can't win anyway.

A few narrower calls worth flagging on their own:
- Budget parsing understands "X lakh", "Xk", "1.5L", and plain numbers ≥ 5000, and doubles per-semester amounts before comparing against the (annual) fee column. `DATA_DICTIONARY.md` calls a silent per-year answer to a per-semester question "the worst failure available" — so this one is treated as a hard requirement, not a nice-to-have.
- Course keywords are OR-matched, not break-on-first-match — an earlier version dropped the second category whenever a query named two courses at once.
- A yes/no question about one college's hostel ("Does North Ridge have a hostel?") is a lookup, not a filter — the hostel filter only fires when the phrasing is asking for a *list* of colleges with hostels.
- Placement started out *without* a hard filter — it ran through TF-IDF ranking and the model's own reasoning alone. Testing turned up no wrong answers, but no true-negative guarantee either, so I added `_apply_placement_filter` alongside the others rather than leave it as the one constraint the model could still talk itself into being wrong about.
- `avg_placement_lpa = 0` gets explained from the `about` field, never presented as the worst package. The eval suite checks this directly against C006.
- A diploma-only result is labelled as a diploma, not silently dropped from an "engineering colleges" question and not silently counted as a degree either. Shivalik (C005) can legitimately surface for adjacent questions; it just has to be labelled honestly when it does.
- Similar college names are disambiguated by requiring a multi-word match rather than one shared token, specifically so "Ganga" alone can't match both Ganga Valley University (C002) and Ganga Institute of Commerce (C014).

**Citations are checked in code, not trusted from the model.** `answer.py` strips out any citation the model returns that wasn't actually part of the retrieved context. Of everything in this repo, this is probably the single line doing the most work to keep a hallucinated citation from ever reaching a student.

## Testing & verification

Real output from real runs — nothing below is aspirational.

**Unit tests.** The parsing layer — budget/cutoff/placement extraction, PII scrubbing, schema validation — is pure functions with no network calls, so it's fully unit-testable on its own. 48 table-driven tests live in `tests/test_parsers.py`:
```
$ pytest tests/test_parsers.py -v
...
48 passed in 3.97s
```

**Eval suite.** `evals/cases.json` has 11 cases, one for each trap called out in `DATA_DICTIONARY.md`: semester-vs-annual fees, the cutoff hard floor, `avg_placement_lpa = 0`, diploma vs. degree, the two Ganga colleges, refusing on a college that doesn't exist, and so on. `evals/run_evals.py` runs each one through the real pipeline — real retrieval, real Gemini call — and checks the result against expected citations and required/forbidden substrings, since exact wording varies run to run and shouldn't be graded on it.
```
[PASS] unit_conversion_semester: What are the fees at Doon Business School per semester?
[PASS] cutoff_hard_floor_true_negative: I scored 40% in Class 12, which engineering colleges can I get into?
[PASS] placement_zero_not_worst: What is the average placement package at Nainital Institute of Medical Sciences?
[PASS] diploma_not_degree: Does Shivalik Government Polytechnic offer a B.Tech degree?
[PASS] similar_name_disambiguation: Does Ganga Institute of Commerce offer an MBA?
[PASS] unknown_course_refusal: Does Terai Technical University offer a B.Des programme?
[PASS] unknown_college_refusal: What are the annual fees at Garhwal Central University?
[PASS] scholarship_freetext: Which colleges offer scholarships or fee concessions for low-income families?
[PASS] government_hostel_filter: List the government colleges that have hostel facilities.
[PASS] extra_costs_beyond_tuition: What is the total yearly cost to study at Rishikesh Institute of Design, including everything?
[PASS] best_college_judgment_call: Which engineering college is best for me? I scored 80% and have a budget of Rs 1.5 lakh/year.

11/11 passed (100%)
```
Worth saying plainly: 11 cases isn't exhaustive, and I wrote both the cases and the code they're testing, so there's an obvious risk of mostly confirming what I already expected to work. The unseen questions Make My Education runs against this are the real test.

**Cost and latency, from that same run**, straight from `logs/queries.jsonl`:
```
Queries measured: 17
Avg input tokens / query: 1211.5
Avg output tokens / query: 132.0
Avg latency / query: 1.243 s
Model used: gemini-3.1-flash-lite
Avg cost / query: INR 0.0436
Cost per 1,000 queries: INR 43.58
```
(17, not 18 — one eval case, the below-cutoff true negative, short-circuits before any LLM call and never gets logged with a cost entry.)

**PII scrubbing.** Emails, phone numbers, Aadhaar numbers, and family-income figures get masked before anything is written to `logs/queries.jsonl`:
```
Input:  Contact testing@mme.com or 9876543210. Aadhaar: 1234 5678 9012. Family income is Rs 400000.
Logged: Contact [EMAIL_MASKED] or [PHONE_MASKED]. Aadhaar: [AADHAAR_MASKED]. Family income [INCOME_MASKED].
```

**What this doesn't cover.** This confirms the deterministic layer — parsing, filtering, schema validation, PII masking — and 11 specific question patterns end to end. It doesn't confirm every phrasing of those same questions gets handled the same way. The CLI output contract under transient API failures (rate limit, timeout, 5xx) is now tested: `answer.py` wraps the LLM call in a try-except and returns `{"answered": false, "reason_if_unanswered": "API error (...)"}` rather than leaking a traceback. Remaining gap: phrasing variation and multilingual inputs are not specifically covered.

## Known limitations

I'd rather list these than have someone else find them first.

- **Existential questions only see the retrieved context.** "Does any college offer X?" is only as reliable as top-k retrieval — if the fact lives in a college that wasn't pulled into context, the system will answer no based on what it *was* shown, not what's true across all 15 rows. Fine at this scale; worth knowing about before it isn't.
- **Hindi and regional-language queries bypass the hard-constraint filters.** Make My Education's primary users are students from Tier 2/3 Indian cities, many of whom will query in Hindi or Hinglish. The model translates and reasons over Hindi input reasonably well, but the budget/cutoff/course/placement regexes are English-only — a Hindi query like "एक लाख बजट" runs on TF-IDF alone, without the programmatic pre-filter that catches a wrong-fee answer before any API call. The fix is to translate the query to English first (a one-line call to the same Gemini model) before running the parser chain. This is the most impactful missing feature for the actual user base.
- **No conversation memory.** Every call to `answer.py` is stateless, which matches the required CLI interface but rules out a multi-turn "and what about hostels there?" follow-up.
- **`usage_metadata` field names have moved between `google-genai` releases before.** The code guards with `getattr(..., 0)`, but if token counts start reading as zero after a version bump, that's the first place to check — `print(vars(resp.usage_metadata))` will show what actually came back.

## What I'd do differently with more time

- Widen `top_k` specifically for existential/exhaustive questions ("does *any* college...", "*all* colleges that...") instead of a fixed default — the most direct fix for the limitation above.
- Add a small reranking step for queries that stack more than two constraints at once, where TF-IDF ordering alone gets less reliable.
- Derive the course-keyword map in `retriever.py` from the dataset's actual course vocabulary instead of a hardcoded dict, so it doesn't silently miss a category I didn't anticipate.
- Add a query classifier that decides *whether retrieval is needed at all* — a question like "what does NAAC stand for" doesn't need a college lookup, and right now everything goes through the same path regardless.

---

## Scaling: 15 → 10,000 → 100,000 colleges, 5M users

- **TF-IDF → a real vector store.** Brute-force cosine over 15 documents is instant; over 100k it isn't. Move to pgvector, Qdrant, or Pinecone once corpus size makes per-query cosine slow, and push the structured filtering into SQL `WHERE` clauses on indexed columns so rows get eliminated before any embedding comparison runs at all.
- **Incremental indexing.** An `updated_at` column so a write re-embeds only the college that changed, not the whole corpus.
- **Caching, exact first, then semantic.** Admissions questions repeat a lot across students. An exact-match cache is nearly free and should go in first; a semantic cache for near-duplicates is worth the added complexity once volume justifies it. At 50k queries/month this is a bigger lever than model choice.
- **What actually breaks first at 50,000 queries/month is latency, not cost.** Token cost at flash pricing stays small even at that volume (see Part D), but one sequential LLM call per query with no caching or streaming will feel slow under peak load in a live chat UI. The exact-question cache is the first fix — cheap to build, and it removes repeated queries from the cost line entirely.

## Where the ambiguity calls got made explicitly

| Ambiguous case | Decision | Why |
|---|---|---|
| "Best college" | Never pick one winner — present 2–3 options with trade-offs | "Best" depends on what the student weighs most; picking a single winner silently penalises whoever cares about something else |
| Budget given per semester vs. per year | Convert and show the arithmetic | `DATA_DICTIONARY.md` calls a silent mismatch here the worst failure in the exercise |
| Unknown college or course | Refuse cleanly, `answered: false` | Guessing from a similar name is exactly the mistake the data dictionary warns about |
| Diploma vs. degree | Label it explicitly; don't silently filter it out or silently count it as a degree | C005 is still relevant to some questions — the failure mode is mislabelling, not inclusion |
| Costs beyond tuition | Surface them when the question is about budget or total cost | A technically-correct number that omits mandatory hostel/mess/studio charges is misleading in practice |
| Yes/no hostel question about a named college | Treated as a lookup, not run through the hostel-list filter | Filtering would wrongly drop the named college from its own context |

---

## Part B — Proof I've shipped something before

**What it was:** a hybrid retrieval and routing search assistant helping students find degrees, colleges, and scholarships — roughly 4 million monthly search queries at peak.

**My part:** I built the hybrid retrieval pipeline (dense embedding search over Qdrant, combined with exact metadata pre-filtering in Postgres) and the citation-validation layer on the backend.

**What broke, twice:**
1. After an upstream model update, the LLM started returning citation IDs that didn't exist — `C999`-style IDs, sometimes references bleeding in from a previous user's session — under heavy prompt compression. The UI dutifully rendered cards pointing at colleges that weren't there.
2. During a scholarship-announcement traffic spike, the model's JSON generation stopped closing properly and got stuck looping under rate-limit pressure from upstream. That ran up **$1,420 in token cost over about 8 hours** before anyone was paged.

**What changed after:** a code-level layer that intersects returned citations against the actual retriever output before anything reaches the UI or database, dropping any hallucinated ID instantly — the direct ancestor of the citation check in `answer.py` here. Also added exponential backoff with jitter on retries, to avoid a thundering-herd effect during rate-limit events, and a two-stage cache (exact match, then semantic) that cut monthly inference cost by 34% and brought p95 latency on repeated queries from 2.2s down to about 80ms.

## Part C — A few honest answers

**Keeping cost down as usage grows:** an exact-question cache first — admissions questions repeat almost verbatim across students, so this is the highest-leverage move, before touching the model at all. I'd default to `gemini-2.5-flash` for reliability and only drop to flash-lite once load testing shows the grounding rate holds up. Stay on TF-IDF until the corpus is actually big enough that brute-force cosine is the real bottleneck, not before.

**Making sure it never states a wrong fee or cutoff:** every number in an answer comes from the retriever, placed into context verbatim from the CSV — the model is never in a position to invent a number it wasn't given. The system prompt requires a citation for every numeric claim, and those citations get re-checked in code against what was actually retrieved, not trusted from the model's own output.

**First thing I'd build if I joined tomorrow:** a short structured intake — budget, score, preferred course, hostel needed, 3–4 questions — captured upfront instead of discovered piecemeal through conversation. Most of the failure modes I'd expect in an open-ended admissions chat come from the system not yet knowing the student's actual constraints, not from the model reasoning badly once it does.

**Measuring whether this is actually helping students:** whether a session ends in a shortlist or an application, not whether the conversation "reads" helpful. I'd track drop-off and time-to-shortlist against the non-AI search flow, and have someone manually review a sample of answered conversations every week specifically for grounding errors — a confident-sounding wrong answer is worse than a visible refusal, and that's exactly the kind of failure that won't show up in aggregate satisfaction numbers.

## Part D — Cost, measured

Model: **gemini-3.1-flash-lite**, via Google AI Studio. Exchange rate used: **1 USD = ₹87** (July 2026). Embedding cost: **₹0** — TF-IDF has no embedding step, at this dataset size or any size up to where the retrieval architecture in Scaling above needs to change.

| Metric | Measured value |
|---|---|
| Avg input tokens / query | 1211.5 |
| Avg output tokens / query | 132.0 |
| Avg end-to-end latency / query | 1.243 s |
| Model + cost per 1M tokens | gemini-3.1-flash-lite — $0.25 input / $1.50 output |
| Cost per 1,000 queries | ₹43.58 |
| One-time embedding cost | ₹0 (TF-IDF) |

These come from `logs/queries.jsonl`, not an estimate — real token counts from the API response, real `time.perf_counter()` latency.

At 50,000 queries/month, using the same per-query averages:
$0.25 × (1211.5 / 1M) × 50,000 + $1.50 × (132 / 1M) × 50,000 = $15.14 + $9.90 ≈ **$25/month (~₹2,178/month)**.

**The 1.4× claim for gemini-2.5-flash, worked out.** Using the same query mix (1211.5 input / 132 output tokens) and gemini-2.5-flash pricing ($0.30/1M input, $2.50/1M output):
$0.30 × (1211.5 / 1M) + $2.50 × (132 / 1M) = $0.000363 + $0.000330 = **$0.000693/query**.
Versus flash-lite: $0.25 × (1211.5 / 1M) + $1.50 × (132 / 1M) = $0.000303 + $0.000198 = **$0.000501/query**.
Ratio: $0.000693 / $0.000501 = **1.38×** (roughly 1.4×) — confirmed.

At that volume, latency is the real bottleneck, not the bill — the fix that matters first is the exact-question cache from Scaling, which removes repeated queries from the cost line entirely rather than trying to shave the per-query rate further.