# Handoff → Phase 3: Generation and Citation Layer

Written 2026-08-08. Phase 1 is complete. This document is for a fresh session
picking up Phase 3 and assumes no memory of the previous ones.

Companion documents: `HANDOFF.md` (full project history, evaluation methodology,
what was tried and failed) and `Project 6.docx` (the original spec, in the repo
root). Read this one first; reach for those when you need the why behind a
number.

---

## 1. What this project is

A RAG system over real SEC EDGAR 10-K filings for five companies (AAPL, MSFT,
JPM, GS, TSLA), with role-based access control modelled on investment-bank
information barriers.

It is a **SEC-filings implementation of "Project 6: RAG Pipeline with Hybrid
Search Over Internal Docs."** Project 6 specifies six phases against generic
internal documentation; this repo implements them against filings and adds a
compliance layer Project 6 does not ask for (RBAC, MNPI/investment-advice
guardrails, SEC 17a-4 audit trail, MCP server).

| Phase | Status |
|---|---|
| Tech stack | Done — `gpt-4o`, `text-embedding-3-small`, ChromaDB, `rank_bm25`, FastAPI, Docker |
| 1. Ingestion & chunking | **Done** — three strategies, dedup, chunk provenance |
| 2. Hybrid retrieval | **Done** — dense + BM25 + RRF + cross-encoder, built to spec |
| 3. Generation & citation | **Done** — bracketed citations, LLM-judge verification, composite confidence, structured "I don't know". Smoke-tested against the live index on all three paths; `citation_accuracy` wired into the eval but **not yet run over the 20-question set** |
| 4. Evaluation | Partial — 20 questions; spec wants 50+ hand-written |
| 5. API & dashboard | Partial — API yes, no query dashboard |
| 6. Portfolio | Partial — case study strong, no demo video |

Agreed build order for what remains: **3 → 4 → 5**.

**Phase 3 outcome (2026-08-08).** Everything below in §3 is implemented behind
`generate_grounded_answer()` in [src/generation/answer.py](src/generation/answer.py),
which the REST route, the MCP server and `run_rag_pipeline()` all call — that
was the architectural decision §4 forced. 78 new unit tests, 249 total, ruff
clean. Verified live on three queries:

| Path | Result |
|---|---|
| Factual (AAPL revenue) | Model emitted `[3]`; judge returned SUPPORTED; accuracy 1.00, coverage 1.00, confidence 0.97 *high* |
| Comparative (JPM vs GS) | Retrieval 0.30 cleared the gate, model refused, `model_refused` structure attached; confidence 0.15 *low* — the top-5 budget bug in §5.2, now visible at query time instead of only in a results file |
| Out-of-corpus | Cross-encoder logits ≈ −11 → relevance 0.00; gate fired **before** generation, no LLM call spent |

Still open: one eval run to produce the first `citation_accuracy` figure
(§5). The metric is wired in and exercised offline, but no run over the
20-question set has been made.

---

## 2. Repo state

- Branch `main`. **Nothing since `ea0aea8` is committed** — Phase 1, the tech
  stack migration, and the evaluation rework are all uncommitted working tree.
- 171 tests pass, ruff clean.
- **Use `./.venv/Scripts/python.exe`, never bare `python`.** The `python` on
  PATH is system Python 3.13 with *some* dependencies (langchain,
  langchain-openai) but not `rank_bm25`, `transformers`, or
  `sentence-transformers`. The partial overlap is the trap: LangChain-only
  scripts run fine, then `pytest` fails collection on five modules and reads as
  a broken suite rather than a wrong interpreter.

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
./.venv/Scripts/python.exe -m ruff check src tests evaluation scripts
```

---

## 3. What Phase 3 asks for

Project 6's own words: this is *"the quality layer most RAG systems skip
entirely."* It also prescribes the case-study line — *"X% faithfulness and **Y%
citation accuracy**"* — which is unwritable today. That sentence is the goal.

Four pieces, in dependency order:

### 3.1 Bracketed citations over numbered context blocks

Instruct the model to answer only from context, cite with `[1]`, `[2]`, and say
explicitly when the context is insufficient.

**Good news: the context blocks are already numbered.**
`format_documents()` in [src/generation/chains.py:35](src/generation/chains.py:35)
emits `[Document 1] (Source: …, Company: AAPL, Section: Item 1A …)` per chunk.
The numbering exists; nothing instructs the model to reference it.

What needs to change is the prompt. `RAG_SYSTEM_PROMPT` in
[src/generation/prompts.py:3](src/generation/prompts.py:3) rule 3 currently asks
for *prose* citations — *"According to Apple's 2024 10-K, Item 7 MD&A…"*. Prose
citations cannot be parsed, so they cannot be verified, so no citation-accuracy
metric can exist. Replace with bracketed references keyed to the block numbers,
and keep the existing rules 5–8 (no investment advice, section attribution)
which are load-bearing for the compliance layer.

### 3.2 Citation verification

Parse `[n]` from the answer, pair each cited claim with the chunk it points at,
send the pair to an LLM-as-judge, flag unsupported citations.

Design decisions worth making deliberately:
- **Claim segmentation.** A citation attaches to a claim, not to the whole
  answer. Sentence-level is the obvious unit; note the project already has a
  sentence splitter at `_SENTENCE_RE` in
  [src/ingestion/semantic_chunking.py](src/ingestion/semantic_chunking.py) that
  handles `$1.5 billion` and `Item 7A.` without splitting on the decimal point.
- **Out-of-range citations.** `[7]` when only 5 chunks were supplied is a
  distinct failure from an unsupported claim, and should be counted separately
  rather than lumped in as "unverified."
- **Uncited claims.** A claim with no citation is not a failed citation; it is
  missing coverage. That distinction is what makes 3.3 meaningful.
- **Cost.** One judge call per claim, ~5–15 claims per answer. Use
  `gpt-4o-mini`, not `gpt-4o` — see §6.

### 3.3 Confidence scorer

Composite of three signals, per Project 6: retrieval confidence, citation
coverage, answer completeness.

**Retrieval confidence needs plumbing that does not exist yet.** The retriever
stack currently throws scores away:
- `RetrieverProtocol.retrieve()` returns `list[Document]`, no scores
  ([src/retrieval/retriever.py:20](src/retrieval/retriever.py:20)).
- `RBACRetriever.retrieve_with_scores()` exists
  ([src/retrieval/retriever.py:82](src/retrieval/retriever.py:82)) but is only on
  that one class, and returns `0.0` on the fallback path.
- `rerank_documents()` computes cross-encoder scores then **discards** them,
  returning documents only
  ([src/retrieval/reranker.py:51](src/retrieval/reranker.py:51)).

So step one of 3.3 is deciding how a score channel reaches generation without
forcing every retriever to change shape. The cross-encoder score is the more
meaningful signal of the two — it is a jointly-scored (query, document)
relevance, not a bi-encoder cosine.

Citation coverage and completeness fall out of 3.2 once claims are segmented.

### 3.4 Structured "I don't know"

Below a retrieval-confidence threshold, return what was found, what was not, and
which documents might be worth checking manually — instead of an answer.

The system already refuses correctly in the plain case; the model reliably says
*"I don't have enough information in the available documents to answer this
question."* What is missing is the structure and the threshold.

---

## 4. Gotchas that will bite

These are specific to this codebase and are the reason a naive implementation
will look right and be wrong.

**The eval bypasses the API route.** `run_rag_pipeline()` in
[evaluation/run_evaluation.py:18](evaluation/run_evaluation.py:18) calls
`get_retriever()` and `query_with_context()` **directly** — it never touches
`src/api/routes/query.py`. Anything implemented only in the route is invisible to
the evaluation. Put citation parsing and verification in the generation layer, or
in a module both call. This single fact decides the architecture.

**The answer is mutated after generation.** In
[src/api/routes/query.py:112-117](src/api/routes/query.py:112),
`apply_financial_disclaimers()` appends disclaimer text and `redact_pii()`
rewrites the body. Both run *after* `query_with_context()`. Parse and verify
citations against the raw generated answer; a parser run after disclaimer
injection will scan text the model never wrote, and PII redaction can alter the
inside of a claim between generation and verification.

**`QueryResponse` has four construction sites.** In
[src/api/routes/query.py](src/api/routes/query.py): injection-blocked (line 38),
no-documents (line 63), LLM-failure (line 75), and success (line 146). Adding
required fields to the schema in
[src/common/schemas.py:85](src/common/schemas.py:85) breaks three of them
silently at runtime unless the new fields have defaults. Give them defaults, and
decide deliberately what confidence means on the paths where nothing was
retrieved or nothing was generated.

**Do not move the eval judge.** `EVAL_JUDGE_MODEL` is pinned to `gpt-4o-mini`
while runtime generation is `gpt-4o`, deliberately: the judge is measurement
apparatus, and changing it invalidates comparison with every result already in
`evaluation/results/`. The citation-verification judge is a *different* judge and
can be chosen freely — just do not repoint the RAGAS one.

**Chunk metadata now carries provenance.** Every chunk has
`chunking_strategy`, `chunk_index` (per source document), and `char_count`,
alongside `ticker`, `section_name`, `filing_type`, `filing_date`. Useful for
rendering a citation as "AAPL 10-K, Item 1A Risk Factors" rather than a chunk id.

---

## 5. Measuring it (the point of the exercise)

Add `citation_accuracy` to the evaluation. It is **not** a RAGAS metric — compute
it separately and merge into the per-question rows.

Concrete edit sites in [evaluation/run_evaluation.py](evaluation/run_evaluation.py):
- `metric_cols` filter set inside `evaluate_with_ragas` — currently hardcodes the
  four RAGAS metric names.
- `METRIC_NAMES` tuple used by `_print_stratified`.
- The `config` block in the saved JSON, so runs record how citations were judged.

**Two measured facts Phase 3 interacts with directly:**

1. **The reranker costs 0.230 context precision** (against a 0.001 run-to-run
   noise floor — retrieval is deterministic, so that metric barely moves between
   identical runs). `ms-marco-MiniLM` was trained on short web passages and
   reorders 512-token filing prose confidently but badly. A confidence score
   would surface this to users at query time instead of leaving it in a results
   file. It is a real defect, not a rounding error.
2. **All three comparative questions score answer relevancy exactly 0.000, in
   every configuration.** With `top_k=5` all five slots fill with one company's
   chunks, the model correctly refuses, and RAGAS scores a refusal as 0. This is
   a retrieval-budget bug — comparatives need per-entity retrieval, not a global
   top-5. Phase 3's structured "I don't know" will make these answers *better*
   without moving the score, so do not read a flat relevancy number there as
   Phase 3 failing.

**Noise floor, per metric, from two identical runs** — deltas below these mean
nothing at n=20:

| Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|
| **0.138** | 0.044 | 0.001 | 0.038 |

---

## 6. Cost and budget

Spend is on the user's OpenAI key and was roughly $3 in the previous session,
against a $10 top-up that had ~$4 remaining before it. **Assume ~$1 or less is
left and confirm before spending.**

Rough costs:
- Full six-config ablation re-run: ~$3.
- Single eval run: ~$0.45 (gpt-4o generation + gpt-4o-mini judging, 20 questions).
- Citation verification adds one judge call per claim — use `gpt-4o-mini`.

**Do not re-ingest the corpus without re-running the whole ablation.** The six
published runs were measured pre-deduplication on 9,572 chunks; `DEDUP_ENABLED`
now defaults to true, so a fresh ingest yields ~8,825 and a different corpus
fingerprint. Re-running one row against a different corpus is worse than
re-running none — the comparison *between* rows is the entire point of that
table. Every result file records a `corpus` fingerprint (chunk count + content
digest) precisely so this mismatch cannot happen silently again.

---

## 7. Decisions already made — do not relitigate

- **`gpt-4o` for generation, `gpt-4o-mini` for the RAGAS judge, local MiniLM for
  judge embeddings.** The judge is deliberately independent of the corpus
  embedding model so the measuring stick does not move when the retriever does.
- **`EMBEDDING_PROVIDER=openai|huggingface`**, default openai
  (`text-embedding-3-small`). The two emit different-sized vectors (1536 vs 384),
  so switching requires `ingest_edgar.py --from-disk --reset`. The Dockerfile
  pins `huggingface` because the image bakes its index at build time and
  embedding with OpenAI there would put a key in the build.
- **Hybrid search stays off by default**, but the old "costs 0.23 recall" claim
  was **retracted** — it did not reproduce on the fixed corpus. The honest
  statement is "not established as helpful."
- **Ground truths are generated from full filing text, never from retrieval**
  (`scripts/generate_ground_truths_from_filings.py`). The earlier version
  answered from this project's own top-20 chunks and then scored that same
  retriever against the result. Do not reintroduce retrieval into ground-truth
  generation.
- **Known limitation:** at the generator's default 240k-char window, gpt-4o-mini
  misses material that is present but diffuse. Verified directly. Use
  `--only "<substring>" --window-chars 60000` to repair a single question
  cheaply. The interpretive stratum is the one to distrust.

---

## 8. Suggested first moves

1. Read `src/generation/prompts.py` and `src/generation/chains.py` end to end —
   they are short, and they are where most of Phase 3 lands.
2. Decide where citation parsing/verification lives so that **both** the API
   route and `run_rag_pipeline()` get it (§4, first gotcha). Everything else
   follows from that choice.
3. Rewrite the prompt for bracketed citations, and eyeball a handful of real
   answers before building the parser — the parser's job is defined by what the
   model actually emits.
4. Build parsing + verification with unit tests over fixture answers (no API
   calls in tests; the existing suite stubs LLMs and embeddings throughout — see
   `tests/unit/test_dedup.py` for the pattern).
5. Only then wire in confidence scoring, which needs the retrieval-score
   plumbing in §3.3.
6. Add `citation_accuracy` to the eval and run **one** config to sanity-check the
   metric before spending on a full sweep.

---

## 9. Commit split, when the time comes

Nothing is committed yet. Reasonable separation:

- EDGAR parser fixes + Dockerfile sample-docs fix (independently valuable)
- Tech stack: OpenAI embeddings + gpt-4o + provider switches
- Phase 1: three chunking strategies, dedup, chunk provenance
- Evaluation: v3 ground truths, stratification, corpus fingerprint, README
- Phase 3: whatever this session produces
