# Handoff — RAG Enterprise retrieval & evaluation work

Updated 2026-08-08 (second session). Everything below is uncommitted working-tree
state on `main`; last pushed commit is still `ea0aea8`.

---

## 0. Relationship to "Project 6"

This project is a **SEC-filings implementation of Project 6: RAG Pipeline with
Hybrid Search Over Internal Docs** (`Project 6.docx` in the repo root). Project 6
specifies six phases; this repo implements them against SEC EDGAR filings and
adds a compliance layer Project 6 does not ask for (RBAC information barriers,
MNPI/investment-advice guardrails, SEC 17a-4 audit trail, MCP server).

Phase status against the spec:

| Phase | Status |
|---|---|
| Tech stack | **Done** — aligned this session (see §2) |
| 1. Ingestion & chunking | **Done** — three strategies, dedup, chunk provenance |
| 2. Hybrid retrieval | **Done** — dense + BM25 + RRF + cross-encoder, built to spec |
| 3. Generation & citation | **Not started** — the largest remaining gap |
| 4. Evaluation | **Partial** — 20 questions, spec wants 50+ hand-written |
| 5. API & dashboard | **Partial** — API yes, no query dashboard |
| 6. Portfolio | **Partial** — case study strong, no demo video |

**Agreed build order for the remaining gaps: Phase 1 → 3 → 4 → 5.** Phase 1 is
complete as of this session.

> **Starting Phase 3 in a fresh session? Read [`handoffTo3.md`](handoffTo3.md)
> first.** It is self-contained — the implementation map, the codebase gotchas
> that decide the architecture, and the measured facts Phase 3 interacts with.
> This document is the background it refers back to.

### What Phase 3 needs (next up)

Project 6 calls this the differentiator — *"the quality layer most RAG systems
skip entirely"* — and prescribes the case-study line *"X% faithfulness and **Y%
citation accuracy**"*, which is unwritable today. Four pieces:

1. Numbered context blocks and bracketed `[1]`/`[2]` citations. The current
   prompt asks for prose citations ("According to Apple's 2024 10-K…"), which
   cannot be parsed or verified.
2. Citation verification — parse each citation, send the claim/citation pair to
   an LLM-as-judge, flag unsupported ones.
3. A composite confidence score: retrieval confidence, citation coverage,
   answer completeness.
4. A structured "I don't know" — what was found, what wasn't, which documents to
   check — behind a retrieval-confidence threshold.

Then add `citation_accuracy` to the eval metrics. Note this connects to a
measured defect: the reranker costs 0.230 context precision (§3), and a
confidence score would surface that to users at query time instead of leaving it
in a results file.

### Other known gaps (lower priority)

- RRF fusion weighting is hardcoded unweighted; the spec wants a tunable
  0.7 dense / 0.3 sparse split (`src/retrieval/fusion.py`).
- Routes are `/query` and `/documents/ingest`; the spec names `POST /v1/ask`,
  `GET /v1/documents`, `POST /v1/ingest`. Cosmetic, but it is what the spec says.
- Phase 4 wants **no-answer** and **ambiguous** question categories. The current
  set has neither — every one of the 20 is answerable by construction.

---

## 1. The goal

Make the retrieval quality claims in this project **true and defensible**, and
enable the techniques that actually earn it.

An external reviewer made three correct criticisms of the evaluation: n=20 is too
small, the ground truths were circular, and there was no stratification. Points 2
and 3 are now fixed. Point 1 is not.

Chasing them uncovered a larger problem — the corpus was missing most of two
filings — which invalidated every retrieval number measured before it.

---

## 2. What is done

### Shipped and pushed (`ea0aea8` and earlier)

Cross-encoder reranking, BM25 hybrid + RRF, landing page redesign,
config-tagged eval results.

### Uncommitted — all 154 tests pass, lint clean

**The evaluation is re-run and the README rewritten.** Specifically:

| Item | State |
|---|---|
| EDGAR parser section-extraction fixes | Done, 4 regression tests. JPM 11%→83%, MSFT 8%→71% |
| Non-circular ground truths (`eval_questions_v3.json`) | Done, n=20, **0 unanswerable** |
| Stratification | Done — 7 exact_figure / 10 interpretive / 3 comparative |
| `eval_config.py` points at v3 | Done |
| Main index rebuilt | 9,572 chunks, `text-embedding-3-small` |
| Semantic index rebuilt | 8,936 chunks, fixed parser + OpenAI embeddings |
| Ablation re-run (6 runs) | Done — see README "Retrieval Pipeline Ablation" |
| README ablation section | Rewritten against the new numbers |
| Pre-fix results marked superseded | `evaluation/results/README.md` |
| Corpus fingerprint in results | Done — chunk count + content digest |

### Phase 1 — ingestion & chunking (this session)

- **Three chunking strategies**, was two. `fixed` (structure-blind character
  windows) is new and exists as a baseline for the recursive splitter to beat.
  `CHUNKING_STRATEGY=fixed|recursive|semantic`.
- **Chunk provenance**: every chunk records `chunking_strategy`, `chunk_index`
  (per source document), and `char_count`. `chunk_index` was removed from
  `enrich_metadata` — it was set there over whatever slice of chunks that call
  received, and would overwrite the value `split_documents` now assigns
  consistently across every ingestion path.
- **Deduplication** (`src/ingestion/dedup.py`), wired into
  `ChromaVectorStore.add_documents` so every path gets it — batch scripts and
  `POST /documents/ingest` alike. Compares against both the existing index and
  earlier chunks in the same batch; the second matters because boilerplate
  repeated inside one document would otherwise survive a first ingest.
- **`scripts/analyze_duplicates.py`** — read-only duplicate report over an
  existing collection, reusing stored vectors, so it costs no API calls.
- **HTML added to the loader registry** (`.html`, `.htm` via `BSHTMLLoader`).
  SEC filings still do *not* go through it — `src/edgar/parser.py` splits them on
  Item boundaries first, which is what makes a chunk attributable to "Item 1A".

**Bug found and fixed in passing:** `create_text_splitter` used
`chunk_overlap or settings.chunk_overlap`, so an explicit `chunk_overlap=0` was
falsy and silently replaced by the configured 50. Now `is None`.

**Measured finding: 7.8% of the corpus (747 of 9,572 chunks) is near-duplicate**
at cosine 0.95; 2.53% is near-identical. Two causes — page furniture that became
its own chunk (`"Apple Inc. | 2025 Form 10-K | 3"`), and filings genuinely
restating disclosures verbatim between Item 1A and the incorporated annual
report. Full table in the README.

> **The six ablation runs predate dedup.** They were measured on the full 9,572
> chunks; `DEDUP_ENABLED` now defaults to true, so a fresh ingest gives ~8,825
> and a different corpus fingerprint. The table was left as measured rather than
> re-running one row against a different corpus — the comparison *between* rows
> is the point. Re-running all six together costs roughly $3.
>
> A related observation worth acting on separately: the page-footer duplicates
> imply the splitter is emitting very short contentless chunks. Dedup removes
> most of them as a side effect, but a minimum-chunk-length filter would address
> the cause rather than the symptom.

### Tech stack alignment (this session)

The stack was audited against the project spec. Two gaps, both now closed:

- **Embeddings** were `all-MiniLM-L6-v2`, spec wanted `text-embedding-3-small`.
  Now provider-switchable via `EMBEDDING_PROVIDER=openai|huggingface`, default
  openai. The two emit different-sized vectors (1536 vs 384), so **switching
  provider requires `python scripts/ingest_edgar.py --from-disk --reset`** — an
  existing collection cannot be queried with the other provider's embeddings.
- **Generation** was Groq `llama-3.3-70b`, spec wanted GPT-4o or Claude Sonnet.
  Now `LLM_PROVIDER=openai` / `OPENAI_MODEL=gpt-4o` by default. Groq and Gemini
  remain as free-tier fallbacks.

The RAGAS judge was deliberately **not** moved with it — `EVAL_JUDGE_MODEL`
stays `gpt-4o-mini` and judge embeddings stay local MiniLM. The judge is
measurement apparatus; changing it invalidates comparison with everything in
`evaluation/results/` and costs more than the runs it grades.

`Dockerfile` pins `EMBEDDING_PROVIDER=huggingface`, because the image bakes its
index at build time and the two providers' vectors are incompatible. Embedding
with OpenAI at build time would require a key in the build; baking locally while
defaulting to openai would ship an image whose index mismatches its own query
path. Generation is still whatever `LLM_PROVIDER` is set to at deploy.

---

## 3. The numbers, and what they support

Full table and reasoning in the README. The short version:

**The noise floor is per-metric, and that matters more than any single delta.**
Two identical dense runs: Precision spread 0.001, Recall 0.038, Relevancy 0.044,
**Faithfulness 0.138**. Retrieval is deterministic so precision barely moves;
faithfulness compounds a non-deterministic generator with a non-deterministic
judge. Any single-run faithfulness delta under ~0.14 means nothing.

| Claim | Status |
|---|---|
| Reranking lifts Answer Relevancy +0.239 | **Real** (floor 0.044) |
| Reranking costs Context Precision −0.230 | **Real** (floor 0.001) — a live defect |
| Hybrid costs −0.23 recall | **Retracted.** Does not reproduce on the fixed corpus; now inside noise |
| HyDE: precision +0.148, relevancy −0.152 vs rerank | Both real |
| Semantic chunking helps | No — relevancy −0.126, nothing else moves |

Two findings worth acting on:

1. **The reranker hurts precision badly.** `ms-marco-MiniLM` was trained on short
   web passages; it reorders 512-token filing prose confidently but not well. It
   ships because relevancy is what users feel, but this is a real regression.
2. **All 3 comparative questions score Answer Relevancy exactly 0.000, in every
   config.** With `top_k=5` all five slots fill with one company's chunks, the
   model correctly refuses, and RAGAS scores a refusal as 0. This is a retrieval
   budget bug — comparatives need per-entity retrieval, not a global top-5.

---

## 4. Known limitation in the ground-truth generator

At the default 240k-char window, gpt-4o-mini **misses material that is present
but diffuse**. Verified directly: asked "what legal proceedings are disclosed by
JPMorgan", it returned `NOT_IN_WINDOW` for a window containing five occurrences
of "legal proceedings" and the `Note 30 — Litigation` heading. Re-running that
one question at `--window-chars 60000` recovered a specific, correct answer.

So `--only` and `--window-chars` now exist:

```bash
python scripts/generate_ground_truths_from_filings.py \
  --input evaluation/datasets/eval_questions_v3.json \
  --output evaluation/datasets/eval_questions_v3.json \
  --only "legal proceedings" --window-chars 60000
```

**Implication not yet chased:** if diffuse material is missed at 240k, other
*interpretive* ground truths may be thin rather than absent — present but
incomplete, which no `NOT_IN_FILING` count would reveal. The interpretive stratum
is the one to distrust. Regenerating the whole set at 60k windows would cost
roughly 4× (~$4) and is the obvious next validation.

---

## 5. What was tried that failed (kept from the previous session — still true)

- **Truncating filings to fit context.** JPM's financials sit at the end;
  truncation drops exactly what most questions ask about.
- **Asking each window "does this answer the question?"** Comparative questions
  came back unanswerable because one window covers one company. Fixed by
  extract-then-synthesize.
- **A shared system prompt across both stages.** The prompt said reply
  `NOT_IN_FILING` while the caller checked for `NOT_IN_WINDOW`, so the sentinel
  survived as a "finding" and poisoned synthesis.
- **`langchain-experimental` for SemanticChunker.** Pulls `langchain-core` to
  1.x against this project's deliberate `<1.0` pin, and the package is sunset.
  Hand-implemented in ~50 lines instead.
- **Gemini as RAGAS judge** — free tier is 20 requests/day. **Groq as judge** —
  100k tokens/day, exhausted by one run, and not comparable to the gpt-4o-mini
  baseline anyway.
- **`semantic_max_chunk_chars = 2000`.** Was above MiniLM's 256-token window so
  chunk tails were silently truncated at embedding time. Now 1200. Note this
  constraint is **gone** on `text-embedding-3-small` (8k window) — 1200 is now a
  deliberate choice to keep the chunking ablation about strategy rather than
  size, not a hard limit.
- **BM25 filtering on `score > 0`.** BM25Okapi's IDF is exactly 0 when a term is
  in half the corpus, so real matches were dropped. Filters on query-term
  overlap now.

---

## 6. Next steps, in order

**Phase 3 is next** (see §0 for the four pieces). Everything below is either part
of it or sequenced behind it.

1. **Phase 3: citations, verification, confidence.** The agreed next phase, and
   the largest gap against Project 6.
2. **Fix the comparative retrieval budget.** Three of twenty questions score a
   hard zero on relevancy for a structural reason. Retrieve per entity mentioned
   in the query and merge, rather than one global top-5. Biggest single
   retrieval win available, and it is what makes the comparative stratum
   meaningful at all.
3. **Investigate the reranker precision regression.** Either a
   financial-domain-appropriate cross-encoder, or rerank-then-restore-order, or
   drop the reranker and keep the dense ordering that scored 0.696 precision.
4. **Phase 4: grow the set past 20**, including the **no-answer** and
   **ambiguous** categories the spec asks for and this set has none of. The
   per-metric noise floor sharpens the case: at n=20 a faithfulness delta must
   exceed 0.14 to be believable.
5. **Re-run the full ablation on a deduplicated corpus** (~$3). Only worth doing
   as a complete set, and cheapest to bundle with whatever Phase 3 adds to the
   metrics.
6. **Hand-verify ~10 ground truths against filing text.** An independent check on
   the LLM-generated references. §4 makes this more valuable, not less.
7. **Phase 5: query dashboard.** Clickable citations, ranked chunks, confidence
   breakdown, hybrid vs dense-only side-by-side. Depends on Phase 3 for the
   citation and confidence data it would display.
8. **Redeploy the live demo.** Cloud Run still serves the pre-parser-fix build
   with no sample documents. The Dockerfile fixes only take effect on rebuild.
   Note the demo will now need an `OPENAI_API_KEY` at runtime for generation, or
   `LLM_PROVIDER=groq` set at deploy to stay on a free tier.
9. **MSFT at 71%** — better than 8%, but the recovered text is mid-section
   fragments. Worth another look if MSFT questions underperform.

### Committing

Nothing since `ea0aea8` is committed. Reasonable split:

- Parser fixes + Dockerfile sample-docs fix (independently valuable)
- Tech stack: OpenAI embeddings + gpt-4o + provider switches
- Evaluation: v3 ground truths, stratification, corpus fingerprint, README
