# Handoff → Phase 4: Evaluation

Written 2026-08-08. Phases 1, 2 and 3 are complete. This document is for a
fresh session picking up Phase 4 and assumes no memory of the previous ones.

Companion documents: `handoffTo3.md` (the previous handoff, plus a record of
what Phase 3 produced), `HANDOFF.md` (full project history, evaluation
methodology, what was tried and failed) and `Project 6.docx` (the original
spec, in the repo root). Read this one first; reach for those when you need
the why behind a number.

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
| 3. Generation & citation | **Done** — bracketed citations, LLM-judge verification, composite confidence, structured "I don't know" |
| **4. Evaluation** | **← you are here.** 20 questions; spec wants 50+ hand-written, including categories this set has none of |
| 5. API & dashboard | Partial — API yes, no query dashboard |
| 6. Portfolio | Partial — case study strong, no demo video |

Agreed build order for what remains: **4 → 5**.

---

## 2. Repo state

- Branch `main`. **Nothing since `ea0aea8` is committed** — Phases 1 and 3,
  the tech stack migration, and the evaluation rework are all uncommitted
  working tree. See §9.
- 249 tests pass, ruff clean.
- **Use `./.venv/Scripts/python.exe`, never bare `python`.** The `python` on
  PATH is system Python 3.13 with *some* dependencies (langchain,
  langchain-openai) but not `rank_bm25`, `transformers`, or
  `sentence-transformers`. The partial overlap is the trap: LangChain-only
  scripts run fine, then `pytest` fails collection on five modules and reads
  as a broken suite rather than a wrong interpreter.

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
./.venv/Scripts/python.exe -m ruff check src tests evaluation scripts
```

---

## 3. What Phase 3 left you (read this before designing Phase 4)

Phase 4 is mostly a dataset problem, but the shape of the dataset now depends
on what Phase 3 made scorable. The short version: **the system can now be
right by refusing**, and the eval cannot yet express that.

New surface, all of it behind one entry point:

```
src/generation/answer.py        generate_grounded_answer()  <- API, MCP and eval all call this
src/generation/citations.py     claim segmentation + [n] parsing (no LLM)
src/generation/verification.py  LLM-as-judge, one call per (claim, block) pair
src/generation/confidence.py    0.5 retrieval / 0.3 coverage / 0.2 completeness
src/generation/insufficient.py  structured "I don't know"
src/retrieval/scores.py         relevance scores carried through the pipeline
src/common/text.py              sentence splitter, shared with semantic chunking
```

Three facts that matter for dataset design:

1. **A refusal now has structure.** `GroundedAnswer.unanswered` carries
   `reason` (`low_retrieval_confidence` when the gate fired before generation,
   `model_refused` when the model declined over decent retrieval), the
   passages consulted, and the filings worth opening by hand. So "the correct
   answer is *no answer*" is finally a checkable outcome rather than a string
   comparison against a refusal sentence.
2. **Confidence is on every query, free.** Citation coverage and completeness
   come from parsing; retrieval relevance comes from the cross-encoder score
   channel. No LLM call. A no-answer question should score `label: "low"` and
   `answer_completeness: 0.0` — that is the assertion to test.
3. **`citation_accuracy` is wired into the eval but has never been run.**
   `run_rag_pipeline()` forces `verify=True`, the metric is merged into the
   per-question rows and aggregated with nulls excluded. It has been exercised
   against synthetic rows offline and smoke-tested on three live queries. No
   run over the 20-question set exists. **That run is the first item of Phase
   4**, not a leftover of Phase 3 — see §9.

Smoke-test evidence, three live queries against the current index:

| Query | Result |
|---|---|
| Apple FY24 revenue | Model emitted `[3]`; judge SUPPORTED; accuracy 1.00, coverage 1.00, confidence 0.97 *high* |
| JPM vs GS credit risk | Retrieval 0.30 cleared the gate, model refused, `model_refused` attached; confidence 0.15 *low* |
| Out-of-corpus nonsense | Logits ≈ −11 → relevance 0.00; gate fired **before** generation, no LLM call spent |

---

## 4. What Phase 4 asks for

Project 6 wants **50+ hand-written questions with ground truth**, spanning
categories the current set does not have. The current set is 20:

| Stratum | n | Notes |
|---|---|---|
| `exact_figure` | 7 | Wants a specific number stated in the filing |
| `interpretive` | 10 | Wants a description synthesized across a discussion |
| `comparative` | 3 | Spans two filings; **all three score answer relevancy exactly 0.000** — see §5 |
| `no_answer` | 0 | **Missing.** The spec asks for it |
| `ambiguous` | 0 | **Missing.** The spec asks for it |

Every one of the 20 is answerable by construction, which means the entire
Phase 3 refusal path is currently unmeasured by the eval that is supposed to
measure the system.

Four pieces, in dependency order:

### 4.1 Decide how a no-answer question is scored

This is the design decision the rest of Phase 4 hangs off, and it is not
obvious. **RAGAS scores a correct refusal as 0.000 answer relevancy.** That is
already visible in the comparative stratum (§5) and it is not a bug in RAGAS —
a refusal genuinely has no relevance to the question. So adding ten no-answer
questions to the existing metric set would *lower every aggregate* precisely
because the system behaved correctly.

Options, roughly in order of how much work they are:

- Score no-answer rows on their own axis and exclude them from
  `answer_relevancy` and `faithfulness`. Needs a per-row metric mask, which
  `evaluate_with_ragas` does not currently have — it runs all four metrics
  over the whole dataset.
- Add a `refusal_correctness` metric computed outside RAGAS, in the same place
  `citation_accuracy` is computed (`run_rag_pipeline`), asserting
  `unanswered is not None` and `confidence.label == "low"`. Cheap: no LLM call
  needed for the assertion itself.
- Report the aggregate per stratum only, never pooled. `_print_stratified`
  already does this for the console; the saved JSON still records a pooled
  mean in `scores`.

Whichever you pick, say so in the results JSON `config` block, because a run
that scored no-answer rows differently is not comparable to one that did not.

### 4.2 Write the questions

50+ total, so ~30 new. Hand-written — the spec says so, and §7 explains why
LLM-generated questions would compromise the measurement chain.

Coverage gaps worth filling, from what the existing 20 do *not* touch:

- **No-answer**: questions whose answer is genuinely absent from these five
  filings. Two flavours worth separating, because Phase 3 distinguishes them:
  plausible-but-absent (a real 10-K topic these filers do not disclose →
  expect `model_refused`) and out-of-corpus (→ expect
  `low_retrieval_confidence`, and no generation call at all).
- **Ambiguous**: questions with more than one defensible reading ("what was
  revenue last year" — whose, and which fiscal year?). The correct behaviour
  is arguably to answer one reading and name the ambiguity; decide what you
  are asserting before writing them.
- **RBAC-scoped**: see the fourth gotcha in §5. Right now the eval never
  exercises a role filter at all.
- **MSFT**, which is recovered at ~71% of its filing text and is the most
  likely company to underperform.

### 4.3 Generate ground truths — from filing text, never retrieval

```bash
./.venv/Scripts/python.exe scripts/generate_ground_truths_from_filings.py \
  --input evaluation/datasets/eval_questions_v4.json \
  --output evaluation/datasets/eval_questions_v4.json
```

Roughly $0.15 per 20 questions on gpt-4o-mini. Known limitation: at the
default 240k-char window the generator **misses material that is present but
diffuse** — verified directly. Repair a single question cheaply with
`--only "<substring>" --window-chars 60000`. The interpretive stratum is the
one to distrust.

No-answer questions need a ground truth field anyway (RAGAS requires it).
Decide what goes in it — a sentinel, or a description of what would have been
needed — and be consistent, because §4.1's scoring depends on it.

### 4.4 Stratify

`scripts/stratify_eval_questions.py` holds a hand-written `QUESTION_TYPES`
dict keyed by **a distinctive substring of each question**. Adding 30
questions means adding 30 entries by hand. Classification is deliberately not
LLM-inferred — see §7. Two new strata (`no_answer`, `ambiguous`) need adding
to that module and they will flow through `_print_stratified` automatically.

---

## 5. Gotchas that will bite

These are specific to this codebase and are the reason a naive implementation
will look right and be wrong.

**The eval never exercises RBAC.** Every one of the 20 questions lists
`"admin"` in its `access_roles`, and `run_rag_pipeline()` passes the whole set
as the user's roles. `RBACRetriever.build_role_filter()` returns `None` for
admin, so the ChromaDB where-clause is never applied on any eval question. The
information-barrier layer — arguably the most distinctive thing in the repo —
is covered by unit tests and by nothing in the eval. Dropping `"admin"` from
new questions fixes it; note that `operations` and `viewer` have restricted
access (`operations` cannot see `sec_filings` at all), so a question tagged
with the wrong role retrieves nothing and reads as a retrieval failure.

**Per-question rows join by position.** `_merge_citation_metrics()` attaches
citation metrics to RAGAS rows by index, on the assumption that RAGAS returns
one row per input in input order and that `results` is index-aligned with
`eval_data`. The loop in `main()` appends an error row when a question raises,
precisely to preserve that alignment — and `evaluate_with_ragas` slices
`eval_data[: len(results)]` on the same assumption. Any filtering, sorting or
partial-run logic added
to `main()` breaks the join silently — every metric would be attached to the
wrong question.

**All three comparative questions score answer relevancy exactly 0.000, in
every configuration.** With `top_k=5` all five slots fill with one company's
chunks, the model correctly refuses, and RAGAS scores a refusal as 0. This is
a retrieval-budget bug — comparatives need per-entity retrieval, not a global
top-5 — and it is now *visible at query time* (the smoke test above shows
confidence 0.15 on exactly this case) but not *fixed*. Growing the set makes
this worse in absolute terms: three broken questions out of twenty becomes
seven or eight out of fifty unless the retrieval budget is fixed first.
Consider fixing it before writing more comparatives.

**The reranker costs 0.230 context precision** (against a 0.001 run-to-run
noise floor — retrieval is deterministic, so that metric barely moves between
identical runs). `ms-marco-MiniLM` was trained on short web passages and
reorders 512-token filing prose confidently but badly. It is a real defect,
not a rounding error, and it is still unfixed.

**Do not move the eval judge.** `EVAL_JUDGE_MODEL` is pinned to `gpt-4o-mini`
while runtime generation is `gpt-4o`, deliberately: the judge is measurement
apparatus, and changing it invalidates comparison with every result already in
`evaluation/results/`. `CITATION_JUDGE_MODEL` is a *different* judge and can be
chosen freely — just do not repoint the RAGAS one.

**An eval run now costs more than it used to.** `run_rag_pipeline()` sets
`verify=True`, which adds one `gpt-4o-mini` call per citation — call it 5–15
per answered question. Budget accordingly (§6).

**The insufficient-context gate can suppress generation entirely.** With
`INSUFFICIENT_CONTEXT_THRESHOLD=0.15`, a question whose retrieval scores below
that returns the structured report and never calls the LLM. That is correct
behaviour, and it means a badly-worded new question can produce a row with no
generated answer at all. Check `generated: false` in the results before
concluding a question is broken. Set the threshold to `0` to disable the gate
for a diagnostic run.

---

## 6. Cost and budget

Spend is on the user's OpenAI key. `handoffTo3.md` estimated ~$1 or less
remaining; this session spent roughly $0.05 on the three-query smoke test.
**Assume ~$1 or less is left and confirm before spending.**

Rough costs at current settings:

| Action | Estimate |
|---|---|
| Single eval run, 20 questions | ~$0.45 base, plus citation verification (~5–15 `gpt-4o-mini` calls/question) |
| Single eval run, 50 questions | ~$1.10 base, plus verification — scale linearly |
| Ground truth generation | ~$0.15 per 20 questions; ~4× that at `--window-chars 60000` |
| Full six-config ablation re-run | ~$3, and only meaningful as a complete set |

**Do not re-ingest the corpus without re-running the whole ablation.** The six
published runs were measured pre-deduplication on 9,572 chunks; `DEDUP_ENABLED`
now defaults to true, so a fresh ingest yields ~8,825 and a different corpus
fingerprint. Re-running one row against a different corpus is worse than
re-running none — the comparison *between* rows is the entire point of that
table. Every result file records a `corpus` fingerprint (chunk count + content
digest) precisely so this mismatch cannot happen silently again. The index
currently on disk is the 9,572-chunk one.

---

## 7. What Phase 4 actually buys, quantitatively

The case for 50+ is not "more is better". At n=20 the measured run-to-run
noise floor is:

| Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|
| **0.138** | 0.044 | 0.001 | 0.038 |

A faithfulness delta below 0.138 is unreadable at n=20 — which is most of the
deltas anyone would want to claim. If that floor is dominated by judge
nondeterminism averaging out across questions, it should fall roughly with
√n: ~0.087 at n=50. **Treat that as a hypothesis, not a result.** Re-measure
it the same way it was measured the first time — two identical runs, diffed —
once the set is grown. If it does not fall, the noise is systematic rather
than sampling, and that is a more interesting finding than the metric it was
blocking.

---

## 8. Decisions already made — do not relitigate

- **Questions and strata are hand-written.** 20 questions is small enough to
  read; an LLM classifier or generator introduces another model's judgment
  into a measurement chain the eval works hard to keep clean. This applies to
  the 30 new ones too.
- **Ground truths are generated from full filing text, never from retrieval**
  (`scripts/generate_ground_truths_from_filings.py`). The earlier version
  answered from this project's own top-20 chunks and then scored that same
  retriever against the result. Do not reintroduce retrieval into ground-truth
  generation.
- **`gpt-4o` for generation, `gpt-4o-mini` for the RAGAS judge, local MiniLM
  for judge embeddings.** The judge is deliberately independent of the corpus
  embedding model so the measuring stick does not move when the retriever does.
- **Citation verification is off at runtime, forced on in the eval.** One LLM
  call per citation on the critical path is not worth it for a served query;
  it is exactly what buys the metric in a measured run.
- **A refusal is labelled `low` confidence regardless of the numeric
  composite.** Retrieval carries half the weight and can score well on a
  refusal, which would land the composite in "medium". The number is
  meaningful; the label is a claim about the answer, and there is no answer.
- **`EMBEDDING_PROVIDER=openai|huggingface`**, default openai
  (`text-embedding-3-small`). The two emit different-sized vectors (1536 vs
  384), so switching requires `ingest_edgar.py --from-disk --reset`. The
  Dockerfile pins `huggingface` because the image bakes its index at build
  time and embedding with OpenAI there would put a key in the build.
- **Hybrid search stays off by default**, but the old "costs 0.23 recall"
  claim was **retracted** — it did not reproduce on the fixed corpus. The
  honest statement is "not established as helpful."

---

## 9. Suggested first moves

1. **Run the eval once, unchanged, on the current 20 questions.** This is the
   cheapest thing in Phase 4 and it validates the `citation_accuracy` plumbing
   under real data before any of it is built on. It also produces the first
   number for Project 6's prescribed case-study line — *"X% faithfulness and
   Y% citation accuracy"* — which is the sentence this whole exercise exists
   to make writable. Confirm the spend first (§6).

   ```bash
   ./.venv/Scripts/python.exe -m evaluation.run_evaluation
   ```

2. **Read the resulting `per_question_scores` before writing any questions.**
   Specifically: where citation accuracy is low, is it the model citing the
   wrong block, or the judge being harsh? A handful of `verdict_counts` and
   reasons will tell you, and it decides whether Phase 4 is a dataset problem
   or a prompt problem.
3. **Decide §4.1** — how a no-answer question is scored — and write it down in
   the results `config` block before generating anything.
4. **Fix the comparative retrieval budget**, or accept that the comparative
   stratum stays broken at a larger n. Per-entity retrieval merged into one
   context, rather than a global top-5.
5. **Write the ~30 new questions**, generate ground truths, stratify, and
   re-run. Keep `eval_questions_v3.json` intact and write `v4` — the v2→v3
   change is documented in `evaluation/results/README.md` as the reason
   earlier runs are not comparable, and the same discipline applies here.
6. **Re-measure the noise floor** (§7) with two identical runs on the grown
   set, before claiming any delta against it.

---

## 10. Commit split, when the time comes

Nothing is committed yet. Reasonable separation:

- EDGAR parser fixes + Dockerfile sample-docs fix (independently valuable)
- Tech stack: OpenAI embeddings + gpt-4o + provider switches
- Phase 1: three chunking strategies, dedup, chunk provenance
- Evaluation: v3 ground truths, stratification, corpus fingerprint, README
- Phase 3: citations, verification, confidence, structured refusal, score
  channel — plus the `citation_accuracy` wiring in the eval harness
- Phase 4: whatever the next session produces
