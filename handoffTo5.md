# Handoff → Phase 5: API & Dashboard

Written 2026-08-08, replacing an earlier draft of the same name. Phases 1–4 are
complete. This document is for a fresh session picking up Phase 5 and assumes
no memory of the previous ones.

Companion documents: `handoffTo4.md` (what Phase 4 was asked to produce),
`handoffTo3.md`, `HANDOFF.md` (full project history, what was tried and failed)
and `Project 6.docx` (the original spec, in the repo root). Read this one
first; reach for those when you need the why behind a number.

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
| 1. Ingestion & chunking | **Done** — three strategies, dedup, page-furniture stripping, chunk provenance |
| 2. Hybrid retrieval | **Done** — dense + BM25 + RRF + cross-encoder + per-entity split |
| 3. Generation & citation | **Done** — bracketed citations, LLM-judge verification, composite confidence, structured "I don't know" |
| 4. Evaluation | **Done** — 54 hand-written questions, 7 strata, refusal scoring, noise floor re-measured |
| **5. API & dashboard** | **← you are here.** API yes; no query dashboard |
| 6. Portfolio | Partial — case study strong, no demo video |

---

## 2. Read this before anything else

**No evaluation run exists on the corpus currently on disk.**

The index was re-ingested at the very end of Phase 4, after a page-furniture
fix landed in the parser. It went from 9,572 chunks (`e705ac4b47e9daf3`) to
**8,232 chunks (`c2f8c13673cf5ca5`)**. Every result file in
`evaluation/results/` predates that change.

This is not a crisis — the numbers in §4 are accurate records of what was
measured, every result file records its own corpus fingerprint, and
`scripts/compare_eval_runs.py` refuses to diff across a fingerprint change. The
guard rails work. But it means:

- **Do not quote a score as current** until you have run the eval once
  (~$0.80, and confirm the spend first — see §7).
- **The six-config ablation table in `README.md` is obsolete.** It was measured
  on 9,572 chunks and is only meaningful as a complete set; re-running one row
  against the new corpus is worse than re-running none, because the comparison
  *between* rows is the entire point of that table.
- Both `README.md` and this document carry banner warnings saying so. Do not
  quietly remove them; replace them with a measurement.

The fix itself is real and worked. `_strip_page_furniture()` in
`src/edgar/parser.py` removes running headers, footers and standalone page
numbers, and `min_chunk_chars=80` drops what is left:

| | AAPL | MSFT | JPM | GS | TSLA |
|---|---|---|---|---|---|
| chunks, was → now | 505 → 460 | 809 → 672 | 3,625 → 3,122 | 3,402 → 2,888 | 1,124 → 984 |
| under 200 chars | 18.2% → 13.7% | 25.6% → 15.3% | 20.7% → 13.0% | 23.7% → **14.6%** | 23.6% → 17.5% |

Goldman's bare running-header chunks went **214 → 0**. Expect GS context recall
to move off its hard 0.000 and the over-refusal rate to improve. Neither is
measured.

---

## 3. Repo state

- Branch `main`. **Nothing since `ea0aea8` is committed** — Phases 1, 3 and 4,
  the tech stack migration and the evaluation rework are all uncommitted
  working tree. See §10.
- 321 tests pass, ruff clean.
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

## 4. What Phase 4 left you

Phase 5 is a UI phase, and the useful thing Phase 4 produced for it is not the
scores — it is that **everything a dashboard would want to show is already on
the response object**, plus a measurement saying how much of it to trust.

```
evaluation/datasets/eval_questions_v4.json  54 questions, 7 strata
evaluation/run_evaluation.py                refusal_correctness, per-row metric mask, index join
scripts/compare_eval_runs.py                diff two runs; refuses mismatched corpora
scripts/stratify_eval_questions.py          + no_answer, out_of_corpus, rbac_blocked, ambiguous
src/retrieval/entities.py                   which companies a question names
src/retrieval/retriever.py                  MultiEntityRetriever + extra_filter threading
```

### 4.1 The numbers, as of the 9,572-chunk corpus

> Superseded by the re-ingest — see §2. Kept because the *shape* is unlikely to
> change even though the values will.

Mean of two identical runs (`eval_20260808_212927`, `eval_20260808_213635`),
except refusal correctness, which is from `eval_20260809_001003` — the run
after the `is_full_refusal` fix:

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Citation Accuracy | Citation Coverage | Refusal Correctness |
|---|---|---|---|---|---|---|
| 0.737 | 0.639 | 0.419 | 0.383 | **0.945** | 0.607 | 0.796 |

Per stratum, which is where the shape is:

| Stratum | n | Refusal Correctness |
|---|---|---|
| `interpretive` | 14 | 0.929 |
| `exact_figure` | 13 | 0.692 |
| `comparative` | 8 | 0.625 |
| `ambiguous` | 6 | 0.500 |
| `no_answer` | 5 | **1.000** |
| `out_of_corpus` | 4 | **1.000** |
| `rbac_blocked` | 4 | **1.000** |

### 4.2 Three facts that matter for a dashboard

1. **The refusal path is the best-measured thing in the system and has nothing
   to look at.** All three unanswerable strata score a perfect 1.000,
   identically across runs. `UnansweredReport` already carries the reason, the
   passages consulted and the filings worth opening by hand; the REST API
   returns it; nothing renders it. Highest-value screen available, already
   plumbed.
2. **The system falsely refuses answerable questions and a user cannot tell.**
   On the old corpus, 11 of 38 answerable questions declined. On the wire a
   false refusal and a genuine "the filing does not say" are the same shape.
   `reason` distinguishes `low_retrieval_confidence` (gate fired, no LLM call)
   from `model_refused` — the one signal a user could act on.
3. **`confidence` is on every response, free.** `ConfidenceScore` carries
   `overall`, `retrieval`, `citation_coverage`, `answer_completeness` and a
   `label`. Showing only `overall` throws away the part that says *why*.

---

## 5. What Phase 5 asks for

Project 6 §5 wants a **query dashboard**. The API half is done: `POST /query`
returns answer, per-claim citations with verdicts, source documents with
provenance, confidence breakdown, guardrail flags and the unanswered report.

Read `src/common/schemas.py` first — the response shape is the spec for the UI,
and it is richer than a typical RAG demo because Phase 3 put it there.

### 5.1 Decide what the dashboard is *for*

Not rhetorical. Two defensible products, wanting different screens:

- **An analyst tool.** Ask, read, click a citation, land on the chunk. Optimises
  for trusting a single answer.
- **A system demonstration.** Show the pipeline working: ranked chunks with
  scores, which stage moved what, the confidence breakdown, the RBAC filter
  that was applied. Optimises for a reviewer understanding the build.

This repo's distinguishing assets — information barriers, citation
verification, structured refusal — argue for the second, and Phase 6 is a
portfolio phase. Pick one and say so; "both" produces a screen that does
neither.

### 5.2 Pick the stack, and mind what already exists

**There is already a frontend, and it is a 780-line Python string.**
`src/main.py` is 973 lines, of which lines 173–952 are one inline HTML document
— `<style>`, `<script>` and all — returned by the `GET /` handler. There is no
build tooling, no `package.json`, no `static/` or `templates/` directory.

Two consequences worth deciding on deliberately:

- The visual language already exists (the filing/terminal palette, the branded
  favicon at `GET /favicon.svg`). A dashboard that ignores it will look like a
  different product bolted on.
- Adding a second screen the same way doubles a file that is already 80% string
  literal. Extracting the existing page into Jinja2 templates *before* adding
  to it is the cheap moment, and it makes the first option below the path of
  least resistance rather than a compromise.

Options:

- **Server-rendered templates from the existing FastAPI app.** No second
  process, no CORS, no second deployment. Jinja2 + a little vanilla JS. Least
  infrastructure, the demo URL keeps working, continuous with what is there.
- **Streamlit as a separate service.** Fastest to write, hardest to deploy
  alongside the API on Cloud Run, wants its own container.
- **A static SPA served by FastAPI.** Most flexible, most work, and the auth
  story (JWT in `localStorage`) needs thinking about.

The complete API surface, read from the OpenAPI schema rather than from the
route decorators — `main.py` includes its routers lazily, so `app.routes` does
not show them and grepping for `@router.post` misses the `/auth` prefix:

| Method | Path | Notes |
|---|---|---|
| `POST` | `/auth/token` | Login → JWT bearer. Not `/auth/login`. |
| `POST` | `/auth/register` | |
| `POST` | `/query` | The one that matters. See `QueryResponse`. |
| `GET` | `/health`, `/health/ready` | |
| `POST` | `/documents/ingest` | |
| `GET` | `/documents/supported-types` | |
| `GET` | `/admin/users` | |
| `GET` | `/`, `/docs`, `/redoc`, `/favicon.svg` | Landing page and branded docs |

`scripts/seed_users.py` creates users per role, and `scripts/start.py` seeds
them itself before launching uvicorn.

### 5.3 Make the RBAC visible

The most distinctive thing in the repo is invisible in every screenshot taken
so far. Logging in as `research` versus `admin` and asking the same question
returns different retrieval, and Phase 4 added four questions that prove it. A
role switcher showing the same question answered differently — including one
that correctly refuses because the barrier held — is the single most convincing
thing this dashboard could do, and the backend already does all of it.

**One thing you will need to add.** `get_information_barriers_for_user()`
returns the active barriers with names and descriptions, but `query.py`
flattens them into a single `guardrail_flags` string —
`"information_barriers: Research-Trading Wall, Research-Compliance Wall"` — and
throws the descriptions away. `QueryResponse` has no structured field for them.
Do not parse that string in the UI; add the field. It is the one place the
response shape is poorer than the data behind it, and it is exactly the data
this screen exists to show.

### 5.4 Do not invent numbers in the UI

Everything displayed should come from the response. In particular: do not
render `confidence.overall` as a percentage bar without the label, and **do not
render a refusal as an error state**. A refusal is a correct outcome that
scores 1.000 in the eval; showing it in red with a warning triangle tells the
user the opposite of what the measurement says.

---

## 6. Gotchas that will bite

**The corpus changed and nothing has been measured on it.** See §2. This is the
first one for a reason.

**The reranker silently refuses answerable questions.** The biggest live
defect, and Phase 4 is what made it visible. `ms-marco-MiniLM` assigns
uniformly negative logits to financial-table prose; `ScoredDocument.relevance`
squashes a logit through a logistic, so the whole set lands under the 0.15
`INSUFFICIENT_CONTEXT_THRESHOLD` and the system declines without an LLM call.
On "What was Apple's total net revenue for its most recent fiscal year?" it
ranked a *Foreign pretax earnings* chunk first at −2.33 and pushed the actual
net-sales table from rank 1 to rank 5, for a set relevance of 0.077. The same
five questions scored 0.286–0.466 with `RERANK_ENABLED=false`. **Do not fix
this by lowering the threshold** — that tunes the gate to the test set and
moves the failure downstream into ungrounded answers. Either a
financial-domain cross-encoder, or rerank-then-restore-order, or drop the
reranker and keep the dense ordering. Re-measure on the new corpus first: fewer
furniture chunks compete for the top-5 slots now, so the size of this may have
changed.

**A two-run noise floor is a lower bound, not a bound.** The n=54 floor in §8
comes from one pair of runs, and a third run on the same corpus and config
already exceeded it on one metric: citation coverage moved 0.032 against an
A–B spread of 0.004. Nothing about the system changed that could explain it —
coverage is computed by parsing, not judged — so the 0.004 was two samples
landing close together. Treat the table as "deltas below this are definitely
noise", never as "deltas above this are definitely real".

**`compare_eval_runs.py` will call a code change a noise floor.** It prints
"identical config — the spread below is the run-to-run noise floor" whenever
the two `config` blocks match, and `config` fingerprints *settings*, not
source. The `is_full_refusal` change moved refusal correctness 0.759 → 0.796
with a byte-identical config block, and the script announced it as noise. If
you change behaviour without changing a setting, say so in the commit.

**Never pool the no-answer questions.** 16 of the 54 expect a refusal. They are
excluded from the RAGAS and citation aggregates and scored on
`refusal_correctness`, because RAGAS scores a correct refusal 0.000 answer
relevancy — correctly, a refusal has no relevance to the question. Pooling them
lowers every aggregate *because the system behaved correctly*. The rule is
written in full into `config.no_answer_scoring` in every run.

**`refusal_correctness` is scored over all 54, not just the 16.** On an
answerable question it catches over-refusal, which is the live defect. If you
find yourself computing it over the refusal strata only, you have deleted the
half that currently matters.

**Per-question rows join by dataset index, not position.** Fixed in Phase 4 —
`_build_per_question()` joins through the index into `eval_data`, and
`evaluate_with_ragas()` returns `{index: scores}` because it now filters rows
out. A guard raises if RAGAS ever returns rows out of order. Do not reintroduce
a positional join; the filter makes it silently wrong.

**The eval judge's `max_tokens` is a coverage setting, not a tuning knob.** At
the old 1024 the judge overran on 4 of 20 questions and RAGAS dropped those
rows rather than truncating — and the dropped rows were the long, multi-claim
answers most at risk of being unfaithful. `eval_judge_max_tokens` is now 4096.
If `faithfulness_n` ever drops below the answerable count, this is why.
**`EVAL_JUDGE_MODEL` is pinned to `gpt-4o-mini` and must not move** — it is
measurement apparatus. `CITATION_JUDGE_MODEL` is a different judge and can be
chosen freely.

**Ambiguity is not handled at all.** Two of the three underspecified questions
were answered about one arbitrarily-chosen company with no flag. There is no
ambiguity-detection mechanism; the `ambiguous` stratum exists to record that.
If the dashboard offers query suggestions, do not let it paper over this.

**The live demo is stale.** Cloud Run still serves the pre-parser-fix build
with no sample documents. It will need an `OPENAI_API_KEY` at runtime for
generation, or `LLM_PROVIDER=groq` at deploy to stay on a free tier.

---

## 7. Cost and budget

Spend is on the user's OpenAI key. Phase 4 spent roughly **$3.25** across two
sessions, against a $2.60 approval. **Assume very little is left and confirm
before spending.**

Phase 5 is mostly UI and should cost almost nothing. The exceptions:

| Action | Estimate |
|---|---|
| Fresh baseline on the current corpus | ~$0.80 — the one thing worth spending on early |
| Single eval run, 54 questions | ~$0.80 (generation + judge + citation verification) |
| Ground truth generation | ~$0.03/question, scaling with filing size — JPM is 6 windows, AAPL 1 |
| Full six-config ablation re-run | ~$3, and only meaningful as a complete set |

Manual dashboard testing hits `POST /query`, a real `gpt-4o` call plus
retrieval each time. Cents, not dollars, but not free — consider a fixture or a
recorded response for UI iteration.

---

## 8. What Phase 4 established, quantitatively

**The noise floor fell an order of magnitude, and the reason retires a
hypothesis.** `handoffTo4.md` §7 predicted the n=20 faithfulness floor of 0.138
would fall roughly with √n to ~0.087 at n=50. Measured at n=54 it is **0.014**.

| | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Refusal Correctness |
|---|---|---|---|---|---|
| Spread, n=20 | 0.138 | 0.044 | 0.001 | 0.038 | — |
| **Spread, n=54** | **0.014** | 0.023 | **0.050** | 0.003 | **0.000** |

Sampling alone predicts a factor of 1.6; the observed factor is nearly 10. The
old floor was mostly the judge dropping rows (§6), not sampling noise. Two
things changed at once — question count and token budget — so the split is not
measured, but sampling can only account for the smaller part.

**Context precision moved the other way, and it is not a retrieval effect.**
Its spread rose from 0.001 to 0.050 despite 53 of 54 questions retrieving
byte-identical contexts across the two runs. The largest single move (+0.500)
had *identical contexts and an identical generated answer*. RAGAS context
precision is an LLM judgment about chunk usefulness, so a deterministic
retriever does not make it a deterministic metric. Do not repeat the inference
that "retrieval is deterministic so this barely moves."

**Refusal correctness did not move at all** across two runs in which 27 of 54
answers were textually different. The most stable metric in the set, and
therefore the best one to measure a change against.

**Two defects were found by slicing per company rather than per stratum**, and
neither was predicted: Goldman Sachs scoring exactly 0.000 context recall on
every question in every run (page furniture — now fixed, unmeasured), and MSFT
underperforming on faithfulness at 0.658 against 0.85–0.92 for everyone else,
consistent with its filing text being recovered at only ~71%. The MSFT one is
still open.

---

## 9. Decisions already made — do not relitigate

- **Questions, strata and refusal expectations are hand-written.** 54 is still
  small enough to read; an LLM classifier or generator introduces another
  model's judgment into a measurement chain the eval works to keep clean.
- **Ground truths come from full filing text, never from retrieval**
  (`scripts/generate_ground_truths_from_filings.py`). Use `--pending-only` when
  growing the set: regenerating an existing reference silently replaces what
  published results were scored against. At the default 240k window the
  generator misses material that is present but diffuse — two questions needed
  `--window-chars 60000` to recover. Retry before believing a NOT_IN_FILING.
- **Ambiguous questions get ground truths naming every defensible reading.**
  Three were hand-reconciled after the generator committed to one arbitrarily;
  scoring against a single reading measures a coin flip, not the system.
- **The three refusal strata stay separate**, not pooled into `no_answer`. They
  fail at different stages — `out_of_corpus` at the retrieval gate, `no_answer`
  only at the model, `rbac_blocked` at the where-clause — and point at
  different fixes.
- **Entity detection is a literal alias match** over five known companies, not
  NER and not an LLM call. Tickers match case-sensitively because lowercase
  "gs" occurs constantly in prose.
- **A comparison gets `retrieval_top_k` slots per company, not a share of one.**
  Splitting five slots two ways starves both halves to keep the context the
  size it would be for a simpler question, which is the wrong thing to hold
  fixed.
- **Citation verification is off at runtime, forced on in the eval.** One LLM
  call per citation is not worth it on a served query; it is exactly what buys
  the metric in a measured run.
- **A refusal is labelled `low` confidence regardless of the numeric
  composite** — but an answer carrying refusal wording *and* cited claims is a
  partial answer (completeness 0.5), not a refusal. See `is_full_refusal()`.
- **`EMBEDDING_PROVIDER=openai|huggingface`**, default openai
  (`text-embedding-3-small`). The two emit different-sized vectors (1536 vs
  384), so switching requires `ingest_edgar.py --from-disk --reset`. The
  Dockerfile pins `huggingface` because the image bakes its index at build time
  and embedding with OpenAI there would put a key in the build.
- **Hybrid search stays off by default**, and the old "costs 0.23 recall" claim
  was retracted — it did not reproduce on the fixed corpus. The honest
  statement is "not established as helpful."

---

## 10. Suggested first moves

1. **Run the eval once to get a current baseline** (~$0.80, confirm first).
   Everything in §4 describes a corpus that is no longer on disk, and the
   parser fix should have moved GS recall and the over-refusal rate. Doing this
   first means every later claim has something to stand on — and it is the
   first row of the ablation re-run you will eventually want anyway.

   ```bash
   ./.venv/Scripts/python.exe -m evaluation.run_evaluation
   ```

2. **Decide §5.1** — analyst tool or system demonstration — before writing any
   markup. It changes every screen.
3. **Read `src/common/schemas.py` and make one real query.** The response is
   the UI spec. Look at an actual `QueryResponse` with citations and an
   `UnansweredReport` before designing around it.

   ```bash
   ./.venv/Scripts/python.exe scripts/start.py
   ```

   Then authenticate at `POST /auth/token` and call `POST /query`.
4. **Build the refusal view first, not last.** It is the best-measured
   behaviour in the system, the data is already on the response, and it is the
   screen most RAG demos do not have. Building it first also stops it being
   rendered as an error state.
5. **Add the role switcher early.** It is what makes the information-barrier
   layer visible, and Phase 4 supplies four questions that demonstrate it.
   Add the structured barriers field (§5.3) while you are in there.
6. **Then Phase 6**: demo video, and redeploy the live demo, which is stale.

---

## 11. Commit split, when the time comes

Nothing is committed yet. Reasonable separation:

- EDGAR parser fixes + Dockerfile sample-docs fix (independently valuable)
- Tech stack: OpenAI embeddings + gpt-4o + provider switches
- Phase 1: three chunking strategies, dedup, chunk provenance
- Evaluation: v3 ground truths, stratification, corpus fingerprint, README
- Phase 3: citations, verification, confidence, structured refusal, score
  channel — plus the `citation_accuracy` wiring in the eval harness
- Phase 4a — retrieval: `entities.py`, `MultiEntityRetriever`, `extra_filter`
  threading, BM25 cache key fix
- Phase 4b — harness: `refusal_correctness`, the no-answer metric mask, the
  index join, `eval_judge_max_tokens`, `compare_eval_runs.py`
- Phase 4c — dataset: `eval_questions_v4.json`, the stratifier's new strata,
  `--pending-only`
- Phase 4d — `is_full_refusal` and its tests
- Phase 4e — page-furniture stripping in the parser, `min_chunk_chars`, and the
  re-ingest. **This one changes the corpus fingerprint**; note that in the
  commit message so the results boundary is findable later.
- Phase 5: whatever the next session produces
