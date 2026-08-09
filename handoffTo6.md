# Handoff → Phase 6: Portfolio

Written 2026-08-08, at the end of Phase 5. This document is for a fresh
session picking up Phase 6 and assumes no memory of the previous ones.

Companion documents: `handoffTo5.md` (what Phase 5 was asked to produce),
`handoffTo4.md`, `handoffTo3.md`, `HANDOFF.md` (full project history, what was
tried and failed) and `Project 6.docx` (the original spec — kept locally in the
repo root but deliberately not in version control, so a fresh clone will not
have it). Read this one first; reach for those when you need the why behind a
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
| 1. Ingestion & chunking | **Done** — three strategies, dedup, page-furniture stripping, chunk provenance |
| 2. Hybrid retrieval | **Done** — dense + BM25 + RRF + cross-encoder + per-entity split |
| 3. Generation & citation | **Done** — bracketed citations, LLM-judge verification, composite confidence, structured "I don't know" |
| 4. Evaluation | **Done** — 54 hand-written questions, 7 strata, refusal scoring, noise floor measured |
| 5. API & dashboard | **Done** — `GET /dashboard`, `GET /access`, structured barriers on the response, templates |
| **6. Portfolio** | **← you are here.** Case study strong, no demo video, live demo stale |

---

## 2. Read this before anything else

**The live demo is deployed and current, but it is not the measured system.**

Two Cloud Run services in `us-central1`, project `rag-enterprise-498519`:

| Service | URL |
|---|---|
| `rag-enterprise` (API + landing page) | https://rag-enterprise-1072425852803.us-central1.run.app |
| `rag-enterprise-dashboard` (Streamlit) | https://rag-enterprise-dashboard-1072425852803.us-central1.run.app |

They know each other through env vars: the dashboard has `RAG_API_URL`, the
API has `DASHBOARD_URL` so its landing page links resolve. Deploy the API
first, then the dashboard with the API's URL, then update the API with the
dashboard's — the dependency is circular and that is the order that breaks it.

**The deployment generates with Gemini and embeds with local MiniLM.** Every
score in the README was measured on `gpt-4o` with `text-embedding-3-small`.
The Dockerfile pins `EMBEDDING_PROVIDER=huggingface` because the image bakes
its index at build time and embedding with OpenAI there would put a key in the
build — so the deployed index is a different corpus too (8,099 chunks against
the 8,232 measured). The demo shows the system working; it is not the
configuration the numbers describe, and both the README and the landing page
say so. Switching to OpenAI is a config change on the service plus a rebuild,
not a code change.

**Secrets are in Secret Manager, not env vars.** `google-api-key` and
`jwt-secret-key`, referenced with `--set-secrets`, with
`1072425852803-compute@developer.gserviceaccount.com` granted
`secretAccessor`. The previous deployment held both in plaintext in the
service config, where they were readable by any project viewer and persisted
in every revision's history; the JWT key was rotated when they moved. **The
old Google API key should be considered compromised** — it sat in that config
and in this repo's deploy history. Delete it in AI Studio if that has not
already happened.

**One code change is waiting on a re-ingest.** `src/ingestion/loaders.py` now
pins UTF-8 for `TextLoader` and `BSHTMLLoader`, which previously defaulted to
`locale.getpreferredencoding()` — cp1252 on Windows, UTF-8 on Linux. The index
on disk was built on Windows, so the sample documents' em dashes are stored as
`â€"` and are visible in any chunk snippet the dashboard renders. The loaders
spell the argument differently (`encoding=` vs `open_encoding=`), so it is a
per-class lookup; passing one keyword to both raises `TypeError` on the HTML
path, which is what `tests/unit/test_loaders.py` pins.

The code is fixed and tested. **The index is not**, because re-ingesting
changes the corpus fingerprint and would invalidate the baseline measured in
§4. Do it deliberately, and re-run the eval in the same sitting. It only
affects `data/sample/` — SEC filings go through `src/edgar/parser.py`, not this
loader.

---

## 3. Repo state

- Branch `phases-1-4`. Phases 1–5 are committed; `main` is behind. See §9.
- 347 tests pass, ruff clean.
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

Run the app — `.claude/launch.json` has a `dev` config on port 8000, or:

```bash
./.venv/Scripts/python.exe scripts/start.py
```

---

## 4. The current baseline

`eval_20260809_015503.json`, **one run**, against the 8,232-chunk corpus
(`c2f8c13673cf5ca5`) currently on disk. This is the first run on this corpus;
everything before it was measured on 9,572 chunks.

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Citation Accuracy | Citation Coverage | Refusal Correctness |
|---|---|---|---|---|---|---|
| 0.698 | 0.682 | 0.378 | 0.335 | **0.939** | 0.661 | **0.852** |

| Stratum | n | Refusal Correctness |
|---|---|---|
| `interpretive` | 14 | 0.929 |
| `exact_figure` | 13 | 0.846 |
| `comparative` | 8 | 0.625 |
| `ambiguous` | 6 | 0.667 |
| `no_answer` | 5 | **1.000** |
| `out_of_corpus` | 4 | **1.000** |
| `rbac_blocked` | 4 | **1.000** |

**It is one run against a two-run mean.** The previous figures were a mean of
two. Only the largest moves are worth reading, and only on metrics stable
enough to carry them.

### 4.1 What the parser fix did and did not do

Page-furniture stripping was aimed at over-refusal and at Goldman Sachs'
context recall. It hit one of the two.

**Over-refusal improved.** Refusal correctness 0.796 → 0.852, `exact_figure`
0.692 → 0.846. As rows rather than as a mean: 11 answerable questions declined
before, **6** now, plus 2 answered that should have declined. Refusal
correctness is the most stable metric in the set — spread 0.000 across two runs
in which 27 of 54 answers were textually different — so this move is safe to
attribute.

**Goldman Sachs is still exactly 0.000, and that retires the explanation.** GS
carried 214 bare running-header chunks (`"Goldman Sachs 2025 Form 10-K | 123"`);
`_strip_page_furniture()` took them to **0**. GS context recall did not move off
0.000 and GS context precision fell 0.271 → 0.050. The furniture was real, it is
gone, and it was not the cause. This is now the best-isolated open defect in the
system: a hard zero on one company and not the other four, with the leading
hypothesis eliminated by measurement.

**Context recall fell overall, 0.383 → 0.335.** AAPL 0.569 → 0.403, JPM 0.267 →
0.167, only MSFT rising. The n=54 recall spread is 0.003, so these are large
relative to run-to-run variation — but it is not a controlled comparison. If
you want to call the stripping a net regression on grounding, measure it
properly first.

Per company, current run:

| | AAPL | MSFT | JPM | GS | TSLA |
|---|---|---|---|---|---|
| Faithfulness | 0.737 | 0.718 | 0.900 | 0.613 | 0.881 |
| Context Recall | 0.403 | 0.346 | 0.167 | **0.000** | 0.506 |
| Context Precision | 0.722 | 0.619 | 0.158 | **0.050** | 0.305 |

---

## 5. What Phase 5 built

**A Streamlit query dashboard** (`dashboard/app.py`, port 8501) — a system
demonstration, not an analyst tool. That choice was made explicitly
(`handoffTo5.md` §5.1 asked for it) on the grounds that the distinctive things
here — information barriers, citation verdicts, structured refusal — are
invisible in a screen optimised for reading one answer. It traces one question
through six stages: identity and the walls in force, the query, the ranked
chunks with scores, the answer or the refusal, the per-claim verdicts, the
confidence breakdown, the guardrail flags.

It was first built as a Jinja2 page served by the API, then rebuilt in
Streamlit because Project 6 §5.2 names Streamlit or React. The Jinja2 landing
page stays; only the dashboard moved. **The consequence is that the deployment
is now two services**, which is what §5.3 of the spec describes and what
`docker-compose.yml` now does — but Cloud Run currently runs one, so §9 has
grown a step.

**A per-request retrieval toggle.** §5.2 asks for "a toggle to compare hybrid
vs. dense-only retrieval side by side," which was impossible while
`get_retriever()` read the stage choice from global settings. `QueryRequest`
now carries `retrieval_mode` (`default | dense | hybrid`), and the response
reports the stages that actually ran in `retrieval`. Only this one stage is
exposed per request: the eval harness measures stages by running under a
configuration, and a per-request knob for each would make "which pipeline
produced this result" a property of the request rather than of the run.

**`GET /documents`** — what is indexed, rolled up from chunk provenance and
filtered by the same where-clause retrieval uses. Research sees 9 documents
and 8,162 chunks; admin sees 14 and 8,232. Reads metadata only
(`get_all_metadata`), because materialising 8,232 chunk bodies to count them
made the listing slower than a real query — 39s against under one.

**`GET /access`** — the caller's roles, accessible departments and active
barriers, with no LLM call and no vector store hit. It exists so the role
switcher can show the barrier moving without spending a `gpt-4o` call per role.

**New fields on the wire.** `QueryResponse.information_barriers`
(name, description, blocked departments) and `.accessible_departments`, both
of which were computed and then flattened into one `guardrail_flags` string;
and `SourceDocument.relevance_score` / `.raw_score` / `.score_type`, which were
declared on the schema and null on every response ever served because the route
mapped documents rather than the scored channel behind them. The flattened
`guardrail_flags` string is unchanged — it is what the 17a-4 audit trail
already records.

**`src/web/`.** `src/main.py` was 973 lines, of which 780 were one inline HTML
string. It is now ~210, and the landing page is a Jinja2 template with
extracted stylesheets. `pyproject.toml` declares `jinja2` and the
template/static files as package data, because the Dockerfile `pip install .`s
the project and a `src.web` installed without its templates is a package of
nothing but `.py`.

**A compose port that never worked.** `docker-compose.yml` published
`8000:8000` while `scripts/start.py` binds `$PORT`, defaulting to 8080. `.env`
sets `APP_PORT=8000`, a variable `start.py` does not read. `make docker-up`
produced a container that started cleanly and was unreachable — a worse
failure than not starting at all. It is now `8000:8080` with `PORT` named
explicitly in the compose environment so the two cannot drift.

### 5.1 Three rules the dashboard keeps — do not break them

- **A refusal is not an error state.** It scores 1.000 on all three
  unanswerable strata. It gets a neutral treatment of its own, and the panel
  distinguishes `low_retrieval_confidence` (gate fired, nothing sent to the
  model) from `model_refused` (retrieval cleared, model still declined),
  because the two point at different fixes.
- **No number without its scale.** `confidence.overall` never appears without
  its label; a null retrieval score renders as *unavailable*, not zero; a
  chunk's relevance sits next to its raw score and the stage that produced it.
- **Nothing is precomputed.** The template is a shell. A test in
  `tests/integration/test_web_routes.py` asserts no score is baked into it.

---

## 6. Gotchas that will bite

**The Goldman Sachs zero has no explanation left.** See §4.1. Do not repeat
the page-furniture hypothesis; it is measured and refuted.

**The reranker silently refuses answerable questions.** Still the biggest live
defect, improved but not fixed — 6 false refusals remain.
`ms-marco-MiniLM` assigns uniformly negative logits to financial-table prose;
`ScoredDocument.relevance` squashes a logit through a logistic, so the whole set
lands under the 0.15 `INSUFFICIENT_CONTEXT_THRESHOLD` and the system declines
without an LLM call. This is now visible on the dashboard as a matter of course:
a chunk at raw −2.35 rendering as 0.087. **Do not fix this by lowering the
threshold** — that tunes the gate to the test set and moves the failure
downstream into ungrounded answers. Either a financial-domain cross-encoder, or
rerank-then-restore-order, or drop the reranker and keep the dense ordering.

**A two-run noise floor is a lower bound, not a bound.** The n=54 floor comes
from one pair of runs, and a third run on the same corpus and config already
exceeded it on one metric. Treat the table as "deltas below this are definitely
noise", never as "deltas above this are definitely real".

**`compare_eval_runs.py` will call a code change a noise floor.** It prints
"identical config — the spread below is the run-to-run noise floor" whenever the
two `config` blocks match, and `config` fingerprints *settings*, not source. If
you change behaviour without changing a setting, say so in the commit.

**Never pool the no-answer questions.** 16 of the 54 expect a refusal. They are
excluded from the RAGAS and citation aggregates and scored on
`refusal_correctness`, because RAGAS scores a correct refusal 0.000 answer
relevancy — correctly, a refusal has no relevance to the question. Pooling them
lowers every aggregate *because the system behaved correctly*.

**`refusal_correctness` is scored over all 54, not just the 16.** On an
answerable question it catches over-refusal, which is the live defect.

**Per-question rows join by dataset index, not position.** A guard raises if
RAGAS returns rows out of order. Do not reintroduce a positional join.

**The eval judge's `max_tokens` is a coverage setting, not a tuning knob.**
`eval_judge_max_tokens` is 4096. If `faithfulness_n` ever drops below the
answerable count, this is why. **`EVAL_JUDGE_MODEL` is pinned to `gpt-4o-mini`
and must not move** — it is measurement apparatus. `CITATION_JUDGE_MODEL` is a
different judge and can be chosen freely.

**Ambiguity is not handled at all.** There is no ambiguity-detection mechanism;
the `ambiguous` stratum exists to record that.

**The six-config ablation table in `README.md` is obsolete** and still carries
its banner. It was measured on 9,572 chunks and is only meaningful as a
complete set; re-running one row against the current corpus is worse than
re-running none. ~$3 for the full set.

---

## 7. Cost and budget

Spend is on the user's OpenAI key. Phase 4 spent roughly $3.25 against a $2.60
approval; Phase 5 spent ~$0.80 on the baseline in §4 plus a few cents of
dashboard testing, approved in advance. **Assume very little is left and
confirm before spending.**

| Action | Estimate |
|---|---|
| Single eval run, 54 questions | ~$0.80 |
| Re-ingest + fresh baseline (see §2) | ~$0.80 plus embedding cost |
| Full six-config ablation re-run | ~$3, and only meaningful as a complete set |
| Ground truth generation | ~$0.03/question, scaling with filing size |

Manual dashboard testing hits `POST /query`, a real `gpt-4o` call plus
retrieval each time. Cents, not dollars, but not free. A refusal on the
`low_retrieval_confidence` path costs nothing — the gate fires before the model
is called — so barrier and out-of-corpus demos are free to exercise.

---

## 8. Decisions already made — do not relitigate

- **The dashboard is a system demonstration, not an analyst tool.** See §5.
- **Server-rendered templates from the existing FastAPI app**, not Streamlit or
  an SPA. No second process, no CORS, no second deployment, and the demo URL
  keeps working.
- **`guardrail_flags` keeps its flattened barrier string** alongside the
  structured field. The audit trail already records it.
- **Questions, strata and refusal expectations are hand-written.** 54 is still
  small enough to read; an LLM classifier introduces another model's judgment
  into a measurement chain the eval works to keep clean.
- **Ground truths come from full filing text, never from retrieval.** Use
  `--pending-only` when growing the set. At the default 240k window the
  generator misses material that is present but diffuse — two questions needed
  `--window-chars 60000`. Retry before believing a NOT_IN_FILING.
- **Ambiguous questions get ground truths naming every defensible reading.**
- **The three refusal strata stay separate**, not pooled into `no_answer`. They
  fail at different stages and point at different fixes.
- **Entity detection is a literal alias match** over five known companies, not
  NER and not an LLM call. Tickers match case-sensitively because lowercase
  "gs" occurs constantly in prose.
- **A comparison gets `retrieval_top_k` slots per company, not a share of one.**
- **Citation verification is off at runtime, forced on in the eval.**
- **A refusal is labelled `low` confidence regardless of the numeric
  composite** — but an answer carrying refusal wording *and* cited claims is a
  partial answer (completeness 0.5), not a refusal. See `is_full_refusal()`.
- **`EMBEDDING_PROVIDER=openai|huggingface`**, default openai. The two emit
  different-sized vectors (1536 vs 384), so switching requires
  `ingest_edgar.py --from-disk --reset`.
- **Hybrid search stays off by default.** The old "costs 0.23 recall" claim was
  retracted; the honest statement is "not established as helpful."

---

## 9. What Phase 6 asks for

Project 6 §6 is the portfolio phase. The case study is strong — the README
carries the measurements, the retracted claims and the open defects. What is
missing:

1. ~~**Redeploy.**~~ Done — both services are live and verified end to end
   (health, the access profile with both walls, the documents listing, and the
   barrier demo declining as `research` and answering as `trading`). See §2 for
   URLs and the Gemini caveat. The one thing left here is deciding whether the
   demo should run OpenAI so it matches the published numbers.
2. **Demo video.** The dashboard is the thing to record, and there are two
   shots. The role switch: ask the ACME trading-desk question as
   `research_analyst`, watch it decline at the gate with the Research-Trading
   Wall shown in force, then switch to `trader_desk` and watch the same
   question answer at high confidence off `trading_desk_procedures.txt`. And
   the hybrid-vs-dense comparison the spec asks for: on the Apple revenue
   question the two columns differ by one rank position, with both leaving a
   Goldman Sachs chunk wrongly at rank 1 — the reranker defect, on screen.
3. **Merge to `main`.** §3. Everything is on `phases-1-4`.
4. Optional, if budget allows: the ablation re-run (§6), or a second run on the
   current corpus so the baseline in §4 is a mean rather than a single sample.
