# Handoff → Phase 6: Portfolio

Written 2026-08-09, at the end of Phase 5. This document is for a fresh
session picking up Phase 6 and assumes no memory of the previous ones.

Companion documents: `handoffTo5.md` (what Phase 5 was asked to produce),
`handoffTo4.md`, `handoffTo3.md`, `HANDOFF.md` (full project history, what was
tried and failed), `CASE_STUDY.md` (the portfolio write-up) and
`Project 6.docx` (the original spec — kept in the repo root but deliberately
not in version control, so a fresh clone will not have it). Read this one
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
| 4. Evaluation | **Done** — 54 hand-written questions, 7 strata, refusal scoring, noise floor measured |
| 5. API & dashboard | **Done** — Streamlit dashboard, `/access`, `/documents`, per-request retrieval mode, two-service deploy |
| **6. Portfolio** | **← you are here.** Case study written, demo deployed. No video; one open decision in §2.1 |

Phase 5 was verified requirement-by-requirement against Project 6 §5, not
against a summary of it. That turned up four gaps that had been assumed met.
If you take one process lesson from this handoff, it is to read the spec text
before declaring a phase done.

---

## 2. Read this before anything else

### The demo is live, and it is *not* the configuration the numbers describe

Two Cloud Run services in `us-central1`, project `rag-enterprise-498519`:

| Service | URL |
|---|---|
| `rag-enterprise` (API + landing page) | https://rag-enterprise-laa65asupq-uc.a.run.app |
| `rag-enterprise-dashboard` (Streamlit) | https://rag-enterprise-dashboard-laa65asupq-uc.a.run.app |

Both also answer on the longer `…-1072425852803.us-central1.run.app` form.
The short form is canonical because it is what has already been shared.

They find each other through environment variables: the dashboard has
`RAG_API_URL`, the API has `DASHBOARD_URL` so its landing-page links resolve.
**The dependency is circular**, so the working order is: deploy the API, read
its URL, deploy the dashboard with it, then update the API with the
dashboard's URL.

**The deployment generates with Gemini and embeds with local MiniLM.** Every
score in this document and in the README was measured on `gpt-4o` with
`text-embedding-3-small`. The Dockerfile pins `EMBEDDING_PROVIDER=huggingface`
because the image bakes its index at build time and embedding with OpenAI
there would put a key in the build — so the deployed index is a *different
corpus* as well (8,099 chunks against the 8,232 measured). The README and the
landing page both say so in as many words. Do not quietly delete those
caveats; a page showing 0.70 faithfulness beside answers no part of that
measurement produced is the exact failure this project's evaluation section
spends its length warning about.

### 2.1 The open decision: should the demo run gpt-4o?

This was asked and answered "keep Gemini (free tier)" before the first deploy,
then reopened at the end of the session. It was left undecided. There are two
levels and they differ enormously in effort:

**Option A — generation only.** Set `LLM_PROVIDER=openai` and add an
`openai-api-key` secret. No rebuild; a service config change, about five
minutes. `gpt-4o` writes the answers, but *retrieval* still runs on MiniLM
embeddings over the build-time index. Closer to the measured system, still not
it — faithfulness and citation accuracy were measured with OpenAI embeddings
driving retrieval.

**Option B — actually the measured system.** The local `chroma_data/` is
362 MB and holds exactly the measured corpus: **8,232 chunks, digest
`c2f8c13673cf5ca5`**. Ship *that* into the image instead of baking a fresh one
(it is currently excluded by `.dockerignore` line 20) and set
`EMBEDDING_PROVIDER=openai`. The deployment then has the same corpus, the same
embeddings and the same generator as `eval_20260809_015503` — the live demo
becomes the thing the README describes. Costs a ~362 MB image increase and a
rebuild.

**The reason to think hard before either.** gpt-4o at `retrieval_top_k=5` runs
roughly 3,500 input tokens and 300 output per query — about **$0.012 a query**.
The dashboard is deployed public with four demo logins and no authentication
of its own. The API rate-limits at 20/minute, which is the only thing between
a bot and the key: sustained, that is ~$14/hour. Before switching, either cap
spend in the OpenAI dashboard, or deploy the dashboard with
`--no-allow-unauthenticated`, or both. Gemini's free tier is why this is not
currently a problem.

### 2.2 Secrets

`google-api-key` and `jwt-secret-key` live in Secret Manager and are wired
with `--set-secrets`, with `1072425852803-compute@developer.gserviceaccount.com`
granted `secretAccessor`. Nothing sensitive remains in the service config.

The previous deployment held both **in plaintext as environment variables**,
readable by any project viewer and retained in every revision's history. The
JWT signing key was rotated during the move. **The old Google API key should
be treated as compromised** — delete it in AI Studio if that has not happened.
There is also a stray version 1 on `google-api-key` from a first attempt;
harmless because deploys pin `:latest`, but worth disabling.

### 2.3 One code change is waiting on a re-ingest

`src/ingestion/loaders.py` now pins UTF-8 for `TextLoader` and `BSHTMLLoader`,
which previously defaulted to `locale.getpreferredencoding()` — cp1252 on
Windows, UTF-8 on Linux. The index on disk was built on Windows, so the sample
documents' em dashes are stored as mojibake and show in any chunk snippet the
dashboard renders. The two loaders spell the argument differently
(`encoding=` vs `open_encoding=`), so it is a per-class lookup; one keyword
passed to both raises `TypeError` on the HTML path, which is what
`tests/unit/test_loaders.py` pins.

The code is fixed and tested. **The index is not**, because re-ingesting
changes the corpus fingerprint and would invalidate the baseline in §4. Do it
deliberately and re-run the eval in the same sitting. It affects only
`data/sample/` — SEC filings go through `src/edgar/parser.py`, not this loader.

---

## 3. Repo state

- `main` is current and pushed. Phases 1–5 are all on it; `origin/main` is at
  the same commit.
- **362 tests pass, ruff clean.**
- **Use `./.venv/Scripts/python.exe`, never bare `python`.** The `python` on
  PATH is system Python 3.13 with *some* dependencies (langchain,
  langchain-openai) but not `rank_bm25`, `transformers`, or
  `sentence-transformers`. The partial overlap is the trap: LangChain-only
  scripts run fine, then `pytest` fails collection on five modules and reads
  as a broken suite rather than a wrong interpreter.

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
./.venv/Scripts/python.exe -m ruff check src tests dashboard evaluation scripts
```

Run it — `.claude/launch.json` has `dev` (API, :8000) and `dashboard` (:8501):

```bash
make docker-up   # both services in containers, the path a reviewer should take
make dev         # API only
make dashboard   # Streamlit only; needs the API up. Runs from dashboard/
```

`make dashboard` runs from inside `dashboard/` because Streamlit resolves
`.streamlit/config.toml` against the working directory, not the script. The
container does the same. Diverging themes one and not the other.

---

## 4. The current baseline

`eval_20260809_015503.json`, **one run**, against the 8,232-chunk corpus
(`c2f8c13673cf5ca5`) on disk. First run on this corpus; everything before it
was measured on 9,572 chunks.

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
`_strip_page_furniture()` took them to **0**. GS context recall did not move
off 0.000 and GS context precision *fell*, 0.271 → 0.050. The furniture was
real, it is gone, and it was not the cause. This is now the best-isolated open
defect in the system: a hard zero on one company and not the other four, with
the leading hypothesis eliminated by measurement.

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

**A Streamlit query dashboard** (`dashboard/app.py`) — a system demonstration,
not an analyst tool. The distinctive things here (information barriers,
citation verdicts, structured refusal) are invisible in a screen optimised for
reading one answer, so it traces one question through six stages: identity and
the walls in force, the query, the ranked chunks with scores, the answer or the
refusal, the per-claim verdicts, the confidence breakdown, the guardrails.

It holds no retrieval logic, no scoring and no thresholds — it is an HTTP
client, so it cannot disagree with the system it shows.

It was first built as a Jinja2 page served by the API, then rebuilt in
Streamlit because Project 6 §5.2 names Streamlit or React. Only the dashboard
moved; the landing page is still Jinja2 in `src/web/`.

**A per-request retrieval toggle.** §5.2 asks for a toggle comparing hybrid
against dense-only side by side, which was impossible while `get_retriever()`
read the stage from global settings. `QueryRequest` carries `retrieval_mode`
(`default | dense | hybrid`) and the response reports the stages that actually
ran in `retrieval` — a toggle that silently failed to take effect would look
exactly like one that made no difference. Only this stage is exposed per
request: the eval harness measures stages by running under a configuration,
and a knob for each would make "which pipeline produced this result" a
property of the request rather than of the run.

**`GET /documents`** — what is indexed, rolled up from chunk provenance and
filtered by the same where-clause retrieval builds, so a listing cannot name a
document the caller could not retrieve. Locally: research sees 9 documents /
8,162 chunks, admin 14 / 8,232. Reads metadata only (`get_all_metadata`) —
materialising 8,232 chunk bodies to count them took 39 seconds, which made a
listing slower than a real query.

**`GET /access`** — roles, accessible departments and active barriers, with no
LLM call and no vector store hit, so the role switcher can show the barrier
moving without spending a generation call per role.

**New fields on the wire.** `information_barriers` (name, description, blocked
departments) and `accessible_departments`, both previously flattened into one
`guardrail_flags` string; `SourceDocument.relevance_score` / `.raw_score` /
`.score_type`, which were declared on the schema and null on every response
ever served because the route mapped documents rather than the scored channel
behind them. The flattened string is unchanged — it is what the 17a-4 audit
trail already records.

**`src/web/`.** `src/main.py` was 973 lines, of which 780 were one inline HTML
string. It is now ~210 and the landing page is a Jinja2 template with
extracted stylesheets. `pyproject.toml` declares `jinja2` and the
template/static files as package data, because the Dockerfile `pip install .`s
the project and a `src.web` installed without its templates is a package of
nothing but `.py`.

**A compose port that never worked.** `docker-compose.yml` published
`8000:8000` while `scripts/start.py` binds `$PORT`, defaulting to 8080; `.env`
sets `APP_PORT`, which `start.py` does not read. `make docker-up` produced a
container that started cleanly and was unreachable. Now `8000:8080` with
`PORT` named explicitly, a healthcheck the dashboard waits on, and `.env`
marked optional so a fresh clone starts instead of refusing to.

### 5.1 Three rules the dashboard keeps — do not break them

- **A refusal is not an error state.** It scores 1.000 on all three
  unanswerable strata. It gets a neutral treatment of its own, and the panel
  distinguishes `low_retrieval_confidence` (gate fired, nothing sent to the
  model) from `model_refused` (retrieval cleared, model still declined),
  because the two point at different fixes.
- **No number without its scale.** `confidence.overall` never appears without
  its label; a null retrieval score renders as *unavailable*, not zero; a
  chunk's relevance sits next to its raw score and the stage that produced it.
- **Nothing is precomputed.** A test in `tests/integration/test_web_routes.py`
  asserts no score is baked into the landing template.

---

## 6. Gotchas that will bite

**The Goldman Sachs zero has no explanation left.** See §4.1. Do not repeat the
page-furniture hypothesis; it is measured and refuted.

**The reranker silently refuses answerable questions.** Still the biggest live
defect — improved but not fixed, 6 false refusals remain. `ms-marco-MiniLM`
assigns uniformly negative logits to financial-table prose;
`ScoredDocument.relevance` squashes a logit through a logistic, so the whole
set lands under the 0.15 `INSUFFICIENT_CONTEXT_THRESHOLD` and the system
declines without an LLM call. Visible on the dashboard as a matter of course: on
the Apple revenue question the actual net-sales table ranks **[3]** at raw
−3.69 while a Goldman Sachs chunk takes slot [1] at +0.34. **Do not fix this by
lowering the threshold** — that tunes the gate to the test set and moves the
failure downstream into ungrounded answers. Either a financial-domain
cross-encoder, or rerank-then-restore-order, or drop the reranker and keep the
dense ordering.

**A two-run noise floor is a lower bound, not a bound.** The n=54 floor comes
from one pair of runs, and a third run on the same corpus and config already
exceeded it on one metric. Treat the table as "deltas below this are definitely
noise", never as "deltas above this are definitely real".

**`compare_eval_runs.py` will call a code change a noise floor.** It prints
"identical config — the spread below is the run-to-run noise floor" whenever
the two `config` blocks match, and `config` fingerprints *settings*, not
source. If you change behaviour without changing a setting, say so in the
commit.

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
its banner. Measured on 9,572 chunks at n=20 against a 0.138 faithfulness noise
floor, and only meaningful as a complete set — re-running one row is worse than
re-running none. ~$3 for the full set.

**The `fixed` chunking strategy has never been evaluated.** It exists in code
as the structure-blind baseline that recursive chunking has to beat, and the
comparison in `CASE_STUDY.md` is two rows on two different indices with one
delta clearing its noise floor. That missing row is the cheapest real result
left in the project.

**The API Dockerfile does `COPY . .` before the build-time ingest**, so editing
*any* file — including `dashboard/`, which the API image never uses —
invalidates a ~25-minute ingest layer. Splitting the COPY so the ingest depends
only on `src/` and `scripts/` is a small fix that was deliberately not made
mid-verification. Cloud Build takes ~26 minutes for the API image and ~1.5 for
the dashboard.

**The palette is duplicated.** `src/web/static/css/tokens.css` and
`dashboard/.streamlit/config.toml` hold the same colours, because the dashboard
is a separate process and cannot import the stylesheet. Both files say so.
Change them together. Streamlit's widget chrome (radio dots, select carets)
stays Streamlit's — the two pages read as one product but are not
pixel-identical, and closing that gap means fragile hashed-class selectors that
break on upgrade.

---

## 7. Cost and budget

Spend is on the user's OpenAI key. Phase 4 spent roughly $3.25 against a $2.60
approval; Phase 5 spent ~$0.80 on the baseline in §4 plus a few cents of
manual testing, approved in advance. **Assume very little is left and confirm
before spending.**

| Action | Estimate |
|---|---|
| Single eval run, 54 questions | ~$0.80 |
| The missing `fixed` chunking row (§6) | ~$0.80, plus an index build |
| Re-ingest + fresh baseline (§2.3) | ~$0.80 plus embedding cost |
| Full six-config ablation re-run | ~$3, and only meaningful as a complete set |
| Ground truth generation | ~$0.03/question, scaling with filing size |
| Running the demo on gpt-4o | ~$0.012/query, uncapped on a public URL — see §2.1 |

A refusal on the `low_retrieval_confidence` path costs nothing: the gate fires
before the model is called, so barrier and out-of-corpus demos are free to
exercise as often as you like.

---

## 8. Decisions already made — do not relitigate

- **The dashboard is a system demonstration, not an analyst tool.**
- **The dashboard is Streamlit, in its own process and its own container.**
  This reversed an earlier decision to server-render it from the FastAPI app.
  The earlier reasoning (no second process, no CORS, no second deployment) was
  sound and is now the accepted cost; Project 6 §5.2 names Streamlit or React,
  and §5.3 describes a compose file with the frontend as its own service.
- **`guardrail_flags` keeps its flattened barrier string** alongside the
  structured field. The audit trail already records it.
- **Only the search stage is overridable per request.** See §5.
- **Questions, strata and refusal expectations are hand-written.** 54 is still
  small enough to read; an LLM classifier introduces another model's judgment
  into a measurement chain the eval works to keep clean.
- **Ground truths come from full filing text, never from retrieval.** Use
  `--pending-only` when growing the set. At the default 240k window the
  generator misses material that is present but diffuse — two questions needed
  `--window-chars 60000`. Retry before believing a NOT_IN_FILING.
- **Ambiguous questions get ground truths naming every defensible reading.**
- **The three refusal strata stay separate**, not pooled into `no_answer`.
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
- **Hybrid search stays off by default.** Measured against dense + rerank on
  the same corpus it lost on all four metrics, every delta at or inside the
  n=20 noise floor. The honest statement is "not established as helpful", not
  "hybrid is worse" — and the old "costs 0.23 recall" claim was retracted when
  it failed to reproduce. RRF is also ordinal, so enabling it discards the
  score channel the confidence layer and the insufficient-context gate both
  read: turning hybrid on silently disables the system's ability to say "I
  don't know."

---

## 9. What Phase 6 asks for

Project 6 §6 is the portfolio phase: record a demo walkthrough, and write the
case study. The case study is done (`CASE_STUDY.md`). What is left:

1. **Decide §2.1** — whether the demo runs gpt-4o, and if so whether it also
   ships the measured index. Read the spend note first; the dashboard is
   public and unauthenticated.

2. **Record the demo walkthrough.** Under four minutes, per the spec. Three
   shots, in this order:
   - **The role switch.** Ask the ACME trading-desk question as
     `research_analyst` — it declines at the retrieval gate with the
     Research-Trading Wall shown in force and `trading` listed among the
     departments it removed. Switch to `trader_desk`; the same question answers
     at ~0.80 confidence off `trading_desk_procedures.txt`. The wall is the
     only thing that changed. This is the strongest 40 seconds in the project.
   - **A refusal that is not an error.** "How many iPhone units did Apple ship
     in fiscal 2025?" — `model_refused`, retrieval scoring 0.716 while
     completeness is 0.000, the passages consulted and the filings worth
     opening by hand. Shows the system declining correctly and saying why.
   - **Hybrid vs. dense side by side**, which the spec asks for by name. On the
     Apple revenue question the two columns differ by one rank position and
     both leave a Goldman Sachs chunk wrongly at rank 1 — the reranker defect,
     on screen, next to a comparison whose aggregate answer is "inside the
     noise floor."

3. **Optional, if budget allows**: the missing `fixed` chunking row (§6), a
   second run on the current corpus so §4 is a mean rather than a single
   sample, or the ablation re-run.

The case study deliberately does not claim hybrid beats dense-only, because
the measurement says otherwise. If someone asks you to reframe it that way,
that is the one thing in this repo worth pushing back on: the negative result
and the retracted claims are what make the rest of the numbers credible.
