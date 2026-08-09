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
| Tech stack | Done — `gpt-4o`, `text-embedding-3-small`, ChromaDB, `rank_bm25`, FastAPI, Docker (the deploy generates with Gemini — §2) |
| 1. Ingestion & chunking | **Done** — three strategies, dedup, page-furniture stripping, chunk provenance |
| 2. Hybrid retrieval | **Done** — dense + BM25 + RRF + cross-encoder + per-entity split |
| 3. Generation & citation | **Done** — bracketed citations, LLM-judge verification, composite confidence, structured "I don't know" |
| 4. Evaluation | **Done** — 54 hand-written questions, 7 strata, refusal scoring, noise floor measured |
| 5. API & dashboard | **Done** — Streamlit dashboard, `/access`, `/documents`, per-request retrieval mode, two-service deploy |
| **6. Portfolio** | Case study written; demo deployed on the measured index (§2.1) with two privilege holes closed (§2.2a). **The walkthrough recording was dropped** (§9) — the live demo stands in for it |

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

**The deployment retrieves as measured and generates with Gemini.** Resolved
2026-08-09; §2.1 records the reasoning. The image now *ships* the measured
index (8,232 chunks, digest `c2f8c13673cf5ca5`) instead of baking a fresh one,
and embeds with `text-embedding-3-small`, so retrieval, ranking, citation
targets and the insufficient-context gate are the measured pipeline. Answer
text comes from Gemini's free tier, not `gpt-4o`.

So of the seven headline metrics, the retrieval- and citation-driven ones are
reproducible on the live demo; faithfulness and answer relevancy are
generator-dependent and were measured on `gpt-4o`. The README and landing page
say exactly this. **Do not upgrade that to "the demo is the measured system"** —
it is two-thirds of the way there, and the remaining third is the part a
reviewer is most likely to check.

### 2.1 The gpt-4o decision — settled, and why it landed where it did

The index moved to OpenAI; generation did not. Both halves were deliberate.

**Why the index moved.** The local `chroma_data/` holds the measured corpus.
`scripts/build_index_dist.py` copies it to `chroma_dist/`, keeps the one live
collection (the working directory also held a semantic-chunking experiment, a
dedup smoke test and eight orphaned segment directories — 361 MB down to 106),
and **fails unless the result is 8,232 chunks at digest `c2f8c13673cf5ca5`**.
The Dockerfile copies that to the default persist path. The build-time ingest
is gone, which took the image from 3.13 GB to 829 MB and Cloud Build from ~26
minutes to ~3.

**Why generation did not.** gpt-4o runs ~3,500 input and 300 output tokens per
query — **~$0.012**. The demo is public with published logins, and the owner's
constraint was no spend after deploy. Embedding a question instead costs
**~$0.000005** (`text-embedding-3-small`, $0.02/1M, questions capped at 1,000
chars), which is what makes an open URL affordable. The OpenAI key is on the
service for embeddings only.

**The claim in §2.1 of the previous handoff was wrong, and it mattered.** It
said "the API rate-limits at 20/minute, which is the only thing between a bot
and the key." There is no rate limiting. `Limiter` is built in
`src/api/middleware.py` and attached to `app.state`, but `SlowAPIMiddleware` is
never added and no route carries `@limiter.limit`, so slowapi's `default_limits`
never apply — verified against the deployment, 40 consecutive requests, zero
429s. Had the demo been switched to gpt-4o on the strength of that sentence,
the only real ceiling would have been the OpenAI account tier: between ~$140
and ~$3,600 a day depending on it. **Check a control exists before citing it as
one.** The setting now carries a comment saying it is not enforced; wiring it
up is a three-line change nobody has made.

Spend is instead bounded by `--max-instances 1 --concurrency 8` on the service,
which caps in-flight requests regardless of caller IP.

### 2.2 Secrets

`google-api-key`, `jwt-secret-key` and `openai-api-key` live in Secret Manager
and are wired with `--set-secrets`, with
`1072425852803-compute@developer.gserviceaccount.com` granted `secretAccessor`.
Nothing sensitive remains in the service config.

The previous deployment held both **in plaintext as environment variables**,
readable by any project viewer and retained in every revision's history. The
JWT signing key was rotated during the move. **The old Google API key should
be treated as compromised** — delete it in AI Studio if that has not happened.
There is also a stray version 1 on `google-api-key` from a first attempt;
harmless because deploys pin `:latest`, but worth disabling.

### 2.2a Two privilege holes, found while pricing the gpt-4o switch

Both are closed and verified against the deployment. Recorded because the
pattern is worth keeping: the barriers were enforced correctly everywhere they
were *checked*, and the holes were both upstream of the check.

**Anyone could register themselves as `admin`.** `POST /auth/register` is
public and took `roles` from the request body; `create_user()` validated only
that the role existed. One unauthenticated request bought a token that read
every department — the information barriers, which are the most distinctive
thing in this project, bypassed without touching them. Self-registration is now
capped at `viewer`; anything else needs an admin token
(`tests/unit/test_registration.py` pins it, including `["viewer","admin"]`,
which a membership test would let through).

While fixing it: `create_user()` returned a detached instance, so reading
`user.roles` raised *after* the commit. A successful registration returned 500.
The exploit looked like it had failed, which is the worst way for one to look.

**The published admin login could write to the index.** `POST /documents/ingest`
requires `admin` — correct nearly everywhere and wrong here, because
`admin_user / admin1234!` is printed in the README and sits on the landing page
as a button, so admin is a *public* role by design. A fabricated upload took
the corpus 8,232 → 8,236. The image now sets `ALLOW_RUNTIME_INGEST=false`, so
the deployment serves a fixed corpus and refuses uploads; admin still *reads*
everything, so the unfiltered-admin contrast still demos. Set in the image
rather than at deploy time, so a future deploy that forgets the flag cannot
silently reopen it.

A corpus a visitor can edit cannot carry a digest in the README. Those two
claims have to be defended together.

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

**Since 2026-08-09 this ships.** The deployment serves the local index, so the
mojibake is now public, and it is not buried: the top-ranked chunk of the
role-switch demo opens with

    ACME FINANCIAL HOLDINGS â€… TRADING DESK PROCEDURES AND CONTROLS

Every sample document carries it — `annual_report_10k.txt` and
`credit_risk_policy.txt` too — so any answer citing one shows it in the
retrieved-chunks panel. It is cosmetic and it is on the first screen a reviewer
sees.

Fixing it is a genuine trilemma rather than a chore, which is why it is still
here: re-ingesting the samples changes the digest, and `c2f8c13673cf5ca5` is
now printed on the landing page and in the README. So the fix costs a re-ingest
*plus* a fresh eval run (~$0.80) *plus* updating every published figure — or it
means dropping the digest claim, which is the thing that makes the deployment
checkable. Cheapest honest version: re-ingest, re-run, republish, all in one
sitting. Do not re-ingest without the eval.

---

## 3. Repo state

- `main` is current and pushed. Phases 1–5 are all on it; `origin/main` is at
  the same commit.
- **383 tests pass, ruff clean.**
- `chroma_dist/` is generated and gitignored. `make docker-up` builds it if
  missing; delete the directory to regenerate. A clone with no `chroma_data/`
  of its own has to build one first (`make demo`) — that is the cost of the
  image and the deployment sharing the published corpus.
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

**Now a two-run mean** against the 8,232-chunk corpus (`c2f8c13673cf5ca5`):
`eval_20260809_015503` and `eval_20260809_202437`, identical config, so their
spread is the current noise floor rather than a result.

| | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Citation Coverage | Refusal Correctness |
|---|---|---|---|---|---|---|---|---|
| run 1 (`015503`) | 0.698 | 0.682 | 0.378 | 0.335 | — | **0.939** | 0.661 | **0.852** |
| run 2 (`202437`) | 0.693 | 0.678 | 0.382 | 0.347 | 0.464 | 0.921 | 0.642 | **0.852** |
| **mean** | **0.696** | **0.680** | **0.380** | **0.341** | 0.464 | **0.930** | **0.652** | **0.852** |
| **noise floor** | *0.005* | *0.004* | *0.004* | *0.012* | *—* | *0.017* | *0.019* | *0.000* |

`answer_correctness` exists only in run 2 — it was added the same day, so it
has a value but no floor, and no delta on it can be called yet. Everywhere it
is absent it means *not measured*, never 0.000.

**This floor is much tighter than the old n=20 one** (faithfulness 0.005 here
against 0.138 there), which is what n=54 buys. It is still two runs: treat it
as "deltas below this are definitely noise", never as "deltas above this are
definitely real".

| Stratum | n | Refusal Correctness |
|---|---|---|
| `interpretive` | 14 | 0.929 |
| `exact_figure` | 13 | 0.846 |
| `comparative` | 8 | 0.625 |
| `ambiguous` | 6 | 0.667 |
| `no_answer` | 5 | **1.000** |
| `out_of_corpus` | 4 | **1.000** |
| `rbac_blocked` | 4 | **1.000** |

The two runs disagree most on citation coverage (0.019) and accuracy (0.017),
and not at all on refusal correctness — which stays the most stable metric in
the set, now across four runs.

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

**The shipped chunking strategy lost its own comparison.** All three strategies
were measured on 2026-08-09 against the same 54-question suite, generator and
judge (`eval_20260809_202437` recursive, `_203012` fixed, `_203636` semantic).
`fixed` — the structure-blind baseline that `recursive` exists to beat — beats
it on faithfulness by 19× the noise floor, context precision by 22× and recall
by 8×. `semantic` beats it on faithfulness too. `recursive` wins citation
accuracy alone.

Nothing was changed on the strength of it. Switching the default invalidates
the shipped index, its digest, the deployment and every published figure, and
these are single runs for `fixed` and `semantic` against a floor transferred
from `recursive`'s pair. **Get a second run per strategy before acting**, and
note that strategy and corpus are inseparable by construction — `fixed` is
6,585 chunks against 8,232, so what won is the pipeline under that strategy,
not chunking in isolation.

The `fixed` index lives in the `rag_enterprise_fixed` collection inside
`chroma_data/`, alongside `rag_enterprise_semantic`. `build_index_dist.py`
keeps only `rag_enterprise`, so none of this reaches the image.

**The rate limit does not exist.** `rate_limit = "20/minute"` is configured,
constructed and never enforced — see §2.1. Do not cite it as a control.

~~The API Dockerfile does `COPY . .` before the build-time ingest~~ — fixed.
The ingest is gone (the index ships instead) and the COPYs are explicit, so
editing `dashboard/` no longer invalidates anything in the API image. Cloud
Build is now ~3 minutes for the API image and ~1.5 for the dashboard.

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
| Single eval run, 54 questions | ~$0.80 (three run 2026-08-09 came in near this each) |
| A second run per chunking strategy (§6) | ~$1.60 for the pair that matters |
| Building an index for one strategy | ~$0.05 in embeddings — the cheap half |
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
  it failed to reproduce.

  **Correction (2026-08-09).** This entry used to add that RRF's ordinality
  "silently disables the system's ability to say 'I don't know.'" That is true
  only with reranking *also* off, which is not a configuration anyone runs.
  The pipeline is `dense (+BM25 → RRF) → cross-encoder rerank → top_k`: the
  hybrid stage does emit `score=None` with `ScoreType.RRF`
  (`retriever.py:218`), but the reranker re-wraps every document with a real
  cross-encoder score immediately after (`reranker.py:64`), so the channel is
  restored before confidence or the gate ever reads it. Verified against the
  deployment — `retrieval_mode=hybrid` returns `score_type: cross_encoder` and
  a retrieval confidence of 0.952, not null.

  The structural cost is real but conditional, and it was stated
  unconditionally. `rerank_enabled=False` plus hybrid is the combination that
  actually loses the score channel.

---

## 9. What Phase 6 asks for

Project 6 §6 is the portfolio phase: record a demo walkthrough, and write the
case study. The case study is done (`CASE_STUDY.md`).

1. ~~Decide §2.1~~ — done, see §2.1.

2. **The walkthrough recording was dropped** by the project owner on
   2026-08-09. This is the one spec item Phase 6 does not deliver; the live
   demo is what a reviewer gets instead. Nothing public promises a video —
   neither `README.md` nor `CASE_STUDY.md` ever mentioned one — so there is no
   dangling claim to clean up.

   **The path below stays, because it is verified behaviour rather than a
   shooting script.** These three are the demonstration whether it is watched
   live, walked through in an interview, or recorded after all, and the
   questions were checked against the deployment on 2026-08-09 — the ones the
   previous handoff named no longer behave as it described. See the note after
   the list.
   - **The role switch.** Ask the ACME trading-desk question as
     `research_analyst`: the Research-Trading Wall is shown in force, `trading`
     is listed among the departments it removed, and the answer says the
     documents do not detail ACME's procedures — sourced only from filings the
     research analyst can see. Switch to `trader_desk`; the same question
     answers at **0.885** off `trading_desk_procedures.txt` at 0.999. The wall
     is the only thing that changed. Still the strongest 40 seconds here.
   - **A refusal that is not an error.** Use **"What was Amazon's total revenue
     in fiscal 2025?"** — `model_refused`, label low, while **retrieval scores
     0.861**. That combination is the shot: retrieval is confident, and the
     model declines anyway because Amazon is not in the corpus. Follow it with
     **"What is Netflix's subscriber count?"** — `low_retrieval_confidence`,
     retrieval 0.000, the gate firing *before* the model is called. Two
     refusals with different causes, which is precisely the distinction the
     panel draws (§5.1) and the reason it draws it.
   - **Hybrid vs. dense side by side**, which the spec asks for by name. On the
     Apple revenue question the two columns differ by one rank position and
     both leave a Goldman Sachs chunk wrongly at rank 1 — the reranker defect,
     on screen, next to a comparison whose aggregate answer is "inside the
     noise floor."

   **Why the old questions were replaced.** Both refusal shots the previous
   handoff named now return *partial answers*, not structured refusals. Shipping
   the measured index made retrieval good enough that the
   insufficient-context gate stops firing on them, so the question reaches the
   model — and Gemini declines *in prose while citing sources*, which
   `is_full_refusal()` correctly classifies as a partial answer at completeness
   0.5, not a refusal (§8). The iPhone question now scores retrieval 0.716 and
   label `medium`; under gpt-4o locally it still gives a clean `model_refused`.
   Nothing is broken — a documented rule met a better retriever — but the
   dashboard renders it as an answer, so describing it as a refusal, in any
   medium, would be describing something the screen is not showing.

3. ~~Optional: the missing `fixed` chunking row, a second run on the current
   corpus, answer correctness~~ — all three done 2026-08-09 (§4, §6). What is
   left in this category: **a second run per chunking strategy**, so the
   comparison that just contradicted the shipped default rests on more than one
   sample each (~$1.60), and the six-config ablation re-run (~$3).

The case study deliberately does not claim hybrid beats dense-only, because
the measurement says otherwise. If someone asks you to reframe it that way,
that is the one thing in this repo worth pushing back on: the negative result
and the retracted claims are what make the rest of the numbers credible.
