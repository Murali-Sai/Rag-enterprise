# Handoff → RAG Enterprise, complete state

Written 2026-08-10, end of session. **Self-contained**: it assumes no memory of
any previous session and no reading of the earlier handoffs. Where an older
document is still the better source it is named, but nothing here depends on
having read it.

Supersedes `HANDOFF.md` (2026-08-08), `handoffTo6.md` and `handoffToGitStuff.md`
as the current description of the system. Those are kept because their commit
bodies and reasoning are still the record of *why* things are the way they are;
this one is the state.

---

## 1. What this project is

A retrieval-augmented generation system over five companies' **real SEC 10-K
filings** — Apple, Microsoft, JPMorgan, Goldman Sachs, Tesla — downloaded from
the EDGAR API and parsed into Items 1, 1A, 7, 7A and 8. On top of retrieval sits
a compliance layer modelled on investment-bank practice: role-based access
enforced as a vector-store `where` clause, information barriers (Chinese Walls),
MNPI and investment-advice guardrails, and a SEC 17a-4 append-only audit trail.

It is an implementation of **"Project 6: RAG Pipeline with Hybrid Search Over
Internal Docs"** (`Project 6.docx`, kept in the repo root and deliberately *not*
in version control — a fresh clone will not have it; read it with a small
`zipfile` + regex script, pandoc is not installed here). All 38 spec
requirements are met except the demo video, which was dropped deliberately and
which nothing public promises.

**What makes it a case study rather than a demo** is that almost every claim in
`CASE_STUDY.md` is a number, and several are negative results. The retracted
claims are load-bearing: they are why the remaining numbers are believable. Do
not "clean them up".

### Stack

Python 3.11+, FastAPI, LangChain 0.3.x (pinned `<1.0`), ChromaDB, `rank_bm25`,
`sentence-transformers` (cross-encoder reranker), RAGAS for evaluation,
Streamlit for the dashboard, Docker, Terraform (AWS reference architecture
only — the live deployment is Cloud Run), GitHub Actions.

---

## 2. State in one paragraph

`main` is at `9b8dc19`, clean, pushed, **CI green**. 412 tests pass with no
`OPENAI_API_KEY` set. Two Cloud Run services are live. Seven commits landed
today: the CI suite was made to actually run, the rate limit was enforced for
the first time, `main` was protected, the chunking claim was re-measured and cut
by a factor of seven, and two retrieval defects were diagnosed and fixed. **The
live deployment is one revision behind the repo** — see §9.1, it is the first
thing to deal with.

---

## 3. Where everything lives

### Repo

`https://github.com/Murali-Sai/Rag-enterprise.git`, branch `main`.

**Two rulesets protect it** (added today):

| Ruleset | Rules | Bypass |
|---|---|---|
| `main: no force-push, no deletion` | `deletion`, `non_fast_forward` | **none** — owner included |
| `main: PR with green CI` | `pull_request` (0 approvals), `required_status_checks`: `lint`, `test (unit)`, `test (integration)` | repository admin, always |

So `git push origin main` still works for the owner; GitHub prints a note that
3 required checks were expected and lets it through. Force-push and branch
deletion are blocked for everyone. Be honest about what the second ruleset is
worth: with a self-bypass it makes PRs the path of least resistance, not a wall.

### Live services

Project `rag-enterprise-498519`, region `us-central1`.

| Service | URL |
|---|---|
| `rag-enterprise` (API + landing page) | https://rag-enterprise-laa65asupq-uc.a.run.app |
| `rag-enterprise-dashboard` (Streamlit) | https://rag-enterprise-dashboard-laa65asupq-uc.a.run.app |

Both also answer on the longer `…-1072425852803.us-central1.run.app` form; the
short form is canonical because it is what has been shared.

They find each other through environment variables — the dashboard has
`RAG_API_URL`, the API has `DASHBOARD_URL`. **The dependency is circular**, so
the working order is: deploy the API, read its URL, deploy the dashboard with
it, then update the API with the dashboard's URL.

Deployed revision **`rag-enterprise-00008-fzj`** (deployed today). Its config:

```
cpu 2, memory 2Gi, containerPort 8080
autoscaling.knative.dev/maxScale = 1        <- load-bearing, see §7.3
run.googleapis.com/startup-cpu-boost = true
env: LLM_PROVIDER=gemini, ENVIRONMENT=production, DASHBOARD_URL=<dashboard>
secrets: GOOGLE_API_KEY, JWT_SECRET_KEY, OPENAI_API_KEY (Secret Manager, :latest)
```

`ALLOW_RUNTIME_INGEST=false` is set **in the Dockerfile**, not as a deploy flag,
so a future deploy that forgets it cannot re-enable writes.

Deploy with:

```bash
gcloud run deploy rag-enterprise --source . --region us-central1
```

No env flags needed — `gcloud run deploy` preserves existing env vars, secrets
and scaling. `.gcloudignore` deliberately does **not** exclude `chroma_dist/`,
which is why the upload is ~106 MB rather than ~1 MB; that is the index going
into the image.

The boot log names the corpus it is serving, which is the fastest way to
confirm a deployment is what it claims:

```
Corpus: rag_enterprise — 8232 chunks, embedded with openai/text-embedding-3-small
```

Read it with:

```bash
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.revision_name="<rev>" AND textPayload:"Corpus"' --limit 5 --format="value(textPayload)" --freshness=60m
```

### Local state that is not in git — **expensive, do not delete**

`chroma_data/` (~470 MB) holds four collections:

| Collection | Chunks | Digest | What |
|---|---|---|---|
| `rag_enterprise` | 8,232 | `c2f8c13673cf5ca5` | **The measured corpus.** Everything published describes this |
| `rag_enterprise_fixed` | 6,585 | `0740fec6381cbacc` | Fixed-size chunking comparison |
| `rag_enterprise_semantic` | 8,936 | `8068f567f5f09588` | Semantic chunking comparison |
| `dedup_smoke_test` | 0 | — | Junk, safe to delete |

`chroma_dist/` (106 MB) is the pruned single-collection copy the image ships.
Regenerate by deleting it and running `make index-dist`; the script refuses to
finish unless the result is 8,232 chunks at that digest.

**Rebuilding `rag_enterprise` costs an ingest plus re-embedding, and the digest
would change — invalidating every published figure.** If it is ever lost, say so
in the docs rather than quietly re-ingesting.

`evaluation/results/` holds 25+ run files and **is** in git, deliberately: the
corpus fingerprint in each is what makes historical claims checkable.

---

## 4. The measured baseline — every number

### 4.1 Shipped configuration

Dense retrieval over `text-embedding-3-small`, cross-encoder rerank
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, 20 candidates → top 5), per-entity
retrieval for multi-company questions, `recursive` chunking, `gpt-4o` for
generation, `gpt-4o-mini` as judge (`eval_judge_max_tokens=4096`), hybrid search
and HyDE **off**.

### 4.2 The three-run baseline (pre-fix pipeline)

Mean of `eval_20260809_015503`, `eval_20260809_202437`, `eval_20260810_214534` —
identical config, same 8,232-chunk corpus at digest `c2f8c13673cf5ca5`, n=54.

| | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Citation Coverage | Refusal Correctness |
|---|---|---|---|---|---|---|---|---|
| **mean (n=3)** | 0.6969 | 0.6806 | 0.3915 | 0.3376 | 0.4625 | **0.9282** | 0.642 | 0.8457 |
| *spread* | *0.007* | *0.004* | *0.035* | *0.016* | *0.003* | *0.017* | *0.040* | *0.019* |

### 4.3 Noise floors — read these before any other number

**Per-metric, widest run-to-run spread observed across the three chunking
strategies.** This is the conservative floor and the one to quote.

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|
| 0.034 | 0.004 | 0.035 | 0.044 | 0.026 | 0.017 | 0.019 |

The floors differ by metric across an order of magnitude. **Context precision
moves 0.035 between runs that differ in nothing at all**, so a reported
precision gain smaller than that is not a gain.

### 4.4 Per stratum (from the shipped baseline)

| Stratum | n | Expected | Faithfulness | Answer Relevancy | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|
| `interpretive` | 14 | answer | 0.827 | 0.793 | 0.929 | 0.929 |
| `exact_figure` | 13 | answer | 0.696 | 0.581 | 0.955 | 0.846 |
| `comparative` | 8 | answer | 0.426 | 0.576 | 0.891 | 0.625 |
| `ambiguous` | 6 | 3 answer, 3 refuse | 0.833 | 0.883 | 0.900 | 0.667 |
| `no_answer` | 5 | refuse | — | — | — | **1.000** |
| `out_of_corpus` | 4 | refuse | — | — | — | **1.000** |
| `rbac_blocked` | 4 | refuse | — | — | — | **1.000** |

**All three unanswerable strata have scored 1.000 in every run ever recorded**,
including today's. The refusal path is the strongest thing in the system.
Comparatives are the weakest.

### 4.5 Per company (context recall)

| | AAPL | MSFT | JPM | GS | TSLA |
|---|---|---|---|---|---|
| baseline | 0.472 | 0.371 | 0.229 | **0.125** | 0.485 |
| after today's fixes | 0.635 | 0.350 | 0.229 | **0.229** | 0.487 |

Those GS figures include comparative questions. **GS-only** (questions whose
`source_filing` is exactly `GS 10-K`): **0.0000 across 12 runs → 0.1250 today.**

### 4.6 The chunking comparison — re-measured today

Two runs per strategy (three for `recursive`), same suite, generator and judge.

| Strategy | Index | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|---|---|
| `recursive` *(shipped, n=3)* | 8,232 | 0.697 | 0.681 | 0.391 | 0.338 | 0.462 | **0.928** | 0.846 |
| `fixed` *(n=2)* | 6,585 | **0.790** | **0.690** | **0.474** | **0.468** | **0.488** | 0.883 | **0.861** |
| `semantic` *(n=2)* | 8,936 | 0.781 | 0.646 | 0.430 | 0.360 | 0.477 | 0.915 | 0.833 |

**The headline correction of the day.** The previous version of this table
claimed `fixed` beat `recursive` by 19×, 22× and 8× the noise floor. The deltas
held almost exactly. The *floors* were wrong — measured on one pair of
`recursive` runs and transferred to the other strategies:

| | claimed 2026-08-09 | measured 2026-08-10 |
|---|---|---|
| Faithfulness | +0.095 vs 0.005 = **19×** | +0.093 vs 0.034 = **2.8×** |
| Context precision | +0.088 vs 0.004 = **22×** | +0.082 vs 0.035 = **2.3×** |
| Context recall | +0.099 vs 0.012 = **8×** | +0.130 vs 0.044 = **3.0×** |

Two deltas that table counted are now noise outright: refusal correctness
(+0.015 vs 0.019) and answer correctness (+0.026 vs 0.026). One new negative
appeared that single runs hid: `semantic` **loses** answer relevancy by 0.034
against a 0.004 floor — **8.6×**, the largest single effect in the table,
pointing opposite to its faithfulness win.

**`recursive` stays the default.** The trade is 0.045 of citation accuracy — the
strongest number in the project — for 0.08–0.13 elsewhere, and at 2–3× the noise
that does not justify invalidating the shipped index, its digest, the deployment
and every published figure. That is a judgment call, recorded as one.

---

## 5. The evaluation apparatus

**54 hand-written, hand-classified questions** in
`evaluation/datasets/eval_questions_v4.json`. Not LLM-generated: 54 is small
enough to read, and a classifier would put another model's judgment inside a
measurement chain built to keep it out. **16 of the 54 correctly have no
answer.**

Three design rules that must not be quietly broken:

1. **Ground truths never come from retrieval.** An earlier version generated
   each reference from this project's own top-20 chunks, then scored that same
   retriever against the result — so anything retrieval systematically missed
   was missing from the reference too and could never count as a miss.
   Regenerating from the complete parsed filing dropped context recall 0.57 →
   ~0.33. The system did not get worse; the measurement stopped grading itself.
2. **Unanswerable questions are never pooled into the aggregates.** RAGAS scores
   a correct refusal as 0.000 answer relevancy — correctly — so pooling them
   would lower every aggregate *because the system behaved correctly*. They are
   scored on refusal correctness instead, and the rule is written into
   `config.no_answer_scoring` of every result file.
3. **The three refusal strata stay separate.** They fail at different stages —
   `out_of_corpus` at the retrieval gate, `no_answer` only at the model,
   `rbac_blocked` at the where-clause — so they point at different fixes.

**Citation accuracy and refusal correctness are not RAGAS metrics.** Citation
accuracy hands a judge one claim and *only* the chunk it cited — showing the
whole context would turn it into a faithfulness check. Refusal correctness is
scored over all 54, not just the 16, so it catches the opposite failure:
declining something the corpus does answer.

### Running an evaluation

There is **no argument parser** — `--help` starts a real run and bills for it.
Configuration comes from the environment:

```bash
CHROMA_COLLECTION=rag_enterprise CHUNKING_STRATEGY=recursive \
  ./.venv/Scripts/python.exe -m evaluation.run_evaluation
```

~7 minutes, **~$0.80 per run**. Results land in
`evaluation/results/eval_<timestamp>.json` with a `config` block and a `corpus`
block (`chunk_count`, `content_digest`) — that fingerprint is what makes a run
comparable to another.

---

## 6. What this session did — seven commits

| Commit | What |
|---|---|
| `00b1ba2` | Integration suite runs in CI; actions to v7; codecov gated on a token; README badge |
| `28aa80c` | The rate limit is enforced for the first time; 10 MB upload cap; 18 tests |
| `d2ed58f` | Records that the limit is per-instance and only exact because `maxScale=1` |
| `a6d4d6b` | Second run per chunking strategy; cuts the claim from 19× to 2.8× |
| `c61e12d` | Two branch-protection rulesets |
| `c692b85` | Entity-scoped rerank queries; over-refusals 6–7 → 4 |
| `9b8dc19` | Single-company questions filtered to that company; GS recall leaves 0.000 |

### 6.1 CI was lying, and how

`.github/workflows/ci.yml` had two jobs, `lint` and `test`, with **`test`
depending on `lint`**. Lint had been failing on formatting — whitespace in three
test files — for at least five commits. While it failed, the test job reported
**skipped, not failed**, so the red badge said "formatting" and the suite was
not running at all. Fixing the formatting let it execute and it immediately
failed on a real defect: a unit test constructed a real embedding client and
needed `OPENAI_API_KEY`.

CI now runs `tests/unit` and `tests/integration` as legs of a matrix, keyless on
purpose. **If lint goes red again, the tests stop running, silently.**

Reproduce the keyless condition before trusting a green local run:

```bash
OPENAI_API_KEY= ./.venv/Scripts/python.exe -m pytest tests/ -q
```

412 pass under it today.

### 6.2 The rate limit existed only on paper

`rate_limit = "20/minute"` sat in settings, a `Limiter` was constructed and
attached to `app.state`, and nothing consulted it — no middleware, no decorated
route. Forty consecutive requests against the deployment returned zero 429s.

It was never the "three-line change" the old handoff called it: slowapi requires
parameters named exactly `request` and `response`, and `/query`, `/auth/token`
and `/auth/register` had all named their Pydantic body `request`. Bodies moved
to `payload`; FastAPI treats a lone Pydantic parameter as the whole body
regardless of name, so nothing changed on the wire.

Now enforced: **`/query` 20/min, auth routes 30/min, per IP**, with
`X-RateLimit-*` and `Retry-After`. Static files and `/health` are unlimited.

Three deliberate choices, all serving "a recruiter can still exercise the whole
demo":

- **Not a blanket limit.** The landing page pulls three assets plus a favicon
  per load, so `SlowAPIMiddleware` with `default_limits` would have spent a
  visitor's budget on stylesheets and locked them out on their fifth reload.
- **Auth is looser than query** (30 vs 20), which reads backwards until you
  watch someone use it: the page logs in once per role-button click and
  switching roles *is* the demonstration. The demo credentials are published on
  that page anyway, so throttling guesses at those accounts protects nothing.
- **Keyed on the first `X-Forwarded-For` hop.** Behind Cloud Run
  `request.client.host` is Google's front end (`169.254.169.126`) for every
  visitor on earth; keying on it would put the whole internet in one bucket.
  The header is spoofable and that is traded away knowingly.

**The bug worth remembering:** `headers_enabled=True` makes slowapi write
`X-RateLimit-*` into the endpoint's `response`, and it *raises* rather than
degrades when there is no such parameter. A wrong password raises before that
injection point — so the first round of tests, which checked that bad
credentials get 401 and the 31st request gets 429, all passed **while every
successful login returned 500**. Rate-limit suites test refusal by instinct.
Only a live request against a running server found it.

### 6.3 The two retrieval fixes

Both came out of "the reranker silently refuses answerable questions", which was
the biggest live defect. It turned out to be two unrelated bugs.

**(a) Comparatives were scored against the whole comparison.**
`MultiEntityRetriever` correctly gives each company its own budget and metadata
filter, then handed each leg the *full* question to score against. A
cross-encoder scores "does this passage answer this query?", and no single chunk
answers "compare A and B" — so every chunk of Apple's filing was judged a
half-answer: logits −3.5 to −7.0, relevances 0.03 down to 0.0008, against a 0.15
gate. Retrieval had found the right passages; the scoring refused them.

`scope_query()` in `src/retrieval/entities.py` now drops the other companies,
the conjunction they leave behind, and the comparative opener. Mechanical on
purpose — that module's whole argument is that entity handling stays a literal
alias match rather than an LLM call, and query rewriting is exactly where that
temptation returns.

Merged confidence on "Apple and Tesla … single-source suppliers" moved
**0.021 → 0.569**; the JPMorgan/Goldman comparative, which already passed, moved
**0.673 → 0.914**.

**(b) Single-company questions were never filtered to that company.**
`MultiEntityRetriever` applied a ticker filter only at **two or more** entities,
so a question naming one searched all five filings *and* the synthetic sample
documents. Asked for Goldman Sachs' total net revenues and return on average
common equity, the top five were two GS chunks, one Tesla, one JPMorgan, and one
from `annual_report_10k.txt` — a fabricated sample reading *"Total Net Revenue:
$38.2B / Return on Equity (ROE): 14.8%"*. No ground-truth claim about Goldman
Sachs can be supported by those. **That is how GS scored a hard 0.000 for twelve
runs.** Banks hid it best because their tables are structurally identical: the
wrong company's figures are not merely retrievable but convincing.

The entity clause is **ANDed with the RBAC department clause**, never
substituted for it — a ticker filter that replaced the barrier would drop the
control this system exists to demonstrate while every retrieval test still
passed. That is pinned by its own test.

### 6.4 Measured effect of the two fixes

| Metric | baseline (n=3) | + scoping | + entity filter | net | floor |
|---|---|---|---|---|---|
| Faithfulness | 0.6969 | 0.7170 | **0.8003** | +0.103 | 0.034 |
| Answer relevancy | 0.6806 | 0.6780 | 0.6826 | +0.002 | 0.004 |
| Context precision | 0.3915 | 0.4087 | 0.4299 | +0.038 | 0.035 |
| Context recall | 0.3376 | 0.3395 | 0.3833 | +0.046 | 0.044 |
| Answer correctness | 0.4625 | 0.4565 | 0.4625 | +0.000 | 0.026 |
| Citation accuracy | 0.9282 | 0.9308 | **0.9543** | +0.026 | 0.017 |
| Refusal correctness | 0.8457 | **0.8889** | 0.8704 | +0.025 | 0.019 |
| GS-only recall | 0.0000 | 0.0000 | **0.1250** | | |
| Answerable declined | 6, 6, 7 | 4 | 5 | | |

Runs: `eval_20260810_234108` (scoping), `eval_20260811_001050` (+ filter).

**Both post-fix rows are single runs.** The README's three-run table is
deliberately left describing the *pre-fix* pipeline until a fresh set of three
exists — quoting a single run as a mean is the exact mistake that cost $2.40 to
correct earlier the same day. **This is the top piece of measurement debt.**

### 6.5 A mistake made and corrected inside one session — worth reading

While diagnosing (a), I measured (b) — the single-entity filter — saw that it
raised slot purity 3.90/5 → 5.00/5 but **lowered mean retrieval confidence**
0.832 → 0.789, admitted zero questions and newly refused two, and rejected it
with the words "purity was the wrong target".

Confidence was the wrong target. It was high because the cross-encoder was
scoring a **fabricated document about another company** as a perfect answer.
Optimising a proxy chose the reading that kept the proxy healthy. The fix that
proxy rejected is the one that later produced +0.103 faithfulness and moved GS
off zero.

The general form: **when a proxy metric and the thing you actually care about
disagree, the proxy is the suspect.** The eval harness measures the real thing
and costs $0.80; that is cheap next to shipping the wrong conclusion.

---

## 7. Open defects, with all the evidence

### 7.1 **The retrieval-confidence gate is uncalibrated — the top open item**

This is now the highest-value defect in the repo, and it is what is left of
both defects above.

`retrieval_confidence()` in `src/generation/confidence.py` computes
`0.6 * best + 0.4 * mean` over `ScoredDocument.relevance`, which
`src/retrieval/scores.py` produces by pushing the cross-encoder logit through a
logistic. The gate refuses below `insufficient_context_threshold = 0.15`.

**A cross-encoder is trained to rank, not to emit a calibrated probability.** A
logistic over a MS MARCO logit is not P(relevant) for 10-K prose. Evidence, all
from today's runs:

- Healthy questions score 0.8–0.99; failures score 0.003–0.03. There is a wide
  gap, so **no threshold separates them** — anything low enough to admit the
  failures disables the gate. Do not "fix" this by lowering 0.15.
- After the entity filter, *"What was Apple's total net revenue for its most
  recent fiscal year?"* retrieves context scoring **1.000 context recall** — the
  right chunks, provably — and the gate refuses it anyway.
- *"What was Apple's revenue last year?"* likewise: recall 0.667, refused.
- Dense financial tables score low while containing the answer; thematic
  questions ("human capital and employee programs") score low across the board.

**The 5 remaining over-refusals** in `eval_20260811_001050`:

| Reason | Question |
|---|---|
| `low_retrieval_confidence` | What was Apple's total net revenue for its most recent fiscal year? |
| `low_retrieval_confidence` | What was Apple's revenue last year? |
| `low_retrieval_confidence` | What does Microsoft disclose about its human capital and employee programs? |
| `low_retrieval_confidence` | How do Microsoft and Tesla differ in their disclosed R&D priorities? |
| `model_refused` | What are Goldman Sachs' principal business segments? |

Four are the gate; one is the model declining despite good retrieval.

Directions worth measuring (none tried yet): calibrate the logit against a
held-out set rather than assuming a logistic; make the gate rank-relative
(compare best to the candidate distribution) rather than absolute; or gate on
something other than the cross-encoder entirely. **Measure, do not reason —
this session refuted three plausible hypotheses by measuring them.**

### 7.2 Goldman Sachs — improved, not fixed

GS-only context recall is **0.125**, up from a hard 0.000 across twelve runs.
Three of four GS questions still score 0.000. The page-furniture hypothesis was
tested and refuted long ago (214 bare running-header chunks went to 0; recall
did not move and GS precision *fell* 0.271 → 0.050). The confusable-companies
cause is now fixed. What remains is unexplained and needs the same treatment:
dump the retrieved contexts per GS question, compare against the ground truth,
and find out which claims are unsupported and why.

Useful fact from today: **all the GS ground-truth figures are present in the
corpus** — `58,283`, `14.3%`, `727,338`, `130,665`, `52,953`, `360,080`,
`Platform Solutions`, `Value-at-Risk` all appear in the 2,888 indexed GS chunks.
So this is not a parsing gap. GS section coverage:

```
Financial Statements and Supplementary Data  1226
Management Discussion and Analysis            833
Business                                      425
Risk Factors                                  403
Quantitative and Qualitative ... Market Risk     1   <- suspicious
```

That single market-risk chunk may be worth a look.

### 7.3 The mojibake ships — untouched

Sample documents carry mangled em dashes from a Windows-encoded ingest, visible
in the first chunk of the headline demo:
`ACME FINANCIAL HOLDINGS â€… TRADING DESK PROCEDURES`. Fixing it needs a
re-ingest **plus** a fresh eval **plus** republishing every figure, because the
corpus digest is printed on the landing page. `handoffTo6.md` §2.3 has the
detail. This was the third of the three defects and was not started.

### 7.4 Ambiguity is not handled at all

Two of three underspecified questions were answered about one arbitrarily chosen
company with no flag that the question admitted others. There is no
ambiguity-detection mechanism; the `ambiguous` stratum exists to record that,
not to claim it is solved.

### 7.5 Synthetic sample documents pollute filing questions

`annual_report_10k.txt` and friends have `department: sec_filings` but no
`ticker`, and their clean prose outranks real filings on financial questions.
The entity filter incidentally excludes them for single-company questions, but a
question naming no company still reaches them. Worth deciding deliberately
whether fabricated documents belong in the same department as real filings.

---

## 8. Gotchas that will bite

**Use `./.venv/Scripts/python.exe`, never bare `python`.** The system Python is
3.13 with *some* dependencies but not `rank_bm25` or `sentence-transformers`.
The partial overlap is the trap: LangChain-only scripts run fine, then pytest
fails collection on five modules and reads as a broken suite.

**`evaluation.run_evaluation` has no `--help`.** It starts a real, billed run.

**`settings` is a singleton built at first import.** Writing to `os.environ`
afterwards reaches subprocesses and `os.execvp` but *not* the already-built
object. This once produced a boot line reporting `text-embedding-3-small` over
an index built with MiniLM — the exact misreport that line exists to prevent.
`scripts/start.py` reads the env directly; anything doing late overrides needs
the same care.

**The rate limit is per instance.** slowapi's default storage is in-memory and
per process, so the limits are exact *only* because
`autoscaling.knative.dev/maxScale = 1`. Raising maxScale to N multiplies every
limit by N — nothing errors, nothing logs. Fixing that needs shared storage
(`storage_uri=`), not a bigger number.

**Two Docker traps, both of which made a test pass while measuring nothing.**
BuildKit reused the cached COPY layer when `chroma_dist/` was moved aside, so a
"fresh clone" image still carried all 106 MB and reported success — use
`--no-cache` when testing the empty-index path. And the optional index COPY
globs on `chroma_dist*`, which matched `chroma_dist_held`, the directory the
test itself had renamed it to; move it **outside the build context**.
`.dockerignore` now excludes `chroma_dist_*/`.

**`compare_eval_runs.py` calls identical `config` blocks a noise floor.** It
fingerprints *settings*, not source. Change behaviour without changing a setting
— which is exactly what today's two retrieval fixes did — and it will call a
real change noise. Say so in the commit.

**A noise floor measured once is a claim, not a constant.** Every multiple in
the old chunking table was a ratio whose denominator had n=1, and the
denominator was where all the error was.

---

## 9. Open work, ranked

### 9.1 **Redeploy — the live demo is behind the repo** *(first, free)*

Revision `rag-enterprise-00008-fzj` was deployed **before** `c692b85` and
`9b8dc19`. The live demo has the rate limit but **not** the two retrieval fixes,
so it still answers Goldman Sachs questions off Tesla and fabricated documents.
One command, §3.

### 9.2 Re-establish the three-run baseline *(~$2.40)*

Three runs of the shipped config so the README's headline table describes the
pipeline that actually ships. Until then the repo's own top-line numbers are the
pre-fix ones, which is stated in the README but is still debt.

### 9.3 The calibration bug *(§7.1)*

Highest value. Blocks both earlier defects from fully closing.

### 9.4 The mojibake *(§7.3)*

Re-ingest + eval + republish. The most expensive of the three original defects
and the only one not started.

### 9.5 Goldman Sachs' remaining 0.000s *(§7.2)*

### 9.6 Optional

Re-run the six-config retrieval ablation (~$3, only meaningful as a complete
set — it is currently measured on 9,572 chunks at n=20 against the retired 0.138
floor, so it describes a corpus that no longer exists). Set a **GCP billing
budget alert at $10** — suggested repeatedly, never done, free, and the one
safety net that watches every meter at once.

---

## 10. Decisions already made — do not relitigate

- **The demo video is dropped.** Nothing public promised one.
- **The deployment generates with Gemini, not `gpt-4o`.** A public
  unauthenticated URL running gpt-4o is ~$0.012/query. Retrieval is the measured
  pipeline; generation is not. The README and landing page say exactly that —
  do not upgrade it to "the demo is the measured system".
- **The image ships the measured index, and the index COPY stays optional.**
  Shipping it briefly broke `make docker-up` on a fresh clone, which is the one
  thing Project 6 §5.3 asks the container to do. Both properties must hold.
- **`/documents/ingest` is disabled in the image.** The demo's admin password is
  published on the landing page, so admin is a public role and must not write to
  a fingerprinted corpus.
- **Self-registration is capped at `viewer`.**
- **RRF weights default to 1.0/1.0**, not the 0.7/0.3 the spec suggests, because
  the hybrid comparison was measured at equal weighting.
- **The case study does not claim hybrid beats dense-only**, because the
  measurement says otherwise. If someone asks for that reframing, it is the one
  thing in this repo worth pushing back on.
- **`recursive` stays the chunking default** (§4.6).
- **The negative results stay.** The retracted 19× claim, the refuted Goldman
  hypothesis, the recall that fell when the measurement stopped grading itself —
  these are why the rest of the numbers are credible.

---

## 11. Cost and budget

Spend is on the owner's OpenAI key. **The stated ceiling is under $10/month to
keep the project running**, and no spend after deployment from public traffic.

**Today's session: about $4.15**, approved in advance with "do what you think is
best, but don't exceed $10":

| Item | Cost |
|---|---|
| 3 chunking confirmation runs | ~$2.40 |
| Scoped-rerank validation run | ~$0.80 |
| Entity-filter validation run | ~$0.80 |
| Accidental partial run (`--help` is not a flag) | ~$0.05 |
| Local + live test queries, embeddings, Cloud Build | ~$0.10 |

Steady-state hosting, measured: **under $1/month.** Artifact Registry ~$0.40,
Secret Manager ~$0.24, Cloud Run $0 idle because both services scale to zero.
The OpenAI key on the service is for *embeddings only*, ~$0.000005/query.

**What would break the ceiling**, in order of likelihood: setting
`min-instances` above 0 (~$50/month, the classic Cloud Run bill shock);
accumulating deploy images (~$1/month per dozen); sustained abuse (~$4/day —
now much harder, see §6.2).

**The owner delegates the judgment and holds the ceiling.** Surface the cost and
what it buys, make the call, report actual spend afterwards.

---

## 12. Verifying it all still works

```bash
OPENAI_API_KEY= ./.venv/Scripts/python.exe -m pytest tests/ -q
./.venv/Scripts/python.exe -m ruff check src tests dashboard evaluation scripts
./.venv/Scripts/python.exe -m ruff format --check src tests
```

412 tests, ruff clean, format clean. The last is the one CI trips on, and when
it trips the test job reports *skipped*.

```bash
gh run list --limit 1          # CI
make docker-up                 # both services in containers — the reviewer path
make dev                       # API only, :8000
make dashboard                 # Streamlit, :8501 — runs from dashboard/
```

`make dashboard` runs from inside `dashboard/` because Streamlit resolves
`.streamlit/config.toml` against the working directory. `.claude/launch.json`
has an `api` entry for the browser preview tooling.

Live smoke test (both services scale to zero, so the first request is slow):

```bash
URL=https://rag-enterprise-laa65asupq-uc.a.run.app
curl -s "$URL/health"
curl -s -X POST "$URL/auth/token" -H 'Content-Type: application/json' \
  -d '{"username":"research_analyst","password":"research1!"}'
# then, with the access_token:
curl -s -X POST "$URL/query" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was Apple total net revenue in fiscal year 2024?"}'
```

Demo users: `research_analyst`/`research1!`, `trader_desk`/`trade1234!`,
`admin_user`/`admin1234!`, `compliance_officer`/`compl1234!`.

To confirm the rate limit is real rather than configured:

```bash
for i in $(seq 1 35); do
  curl -s -o /dev/null -w "%{http_code} " -X POST "$URL/auth/token" \
    -H 'Content-Type: application/json' -d '{"username":"nobody","password":"wrong"}'
done
```

Expect 30 × 401 then 429. If you get 35 × 401, the limiter is inert again and
§6.2 is the section to read.
