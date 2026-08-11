# Handoff → RAG Enterprise, complete state

Written 2026-08-11, end of session. **Self-contained**: it assumes no memory of
any previous session and no reading of the earlier handoffs.

Supersedes `handoffTo7.md` (2026-08-11, morning) as the current description of
the system. That document and its predecessors are kept because their commit
bodies and reasoning remain the record of *why* things are the way they are —
but **`handoffTo7.md` §7.1 is now actively wrong** and is corrected in §6.1
below. Do not follow its guidance on the confidence gate.

---

## 1. What this project is

A retrieval-augmented generation system over five companies' **real SEC 10-K
filings** — Apple, Microsoft, JPMorgan, Goldman Sachs, Tesla — downloaded from
the EDGAR API and parsed into Items 1, 1A, 7, 7A and 8. On top of retrieval sits
a compliance layer modelled on investment-bank practice: role-based access
enforced as a vector-store `where` clause, information barriers (Chinese Walls),
MNPI and investment-advice guardrails, and a SEC 17a-4 append-only audit trail.

It implements **"Project 6: RAG Pipeline with Hybrid Search Over Internal Docs"**
(`Project 6.docx`, in the repo root and deliberately *not* in version control —
a fresh clone will not have it; read it with a `zipfile` + regex script, pandoc
is not installed here). Compliance against that spec is audited in §8.

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

`main` is clean, pushed, **CI green with all three legs actually running**. 420
tests pass with no `OPENAI_API_KEY` set. Two Cloud Run services are live —
revision `rag-enterprise-00009-hgr`, deployed and smoke-tested this session.

Five commits landed. The retrieval confidence gate was measured and
recalibrated; the three-run baseline was re-established on the pipeline that
actually ships (**faithfulness 0.697 → 0.853, refusal correctness 0.846 →
0.944**); a free retrieval eval was added that reproduces the paid per-company
ranking for $0.00004; and the `/v1` endpoint names Project 6 §5.1 asks for were
mounted over the existing handlers.

**The deployed revision predates the last two commits.** It has the gate fix and
the baseline, not the free eval script (which never runs in the image) or the
`/v1` routes. Redeploy when convenient; nothing user-facing regressed, the new
paths are simply absent live.

The top three items on the previous handoff's ranked list are closed, as is
§8.1 gap 1.

---

## 3. Where everything lives

### Repo

`https://github.com/Murali-Sai/Rag-enterprise.git`, branch `main`.

**Two rulesets protect it**:

| Ruleset | Rules | Bypass |
|---|---|---|
| `main: no force-push, no deletion` | `deletion`, `non_fast_forward` | **none** — owner included |
| `main: PR with green CI` | `pull_request` (0 approvals), `required_status_checks`: `lint`, `test (unit)`, `test (integration)` | repository admin, always |

So `git push origin main` still works for the owner; GitHub prints a
"Bypassed rule violations" note and lets it through. Force-push and branch
deletion are blocked for everyone. Be honest about what the second ruleset is
worth: with a self-bypass it makes PRs the path of least resistance, not a wall.

### Live services

Project `rag-enterprise-498519`, region `us-central1`.

| Service | URL |
|---|---|
| `rag-enterprise` (API + landing page) | https://rag-enterprise-laa65asupq-uc.a.run.app |
| `rag-enterprise-dashboard` (Streamlit) | https://rag-enterprise-dashboard-laa65asupq-uc.a.run.app |

They find each other through environment variables — the dashboard has
`RAG_API_URL`, the API has `DASHBOARD_URL`. **The dependency is circular**, so
the working order is: deploy the API, read its URL, deploy the dashboard with
it, then update the API with the dashboard's URL.

Deployed revision **`rag-enterprise-00009-hgr`**. Its config:

```
cpu 2, memory 2Gi, containerPort 8080
autoscaling.knative.dev/maxScale = 1        <- load-bearing, see §9
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
into the image. The deploy takes 8–12 minutes, most of it the upload.

The boot log names the corpus it is serving, which is the fastest way to
confirm a deployment is what it claims:

```
Corpus: rag_enterprise — 8232 chunks, embedded with openai/text-embedding-3-small
```

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

`evaluation/results/` holds 28 run files and **is** in git, deliberately: the
corpus fingerprint in each is what makes historical claims checkable.

---

## 4. The measured baseline — every number

### 4.1 Shipped configuration

Dense retrieval over `text-embedding-3-small`, cross-encoder rerank
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, 20 candidates → top 5), per-entity
retrieval for multi-company questions, entity-filtered single-company
questions, insufficient-context gate at **0.001**, `recursive` chunking,
`gpt-4o` for generation, `gpt-4o-mini` as judge (`eval_judge_max_tokens=4096`),
hybrid search and HyDE **off**.

### 4.2 The three-run baseline

Mean of `eval_20260811_195440`, `_20260811_200218`, `_20260811_200901` —
identical config, 8,232-chunk corpus at digest `c2f8c13673cf5ca5`, n=54.

| | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Citation Coverage | Refusal Correctness |
|---|---|---|---|---|---|---|---|---|
| **mean (n=3)** | **0.8529** | 0.7628 | 0.4289 | 0.3860 | 0.5255 | **0.9359** | 0.7287 | **0.9444** |
| *spread* | *0.0629* | *0.0126* | *0.0076* | *0.0211* | *0.0073* | *0.0184* | *0.0223* | *0.0000* |
| *previous (n=3)* | *0.6969* | *0.6806* | *0.3915* | *0.3376* | *0.4625* | *0.9282* | *0.6417* | *0.8457* |

Regenerate this table at any time with:

```bash
./.venv/Scripts/python.exe -m scripts.summarize_runs evaluation/results/eval_20260811_195440.json evaluation/results/eval_20260811_200218.json evaluation/results/eval_20260811_200901.json
```

**Which deltas are claims and which are not**, against the conservative floor
(the wider of the two baselines' spreads):

| Metric | Δ | Floor | Verdict |
|---|---|---|---|
| Faithfulness | +0.156 | 0.063 | 2.5× — claim |
| Answer relevancy | +0.082 | 0.013 | 6.5× — claim |
| Refusal correctness | +0.099 | 0.019 | 5.2× — claim |
| Citation coverage | +0.087 | 0.040 | 2.2× — claim |
| Answer correctness | +0.063 | 0.026 | 2.4× — claim |
| Context precision | +0.037 | 0.035 | 1.1× — **not a claim** |
| Context recall | +0.048 | 0.044 | 1.1× — **not a claim** |
| Citation accuracy | +0.008 | 0.018 | noise — unchanged, and that is the point |

**The baseline covers three fixes together** — entity-scoped rerank queries,
the single-company entity filter, and the gate — and cannot apportion the gain
among them. Do not attribute all of it to the gate.

### 4.3 Noise floors — read these before any other number

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|
| **0.063** | 0.013 | 0.035 | 0.044 | 0.026 | 0.018 | 0.019 |

**The faithfulness floor tripled this session and this is the most important
correction on the page.** The previous baseline's spread was 0.0072 and the
conservative cross-strategy figure was 0.034; three runs of the current pipeline
spread **0.0629**. Every faithfulness claim in this repo has been running on a
denominator that was too small. The floors above take the widest observed value
per metric and are the ones to quote.

Context precision moved the other way — 0.035 → 0.0076 — but the table keeps
0.035 because it is the wider observation and this project quotes conservative
floors.

### 4.4 Per stratum

| Stratum | n | Expected | Faithfulness | Answer Relevancy | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|
| `interpretive` | 14 | answer | 0.942 | 0.838 | 0.966 | 0.929 |
| `exact_figure` | 13 | answer | 0.840 | 0.634 | 0.921 | **1.000** |
| `comparative` | 8 | answer | 0.710 | 0.796 | 0.887 | **1.000** |
| `ambiguous` | 6 | 3 answer, 3 refuse | 0.870 | 0.883 | 0.900 | 0.667 |
| `no_answer` | 5 | refuse | — | — | — | **1.000** |
| `out_of_corpus` | 4 | refuse | — | — | — | **1.000** |
| `rbac_blocked` | 4 | refuse | — | — | — | **1.000** |

**All three unanswerable strata have scored 1.000 in every run ever recorded.**
Comparatives were the weakest stratum in every previous baseline (0.426
faithfulness, 0.625 refusal correctness) and are no longer.

**The three refusal errors are identical in all three runs** — which is why the
refusal-correctness spread is exactly 0.000:

| Error | Question | Cause |
|---|---|---|
| declined, should have answered | What are Goldman Sachs' principal business segments? | `model_refused` — the model, not the gate |
| answered, should have declined | How much did the bank set aside for credit losses? | §7.3 ambiguity, unhandled |
| answered, should have declined | What are the company's main risks? | §7.3 ambiguity, unhandled |

**Zero `low_retrieval_confidence` refusals in 162 question-runs.**

### 4.5 Per company

| | AAPL | MSFT | JPM | GS | TSLA |
|---|---|---|---|---|---|
| Faithfulness | 0.827 | 0.874 | 0.788 | 0.847 | 0.934 |
| Context Recall | 0.676 | 0.320 | 0.198 | **0.125** | 0.471 |
| Context Precision | 0.737 | 0.643 | 0.207 | **0.000** | 0.225 |
| Refusal Correctness | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 |

### 4.6 The chunking comparison — **pre-fix, do not mix**

| Strategy | Index | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|---|---|
| `recursive` *(shipped, n=3)* | 8,232 | 0.697 | 0.681 | 0.391 | 0.338 | 0.462 | **0.928** | 0.846 |
| `fixed` *(n=2)* | 6,585 | **0.790** | **0.690** | **0.474** | **0.468** | **0.488** | 0.883 | **0.861** |
| `semantic` *(n=2)* | 8,936 | 0.781 | 0.646 | 0.430 | 0.360 | 0.477 | 0.915 | 0.833 |

**This table is measured on the pre-fix pipeline and the `recursive` row is
deliberately the old baseline, not the 0.853 headline.** `fixed` and `semantic`
have never been run against the entity fixes or the recalibrated gate.
Substituting the new numbers would compare three chunking strategies across two
different pipelines and manufacture a chunking result out of a retrieval change.
Re-running the other two costs ~$3.20. Until then this comparison is valid only
against itself.

`recursive` stays the default. The trade is 0.045 of citation accuracy — the
strongest number in the project — for 0.08–0.13 elsewhere, at 2–3× noise, and
that does not justify invalidating the shipped index, its digest, the deployment
and every published figure. A judgment call, recorded as one.

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
   `out_of_corpus` **at the model** (it was the gate until the gate was measured
   and found not to be doing that job), `no_answer` at the model,
   `rbac_blocked` at the where-clause.

**Citation accuracy and refusal correctness are not RAGAS metrics.** Citation
accuracy hands a judge one claim and *only* the chunk it cited — showing the
whole context would turn it into a faithfulness check. Refusal correctness is
scored over all 54, not just the 16, so it catches the opposite failure:
declining something the corpus does answer.

### 5.1 New this session: a held-out probe set

`evaluation/datasets/gate_calibration_v1.json` — 49 items, **held out from the
eval suite on purpose**. Out-of-corpus items are labelled by construction (the
company is not one of the five). In-corpus items carry a literal `evidence`
string that `scripts/calibrate_gate.py` must find in that ticker's chunks before
the item is allowed to influence anything; items whose evidence is missing are
dropped and reported. That check is lexical and independent of ranking, so it
does not repeat design rule 1's mistake. It has already earned its keep: it
caught that Apple writes "single-source" hyphenated.

The suite is the test set. Tuning a threshold on it and then reporting against
it is the same self-grading that cost 0.24 of apparent recall to undo.

### 5.2 Running things

There is **no argument parser** on the eval — `--help` starts a real run and
bills for it. Configuration comes from the environment:

```bash
CHROMA_COLLECTION=rag_enterprise CHUNKING_STRATEGY=recursive \
  ./.venv/Scripts/python.exe -m evaluation.run_evaluation
```

~7–10 minutes, **~$0.80 per run**. Results land in
`evaluation/results/eval_<timestamp>.json` with a `config` block and a `corpus`
block (`chunk_count`, `content_digest`) — that fingerprint is what makes a run
comparable to another.

Three analysis tools:

| Script | Does | Cost |
|---|---|---|
| `scripts/summarize_runs.py` | Mean + spread + per-stratum over N runs. **Use this for any baseline claim** | free |
| `scripts/compare_eval_runs.py` | Diffs exactly two runs, per metric and per question | free |
| `scripts/calibrate_gate.py` | Gate score distribution over the held-out probe set | ~$0.0003 |
| `scripts/eval_retrieval_free.py` | Answer-figure hit rate and entity purity per company, no LLM | ~$0.00004 |
| `scripts/probe_refusal.py` | Model's unaided refusal rate with the gate held open | ~$0.30 |

`summarize_runs.py` exists because the mean/spread tables were assembled by
hand, which is how a single run got published as a mean twice. It was validated
by reproducing the previous baseline's published figures exactly before being
used on new data, and it refuses to average across different corpora.

---

## 6. What this session did — two commits

| Commit | What |
|---|---|
| `1cae01e` | The gate measured and recalibrated; held-out probe set; two measurement scripts |
| `792e80e` | Three-run baseline on the shipped pipeline; README + CASE_STUDY; `summarize_runs.py` |

### 6.1 The confidence gate — and why `handoffTo7.md` §7.1 was wrong

**Read this before touching `insufficient_context_threshold`.**

`handoffTo7.md` §7.1 named the uncalibrated gate the top open defect and said,
in bold, *"Do not 'fix' this by lowering 0.15."* Its reasoning: the four
`out_of_corpus` questions score 0.0003–0.0029 while healthy questions score
0.8–0.99, so anything low enough to admit the over-refusals disables the gate.

That reasoning rested on four data points and they were luck. The four
out-of-corpus questions happened to occupy the four lowest scores in the suite,
which looks like clean separation at n=4.

**Measurement 1 — `scripts/calibrate_gate.py` over 49 held-out probes.** The
labels overlap across nearly the whole range:

| Score | Question | In corpus? |
|---|---|---|
| 0.9750 | Citigroup's standardized CET1 capital ratio | no |
| 0.8994 | Bank of America net interest income | no |
| 0.7342 | Salesforce remaining performance obligation | no |
| 0.2962 | the current price of Bitcoin | no |
| 0.0031 | what Microsoft says about its dividend | **yes** |

Ten of 24 out-of-corpus probes already cleared 0.15. The corpus genuinely
discusses CET1 ratios, net interest income and remaining performance
obligations — for other companies. **A cross-encoder scores topic match and
carries no representation of which company a passage is about.** This is the
same blindness that had Goldman questions answered off Tesla's filing, showing
up on the gate side instead of the retrieval side.

Rank-relative gating was also tested and is **worse**: candidate-set gap and
z-score overlap completely between the two groups.

**Measurement 2 — `scripts/probe_refusal.py` with the gate held open.** The
model refused **24 of 24** out-of-corpus questions unaided, including the one
scoring 0.975. The gate had never been what made those refusals correct. The
suite could not reveal this because all four of its out-of-corpus questions fall
below the gate and none had ever reached the model.

**So the gate stopped being a correctness mechanism and became a cost guard** —
decline to pay for a generation certain to be a refusal — at **0.001**, below
the lowest answerable probe (0.0031) with margin. It still short-circuits 12 of
24 out-of-corpus probes without an LLM call. The binding constraint is that it
must never be why an answerable question is declined; a test pins that.

**`CASE_STUDY.md` predicted this fix would fail**, saying it "tunes the gate to
the test set and relocates the failure downstream into ungrounded answers". The
held-out set answers the first half. The second is refuted outright: faithfulness
rose 0.156, the largest effect measured in this project. Admitting those
questions did not produce ungrounded answers, because the context had been good
enough to ground them all along — which is what a refused question retrieving
1.000 context recall had been saying.

### 6.2 The baseline

Three runs, §4.2. Retires the previous handoff's "top piece of measurement
debt": until now the README's headline table described the pre-fix pipeline
while the repo shipped a different one.

### 6.3 Two findings that cost nothing

**Goldman's single market-risk chunk is correct, not a parser bug.** GS indexes
1 chunk under `Quantitative and Qualitative Disclosures About Market Risk`
against 4–8 for the non-banks, which `handoffTo7.md` §7.2 flagged as
"suspicious". Both banks incorporate Item 7A by reference — GS points to Item 7,
JPMorgan to pages 133–142 of its Annual Report — and JPMorgan shows the same
1-chunk stub for Items 7 and 8 beside a 2,705-chunk incorporated Annual Report.
One chunk is the correct parse of a cross-reference. **Hypothesis eliminated.**

**The mojibake is 6 characters, not an ingest-wide problem.** `handoffTo7.md`
§7.3 described mangled em dashes from "a Windows-encoded ingest" and estimated
the fix at a re-ingest plus a fresh eval plus republishing every figure. Measured
across the whole index: the five real filings contain 7,269 non-ASCII characters
and **every one is legitimate** (U+2019, U+2022, U+2014, U+201C/D, U+00AE). The
corruption is exactly 6 mangled em dashes (`â` + `€` + `"`, the UTF-8 `E2 80 94`
misdecoded as cp1252) confined to the **17 synthetic sample chunks**. That
collapses §7.3 into §7.4: if the fabricated documents leave the corpus, the
mojibake leaves with them.

---

## 7. Open defects

### 7.1 Goldman Sachs — the best-isolated open defect

GS-only context recall is **0.125**; three of four GS questions score 0.000, and
GS context precision is **0.000**. The retrieved chunks *are* Goldman's — the
entity filter guarantees it — but they are not the chunks the ground truth is
built from. So this is now a **ranking failure inside one company's filing**,
which is a much narrower problem than it was.

Three hypotheses are dead: page furniture (214 running-header chunks went to 0;
recall did not move and GS precision *fell* 0.271 → 0.050), confusable companies
(fixed, §6.1 of `handoffTo7.md`), and the market-risk section (§6.3 above).

All GS ground-truth figures are present in the corpus — `58,283`, `14.3%`,
`727,338`, `130,665`, `52,953`, `360,080`, `Platform Solutions`,
`Value-at-Risk` all appear among the 2,888 indexed GS chunks. GS section
coverage: Financial Statements 1,226 / MD&A 833 / Business 425 / Risk Factors
403 / Market Risk 1.

**Diagnosed 2026-08-11, retrieval only, free.** Locating each ground-truth
figure in the corpus and then asking where the pipeline loses it splits the
failure into three stages — not in the candidate set, in the candidate set but
dropped by the reranker, or in the final top-5. The two failing GS questions
fail at *different* stages:

| Question | Figures | Where lost |
|---|---|---|
| total net revenues and ROE | `15.0%`, `58,283` | **in candidates at ranks 3 and 6, dropped by the reranker** |
| quantitative market-risk disclosures | 7 figures | **never retrieved at all** — not in the 20-candidate set |

The second is the interesting one. Its top-ranked chunk scores **0.994** and is
GS's entire Item 7A: a 272-character cross-reference reading *"…are set forth
in Management's Discussion and Analysis … in Part II, Item 7."* A perfect
*title* match containing no answer, crowding out the section that holds one.

**The fixed candidate budget is a real contributing cause, confirmed and then
found insufficient.** `rerank_candidate_k` is 20 regardless of filing size —
4.3% of Apple's 460 chunks but 0.7% of Goldman's 2,888. Sweeping it with
`scripts/eval_retrieval_free.py`:

| | k=20 | k=50 | k=100 |
|---|---|---|---|
| GS | 0.067 | 0.117 | **0.167** |
| JPM | 0.071 | 0.107 | **0.143** |
| AAPL | 0.500 | 0.500 | 0.500 |
| MSFT | 0.210 | 0.210 | 0.210 |
| TSLA | 0.306 | 0.250 | 0.250 |
| overall | 0.242 | 0.239 | **0.248** |

Exactly the predicted shape: the two starved filings gain 150% and 101% while
the two small ones do not move at all, because 20 candidates already covered
their whole relevant region. **Do not ship this as a fix.** TSLA *loses* 0.056,
and overall nets flat — a larger pool is also more opportunity for a weak ranker
to promote the wrong chunk, so recall gained at the candidate stage is given
back at the rerank stage.

And at k=100, six of the seven market-risk figures *still* never enter the
candidate set, while the cross-reference stub still ranks first. So the binding
constraint is not the budget: **neither the bi-encoder nor the cross-encoder
represents financial tables well.** The question is thematic and the answer is a
table of numbers, and a table of numbers does not embed like the sentence asking
about it. That is the same root cause as the reranker's uniformly negative
logits on financial-table prose (§7.2), reached from the opposite direction.

Worth knowing: **174 chunks corpus-wide are short cross-references**, and 90% of
them are in the two banks (GS 109, JPM 47, against AAPL 3). Removing them is a
re-ingest and would only free slots, not surface the tables, so it is worth
doing *with* §10.1 rather than for its own sake.

**Next steps, in order of expected value:** a financial-domain embedding model
or reranker; table-aware chunking that prepends section context to numeric
tables so a table embeds nearer the question that asks for it; and only then
a size-proportional candidate budget, which is worth revisiting once the ranker
can exploit a bigger pool.

### 7.2 The reranker's precision cost is still real

Cross-encoder reranking moved answer relevancy +0.239 against a 0.044 floor and
context precision **−0.230**. `ms-marco-MiniLM` was trained on short web
passages and assigns uniformly negative logits to financial-table prose. The
*refusal* half of this defect is closed (§6.1); the precision half is not. The
untried fix is a financial-domain cross-encoder.

### 7.3 Ambiguity is not handled at all

Two of three underspecified questions are answered about one arbitrarily chosen
company with no flag that the question admitted others. This is the **only
answerable stratum that has not moved** across every fix — 0.667 in every
baseline — and it is the source of 2 of the 3 remaining refusal errors. There is
no ambiguity-detection mechanism; the `ambiguous` stratum exists to record that.

### 7.4 Synthetic sample documents pollute filing questions

`annual_report_10k.txt` and friends have `department: sec_filings` but no
`ticker`, and their clean prose outranks real filings on financial questions.
The entity filter incidentally excludes them for single-company questions, but a
question naming no company still reaches them. **Decide deliberately whether
fabricated documents belong in the same department as real filings** — and note
this also disposes of the mojibake (§6.3).

---

## 8. Project 6 spec compliance — audited 2026-08-11

Audited against `Project 6.docx` by reading the code, not by trusting the
previous handoff, which claimed "all 38 requirements met except the demo video".
That was optimistic by four items.

### 8.1 Genuinely unmet

| # | Requirement | State | Cost |
|---|---|---|---|
| 1 | **§5.1** — `POST /v1/ask`, `GET /v1/documents`, `POST /v1/ingest` | **Closed 2026-08-11.** `src/api/routes/v1.py` mounts all three as aliases over the existing handlers; `/query`, `/documents` and `/documents/ingest` are unchanged because the dashboard, landing page and every published example use them | done |
| 2 | **§1.1** — metadata carrying *"source file, section heading, page number"* | `source_file` and `section_name` present; **no `page` key on any chunk** (verified across 4,000). `parser.py` computes page numbers only to strip running headers, then discards them | Re-ingest → new digest → invalidates every published figure. Or document that page numbers are stripped as furniture by design for text-based EDGAR filings |
| 3 | **§5.3** — compose with *"the API service, ChromaDB, and the frontend"* | Two services, `api` and `dashboard`; Chroma runs embedded inside the API container | ~1 hr, or defend the embedded choice in writing |
| 4 | **§6.1** — demo video under 4 minutes | Deliberately dropped; nothing public promises one | Owner's call |

### 8.2 Deliberate deviations where measurement contradicts the spec

- **§6.2** asks the case study to *"Explain why hybrid beats dense-only."* It
  does not, here — every hybrid delta lands inside its noise floor, and the case
  study says "not established as helpful" instead. **This is the one thing in
  the repo worth pushing back on if someone asks for the reframing.**
- **§2.3** suggests 0.7/0.3 RRF weights; the default is 1.0/1.0 because that is
  how the comparison was measured. The spec only *requires* configurability,
  which is met.
- **§2.1** says "start with k=10"; the pipeline runs candidate_k=20 → rerank →
  top 5, which §2.4 asks for directly.
- **Generation model**: the deployment uses Gemini rather than GPT-4o/Claude
  Sonnet, for cost (§11). Evaluation uses `gpt-4o`.

### 8.3 Verified present — do not re-audit

All four loader formats (`.pdf`, `.txt`, `.md`, `.html`); 0.95 dedup; three
switchable chunking strategies with `chunking_strategy` / `chunk_index` /
`char_count` per chunk; BM25 kept in sync via `reset_bm25_cache()`; RRF with
configurable weights; cross-encoder 20 → 5; bracketed citations; LLM-judge
citation verification; three-signal composite confidence; structured "I don't
know"; 54 hand-written questions (>50) covering lookups, **8 multi-hop
comparatives**, no-answer and ambiguous; answer correctness / faithfulness /
retrieval relevance / citation accuracy; chunking comparison report; OpenAPI
docs; dashboard with **dense-vs-hybrid side-by-side compare**, ranked chunks and
per-dimension confidence; raw filings retained in `data/edgar` for re-index
without re-download; seed scripts.

---

## 9. Gotchas that will bite

**Use `./.venv/Scripts/python.exe`, never bare `python`.** The system Python is
3.13 with *some* dependencies but not `rank_bm25` or `sentence-transformers`.
The partial overlap is the trap: LangChain-only scripts run fine, then pytest
fails collection on five modules and reads as a broken suite.

**`evaluation.run_evaluation` has no `--help`.** It starts a real, billed run.

**Do not pipe a long-running eval through `tail`.** It buffers the whole stream,
so you get no progress for ten minutes and cannot tell a slow run from a hung
one. Redirect to a file instead.

**`settings` is a singleton built at first import.** Writing to `os.environ`
afterwards reaches subprocesses and `os.execvp` but *not* the already-built
object. This once produced a boot line reporting `text-embedding-3-small` over
an index built with MiniLM. `scripts/start.py` reads the env directly.

**Do not run two evals concurrently against `chroma_data/`.** It is
SQLite-backed and irreplaceable — rebuilding it changes the digest and
invalidates every published figure. Runs take ~8 minutes; chain them.

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

**`compare_eval_runs.py` calls identical `config` blocks a noise floor.** It
fingerprints *settings*, not source. Change behaviour without changing a
setting — which is what the two retrieval fixes did — and it will call a real
change noise. The gate change does move a setting, so it is visible.

**A noise floor measured once is a claim, not a constant.** Every multiple in
the retracted chunking table was a ratio whose denominator had n=1. The
faithfulness floor tripled this session for the same reason.

**Windows consoles are cp1252.** Printing U+2014 or U+2011 from a script raises
`UnicodeEncodeError`, and legitimate Unicode in the corpus renders as `�` in
terminal output — which looks exactly like data corruption and is not. Check
codepoints before believing an encoding bug.

---

## 10. Open work, ranked

### 10.1 Decide the fabricated-documents question *(§7.4)*

Removing them fixes §7.4 and the mojibake at once, but changes the corpus digest
and invalidates published figures — so it is the same expensive class as a
re-ingest, and worth doing *with* any other re-ingest rather than alone.

### 10.2 Goldman's remaining 0.000s *(§7.1)*

Free to investigate. Best-isolated defect in the project, three hypotheses down.

### 10.3 Ambiguity detection *(§7.3)*

The only answerable stratum that has never moved, and 2 of 3 remaining refusal
errors.

### 10.4 Page-number metadata *(§8.1 #2)*

Only worth doing bundled with another re-ingest.

### 10.5 Optional

Re-run `fixed` and `semantic` against the current pipeline (~$3.20) so the
chunking table describes one system rather than two. Re-run the six-config
retrieval ablation (~$3, only meaningful as a complete set). Set a **GCP billing
budget alert at $10** — suggested across four handoffs, still never done, free,
and the one safety net that watches every meter at once:

```bash
gcloud billing budgets create --billing-account=$(gcloud beta billing projects describe rag-enterprise-498519 --format="value(billingAccountName)" | cut -d/ -f2) --display-name="rag-enterprise $10 ceiling" --budget-amount=10USD --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
```

---

## 11. Decisions already made — do not relitigate

- **The demo video is dropped.** Nothing public promised one.
- **The deployment generates with Gemini, not `gpt-4o`.** A public
  unauthenticated URL running gpt-4o is ~$0.012/query. Retrieval is the measured
  pipeline; generation is not. The README and landing page say exactly that.
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
- **The gate is a cost guard, not a correctness mechanism** (§6.1). Raising it
  back toward 0.15 buys nothing and costs answerable questions.
- **The negative results stay.** The retracted 19× claim, the refuted Goldman
  hypotheses, the recall that fell when the measurement stopped grading itself,
  and now the gate guidance this session had to overturn — these are why the
  rest of the numbers are credible.

---

## 12. Cost and budget

Spend is on the owner's OpenAI key. **The stated ceiling is under $10/month to
keep the project running**, and no spend after deployment from public traffic.

**This session: about $2.75**, against $3.20 approved in advance:

| Item | Cost |
|---|---|
| 3 baseline runs | ~$2.40 |
| Refusal probe (24 generations, gate open) | ~$0.30 |
| Gate calibration probe (retrieval only, 49 questions) | ~$0.0003 |
| Live smoke tests, Cloud Build | ~$0.05 |

Steady-state hosting, measured: **under $1/month.** Artifact Registry ~$0.40,
Secret Manager ~$0.24, Cloud Run $0 idle because both services scale to zero.
The OpenAI key on the service is for *embeddings only*, ~$0.000005/query.

**What would break the ceiling**, in order of likelihood: setting
`min-instances` above 0 (~$50/month, the classic Cloud Run bill shock);
accumulating deploy images (~$1/month per dozen); sustained abuse (~$4/day —
now much harder, the rate limit is enforced).

**The owner delegates the judgment and holds the ceiling.** Surface the cost and
what it buys, make the call, report actual spend afterwards.

---

## 13. Verifying it all still works

```bash
OPENAI_API_KEY= ./.venv/Scripts/python.exe -m pytest tests/ -q
./.venv/Scripts/python.exe -m ruff check src tests dashboard evaluation scripts
./.venv/Scripts/python.exe -m ruff format --check src tests
```

420 tests, ruff clean, format clean. The last is the one CI trips on, and **when
it trips the test job reports *skipped*, not failed** — a red badge saying
"formatting" can hide a suite that never ran. Check the job list, not the badge:

```bash
gh run view <run-id> --json jobs --jq '.jobs[] | "\(.name): \(.conclusion)"'
```

```bash
gh run list --limit 1          # CI
make docker-up                 # both services in containers — the reviewer path
make dev                       # API only, :8000
make dashboard                 # Streamlit, :8501 — runs from dashboard/
```

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

**The two queries worth running after any retrieval change**, because each one
regressed silently once:

| Question | Must see |
|---|---|
| *What was Apple's total net revenue for its most recent fiscal year?* | An answer of $416,161 million, not a refusal. All five sources AAPL |
| *What was Goldman Sachs' total net revenues and return on average common equity?* | All five sources **GS** — not Tesla, JPMorgan, or `annual_report_10k.txt` |

Demo users: `research_analyst`/`research1!`, `trader_desk`/`trade1234!`,
`admin_user`/`admin1234!`, `compliance_officer`/`compl1234!`.

To confirm the rate limit is real rather than configured:

```bash
for i in $(seq 1 35); do
  curl -s -o /dev/null -w "%{http_code} " -X POST "$URL/auth/token" \
    -H 'Content-Type: application/json' -d '{"username":"nobody","password":"wrong"}'
done
```

Expect 30 × 401 then 429. If you get 35 × 401, the limiter is inert again.
