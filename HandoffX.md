# Handoff → RAG Enterprise, complete state

Written 2026-08-11, end of session. **Self-contained**: assumes no memory of any
previous session and no reading of the earlier handoffs.

---

## Addendum, 2026-08-12 — read this first, it supersedes parts of the page below

Where this addendum and the body disagree, the addendum wins.

**Live revisions are now API `rag-enterprise-00012-2lq`, dashboard
`rag-enterprise-dashboard-00006-4zs`** (supersedes §3). `main` is past
`a6dd19f`, clean, pushed, CI-checked on push.

**§10.1 keep-warm: done.** Cloud Scheduler job `rag-enterprise-keepwarm`
(us-central1, `*/10 * * * *`, GET `/health`) exists, is ENABLED, and its first
scheduled fire succeeded; `/health` answers in ~0.17s warm.
`cloudscheduler.googleapis.com` had to be enabled and now is. The
`HF_HUB_OFFLINE=1` follow-up is also done: the Dockerfile now bakes the
fallback MiniLM embedder *and then* goes offline, so boot no longer depends on
huggingface.co — the embedder bake first, or the fresh-clone ingest path
breaks.

**§7.3 ambiguity: closed.** `src/retrieval/ambiguity.py` refuses
entity-underspecified questions before generation with a third structured
reason, `ambiguous_entity`, from the shared entry point. Two mechanical
signals, both gated on `detect_entities()` finding nothing: a definite
reference ("the company", "the bank"), or a financial-metric question naming
nothing capitalized at all — the second guard keeps out-of-corpus questions
(Netflix, NVIDIA) on their measured, working path through the model. A
suite-level test pins that exactly the three ambiguous-refuse questions fire
and the other fifty-one do not. Confirmed end-to-end by **one paid run**,
`eval_20260812_180308`: ambiguous stratum 0.667 → **1.000**, refusal
correctness 0.944 → **0.981** (53/54; the survivor is the known Goldman
`model_refused`). Answerable aggregates all inside their §4.3 floors. This is
one run, not a new three-run baseline — bought as a wiring check because the
detector is deterministic and test-pinned, so do not quote 0.981 as a mean.

**§7.1/§7.2 reranker: diagnosis validated, fix not shipped.** Free-eval screen
of drop-in cross-encoders, all at k=20, baseline overall 0.242:

| Model | overall Δ | GS Δ | JPM Δ | MSFT Δ | TSLA Δ | ms / 20 pairs |
|---|---|---|---|---|---|---|
| ms-marco-L6 (shipped) | — | — | — | — | — | 54 |
| ms-marco-L12 | −0.018 | 0 | 0 | 0 | −0.083 | ~2× |
| mxbai-rerank-xsmall | +0.033 | 0 | +0.095 | +0.050 | −0.042 | unmeasured |
| **bge-reranker-base** | **+0.134** | +0.067 | +0.095 | +0.300 | −0.042 | **4,531** |

bge-base recovers exactly the chunks §7.1 diagnosed ms-marco as dropping (GS
revenues/ROE 0.33 → 0.67) while the never-retrieved market-risk table stays at
0 — the reranker was one of the two failures, and the candidate-set failure
remains. **Not shipped**: 84× rerank latency (~3.4s → ~8s warm queries on 2
vCPU), ~1.1 GB heavier image, and bge's logit distribution maps to a 0.50
*minimum* confidence, so the 0.001 gate would be inert and needs
recalibration on the probe set before any swap. The forward path is a
quantized/ONNX bge-class reranker, and it is worth a paid run only after the
latency problem is solved. Screen JSONs were scratch; the numbers above are
the record.

**§7.4 decided (2026-08-12): the fabricated documents leave `sec_filings`.**
Not executed — removal changes the corpus digest, so it is bundled with the
next deliberate re-ingest along with §8.1 #2's page metadata. Until then §7.4's
behaviour stands.

**Cost this session: ~$0.85** (one eval run ~$0.80, Cloud Build ~$0.05; the
reranker screen was four free-eval runs at ~$0.00016 total). Keep-warm pings
are inside Cloud Run's request-based free behaviour and Scheduler's free tier.

Supersedes `handoffTo8.md`, written earlier the same day. Its §4 (the measured
baseline) and §5 (the evaluation apparatus) are still accurate and are repeated
here; everything else moved. The older handoffs stay in the repo because their
commit bodies are the record of *why* things are the way they are — but
**`handoffTo7.md` §7.1 is actively wrong** and is corrected in §6.1 below. Do
not follow its guidance on the confidence gate.

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
is not installed here). Spec compliance is audited in §8.

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

`main` is at **`707e0ba`**, clean, pushed, **CI green on all three legs**. 437
tests pass with no `OPENAI_API_KEY` set. Both Cloud Run services are live and
**current with the repo**: API `rag-enterprise-00011-kpv`, dashboard
`rag-enterprise-dashboard-00005-9bg`. A **$10/month billing alert now exists**
(§12). Ten commits landed this session; the headline is **faithfulness 0.697 →
0.853 and refusal correctness 0.846 → 0.944**, re-established as a proper
three-run baseline. Every ranked item from the previous handoff's top three is
closed, plus the only cheap spec gap.

---

## 3. Where everything lives

### Repo

`https://github.com/Murali-Sai/Rag-enterprise.git`, branch `main`.

**Two rulesets protect it**:

| Ruleset | Rules | Bypass |
|---|---|---|
| `main: no force-push, no deletion` | `deletion`, `non_fast_forward` | **none** — owner included |
| `main: PR with green CI` | `pull_request` (0 approvals), `required_status_checks`: `lint`, `test (unit)`, `test (integration)` | repository admin, always |

`git push origin main` still works for the owner; GitHub prints a "Bypassed rule
violations" note and lets it through. Force-push and branch deletion are blocked
for everyone. Be honest about what the second ruleset is worth: with a
self-bypass it makes PRs the path of least resistance, not a wall.

### Live services

Project `rag-enterprise-498519`, region `us-central1`.

| Service | URL | Revision |
|---|---|---|
| `rag-enterprise` (API + landing page) | https://rag-enterprise-laa65asupq-uc.a.run.app | `00011-kpv` |
| `rag-enterprise-dashboard` (Streamlit) | https://rag-enterprise-dashboard-laa65asupq-uc.a.run.app | `00005-9bg` |

They find each other through environment variables — the dashboard has
`RAG_API_URL`, the API has `DASHBOARD_URL`. **The dependency is circular**, so
the working order is: deploy the API, read its URL, deploy the dashboard with
it, then update the API with the dashboard's URL. Both are currently correct.

API config:

```
cpu 2, memory 2Gi, containerPort 8080
autoscaling.knative.dev/maxScale = 1        <- load-bearing, see §9 and §12
run.googleapis.com/startup-cpu-boost = true
env: LLM_PROVIDER=gemini, ENVIRONMENT=production, DASHBOARD_URL=<dashboard>
secrets: GOOGLE_API_KEY, JWT_SECRET_KEY, OPENAI_API_KEY (Secret Manager, :latest)
```

`ALLOW_RUNTIME_INGEST=false` and `WARM_MODELS_ON_STARTUP=true` are set **in the
Dockerfile**, not as deploy flags, so a deploy that forgets them cannot change
either property.

```bash
gcloud run deploy rag-enterprise --source . --region us-central1
gcloud run deploy rag-enterprise-dashboard --source ./dashboard --region us-central1
```

No env flags needed — `gcloud run deploy` preserves existing env vars, secrets
and scaling. `.gcloudignore` deliberately does **not** exclude `chroma_dist/`,
which is why the upload is ~106 MB rather than ~1 MB; that is the index going
into the image. **A deploy takes 12–25 minutes**, most of it upload and build,
and `gcloud builds list` does not show these builds (Cloud Run source deploys
are regional) — so an empty build list is not evidence the deploy failed. Watch
the revision name instead.

The boot log names the corpus, which is the fastest way to confirm a deployment
is what it claims:

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

`evaluation/results/` holds 32 run files and **is** in git, deliberately: the
corpus fingerprint in each is what makes historical claims checkable.

---

## 4. The measured baseline — every number

### 4.1 Shipped configuration

Dense retrieval over `text-embedding-3-small`, cross-encoder rerank
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, 20 candidates → top 5), per-entity
retrieval for multi-company questions, entity-filtered single-company questions,
insufficient-context gate at **0.001**, `recursive` chunking, `gpt-4o` for
generation, `gpt-4o-mini` as judge (`eval_judge_max_tokens=4096`), **hybrid
search and HyDE off**.

### 4.2 The three-run baseline

Mean of `eval_20260811_195440`, `_20260811_200218`, `_20260811_200901` —
identical config, 8,232-chunk corpus at `c2f8c13673cf5ca5`, n=54.

| | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Citation Coverage | Refusal Correctness |
|---|---|---|---|---|---|---|---|---|
| **mean (n=3)** | **0.8529** | 0.7628 | 0.4289 | 0.3860 | 0.5255 | **0.9359** | 0.7287 | **0.9444** |
| *spread* | *0.0629* | *0.0126* | *0.0076* | *0.0211* | *0.0073* | *0.0184* | *0.0223* | *0.0000* |
| *previous (n=3)* | *0.6969* | *0.6806* | *0.3915* | *0.3376* | *0.4625* | *0.9282* | *0.6417* | *0.8457* |

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

**The baseline covers three fixes together** — entity-scoped rerank queries, the
single-company entity filter, and the gate — and cannot apportion the gain among
them. Do not attribute all of it to the gate.

### 4.3 Noise floors — read these before any other number

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|
| **0.063** | 0.013 | 0.035 | 0.044 | 0.026 | 0.018 | 0.019 |

**The faithfulness floor tripled and this is the most important correction on
the page.** The previous baseline spread 0.0072 and the conservative
cross-strategy figure was 0.034; three runs of the current pipeline spread
**0.0629**. Every faithfulness claim in this repo had a denominator that was too
small. The floors above take the widest observed value per metric.

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
Comparatives were the weakest stratum in every previous baseline (0.426 / 0.625)
and are no longer.

**The three refusal errors are identical in all three runs** — which is why the
spread is exactly 0.000:

| Error | Question | Cause |
|---|---|---|
| declined, should have answered | Goldman Sachs' principal business segments | `model_refused` — the model, not the gate |
| answered, should have declined | How much did the bank set aside for credit losses? | §7.3 ambiguity, unhandled |
| answered, should have declined | What are the company's main risks? | §7.3 ambiguity, unhandled |

**Zero `low_retrieval_confidence` refusals in 162 question-runs.**

### 4.5 Per company

| | AAPL | MSFT | JPM | GS | TSLA |
|---|---|---|---|---|---|
| Chunks | 460 | 672 | 3,122 | 2,888 | 984 |
| Faithfulness | 0.827 | 0.874 | 0.788 | 0.847 | 0.934 |
| Context Recall | 0.676 | 0.320 | 0.198 | **0.125** | 0.471 |
| Context Precision | 0.737 | 0.643 | 0.207 | **0.000** | 0.225 |
| Refusal Correctness | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 |

Context recall correlates with filing size at Spearman **ρ = −0.8**. That is
five points and cannot be established at n=5; see §7.1 for what was and was not
proved about it.

### 4.6 The chunking comparison — **pre-fix, do not mix**

| Strategy | Index | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|---|---|
| `recursive` *(shipped, n=3)* | 8,232 | 0.697 | 0.681 | 0.391 | 0.338 | 0.462 | **0.928** | 0.846 |
| `fixed` *(n=2)* | 6,585 | **0.790** | **0.690** | **0.474** | **0.468** | **0.488** | 0.883 | **0.861** |
| `semantic` *(n=2)* | 8,936 | 0.781 | 0.646 | 0.430 | 0.360 | 0.477 | 0.915 | 0.833 |

**Measured on the pre-fix pipeline, and the `recursive` row is deliberately the
old baseline, not the 0.853 headline.** `fixed` and `semantic` have never been
run against the entity fixes or the recalibrated gate. Substituting the new
numbers would compare three chunking strategies across two different pipelines
and manufacture a chunking result out of a retrieval change. Re-running the
other two costs ~$3.20. Until then this table is valid only against itself.

`recursive` stays the default: the trade is 0.045 of citation accuracy — the
strongest number in the project — for 0.08–0.13 elsewhere at 2–3× noise, and
that does not justify invalidating the shipped index, its digest, the deployment
and every published figure.

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
accuracy hands a judge one claim and *only* the chunk it cited. Refusal
correctness is scored over all 54, so it catches the opposite failure: declining
something the corpus does answer. Both are computed in `run_rag_pipeline()`;
refusal correctness needs **no LLM call at all**.

### 5.1 The held-out probe set

`evaluation/datasets/gate_calibration_v1.json` — 49 items, **held out from the
eval suite on purpose**. Out-of-corpus items are labelled by construction (the
company is not one of the five). In-corpus items carry a literal `evidence`
string that `scripts/calibrate_gate.py` must find in that ticker's chunks before
the item is allowed to influence anything; items whose evidence is missing are
dropped and reported. The check is lexical and independent of ranking, so it
does not repeat rule 1's mistake. It has already earned its keep: it caught that
Apple writes "single-source" hyphenated.

The suite is the test set. Tuning a threshold on it and then reporting against
it is the same self-grading that cost 0.24 of apparent recall to undo.

### 5.2 Running things

There is **no argument parser** on the eval — `--help` starts a real run and
bills for it. Configuration comes from the environment:

```bash
CHROMA_COLLECTION=rag_enterprise CHUNKING_STRATEGY=recursive \
  ./.venv/Scripts/python.exe -m evaluation.run_evaluation
```

~8–10 minutes, **~$0.80 per run**.

| Script | Does | Cost |
|---|---|---|
| `scripts/eval_retrieval_free.py` | **Answer-figure hit rate + entity purity per company, no LLM** | ~$0.00004 |
| `scripts/summarize_runs.py` | Mean + spread + per-stratum over N runs. **Use for any baseline claim** | free |
| `scripts/compare_eval_runs.py` | Diffs exactly two runs, per metric and per question | free |
| `scripts/calibrate_gate.py` | Gate score distribution over the held-out probe set | ~$0.0003 |
| `scripts/probe_refusal.py` | The model's unaided refusal rate with the gate held open | ~$0.30 |

**`eval_retrieval_free.py` is the one to reach for first.** It reproduces the
paid per-company ranking exactly — Spearman **ρ = 1.00** against RAGAS context
recall, and **ρ = −0.80** against chunk count, the same value RAGAS gives — for
1/20,000th of the price. Absolute values are *not* comparable to RAGAS (0.242 vs
0.386 overall, because it is a stricter literal test) so never put them in the
same table. It is trustworthy in one direction: a change that **drops** hit rate
has broken something; a change that raises it is a hypothesis worth $0.80 to
confirm.

`summarize_runs.py` exists because the mean/spread tables used to be assembled
by hand, which is how a single run got published as a mean twice. It was
validated by reproducing the previous baseline's published figures exactly
before being used on new data, and it refuses to average across corpora.

---

## 6. What this session did — ten commits

| Commit | What |
|---|---|
| `1cae01e` | Gate measured and recalibrated; held-out probe set; two measurement scripts |
| `792e80e` | Three-run baseline on the shipped pipeline; README + CASE_STUDY |
| `7b38c44` | `handoffTo8.md`; spec audit |
| `63c408c` | The free retrieval eval |
| `74c28b2` | `/v1` routes + shared rate-limit bucket |
| `fc859b4` | One-character ticker fix |
| `7072edb` | Goldman diagnosis |
| `615027e` | Cross-encoder warmup at boot |
| `57d4893` | Dashboard spinner; cost estimate corrected |
| `707e0ba` | Formatting |

### 6.1 The confidence gate — and why `handoffTo7.md` §7.1 was wrong

**Read this before touching `insufficient_context_threshold`.**

`handoffTo7.md` §7.1 named the uncalibrated gate the top open defect and said,
in bold, *"Do not 'fix' this by lowering 0.15."* Its reasoning: the four
`out_of_corpus` questions score 0.0003–0.0029 while healthy questions score
0.8–0.99, so anything low enough to admit the over-refusals disables the gate.

That reasoning rested on four data points and they were luck.

**Measurement 1 — `scripts/calibrate_gate.py` over 49 held-out probes.** The
labels overlap across nearly the whole range:

| Score | Question | In corpus? |
|---|---|---|
| 0.9750 | Citigroup's standardized CET1 capital ratio | no |
| 0.8994 | Bank of America net interest income | no |
| 0.7342 | Salesforce remaining performance obligation | no |
| 0.2962 | the current price of Bitcoin | no |
| 0.0031 | what Microsoft says about its dividend | **yes** |

Ten of 24 out-of-corpus probes already cleared 0.15. **A cross-encoder scores
topic match and carries no representation of which company a passage is about**
— the corpus really does discuss CET1 ratios and net interest income, for other
banks. Rank-relative gating was also tested and is *worse*: candidate-set gap
and z-score overlap completely.

**Measurement 2 — `scripts/probe_refusal.py` with the gate held open.** The
model refused **24 of 24** out-of-corpus questions unaided, including the one
scoring 0.975. The gate had never been what made those refusals correct — the
suite could not reveal this because all four of its out-of-corpus questions fall
below the gate and none had ever reached the model.

**So the gate stopped being a correctness mechanism and became a cost guard** at
**0.001**, below the lowest answerable probe (0.0031) with margin. It still
short-circuits 12 of 24 out-of-corpus probes without an LLM call. A test pins
that it can never be why an answerable question is declined.

`CASE_STUDY.md` predicted this fix would "relocate the failure downstream into
ungrounded answers". Refuted outright: faithfulness rose 0.156, the largest
effect measured in this project. The context had been good enough to ground
those questions all along.

### 6.2 Everything else

- **`/v1` routes** (§8). The interesting part is that **slowapi keys a limit on
  the request path**, so an alias that merely delegates gets its own budget —
  measured, 21 requests alternating between `/query` and `/v1/ask` all returned
  200 against a 20/minute limit. That would have silently doubled what one
  client can spend on the only route that reaches an LLM. Both routes now carry
  `limiter.shared_limit(..., scope=QUERY_LIMIT_SCOPE)` and delegate to an
  undecorated `answer_query`.
- **One-character tickers.** `\bC\b` (Citigroup) matches 18 times in the corpus
  and `\bF\b` (Ford) 12, against zero for every multi-character candidate.
  Tickers below `MIN_TICKER_MATCH_LENGTH = 2` now match by company name only,
  and `unmatchable_tickers()` fails a test if a registry entry is both too short
  and alias-less. **Do this before adding Citigroup or Ford.**
- **Startup warmup.** `WARM_MODELS_ON_STARTUP=true` in the Dockerfile loads the
  cross-encoder in a background thread at boot. Not awaited (that would move the
  seconds into container startup, where Cloud Run holds the triggering request)
  and failures are swallowed (the lazy path still works). Measured effect was
  **smaller than expected** — see §10.1.
- **Dashboard.** It authenticated on first page load inside the sidebar block
  with a 120s timeout and **no spinner**, so a cold API meant up to two minutes
  of bare header. Reproduced in a browser at 25 seconds. Now spinner + a caption
  explaining the API sleeps when idle.

### 6.3 Two findings that cost nothing

**Goldman's single market-risk chunk is correct, not a parser bug.** GS indexes
1 chunk under `Quantitative and Qualitative Disclosures About Market Risk`
against 4–8 for the non-banks. Both banks incorporate Item 7A by reference — GS
points to Item 7, JPMorgan to pages 133–142 of its Annual Report — and JPMorgan
shows the same 1-chunk stub for Items 7 and 8 beside a 2,705-chunk incorporated
Annual Report. **Hypothesis eliminated.**

**The mojibake is 6 characters, not an ingest-wide problem.** `handoffTo7.md`
§7.3 estimated a re-ingest plus a fresh eval plus republishing every figure.
Measured: the five real filings contain 7,269 non-ASCII characters and **every
one is legitimate** (U+2019, U+2022, U+2014, U+201C/D, U+00AE). The corruption
is exactly 6 mangled em dashes (`â` + `€` + `"`, UTF-8 `E2 80 94` misdecoded as
cp1252) confined to the **17 synthetic sample chunks**. If the fabricated
documents leave the corpus, the mojibake leaves with them.

---

## 7. Open defects

### 7.1 Goldman Sachs — diagnosed, and the obvious fix ruled out

GS-only context recall is **0.125**; three of four GS questions score 0.000 and
GS context precision is **0.000**. The retrieved chunks *are* Goldman's — the
entity filter guarantees it — but not the chunks the ground truth is built from.
So this is a **ranking failure inside one company's filing**.

Tracing where each ground-truth figure is lost splits the two failing questions
into different stages:

| Question | Figures | Where lost |
|---|---|---|
| total net revenues and ROE | `15.0%`, `58,283` | in candidates at ranks 3 and 6, **dropped by the reranker** |
| quantitative market-risk disclosures | 7 figures | **never retrieved at all** |

The second question's top-ranked chunk scores **0.994** and is GS's entire Item
7A: 272 characters reading *"…are set forth in Management's Discussion and
Analysis … in Part II, Item 7."* A perfect *title* match with no answer in it.

**The fixed candidate budget was confirmed as a cause and rejected as a fix.**
`rerank_candidate_k` is 20 regardless of filing size — 4.3% of Apple's 460
chunks but 0.7% of Goldman's 2,888. Swept with the free eval:

| | k=20 | k=50 | k=100 |
|---|---|---|---|
| GS | 0.067 | 0.117 | **0.167** |
| JPM | 0.071 | 0.107 | **0.143** |
| AAPL | 0.500 | 0.500 | 0.500 |
| MSFT | 0.210 | 0.210 | 0.210 |
| TSLA | 0.306 | 0.250 | 0.250 |
| overall | 0.242 | 0.239 | 0.248 |

Exactly the predicted shape — the two starved filings gain 150% and 101% while
the two small ones do not move at all. **Do not ship it.** TSLA *loses* 0.056
and overall nets flat: a larger pool is also more opportunity for a weak ranker
to promote the wrong chunk. And at k=100, six of the seven market-risk figures
still never enter the candidate set.

So the binding constraint is neither the budget nor the filtering. **Neither the
bi-encoder nor the cross-encoder represents financial tables well.** The question
is thematic, the answer is a table of numbers, and a table does not embed like
the sentence asking for it. Same root cause as §7.2, from the opposite
direction.

Also measured: **174 chunks corpus-wide are short cross-references**, 90% in the
two banks (GS 109, JPM 47, AAPL 3). Removing them needs a re-ingest and would
only free slots, not surface the tables.

**Next steps by expected value:** a financial-domain embedding model or
reranker; table-aware chunking that prepends section context to numeric tables;
then a size-proportional candidate budget, once the ranker can exploit a bigger
pool.

### 7.2 The reranker's precision cost is still real

Cross-encoder reranking moved answer relevancy +0.239 against a 0.044 floor and
context precision **−0.230**. `ms-marco-MiniLM` was trained on short web
passages. The *refusal* half of this defect is closed (§6.1); the precision half
is not. The untried fix is a financial-domain cross-encoder.

### 7.3 Ambiguity is not handled at all

Two of three underspecified questions are answered about one arbitrarily chosen
company with no flag. This is the **only answerable stratum that has not moved**
across every fix — 0.667 in every baseline — and the source of 2 of the 3
remaining refusal errors.

### 7.4 Synthetic sample documents pollute filing questions

`annual_report_10k.txt` and friends have `department: sec_filings` but no
`ticker`, and their clean prose outranks real filings on financial questions.
The entity filter incidentally excludes them for single-company questions, but a
question naming no company still reaches them. **Decide deliberately whether
fabricated documents belong in the same department as real filings** — this also
disposes of the mojibake (§6.3).

---

## 8. Project 6 spec compliance

Audited 2026-08-11 by reading the code, not by trusting the previous handoff's
claim that all 38 requirements were met except the demo video. That was
optimistic by four items; one is now closed.

### 8.1 Unmet

| # | Requirement | State | Cost |
|---|---|---|---|
| 1 | **§5.1** — `POST /v1/ask`, `GET /v1/documents`, `POST /v1/ingest` | **CLOSED.** `src/api/routes/v1.py` mounts all three as aliases over the existing handlers; the unversioned paths are unchanged because the dashboard, landing page and every published example use them | done |
| 2 | **§1.1** — metadata carrying *"source file, section heading, page number"* | `source_file` and `section_name` present; **no `page` key on any chunk** (verified across 4,000). `parser.py` computes page numbers only to strip running headers, then discards them | Re-ingest → new digest. Or document that page numbers are stripped as furniture by design for text-based EDGAR filings |
| 3 | **§5.3** — compose with *"the API service, ChromaDB, and the frontend"* | Two services; Chroma runs embedded inside the API container | ~1 hr, or defend the embedded choice in writing |
| 4 | **§6.1** — demo video under 4 minutes | Deliberately dropped | Owner's call |

### 8.2 Deliberate deviations where measurement contradicts the spec

- **§6.2** asks the case study to *"Explain why hybrid beats dense-only."* It
  does not, here — every hybrid delta lands inside its noise floor. **This is
  the one thing in the repo worth pushing back on if someone asks for the
  reframing.**
- **§2.3** suggests 0.7/0.3 RRF weights; the default is 1.0/1.0 because that is
  how the comparison was measured. The spec only *requires* configurability.
- **§2.1** says "start with k=10"; the pipeline runs candidate_k=20 → rerank →
  top 5, which §2.4 asks for directly.
- **Generation model**: the deployment uses Gemini rather than GPT-4o for cost.
  Evaluation uses `gpt-4o`.

### 8.3 Verified present — do not re-audit

All four loader formats (`.pdf`, `.txt`, `.md`, `.html`); 0.95 dedup; three
switchable chunking strategies with `chunking_strategy` / `chunk_index` /
`char_count` per chunk; BM25 kept in sync via `reset_bm25_cache()`; RRF with
configurable weights; cross-encoder 20 → 5; bracketed citations; LLM-judge
citation verification; three-signal composite confidence; structured "I don't
know"; 54 hand-written questions covering lookups, **8 multi-hop comparatives**,
no-answer and ambiguous; answer correctness / faithfulness / retrieval relevance
/ citation accuracy; chunking comparison report; OpenAPI docs; dashboard with
**dense-vs-hybrid side-by-side compare**, ranked chunks and per-dimension
confidence; raw filings retained in `data/edgar`; seed scripts.

Note for the free-eval question: **§4.2 mandates an LLM judge for exactly two of
its four metrics** — answer correctness and citation accuracy, both claims about
generated text. *"Retrieval relevance (were the right chunks retrieved?)"* is
left open, and `eval_retrieval_free.py` answers it more literally than RAGAS
context precision does. **RAGAS is this project's choice, not the spec's** — the
docx never names it.

---

## 9. Gotchas that will bite

**Use `./.venv/Scripts/python.exe`, never bare `python`.** The system Python is
3.13 with *some* dependencies but not `rank_bm25` or `sentence-transformers`.
The partial overlap is the trap.

**`evaluation.run_evaluation` has no `--help`.** It starts a real, billed run.

**Do not pipe a long-running eval or deploy through `tail` or `Select-Object`.**
It buffers the whole stream, so you get no progress for ten minutes and cannot
tell a slow run from a hung one. Redirect to a file.

**Do not run two evals concurrently against `chroma_data/`.** It is
SQLite-backed and irreplaceable.

**`settings` is a singleton built at first import.** Writing to `os.environ`
afterwards reaches subprocesses but *not* the already-built object.

**FastAPI includes routers lazily here.** `app.routes` lists 8 entries rather
than the real surface and `api_router.routes` holds opaque `_IncludedRouter`
objects. Read `app.openapi()["paths"]` instead.

**slowapi keys rate limits on the request path**, not the decorated function. A
second route delegating to a limited handler gets its own budget. Use
`shared_limit` with an explicit scope.

**The rate limit is per instance** — slowapi's default storage is in-memory, so
limits are exact *only* because `maxScale = 1`.

**Two Docker traps.** BuildKit reused the cached COPY layer when `chroma_dist/`
was moved aside, so a "fresh clone" image still carried all 106 MB — use
`--no-cache` when testing the empty-index path. And the optional index COPY globs
on `chroma_dist*`, which matched `chroma_dist_held`; move it outside the build
context.

**`compare_eval_runs.py` calls identical `config` blocks a noise floor.** It
fingerprints *settings*, not source.

**A noise floor measured once is a claim, not a constant.** The faithfulness
floor tripled this session.

**Windows consoles are cp1252.** Printing U+2014 from a script raises
`UnicodeEncodeError`, and legitimate Unicode renders as `�` — which looks
exactly like data corruption and is not. Check codepoints before believing an
encoding bug.

**Watch out for decorator placement when adding a function above `lifespan`.**
Inserting one between `@asynccontextmanager` and `lifespan` applies the
decorator to the wrong function; every `TestClient` test then fails with "a
coroutine was expected", which reads like an async bug.

---

## 10. Open work, ranked

### 10.1 Keep an instance warm *(free, highest user-visible value)*

Measured on the live deployment:

| | Cold | Warm |
|---|---|---|
| Landing page | **50.9s** | 0.10s |
| First query | 23.1s → **19.1s** after the warmup | 3.4–4.9s |

**The startup warmup helped less than predicted** — about 4 seconds, not 20. The
model is already baked into the image (`RUN python -c "...get_reranker()"`), so
the remaining delay is HuggingFace cache-validation round trips plus torch
construction, and the measurement hit the API seconds after deploy, which is the
worst case. A real visitor reads the landing page and signs in first, so the
realistic gain is larger — but that has not been measured and should not be
claimed.

The ~50s container boot is untouched by any of this. What hides it is keeping an
instance alive: Cloud Run holds one for ~15 minutes after a request and
request-based billing does not charge for idle CPU, so a ping every 10 minutes
costs effectively nothing.

```bash
gcloud scheduler jobs create http rag-enterprise-keepwarm \
  --location=us-central1 --schedule="*/10 * * * *" \
  --uri="https://rag-enterprise-laa65asupq-uc.a.run.app/health" --http-method=GET
```

**Not `min-instances=1`**, which is ~$50/month against a $10 ceiling.

Second, cheaper follow-up: `HF_HUB_OFFLINE=1` in the Dockerfile would drop the
cache-validation round trips and remove a boot-time dependency on
huggingface.co being reachable — a real availability risk for a demo.

### 10.2 Goldman's remaining 0.000s *(§7.1)*

Free to investigate, three hypotheses down, now narrowed to table retrieval.

### 10.3 Ambiguity detection *(§7.3)*

The only answerable stratum that has never moved; 2 of 3 remaining refusal
errors.

### 10.4 Decide the fabricated-documents question *(§7.4)*

Fixes §7.4 and the mojibake at once, but changes the digest — so bundle it with
any other re-ingest, including §8.1 #2's page metadata.

### 10.5 Adding more companies

Analysed but not started. **The value is measurement power, not a better
system** — no user gets a better answer because the index has 15 companies.

The strongest reason: per-company recall correlates with filing size at ρ = −0.8
(§4.5), which would reframe Goldman from a company-specific mystery into "large
filings retrieve badly". At n=5 that cannot be settled, **and size is confounded
with sector** — the two big filings are both banks. Breaking that confound needs
specific companies, not more of them: **large non-banks** (Amazon, Walmart,
Berkshire) and **small financials** (a regional bank, an insurer). Six chosen
that way beat ten chosen for coverage.

Second reason: the comparative stratum is **structurally saturated**. Five
companies admit only 10 pairs and 8 comparatives already exist.

Costs: embeddings ~$0.06 and EDGAR is free; the digest changes; **the real cost
is ~5 hand-written questions per new company**, and the no-LLM-generated rule
means it cannot be shortcut. Every future eval run also gets ~90% more
expensive. Do it in a separate collection (`rag_enterprise_v2`) so
`rag_enterprise` and its published digest survive, and measure with
`eval_retrieval_free.py` rather than paid runs.

**Before ingesting Citigroup or Ford, note §6.2's ticker fix is already in
place** — but check `unmatchable_tickers()` returns empty after editing the
registry.

### 10.6 Optional

Re-run `fixed` and `semantic` against the current pipeline (~$3.20) so the
chunking table describes one system rather than two. Re-run the six-config
retrieval ablation (~$3, only meaningful as a complete set).

---

## 11. Decisions already made — do not relitigate

- **The demo video is dropped.** Nothing public promised one.
- **The deployment generates with Gemini, not `gpt-4o`.** Retrieval is the
  measured pipeline; generation is not. The README and landing page say exactly
  that.
- **The image ships the measured index, and the index COPY stays optional.**
  Both properties must hold — Project 6 §5.3 asks the container to work on a
  fresh clone.
- **`/documents/ingest` is disabled in the image.** The admin password is
  published, so admin is a public role and must not write to a fingerprinted
  corpus.
- **Self-registration is capped at `viewer`.**
- **RRF weights default to 1.0/1.0**, because the hybrid comparison was measured
  at equal weighting.
- **The case study does not claim hybrid beats dense-only**, because the
  measurement says otherwise.
- **`recursive` stays the chunking default** (§4.6).
- **The gate is a cost guard, not a correctness mechanism** (§6.1). Raising it
  back toward 0.15 buys nothing and costs answerable questions.
- **`/v1` routes are aliases, not renames.** The unversioned paths stay.
- **The negative results stay.** The retracted 19× claim, the four refuted
  Goldman hypotheses, the recall that fell when the measurement stopped grading
  itself, and the gate guidance this session had to overturn — these are why the
  rest of the numbers are credible.

---

## 12. Cost and budget

**The stated ceiling is under $10/month.**

**This session: about $2.75**, against $3.20 approved in advance: 3 baseline
runs (~$2.40), the refusal probe (~$0.30), the gate calibration probe
(~$0.0003), live smoke tests and Cloud Build (~$0.05). Everything after the
baseline was free.

Steady-state hosting, measured: **under $1/month.** Artifact Registry ~$0.40,
Secret Manager ~$0.24, Cloud Run $0 idle because both services scale to zero.

### 12.1 What public traffic actually costs

Earlier handoffs put sustained abuse at "~$4/day". **That is about an order of
magnitude low.**

Per query on the live configuration: ~$0.0017 Gemini generation (~3,000 input,
~300 output tokens), ~$0.00018 Cloud Run (2 vCPU and 2 GiB for the measured
3.4s), ~$0.000005 OpenAI embedding. **≈ $0.002 per query.**

| Scenario | Queries | Cost |
|---|---|---|
| A recruiter tries it properly | 10 | $0.02 |
| Shared in a team Slack | 250 | $0.50 |
| A LinkedIn post that lands | ~600 | $1.20 |
| Front page of Hacker News for a day | ~6,000 | ~$12 |
| **Saturated 24/7** | ~25,000/day | **~$50/day** |

**The rate limit is not what caps this.** It keys on the first
`X-Forwarded-For` hop, which is spoofable and was knowingly traded away, and the
demo credentials are published, so a token is free to mint.
**`autoscaling.knative.dev/maxScale = 1` is the actual cost cap** — one instance
at ~3.4s per query serves about 25,000 a day and nothing above that is
reachable. It is doing more work than the rate limit is, which is a third reason
not to raise it.

*The Gemini token pricing above is from memory and is the one figure here worth
re-checking; the rest is measured or from published GCP rates.*

### 12.2 The budget alert exists

Created 2026-08-11 after five consecutive handoffs recommended it: budget
`3b36cbd7-514f-463c-a833-596367fc5c62` on billing account
`01B00D-542937-FBEC97`, **$10/month**, scoped to this project, alerting at 50%,
90% and 100% of current spend. `billingbudgets.googleapis.com` had to be enabled
and now is.

**It notifies; it does not cap.** Billing data lags hours, so it converts an
unbounded silent risk into "you find out the same day". If it fires, take
traffic off the service rather than waiting.

---

## 13. Verifying it all still works

```bash
OPENAI_API_KEY= ./.venv/Scripts/python.exe -m pytest tests/ -q
./.venv/Scripts/python.exe -m ruff check src tests dashboard evaluation scripts
./.venv/Scripts/python.exe -m ruff format --check src tests
```

437 tests, ruff clean. The format check is the one CI trips on, and **when it
trips the test job reports *skipped*, not failed** — a red badge saying
"formatting" can hide a suite that never ran. Check the job list, not the badge:

```bash
gh run view <run-id> --json jobs --jq '.jobs[] | "\(.name): \(.conclusion)"'
```

```bash
make docker-up                 # both services in containers — the reviewer path
make dev                       # API only, :8000
make dashboard                 # Streamlit, :8501 — runs from dashboard/
```

Live smoke test (**the first request after an idle period takes ~50s**):

```bash
URL=https://rag-enterprise-laa65asupq-uc.a.run.app
curl -s "$URL/health"
curl -s -X POST "$URL/auth/token" -H 'Content-Type: application/json' \
  -d '{"username":"research_analyst","password":"research1!"}'
curl -s -X POST "$URL/v1/ask" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was Apple total net revenue in fiscal year 2025?"}'
```

**The two queries worth running after any retrieval change**, because each
regressed silently once:

| Question | Must see |
|---|---|
| *What was Apple's total net revenue for its most recent fiscal year?* | $416,161 million, not a refusal. All five sources AAPL |
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
