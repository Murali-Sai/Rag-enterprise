# RAG over SEC 10-K filings, with information barriers and a measured refusal path

**I built a retrieval-augmented generation system over five companies' real SEC
10-K filings that scores 0.698 faithfulness and 0.939 citation accuracy on a
54-question hand-written evaluation suite — and correctly declines 85.2% of the
questions it should decline, including every question whose answer is not in the
corpus or is behind an access barrier.**

Shipped configuration: dense retrieval over `text-embedding-3-small` with a
cross-encoder reranker, `gpt-4o` for generation, `gpt-4o-mini` as judge.
Run `eval_20260809_015503`, against the 8,232-chunk index (`c2f8c13673cf5ca5`).

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Citation Accuracy | Citation Coverage | Refusal Correctness |
|---|---|---|---|---|---|---|
| 0.698 | 0.682 | 0.378 | 0.335 | **0.939** | 0.661 | **0.852** |

Two of those columns are not RAGAS metrics and are the ones I care most about.
**Citation accuracy** is measured by handing a judge one claim and *only* the
chunk it cited — showing the whole context would turn it into a faithfulness
check, where the citation passes because the corpus supports the claim rather
than because the cited chunk does. **Refusal correctness** is scored over all 54
questions, not just the 16 unanswerable ones, so it catches the opposite failure:
declining something the corpus does answer.

---

## What it is

A RAG system over the most recent 10-K filings for Apple, Microsoft, JPMorgan,
Goldman Sachs and Tesla, downloaded from the SEC EDGAR API and parsed into Items
1, 1A, 7, 7A and 8 — not synthetic demo documents. On top of that sits a
compliance layer modelled on investment-bank information barriers: role-based
access enforced as a vector-store `where` clause, MNPI and investment-advice
guardrails, and a SEC 17a-4 append-only audit trail.

The thing that makes it a case study rather than a demo is that almost every
claim in this document is a number, and several of them are negative results.

---

## The evaluation suite is the actual work

54 hand-written questions across seven strata. Hand-written and hand-classified,
not LLM-generated: 54 is small enough to read, and a classifier would put another
model's judgment inside a measurement chain built to keep it out.

| Stratum | n | Expected |
|---|---|---|
| `interpretive` | 14 | answer |
| `exact_figure` | 13 | answer |
| `comparative` | 8 | answer |
| `ambiguous` | 6 | 3 answer, 3 refuse |
| `no_answer` | 5 | refuse |
| `out_of_corpus` | 4 | refuse |
| `rbac_blocked` | 4 | refuse |

**16 of the 54 correctly have no answer,** and that design choice drove three
others.

*Ground truths never come from retrieval.* An earlier version generated each
reference from this project's own top-20 chunks, then scored that same retriever
against the result — so anything retrieval systematically missed was missing from
the reference too and could never be counted as a miss. Regenerating from the
complete parsed filing dropped context recall from 0.57 to ~0.33. The system did
not get worse; the measurement stopped grading itself.

*The unanswerable questions are never pooled into the aggregates.* RAGAS scores a
correct refusal as 0.000 answer relevancy — correctly, since a refusal has no
relevance to the question. Pooling them would lower every aggregate *because the
system behaved correctly*. They are scored on refusal correctness instead, and
the rule is written into `config.no_answer_scoring` of every result file, because
a run that pooled them is not comparable to one that did not and nothing in the
numbers reveals which it was.

*The three refusal strata stay separate.* They fail at different stages —
`out_of_corpus` at the retrieval gate, `no_answer` only at the model,
`rbac_blocked` at the where-clause — and so they point at different fixes.

Per stratum, which is where the shape shows:

| Stratum | n | Faithfulness | Answer Relevancy | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|
| `interpretive` | 14 | 0.827 | 0.793 | 0.929 | 0.929 |
| `exact_figure` | 13 | 0.696 | 0.581 | 0.955 | 0.846 |
| `comparative` | 8 | 0.426 | 0.576 | 0.891 | 0.625 |
| `ambiguous` | 6 | 0.833 | 0.883 | 0.900 | 0.667 |
| `no_answer` | 5 | — | — | — | **1.000** |
| `out_of_corpus` | 4 | — | — | — | **1.000** |
| `rbac_blocked` | 4 | — | — | — | **1.000** |

The refusal path is the strongest thing in the system: all three unanswerable
strata score a perfect 1.000, and have in every run. Comparatives are the
weakest, at 0.426 faithfulness.

---

## What the retrieval ablation actually showed

Six configurations, same 20 questions, same 9,572-chunk corpus, one run each
except the baseline. **Read every row against the noise floor below it — most of
this table says nothing, and knowing which rows those are is the point.**

| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Dense only (mean of 2) | 0.645 | 0.480 | **0.696** | 0.327 |
| **+ Cross-encoder rerank** *(shipped)* | **0.770** | **0.719** | 0.466 | **0.350** |
| + BM25 hybrid (RRF) | 0.679 | 0.678 | 0.463 | 0.312 |
| + HyDE | 0.600 | 0.567 | 0.614 | 0.321 |
| + Semantic chunking¹ | 0.722 | 0.593 | 0.435 | 0.347 |

¹ Different index (8,936 chunks vs 9,572) — a chunking change rebuilds the
corpus, so this row is not strictly same-corpus comparable.

**Per-metric noise floor**, from running the dense baseline twice under identical
configuration:

| | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| **Run-to-run spread (n=20)** | **0.138** | 0.044 | **0.001** | 0.038 |

The spread differs by metric across two orders of magnitude. Faithfulness swings
0.138 because it compounds a non-deterministic generator with a non-deterministic
judge, so **any faithfulness delta in that table below ~0.14 says nothing at
all.** That single row retired several claims an earlier version of this project
had made.

### Hybrid search: measured, and not established as helpful

This is the result I expected to go the other way. BM25 fused with dense
retrieval via reciprocal rank fusion should help on 10-K prose, which is dense
with exact tokens — ticker symbols, "CET1", "Value at Risk", line-item names —
that lexical matching handles and embeddings blur.

It did not:

| vs. shipped dense + rerank | Δ Faithfulness | Δ Relevancy | Δ Precision | Δ Recall |
|---|---|---|---|---|
| + BM25 hybrid (RRF) | −0.091 | −0.041 | −0.003 | −0.038 |
| *noise floor* | *0.138* | *0.044* | *0.001* | *0.038* |

Every delta is negative, and every one lands at or below its own noise floor. The
honest statement is **"not established as helpful"** — not "hybrid is worse."
One run at n=20 cannot support either claim, and I retracted an earlier,
stronger claim I had made in this space (that hybrid "costs 0.23 recall") when it
failed to reproduce on the fixed corpus.

Hybrid ships **off** by default. Beyond the numbers, there is a structural cost:
RRF is deliberately ordinal — it fuses ranks and discards the underlying scores,
so its output says "this came first" and nothing about how relevant first was.
That removes the score channel the confidence layer and the insufficient-context
gate both read, so turning hybrid on silently disables the system's ability to
say "I don't know." A retrieval gain would have to be large to be worth that, and
no gain was measured at all.

### The reranker: the one clearly real effect, and it cuts both ways

Cross-encoder reranking moved answer relevancy **0.480 → 0.719**, a delta of
0.239 against a 0.044 floor — the largest readable effect in the table by a wide
margin. It also moved context precision **0.696 → 0.466**, a loss of 0.230
against a 0.001 floor.

That precision loss is not cosmetic; it is the system's biggest live defect.
`ms-marco-MiniLM` assigns uniformly negative logits to financial-table prose, and
because relevance is a logistic squash of that logit, the whole retrieved set can
land under the 0.15 insufficient-context threshold — at which point the system
declines *without ever calling the LLM*. On "What was Apple's total net revenue
for its most recent fiscal year?" it ranked a *Foreign pretax earnings* chunk
first at −2.33 and pushed the actual net-sales table from rank 1 to rank 5, for a
set relevance of 0.077.

The tempting fix — lower the threshold — is the wrong one. It tunes the gate to
the test set and relocates the failure downstream into ungrounded answers. The
real options are a financial-domain cross-encoder, rerank-then-restore-order, or
dropping the reranker and keeping the dense ordering.

---

## Chunking strategy comparison

Three strategies are implemented and switchable by `CHUNKING_STRATEGY`:

| Strategy | How it splits |
|---|---|
| `fixed` | Character windows with overlap, structure-blind — cuts through a sentence, a table row, or a dollar figure |
| `recursive` *(shipped)* | Paragraph → line → sentence → word → character, falling through only where the current boundary does not occur |
| `semantic` | Splits where consecutive sentence embeddings diverge |

Chunking is baked into an index at ingestion time, so comparing strategies means
building a collection per strategy and pointing the eval at each — not flipping a
setting on a live index. Every chunk carries the strategy that produced it, so a
collection can never be misattributed after the fact.

**The measured comparison, and its limits:**

| Strategy | Index | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|
| `recursive` | 9,572 chunks (`e705ac4b`) | 0.770 | **0.719** | 0.466 | 0.350 |
| `semantic` | 8,936 chunks (`8068f567`) | 0.722 | 0.593 | 0.435 | 0.347 |
| `fixed` | — | *not measured* | | | |

**What this data can and cannot support.** One run each, n=20. Of the four
deltas, only answer relevancy (0.126) clears its noise floor (0.044); faithfulness
(0.048) is well under 0.138, and precision and recall are inside their floors.
And the two rows are different corpora — a chunking change rebuilds the index by
definition — so even the relevancy delta is confounded with the corpus change.

So the honest conclusion is narrow: **recursive is shipped, semantic was not
shown to beat it, and `fixed` — the structure-blind baseline that exists
specifically so structure-awareness has something to prove itself against — was
never run.** That is a gap in the evaluation, not a finding. Anyone claiming
"structure-aware chunking helps" on this evidence would be overreading it, and
the missing `fixed` row is exactly the one that would settle it.

What *is* solidly established about chunking here came from a different
measurement: about a fifth of every filing's chunks were under 200 characters,
and stripping page furniture (running headers, footers, standalone page numbers)
plus dropping sub-80-character stubs took the index from 9,572 to 8,232 chunks.
See below for what that did and did not fix.

---

## Three defects the measurement found, and one hypothesis it killed

**Goldman Sachs scores exactly 0.000 context recall — and the obvious explanation
is wrong.** Slicing per company rather than per stratum surfaced a hard failure
the aggregate hid. GS carried 214 bare running-header chunks
(`"Goldman Sachs 2025 Form 10-K | 123"`), and on the GS revenue question two of
five retrieved slots were page numbers. That looked conclusive. I stripped the
furniture — those 214 chunks went to **0** — and re-measured:

| | AAPL | MSFT | JPM | GS | TSLA |
|---|---|---|---|---|---|
| Faithfulness | 0.737 | 0.718 | 0.900 | 0.613 | 0.881 |
| Context Recall | 0.403 | 0.346 | 0.167 | **0.000** | 0.506 |
| Context Precision | 0.722 | 0.619 | 0.158 | **0.050** | 0.305 |

GS recall did not move off 0.000, and GS context precision *fell*, 0.271 → 0.050.
The furniture was real and it is gone; it was not the cause. This is now the
best-isolated open defect in the project — a hard zero on one company and not the
other four, with the leading hypothesis eliminated by measurement rather than
left untested.

**The parser fix did work on what it was second-aimed at.** Refusal correctness
moved 0.796 → 0.852 and `exact_figure` 0.692 → 0.846. Counted as rows rather than
means: 11 answerable questions were declined before, **6** now. That attribution
is safe because refusal correctness is the most stable metric in the set — it did
not move at all across two runs in which 27 of 54 answers were textually
different.

**And it cost something.** Aggregate context recall fell 0.383 → 0.335, with AAPL
0.569 → 0.403 and JPM 0.267 → 0.167. Against a measured n=54 recall spread of
0.003 those are large, though comparing one run to a two-run mean across a corpus
change can only ever be suggestive. Both facts are the measurement, and both are
in the repo.

**Ambiguity is not handled at all.** Two of three underspecified questions were
answered about one arbitrarily chosen company with no flag that the question
admitted others. There is no ambiguity-detection mechanism; the `ambiguous`
stratum exists to record that, not to claim it is solved.

---

## The measurement apparatus is part of the system

**A judge token budget masqueraded as sampling noise for weeks.** The n=20
faithfulness noise floor was 0.138. I predicted it would fall with √n to roughly
0.087 at n=50. Measured at n=54 it is **0.014** — an order of magnitude, where
sampling alone predicts a factor of 1.6.

The cause was in the run files. `max_tokens=1024` was being applied to the judge
as well as to generation, and RAGAS faithfulness emits one structured verdict per
extracted statement, so it overran on 4 of 20 questions. RAGAS drops an overrun
row rather than truncating it, *which* four varied between runs, and the dropped
rows were the long multi-claim answers — the ones most at risk of being
unfaithful. The mean was being taken over the easy ones. At
`eval_judge_max_tokens=4096`, both 54-question runs score 38 of 38.

**Context precision moved the other way, and it is not a retrieval effect.** Its
spread rose from 0.001 at n=20 to 0.050 at n=54, despite 53 of 54 questions
retrieving byte-identical contexts across the two runs. The largest single move
(+0.500) had identical contexts *and* an identical generated answer. RAGAS context
precision is an LLM judgment about chunk usefulness, so a deterministic retriever
does not make it a deterministic metric — an inference I had made and had to
withdraw.

**A two-run noise floor is a lower bound, not a bound.** A third run on the same
corpus and config later exceeded the floor on one metric. The table says "deltas
below this are definitely noise." It never says "deltas above this are definitely
real."

Supporting apparatus, built because each of these produced a wrong conclusion
first:

- **Corpus fingerprinting.** Every result file records chunk count and a digest
  over the sorted per-chunk content hashes. `compare_eval_runs.py` refuses to
  diff across a fingerprint change. Two runs with the same config and different
  corpora once read as noise, because nothing in the file could tell them apart.
- **The judge is pinned.** `EVAL_JUDGE_MODEL` is fixed at `gpt-4o-mini` because
  every stored result was measured with it. The *citation* judge is a separate,
  freely chosen model — measurement apparatus and system component are not the
  same thing.
- **Per-question rows join by dataset index, not position,** with a guard that
  raises if RAGAS returns rows out of order. RAGAS filters rows out, which makes
  a positional join silently wrong rather than loudly broken.

---

## The compliance layer, and making it visible

Role-based access is enforced as a ChromaDB `where` clause built from the asking
role — a research analyst never receives a trading-desk document, even in raw
retrieval results — and information barriers are absolute: holding `trading`
alongside `research` does not buy back the trading department.

The demonstration is a single question asked twice:

> *"What pre-trade controls and order entry limits does ACME Financial Holdings'
> internal trading desk procedures manual set?"*

As **research_analyst**: declines at the retrieval gate, `low_retrieval_confidence`,
with the Research-Trading Wall shown in force and `trading` listed among the
departments it removed. No LLM call is made, so the refusal costs nothing.

As **trader_desk**: answers at 0.877 confidence off `trading_desk_procedures.txt`,
with five cited claims.

The wall is the only thing that changed.

A Streamlit dashboard traces one question through six stages — identity and the
barriers in force, the query, the ranked chunks with their raw and normalised
scores, the answer or the structured refusal, the per-claim verdicts, the
confidence breakdown, the guardrail flags. It holds no retrieval logic and no
thresholds: it is an HTTP client, so it cannot disagree with the system it
shows.

It also runs the same question through dense and hybrid retrieval in adjacent
columns. On the Apple revenue question the two differ by a single rank
position — hybrid lifts the correct net-sales table from third to second — and
both leave a Goldman Sachs chunk wrongly at rank 1. That is the reranker
defect, reproducible on demand, next to a comparison whose aggregate answer is
"inside the noise floor."

Three rules the layout keeps:

- **A refusal is not an error state.** It scores 1.000 on all three unanswerable
  strata; rendering it in red would tell the reader the opposite of what the
  measurement says. The panel distinguishes the retrieval gate firing from the
  model declining over good retrieval, because those point at different fixes.
- **No number without its scale.** The confidence composite never appears without
  its label; a null retrieval score renders as *unavailable*, never as zero; a
  chunk's relevance sits beside its raw score and the stage that produced it. The
  reranker's negative logits are visible here as a matter of course.
- **Nothing is precomputed.** A test asserts no score is baked into the template,
  because a number in the markup is a number nobody measured.

---

## What I would do next

1. **Run the `fixed` chunking baseline.** It is the missing row that would make
   the chunking comparison mean something, and it exists in code precisely so
   structure-awareness has to prove itself.
2. **Isolate the Goldman Sachs zero.** The hypothesis is dead and the failure is
   sharply scoped: one company, every question, every run.
3. **Replace or reorder the reranker.** Six false refusals remain, and the fix is
   not lowering the threshold.
4. **Re-measure the ablation on the current corpus at n=54.** Every row in it was
   measured at n=20 against a 0.138 faithfulness floor, on an index that no
   longer exists. It is only meaningful as a complete set.

---

### Stack

`gpt-4o` generation · `text-embedding-3-small` · ChromaDB · `rank_bm25` ·
`ms-marco-MiniLM` cross-encoder · FastAPI · SQLite · Jinja2 · Docker · Cloud Run ·
RAGAS · MCP server

347 tests, ruff clean.
