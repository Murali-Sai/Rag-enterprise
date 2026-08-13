# RAG over SEC 10-K filings, with information barriers and a measured refusal path

**I built a retrieval-augmented generation system over five companies' real SEC
10-K filings that scores 0.853 faithfulness and 0.936 citation accuracy on a
54-question hand-written evaluation suite — and correctly declines 94.4% of the
questions it should decline, including every question whose answer is not in the
corpus or is behind an access barrier.**

Shipped configuration: dense retrieval over `text-embedding-3-small` with a
cross-encoder reranker, per-entity retrieval, `gpt-4o` for generation,
`gpt-4o-mini` as judge. **Means over three runs** (`eval_20260811_195440`,
`_20260811_200218`, `_20260811_200901`) against the 8,232-chunk index
(`c2f8c13673cf5ca5`), with the per-metric spread across those runs beneath.

| | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Citation Accuracy | Citation Coverage | Refusal Correctness |
|---|---|---|---|---|---|---|---|
| **mean (n=3)** | **0.853** | 0.763 | 0.429 | 0.386 | **0.936** | 0.729 | **0.944** |
| *spread* | *0.063* | *0.013* | *0.008* | *0.021* | *0.018* | *0.022* | *0.000* |
| *previous baseline (n=3)* | *0.697* | *0.681* | *0.391* | *0.338* | *0.928* | *0.642* | *0.846* |

The spread row is the more useful half. Faithfulness moves **0.063** between
identical runs — nine times what the previous baseline suggested, and the single
most important correction on this page, because it is the denominator under every
faithfulness claim made anywhere in this document. Context precision, by
contrast, tightened from 0.035 to 0.008.

Against the previous baseline, five metrics clear their own floor: faithfulness
+0.156 (2.5×), answer relevancy +0.082 (6.5×), refusal correctness +0.099 (5.2×),
citation coverage +0.087 (2.2×), answer correctness +0.063 (2.4×). **Context
precision (+0.037) and context recall (+0.048) do not** — both land at roughly
1.1× their floors, and neither is a claim. Citation accuracy moved +0.008 against
a 0.018 floor: unchanged, which is the part worth stating, because it means the
gains did not come out of the number this document leans on hardest.

These were single-run figures until 2026-08-10, and quoting the best of three for
citation accuracy would have been worth about 0.011 of flattery on that number.

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
`out_of_corpus` at the model (it was the retrieval gate until the gate was
measured and found not to be doing that job), `no_answer` at the model,
`rbac_blocked` at the where-clause — and so they point at different fixes.

Per stratum, which is where the shape shows:

| Stratum | n | Faithfulness | Answer Relevancy | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|
| `interpretive` | 14 | 0.942 | 0.838 | 0.966 | 0.929 |
| `exact_figure` | 13 | 0.840 | 0.634 | 0.921 | **1.000** |
| `comparative` | 8 | 0.710 | 0.796 | 0.887 | **1.000** |
| `ambiguous` | 6 | 0.870 | 0.883 | 0.900 | 0.667 |
| `no_answer` | 5 | — | — | — | **1.000** |
| `out_of_corpus` | 4 | — | — | — | **1.000** |
| `rbac_blocked` | 4 | — | — | — | **1.000** |

The refusal path is the strongest thing in the system: all three unanswerable
strata score a perfect 1.000, and have in every run ever recorded. Comparatives
used to be the weakest stratum by a wide margin — 0.426 faithfulness and 0.625
refusal correctness — and are now 0.710 and 1.000, because a comparison is two
questions and was finally retrieved and scored as two.

**`ambiguous` was the one answerable stratum that had not moved** — 0.667 in
every baseline, before and after every retrieval fix, because it was never a
retrieval failure: the pipeline answered underspecified questions about
whichever company ranked best, fluently and with no flag. It is now handled
mechanically (see *Ambiguity*, below) and scored **1.000 on one confirming run**
(`eval_20260812_180308`), taking overall refusal correctness 0.944 → **0.981**
on that run. The remaining refusal error is the one Goldman question the model
declines despite good retrieval. Single run, not a new three-run baseline; the
detector is deterministic and a suite-level test pins exactly which questions it
touches, which is why one run was bought rather than three.

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

Hybrid ships **off** by default, on the numbers alone: no gain was measured, and
the cost of carrying a second retriever is not free.

I also used to claim a structural cost here, and it was overstated. RRF is
deliberately ordinal — it fuses ranks and discards the underlying scores, so its
output says "this came first" and nothing about how relevant first was, and the
fusion stage does hand back documents with no score attached. I wrote that this
removes the channel the confidence layer and the insufficient-context gate read,
and so "silently disables the system's ability to say 'I don't know.'"

That holds only if reranking is off too. The pipeline is `dense (+BM25 → RRF) →
cross-encoder rerank → top_k`, and the reranker rescores every surviving
document, so the channel is rebuilt before anything downstream reads it.
Querying the deployment with `retrieval_mode=hybrid` returns cross-encoder
scores and a retrieval confidence of 0.952 — not the null the claim predicted.

I am leaving the correction visible rather than editing the claim away, because
it is the same failure the retracted "costs 0.23 recall" line was: a mechanism
that sounded right, reasoned about but never measured. The difference is that
this one survived longer, since it was an argument about architecture rather
than a number, and nothing in the test suite contradicts an unstated
assumption. Checking it took one HTTP request.

### The reranker: the one clearly real effect, and it cuts both ways

Cross-encoder reranking moved answer relevancy **0.480 → 0.719**, a delta of
0.239 against a 0.044 floor — the largest readable effect in the table by a wide
margin. It also moved context precision **0.696 → 0.466**, a loss of 0.230
against a 0.001 floor.

That precision loss is not cosmetic; it was for a long time the system's biggest
live defect. `ms-marco-MiniLM` assigns uniformly negative logits to
financial-table prose, and because relevance is a logistic squash of that logit,
the whole retrieved set can land under the 0.15 insufficient-context threshold —
at which point the system declines *without ever calling the LLM*. On "What was
Apple's total net revenue for its most recent fiscal year?" it ranked a *Foreign
pretax earnings* chunk first at −2.33 and pushed the actual net-sales table from
rank 1 to rank 5, for a set relevance of 0.077.

I wrote here that the tempting fix — lower the threshold — was the wrong one,
because it "tunes the gate to the test set and relocates the failure downstream
into ungrounded answers." Both halves were testable and I did not test either.
Both turned out to be wrong, and the correction is the third entry in this
document's running theme.

**On tuning to the test set.** That objection is about method, and method has an
answer: don't tune on the test set. `evaluation/datasets/gate_calibration_v1.json`
is a 49-item probe set held out from the suite — out-of-corpus items labelled by
construction, in-corpus items carrying a literal string that must be found in
that company's chunks before the item is allowed to influence anything. On it,
the premise the 0.15 threshold rested on collapses. Out-of-corpus questions score
across the whole range: Citigroup's CET1 ratio **0.9750**, Bank of America net
interest income 0.8994, the price of Bitcoin 0.2962 — against an answerable
question about Microsoft's dividend at **0.0031**. Ten of 24 out-of-corpus probes
already cleared 0.15.

The reason is the one this document keeps rediscovering in different costumes: a
cross-encoder scores *topic* match and has no representation of which company a
passage is about. The corpus really does discuss CET1 ratios and net interest
income — for other banks. It is the same blindness that had Goldman Sachs
questions answered off Tesla's filing, appearing on the gate side rather than the
retrieval side.

**On relocating the failure downstream.** That objection is empirical, and it is
simply false. With the gate held open, the model refused **24 of 24**
out-of-corpus questions on its own, including the one the gate scored 0.975. The
gate had never been what made those refusals correct; the suite could not show
this, because all four of its out-of-corpus questions sit below the gate and none
had ever reached the model. And the predicted downstream damage went the other
way: over a fresh three-run baseline, faithfulness rose **0.697 → 0.853**, the
largest effect measured in this project, at 2.5× its own 0.063 noise floor.
Refusal correctness rose 0.846 → 0.944 with a run-to-run spread of exactly zero,
and all three unanswerable strata stayed at 1.000.

Admitting those questions did not produce ungrounded answers. The context had
been good enough to ground them the whole time — which is what a refused question
retrieving **1.000 context recall** had been saying all along.

So the gate stops being a correctness mechanism and becomes a cost guard, pinned
at 0.001, below the lowest answerable probe. The options I listed instead — a
financial-domain cross-encoder, rerank-then-restore-order, dropping the reranker —
remain open and remain unmeasured. What is now measured is that none of them were
needed for this particular failure.

I am leaving the wrong version above rather than deleting it, for the same reason
as the RRF claim before it: it is the same failure in a new place. A mechanism
that sounded right, argued from plausible premises, never checked. What made it
durable this time is that it came with a number attached — "0.003–0.03 against a
0.15 gate" was real, and it made a claim resting on four data points look like a
measurement.

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

**The measured comparison.** All three strategies, on the same 54-question
suite, the same `gpt-4o` generator and the same `gpt-4o-mini` judge. **Two runs
per strategy** (three for `recursive`), added 2026-08-10 — the first version of
this table had one run each and said something considerably stronger.

> **This table is measured on the pre-fix pipeline** and is left that way
> deliberately. The `recursive` row is the *old* baseline, not the 0.853 headline
> above, because `fixed` and `semantic` were never re-run against the entity
> fixes or the recalibrated gate. Swapping in the new `recursive` figures would
> compare three strategies across two different pipelines and manufacture a
> chunking result out of a retrieval change — which is a more subtle version of
> the error that produced the retracted 19× claim below. Re-running the other two
> strategies costs about $3.20; until then this comparison is valid only against
> itself.

Means, with each strategy's own run-to-run spread beside it:

| Strategy | Index | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|---|---|
| `recursive` *(shipped, n=3)* | 8,232 (`c2f8c136`) | 0.697 | 0.681 | 0.391 | 0.338 | 0.462 | **0.928** | 0.846 |
| `fixed` *(n=2)* | 6,585 (`0740fec6`) | **0.790** | **0.690** | **0.474** | **0.468** | **0.488** | 0.883 | **0.861** |
| `semantic` *(n=2)* | 8,936 (`8068f567`) | 0.781 | 0.646 | 0.430 | 0.360 | 0.477 | 0.915 | 0.833 |
| *widest spread across strategies* | | *0.034* | *0.004* | *0.035* | *0.044* | *0.026* | *0.017* | *0.019* |

**The second run cost the headline claim, and that is the finding.** The
previous version of this table quoted floors of 0.005, 0.004 and 0.012 and
concluded that `fixed` beat `recursive` by **19×, 22× and 8×** them. Those
floors came from a single pair of `recursive` runs and were transferred to the
other two strategies "on the assumption that variance is comparable across
strategies, which is reasonable and unverified." It was unverified and it was
wrong: measured per strategy, faithfulness varies 0.034 run-to-run on
`semantic` against 0.007 on `recursive`, and context precision varies 0.035 on
`recursive` itself — nine times the 0.004 that row claimed.

Re-scored against the widest spread actually observed:

| | claimed 2026-08-09 | measured 2026-08-10 |
|---|---|---|
| Faithfulness | +0.095 vs 0.005 floor = **19×** | +0.093 vs 0.034 floor = **2.8×** |
| Context precision | +0.088 vs 0.004 floor = **22×** | +0.082 vs 0.035 floor = **2.3×** |
| Context recall | +0.099 vs 0.012 floor = **8×** | +0.130 vs 0.044 floor = **3.0×** |

The deltas barely moved. The floors moved by an order of magnitude, and the
floors were the part that was measured once.

**What survives.** `fixed` still beats the shipped default on all three, and
the direction was never really in doubt — 2–3× a floor is a real effect. It is
just not the rout the first table described. Two deltas that first table
counted have since dissolved into noise: refusal correctness (+0.015 against a
0.019 floor) and answer correctness (+0.026 against 0.026), both of which the
earlier version either bolded or declined to call. Declining to call
`answer_correctness` was the right instinct and it is now measured rather than
intuited.

`recursive` keeps citation accuracy — 0.928 against fixed's 0.883, a real
0.045 at 2.6× its floor. Its chunks are larger and paragraph-aligned, so a
cited claim more often sits whole inside one chunk. And `semantic` picked up a
clear negative it did not have before: it *loses* answer relevancy by 0.034
against a 0.004 floor, **8.6×**, which is the largest single effect anywhere in
this table and points the opposite way from its faithfulness win.

**Why the default did not change.** The trade is now legible: give up 0.045 of
citation accuracy — the strongest number this project has, and the one the
headline claim rests on — to buy 0.08–0.13 across faithfulness, precision and
recall. That is a genuine engineering trade rather than a mistake to correct,
and at 2–3× the noise it does not carry enough to justify invalidating the
shipped index, its digest, the running deployment and every figure in this
document. `recursive` stays. What changed is that the choice is now measured
instead of assumed, and the case for revisiting it is written down.

**What this still cannot support.** Strategy and corpus cannot be separated — a
chunking change rebuilds the index by definition, so `fixed` is 6,585 chunks
against recursive's 8,232, and "fixed wins" means the pipeline under that
strategy wins, not that structure-blindness is better in the abstract. And two
runs is enough to show a one-run floor was too tight; it is not enough to
pin variance properly. The floor used above is the widest observed rather than
any strategy's own, which is the conservative reading and the one worth
quoting. Against `recursive`'s own 0.007 faithfulness spread the same delta
reads as 13× — the choice between those two numbers is a judgment call, and
quoting the flattering one is how the first version of this table happened.

This is also the clearest argument in the project for the spec's instruction to
build the comparison at all: the missing row was not a formality. It was the
row that contradicted the design.

What *is* solidly established about chunking here came from a different
measurement: about a fifth of every filing's chunks were under 200 characters,
and stripping page furniture (running headers, footers, standalone page numbers)
plus dropping sub-80-character stubs took the index from 9,572 to 8,232 chunks.
See below for what that did and did not fix.

---

## Three defects the measurement found, and four hypotheses it killed

**Goldman Sachs scored exactly 0.000 context recall for twelve runs — and the
obvious explanation was wrong.** Slicing per company rather than per stratum
surfaced a hard failure the aggregate hid. GS carried 214 bare running-header chunks
(`"Goldman Sachs 2025 Form 10-K | 123"`), and on the GS revenue question two of
five retrieved slots were page numbers. That looked conclusive. I stripped the
furniture — those 214 chunks went to **0** — and re-measured:

| | AAPL | MSFT | JPM | GS | TSLA |
|---|---|---|---|---|---|
| Faithfulness | 0.737 | 0.718 | 0.900 | 0.613 | 0.881 |
| Context Recall | 0.403 | 0.346 | 0.167 | **0.000** | 0.506 |
| Context Precision | 0.722 | 0.619 | 0.158 | **0.050** | 0.305 |

GS recall did not move off 0.000, and GS context precision *fell*, 0.271 → 0.050.
The furniture was real and it is gone; it was not the cause.

**A second hypothesis was found and fixed, and it moved Goldman off zero.** The
hard zero held across twelve runs. Its cause was that `MultiEntityRetriever`
applied a ticker filter only when a question named *two or more* companies, so a
question naming one searched all five filings plus the synthetic sample
documents. Asked for Goldman's total net revenues and return on average common
equity, retrieval returned two GS chunks, one Tesla, one JPMorgan, and one from a
fabricated sample reading *"Total Net Revenue: $38.2B / Return on Equity (ROE):
14.8%"* — a perfect-looking answer about a company that is not Goldman Sachs.
Banks hid it best precisely because their tables are structurally identical:
every one of them reports total net revenues and a return on equity, so the wrong
company's figures are not merely retrievable but convincing.

GS-only context recall is **0.125** on the current baseline. Three of four GS
questions still score 0.000, and GS context precision is 0.000 — the chunks are
Goldman's, which the filter now guarantees, but they are not the chunks the
ground truth is built from. Every GS ground-truth figure *is* in the corpus
(`58,283`, `14.3%`, `727,338`, `Platform Solutions`, `Value-at-Risk` all appear
among 2,888 indexed GS chunks), so what remains is a ranking failure inside one
company's filing.

A third hypothesis died on 2026-08-11. GS indexes only **one** chunk under
`Quantitative and Qualitative Disclosures About Market Risk`, against 4–8 for the
non-banks, which looked like a parser bug on the section most relevant to a
trading desk. It is not: GS and JPMorgan both incorporate Item 7A by reference
(*"…are set forth in Management's Discussion and Analysis … in Part II, Item 7"*),
and JPMorgan shows the identical 1-chunk stub for Items 7 and 8 beside a
2,705-chunk incorporated Annual Report. One chunk is the correct parse of a
cross-reference.

**A fourth hypothesis was tested, confirmed, and still turned out not to be the
fix — which is the most useful result of the four.** `rerank_candidate_k` is 20
for every question regardless of filing size: 4.3% of Apple's 460 chunks, and
0.7% of Goldman's 2,888. If large filings retrieve badly because their candidate
budget is proportionally starved, raising it should help them and leave the small
ones alone. Swept with a free retrieval-only eval:

| | k=20 | k=50 | k=100 |
|---|---|---|---|
| GS | 0.067 | 0.117 | **0.167** |
| JPM | 0.071 | 0.107 | **0.143** |
| AAPL | 0.500 | 0.500 | 0.500 |
| MSFT | 0.210 | 0.210 | 0.210 |
| TSLA | 0.306 | 0.250 | 0.250 |
| overall | 0.242 | 0.239 | 0.248 |

The predicted shape appears exactly: the two starved filings gain 150% and 101%,
the two small ones do not move at all. And it is still not shippable — TSLA
*loses* 0.056 and the overall figure nets flat, because a larger candidate pool
is also more opportunity for a weak ranker to promote the wrong chunk. Recall
won at the candidate stage is handed back at the rerank stage.

Tracing where each ground-truth figure is lost says why. The two failing GS
questions fail at different stages: one has its figures in the candidate set at
ranks 3 and 6 and the *reranker* drops them; the other never retrieves them at
all, and its top-ranked chunk scores **0.994** while being GS's entire Item 7A —
272 characters reading *"…are set forth in Management's Discussion and Analysis
… in Part II, Item 7."* A perfect title match with no answer in it. At k=100,
six of that question's seven figures still never enter the candidate set.

So the binding constraint is not the budget and not the filtering. **Neither the
bi-encoder nor the cross-encoder represents financial tables well.** The question
is thematic; the answer is a table of numbers; a table of numbers does not embed
like the sentence asking about it. That is the same root cause as the reranker's
uniformly negative logits on financial-table prose, arrived at from the opposite
direction — which is the strongest evidence in this document that the reranker,
not the corpus, is what Goldman is actually stuck behind.

This remains the best-isolated open defect in the project — one company far below
the other four, with four hypotheses now eliminated by measurement rather than
left untested, and the fourth eliminated *after* being confirmed.

**The parser fix did work on what it was second-aimed at.** Refusal correctness
moved 0.796 → 0.852 and `exact_figure` 0.692 → 0.846. Counted as rows rather than
means: 11 answerable questions were declined before, **6** now. That attribution
is safe because refusal correctness is among the most stable metrics in the set:
it held at 0.852 across two runs in which 27 of 54 answers were textually
different, and a third on 2026-08-10 came in at 0.833. Its spread is 0.019, not
the 0.000 this paragraph claimed when two runs were all there was — a 0.056
move still clears it threefold, but "did not move at all" was a statement about
sample size wearing the costume of a statement about the metric.

> The current pipeline's three runs give a refusal-correctness spread of 0.000
> again — the same three questions wrong in all three. That is a stronger claim
> than the one retracted here, because it is three runs rather than two and
> because the *identity* of the failures is stable, not just the count. It is
> still not a claim that the metric cannot move.

**And it cost something.** Aggregate context recall fell 0.383 → 0.335, with AAPL
0.569 → 0.403 and JPM 0.267 → 0.167. Against a measured n=54 recall spread of
0.003 those are large, though comparing one run to a two-run mean across a corpus
change can only ever be suggestive. Both facts are the measurement, and both are
in the repo.

**A wrong answer that scored well, and the fix that followed.** Driving the
dashboard by hand — not running a test — turned up the most instructive failure
in the project. Asked for Goldman's total net revenues and return on average
common equity, the system answered *"$58,283 million … 2.1%"*, with citations
and high confidence. Every number was real and every number was in the
retrieved chunk. It was still wrong: 2.1% belongs to the Platform Solutions
segment, the word "Total" is where the firmwide table starts, and the firmwide
figure — 15.0% over $108,726 million of average common equity — is the first
row of the **next** chunk, which was never retrieved. The three segments sum to
the firmwide row exactly, so the corpus was complete and correct; a table
crossed a chunk boundary and the half that survived read like an answer.

No amount of reranking fixes that, because the retrieved chunk really is
relevant — what is missing is its continuation. The fix reads the next row, the
way a person would: after reranking selects the final set, a chunk that looks
like a table gets the following chunk appended, matched on source, section and
`chunk_index + 1`. Query-time by choice — chunking is baked into an index, so
re-chunking would mean re-embedding, a new corpus digest, and the invalidation
of every published figure here.

Measured against the identical config without it: context precision **+0.066**,
citation coverage **+0.058**, answer correctness **+0.053**, answer relevancy
**+0.036**, each clearing its noise floor; citation accuracy paid **−0.026**
against an 0.018 floor, a small real cost, because a stitched chunk puts more
text behind a single citation number. The largest single movement was another
split table — *"JPMorgan's total assets and CET1 capital ratio"*, answer
correctness +0.577.

**The precision result refuted the prediction written into the code.** Longer
chunks were expected to dilute relevance and cost precision. Precision rose,
because RAGAS context precision asks whether the retrieved context is *useful
for answering*, and a table with its continuation is more useful than a table
without one. Completion beat dilution — which is not what the intuition said,
and is why the run was bought.

It narrows the Goldman defect rather than closing it. The evaluation phrases
that question differently, retrieves a chunk set containing neither figure, and
is unchanged at 0.333: expansion extends what was retrieved and cannot rescue a
ranking that never surfaced the table.

**Ambiguity.** For most of this project's life, two of three underspecified
questions were answered about one arbitrarily chosen company with no flag that
the question admitted others, and the stratum existed to record that. It is now
handled, and the shape of the fix is the finding: this was the only defect on
the answerable side that no retrieval change ever touched, because retrieval was
doing its job — it was asked an underspecified question and answered it. The
detector (`src/retrieval/ambiguity.py`) is two literal signals gated on entity
detection finding nothing: a definite reference to an unnamed company ("the
bank" matches two filings, "the company" five), or a company-scoped financial
metric with no subject at all. No LLM call, same argument as entity detection:
a classifier in front of retrieval would put another model's judgment inside
the measurement chain. Both signals bias deliberately toward silence — a missed
ambiguous question falls through to the old behaviour, while a false positive
would refuse an answerable question, which is the failure the answerable half
of `refusal_correctness` exists to catch. The stratum moved 0.667 → **1.000**
on one confirming run, the first time it moved at all.

---

## The measurement apparatus is part of the system

**A judge token budget masqueraded as sampling noise for weeks.** The n=20
faithfulness noise floor was 0.138. I predicted it would fall with √n to roughly
0.087 at n=50. Measured at n=54 it is **0.014** — an order of magnitude, where
sampling alone predicts a factor of 1.6.

A caveat this section did not carry until 2026-08-10: that 0.014 is
`recursive`'s spread on its own corpus, and a floor measured on one
configuration is not a property of the metric. `semantic` varies 0.034 on
faithfulness over the same suite and judge. The judge-budget finding holds —
tightening it collapsed the floor by an order of magnitude, and both runs now
score 38 of 38 — but "the faithfulness floor is 0.014" was over-general, and
believing it is what put a 19× claim in the chunking table above.

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
