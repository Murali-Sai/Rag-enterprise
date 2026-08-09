# Evaluation results

One JSON per run: aggregate scores, per-question scores, the retrieved contexts
and generated answers, and the configuration that produced them.

## Runs on `eval_questions_v4.json` are a new generation

v4 keeps v3's 20 questions byte-for-byte and adds 34, so the per-question rows
for the original 20 remain comparable across the boundary. **The aggregates do
not.** Three things changed at once, and each moves them on its own.

**A question can now correctly have no answer.** 16 of the 54 carry
`expected_behavior: "refuse"` — a real 10-K topic the filer does not disclose
(`no_answer`), a subject no filing covers (`out_of_corpus`), content the asking
role may not read (`rbac_blocked`), or a question with no defensible single
reading (`ambiguous`). Those rows are excluded from the RAGAS and citation
aggregates and scored on `refusal_correctness` instead. RAGAS scores a correct
refusal as 0.000 answer relevancy, and correctly so — a refusal has no
relevance to the question — so pooling them would lower every aggregate
precisely because the system behaved correctly. The full statement of the rule
is saved in each run's `config.no_answer_scoring`, because a run that pooled
them is not comparable to one that did not and nothing in the numbers reveals
which it was.

`refusal_correctness` is computed over *every* question, not just the 16. On an
answerable question it measures the opposite failure — declining something the
corpus does answer — which is the live defect here and which none of the RAGAS
four can distinguish from a genuine miss.

**The judge's token budget was raised from 1024 to 4096.** RAGAS faithfulness
extracts the statements in an answer and emits a structured verdict for each,
so its output scales with answer length; at 1024 it overran on 4 of 20
questions in the first measured run and RAGAS drops an overrun row rather than
truncating it. The dropped rows are the long, multi-claim answers — the ones
most at risk of being unfaithful — so the mean was taken over the easy ones,
and *which* rows overran varied between runs. This is a coverage fix, not a
change of judge: `eval_judge_model` is still pinned to `gpt-4o-mini`. Expect
measured faithfulness to move, and read `faithfulness_n` alongside it.

**Per-entity retrieval is on.** A question naming two of the five companies now
retrieves `retrieval_top_k` chunks for each and merges them, instead of ranking
one global top-k that filled every slot from whichever filing scored better.
This changes what is retrieved for the comparative stratum and nothing else.

## The noise floor at n=54

`eval_20260808_212927.json` and `eval_20260808_213635.json` are the same
configuration run twice, back to back, against the same corpus. Diff them with
`scripts/compare_eval_runs.py`.

| | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|---|
| Run A | 0.744 | 0.627 | 0.394 | 0.381 | 0.940 | 0.759 |
| Run B | 0.730 | 0.650 | 0.444 | 0.384 | 0.950 | 0.759 |
| **Spread (n=54)** | **0.014** | 0.023 | **0.050** | 0.003 | 0.010 | **0.000** |
| Spread (n=20) | 0.138 | 0.044 | 0.001 | 0.038 | — | — |

**The √n hypothesis is wrong, and the reason matters more than the number.**
The previous handoff predicted the 0.138 faithfulness floor would fall roughly
with √n to about 0.087 at n=50. It fell to **0.014** — an order of magnitude,
not the factor of 1.6 sampling alone predicts. So the old floor was never
mostly sampling noise.

The mechanism is visible in the run files. At `max_tokens=1024` the judge
overran on 4 of 20 questions and RAGAS dropped those rows; *which* four varied
between runs, and dropping a row moves a 20-row mean by far more than any real
effect. Both runs above score 38 of 38 (`faithfulness_n: 38`). Two things
changed at once — the question count and the token budget — so the split
between them is not measured, but sampling can only account for the smaller
part of it. **A faithfulness delta above ~0.015 is now readable, where the
previous floor said nothing below 0.14 could be claimed at all.**

**Context precision moved the opposite way, and it is not a retrieval effect.**
Its spread rose from 0.001 to 0.050. Retrieval is still deterministic — 53 of
54 questions retrieved byte-identical contexts across the two runs — and the
largest single move (+0.500, on Microsoft's Intelligent Cloud segment) has
*identical contexts and an identical generated answer* in both runs. The
variance is entirely in the judge. RAGAS context precision is an LLM judgment
about whether each retrieved chunk was useful, so a deterministic retriever
does not make it a deterministic metric; the earlier inference that it "barely
moves because retrieval is deterministic" does not hold. Treat 0.001 as a
lucky draw at n=20 rather than a property of the metric.

**Refusal correctness did not move at all**, per stratum or in aggregate. The
decision to answer or decline did not flip on a single one of the 54 questions
across two runs, even though 27 of the 54 generated textually different
answers. It is the most stable metric in the set.

**A zero floor makes the comparison script's headline misleading.**
`compare_eval_runs.py` prints "identical config — the spread below is the
run-to-run noise floor" whenever the two config blocks match, and `config`
fingerprints settings, not source. A run that changed abstention logic is
indistinguishable from a run that changed nothing.
`eval_20260809_001003.json` is exactly that case: same config, same corpus,
`refusal_correctness` 0.759 → **0.796**, entirely from narrowing
`answer_completeness()` to treat a refusal carrying cited claims as partial
(0.5) rather than full (0.0). Two questions moved, both `expected_behavior:
"answer"`, both from 0.0 to 1.0 — JPMorgan's CET1 ratio and Tesla's net income,
each of which had stated a verified, cited figure and been scored as though it
had answered nothing. Against a 0.000 floor that delta is the whole effect,
measured exactly; the 16 `refuse` rows held at 14/16, unchanged.

## Runs before 2026-08-08 19:00 UTC are superseded

Everything without a `corpus` key was measured against an index that was missing
most of two of the five filings. EDGAR section extraction was silently returning
near-empty sections — 11% of JPMorgan's 10-K, 8% of Microsoft's — so those runs
measured retrieval over a corpus that did not contain what the questions asked
about. Fixing it took the index from 6,394 chunks to 9,572.

Two further changes landed at the same time, and each moves scores on its own:

- **Ground truths are no longer produced by retrieval.** Runs against
  `eval_questions_v2.json` were scored on references generated from this
  project's own top-20 chunks. Context Recall around 0.57 there is not
  comparable to ~0.33 on `eval_questions_v3.json`; the corpus changed, and so
  did what counts as a correct answer.
- **Embeddings moved to `text-embedding-3-small`.** 1536-dim replacing 384-dim
  MiniLM, which changes retrieval outright.

So the superseded runs need re-running, not re-interpreting. They are kept
because the deltas *within* that generation were measured consistently against
each other, and because the record of what was believed and later retracted is
worth more than a clean directory.

## Reading a current run

`config` records every knob that changes what gets retrieved or generated.
`corpus` records what was indexed:

```json
"corpus": { "chunk_count": 9572, "content_digest": "e705ac4b47e9daf3" }
```

The digest is over the sorted per-chunk content hashes. Two runs with the same
config and different digests were measured against different corpora — which is
precisely the failure above, and it read as noise at the time because nothing in
the file could distinguish the two. Chunk count catches a re-ingest; the digest
catches a re-ingest that kept the count but changed the text.

A `chunk_count` of 8,936 is the semantically-chunked index
(`rag_enterprise_semantic`), not a regression in the main one.
