"""RAGAS evaluation pipeline for the RAG system.

Usage:
    python -m evaluation.run_evaluation
"""

import json
from datetime import UTC, datetime

from evaluation.eval_config import EVAL_EMBEDDING_MODEL, EVAL_QUESTIONS_PATH, RESULTS_DIR

RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    # Added 2026-08-09. Runs before this date have no answer_correctness column
    # at all; treat its absence as "not measured", never as a zero.
    "answer_correctness",
)
# Computed in run_rag_pipeline, not by RAGAS. citation_accuracy is the metric
# Project 6 asks the case study to state ("X% faithfulness and Y% citation
# accuracy") and the reason Phase 3 exists; coverage and confidence are kept
# beside it because accuracy alone can be gamed by citing almost nothing.
CITATION_METRIC_NAMES = ("citation_accuracy", "citation_coverage", "confidence")
# The axis a question with no answer is scored on. Computed here, from the
# structure of the refusal — no LLM call, see _refusal_correctness.
BEHAVIOUR_METRIC_NAMES = ("refusal_correctness",)
METRIC_NAMES = RAGAS_METRIC_NAMES + CITATION_METRIC_NAMES + BEHAVIOUR_METRIC_NAMES

# What a question expects the system to do. Absent means "answer" — every
# question written before v4 is answerable by construction.
ANSWER = "answer"
REFUSE = "refuse"

# How the two scoring populations are described in the saved config block. A
# run that scored no-answer rows differently is not comparable to one that did
# not, and the file has to say which it was.
NO_ANSWER_SCORING = (
    "Questions with expected_behavior='refuse' are excluded from the RAGAS and "
    "citation aggregates and scored only on refusal_correctness. RAGAS scores a "
    "correct refusal as 0.000 answer relevancy — correctly, a refusal has no "
    "relevance to the question — so pooling them would lower every aggregate "
    "precisely because the system behaved correctly. refusal_correctness is "
    "computed over every question, including answerable ones, where it measures "
    "the opposite failure: refusing something the corpus does answer."
)


def load_eval_dataset() -> list[dict]:
    with open(EVAL_QUESTIONS_PATH) as f:
        return json.load(f)


def expected_behavior(item: dict) -> str:
    return item.get("expected_behavior", ANSWER)


def scored_by_ragas(item: dict) -> bool:
    """Whether a question's answer is the kind of thing RAGAS can grade."""
    return expected_behavior(item) == ANSWER


def run_rag_pipeline(question: str, user_roles: set[str]) -> dict:
    """Run the RAG pipeline for a single question and return results.

    Uses get_retriever() rather than RBACRetriever directly, so this measures
    whatever the app actually serves — including re-ranking when
    RERANK_ENABLED=true (the default). Run with RERANK_ENABLED=false to get
    a before/after comparison, the same pattern used for the RETRIEVAL_TOP_K
    tuning run.

    Generation goes through generate_grounded_answer(), the same entry point
    the REST API and MCP server use. Citation parsing and confidence scoring
    therefore measure the shipped code path rather than a re-implementation
    that happens to agree with it today.

    Citation verification is forced on here regardless of the runtime
    default: the judge calls are the cost of the citation_accuracy metric,
    and a run that skipped them would record `null` and look like a metric
    that failed rather than one that was never asked for.
    """
    from src.generation.answer import generate_grounded_answer
    from src.retrieval.retriever import get_retriever, retrieve_scored

    retriever = get_retriever(user_roles=user_roles)
    scored = retrieve_scored(retriever, question)

    contexts = [item.document.page_content for item in scored]

    if not scored:
        # Retrieval returned nothing at all — most often an RBAC filter that
        # excludes every department the question's subject lives in. Reported
        # as a refusal rather than an error, because that is what it is: the
        # system declined for want of anything it was allowed to read.
        return {
            "question": question,
            "answer": "No relevant documents found.",
            "contexts": [],
            "citation_accuracy": None,
            "citation_coverage": 0.0,
            "confidence": 0.0,
            "confidence_label": "low",
            "citations": None,
            "unanswered": {"reason": "no_documents_retrieved"},
            "generated": False,
        }

    grounded = generate_grounded_answer(question, scored, verify=True)

    return {
        "question": question,
        "answer": grounded.answer,
        "contexts": contexts,
        # Not RAGAS metrics — computed here and merged into the per-question
        # rows below. RAGAS scores the answer against a ground truth; these
        # score the answer against the chunks it claims to have used, which
        # is a different question and needs a different apparatus.
        "citation_accuracy": grounded.citations.accuracy,
        "citation_coverage": grounded.citations.coverage,
        "confidence": grounded.confidence.overall,
        # Carried separately from the numeric composite because the label is
        # the thing refusal_correctness asserts on: a refusal over good
        # retrieval can score into "medium" numerically while the label
        # correctly stays "low". See src/generation/confidence.py.
        "confidence_label": grounded.confidence.label,
        "citations": grounded.citations.as_dict(),
        "unanswered": grounded.unanswered.as_dict() if grounded.unanswered else None,
        # False when the low-confidence gate fired and no LLM call was made,
        # which is the difference between "the model declined" and "the model
        # was never asked" — different fixes, and indistinguishable from the
        # answer text alone.
        "generated": grounded.generated,
    }


def _refusal_correctness(result: dict, item: dict) -> float | None:
    """Did the system answer when it should, and decline when it should?

    Scored on both populations, because abstention has two failure modes and
    a metric that watched only one would reward the other. Over-refusal is
    the live risk here: the first measured run declined "What was Apple's
    total net revenue" — a figure that is in the index — because the
    cross-encoder scored every candidate below the insufficient-context
    threshold. Nothing in the RAGAS four says that happened; answer relevancy
    just reads 0.000, the same as a genuine miss.

    A refusal counts as correct only if it is *labelled* low confidence as
    well as structured. Returning the "I don't know" report while still
    claiming to be confident is a distinct bug from failing to refuse.

    None for a row that errored: no behaviour was observed, and scoring it 0
    would blame the system for the harness falling over.
    """
    if result.get("error"):
        return None

    refused = result.get("unanswered") is not None
    if expected_behavior(item) == REFUSE:
        return 1.0 if refused and result.get("confidence_label") == "low" else 0.0
    return 0.0 if refused else 1.0


def _weight_ratio(dense: float, sparse: float) -> str:
    """Fingerprint the fusion weighting by its ratio.

    RRF output is a ranking, so scaling both weights reorders nothing — 0.7/0.3
    and 7/3 are the same configuration. Recording the raw pair would make them
    fingerprint as two different runs, and `compare_eval_runs.py` would then
    refuse to call their spread a noise floor when that is exactly what it is.
    """
    if sparse == 0:
        return "dense_only" if dense else "none"
    return f"{dense / sparse:.4g}:1"


def _corpus_fingerprint() -> dict:
    """Identify the indexed corpus a run was measured against.

    Config tagging alone is not enough. Fixing the EDGAR parser took the index
    from 6,394 chunks to 9,572 — JPM and MSFT had been ~90% missing — so every
    earlier result measured a different corpus while recording an identical
    config. Two such runs are indistinguishable after the fact and read as
    noise. Chunk count catches a re-ingest; the digest catches a re-ingest that
    kept the count but changed the text.
    """
    import hashlib

    from src.retrieval.vector_store import get_vector_store

    documents = get_vector_store().get_all_documents()
    digest = hashlib.sha256()
    for content_hash in sorted(
        hashlib.sha256(d.page_content.encode("utf-8")).hexdigest() for d in documents
    ):
        digest.update(content_hash.encode("ascii"))

    return {"chunk_count": len(documents), "content_digest": digest.hexdigest()[:16]}


def _as_score(value) -> float | None:  # noqa: ANN001
    """RAGAS emits NaN when a metric can't be computed for a row; JSON can't
    represent that, and 0.0 would silently read as a genuine zero score."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN != NaN


def evaluate_with_ragas(results: list[dict], eval_data: list[dict]) -> tuple[dict, dict[int, dict]]:
    """Run RAGAS over the answerable questions.

    Returns (aggregate scores, {eval_data index: {metric: score}}).

    Keyed by index rather than returned as a parallel list because the two are
    no longer the same length: questions expecting a refusal are not sent to
    RAGAS at all. They cost judge calls to produce four numbers that mean
    nothing on a refusal — faithfulness over an answer that asserts nothing,
    context recall against a ground truth that says the corpus is silent —
    and any of them landing in a pooled mean would punish correct behaviour.

    The index is carried through explicitly so nothing downstream has to
    assume the rows line up by position. That assumption used to hold only
    because main() appends an error row for every question that raises, and
    it broke silently — every metric attached to the wrong question — for any
    filtering added anywhere in the pipeline. Filtering is now exactly what
    this function does.
    """
    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            # Deprecated in RAGAS 0.4.x in favour of ragas.metrics.collections,
            # which exposes a module rather than the metric instance
            # evaluate() takes. Keep this import until that migration is done
            # as its own change — swapping metric implementations mid-baseline
            # would move scores for a reason that is not the system.
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        from src.config import settings
        from src.generation.llm_factory import create_llm

        # Judge model — defaults to OpenAI gpt-4o-mini (the historical
        # baseline in evaluation/results/); EVAL_JUDGE_PROVIDER overrides it
        # for a run where that's unavailable. See config.py for the caveat.
        # Pinned to eval_judge_model, not the runtime generation model, so
        # the apparatus stays fixed while the system under test changes.
        print(f"Judge model: {settings.eval_judge_provider.value}/{settings.eval_judge_model}")
        ragas_llm = LangchainLLMWrapper(
            create_llm(
                settings.eval_judge_provider,
                model=settings.eval_judge_model,
                # Not the 1024 generation gets: faithfulness emits one verdict
                # per extracted statement, and an overrun drops the row.
                max_tokens=settings.eval_judge_max_tokens,
            )
        )
        # Fixed local judge embeddings — see EVAL_EMBEDDING_MODEL in eval_config.
        ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=EVAL_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        )

        # Only the answerable questions. `scored` keeps the original index
        # alongside each row so the scores can be put back where they belong.
        # Errored rows are excluded alongside the no-answer ones: the "answer"
        # is an exception message, and grading it would fold a harness failure
        # into the score as if the system had produced a bad answer.
        scored = [
            (i, results[i])
            for i, item in enumerate(eval_data)
            if i < len(results) and scored_by_ragas(item) and not results[i].get("error")
        ]
        skipped = len(eval_data) - len(scored)
        if skipped:
            print(f"Excluding {skipped} question(s) from RAGAS — see NO_ANSWER_SCORING")

        if not scored:
            return {}, {}

        ragas_data = {
            "question": [r["question"] for _, r in scored],
            "answer": [r["answer"] for _, r in scored],
            "contexts": [r["contexts"] for _, r in scored],
            "ground_truth": [eval_data[i]["ground_truth"] for i, _ in scored],
        }

        dataset = Dataset.from_dict(ragas_data)

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ]

        score = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        # RAGAS 0.4.x: use to_pandas() and take column means
        try:
            df = score.to_pandas()
            metric_cols = [c for c in df.columns if c in set(RAGAS_METRIC_NAMES)]
            # Mean over the rows that produced a score. A metric RAGAS could
            # not compute comes back NaN and is left out rather than counted
            # as zero — see _as_score.
            aggregate = {}
            for column in metric_cols:
                values = [v for v in (_as_score(v) for v in df[column]) if v is not None]
                if values:
                    aggregate[column] = sum(values) / len(values)
                    aggregate[f"{column}_n"] = len(values)

            # Keep the per-question scores. An aggregate cannot distinguish a
            # broad regression from two catastrophic queries dragging the mean
            # down, and at n=20 that difference decides whether a delta means
            # anything. Discarding these is why earlier ablations could only be
            # argued about in aggregate.
            by_index: dict[int, dict] = {}
            for position, (_, row) in enumerate(df.iterrows()):
                index, result = scored[position]
                # RAGAS returns one row per input in input order. Verified
                # rather than assumed, because a silent reordering would
                # attach every score to the wrong question and still produce a
                # plausible-looking file.
                returned = str(df["user_input"].iloc[position])
                if returned.strip() != result["question"].strip():
                    raise RuntimeError(
                        "RAGAS returned rows out of order — refusing to join "
                        f"scores by position.\n  expected: {result['question']!r}"
                        f"\n  got:      {returned!r}"
                    )
                by_index[index] = {c: _as_score(row[c]) for c in metric_cols}

            return aggregate, by_index
        except RuntimeError:
            raise
        except Exception:
            # Fallback: try attribute access on Result object
            return {
                name: float(getattr(score, name, 0.0) or 0.0) for name in RAGAS_METRIC_NAMES
            }, {}

    except ImportError:
        print("RAGAS not installed. Install with: pip install 'rag-enterprise[eval]'")
        print("Running basic evaluation instead...")
        return run_basic_evaluation(results, eval_data), {}


def run_basic_evaluation(results: list[dict], eval_data: list[dict]) -> dict:
    """Basic evaluation without RAGAS — checks if answers contain key terms."""
    scores = {
        "questions_evaluated": len(results),
        "answers_with_context": sum(1 for r in results if r["contexts"]),
        "non_empty_answers": sum(1 for r in results if r["answer"] and len(r["answer"]) > 20),
    }

    # Simple keyword overlap scoring
    keyword_scores = []
    for result, eval_item in zip(results, eval_data, strict=False):
        gt_words = set(eval_item["ground_truth"].lower().split())
        answer_words = set(result["answer"].lower().split())
        if gt_words:
            overlap = len(gt_words & answer_words) / len(gt_words)
            keyword_scores.append(overlap)

    if keyword_scores:
        scores["avg_keyword_overlap"] = sum(keyword_scores) / len(keyword_scores)

    return scores


def _aggregate_citation_metrics(rows: list[dict]) -> dict:
    """Mean each citation metric over the questions that produced one.

    Questions with no citations at all report citation_accuracy=None and are
    left out of that mean rather than counted as zero. An answer that cited
    nothing has no citation accuracy — its failure shows up in coverage,
    which is where it belongs. Averaging a None as 0.0 would let a refusal
    depress the metric that is supposed to describe the citations that exist.

    Takes the answerable rows only, for the same reason RAGAS does. A correct
    refusal cites nothing and is labelled low confidence; folding its 0.0
    coverage into the mean would make the citation numbers fall as the
    no-answer stratum grows, which describes the dataset rather than the
    system.
    """
    aggregate: dict[str, float | int] = {}
    for metric in CITATION_METRIC_NAMES:
        values = [r[metric] for r in rows if r.get(metric) is not None]
        if values:
            aggregate[metric] = sum(values) / len(values)
            aggregate[f"{metric}_n"] = len(values)
    return aggregate


def _build_per_question(
    eval_data: list[dict],
    results: list[dict],
    ragas_by_index: dict[int, dict],
) -> list[dict]:
    """One row per question, whatever it was scored on.

    Every question appears, including the ones RAGAS never saw — a no-answer
    question that vanished from the report would take the whole refusal path
    with it, which is the gap Phase 4 exists to close. Its RAGAS columns are
    None, which the stratified table renders as "-" and the aggregates skip.

    `results` is index-aligned with `eval_data` by construction: main() appends
    an error row for any question that raises. Everything else joins through
    that index rather than through position in some later list.
    """
    rows = []
    for i, item in enumerate(eval_data):
        result = results[i] if i < len(results) else {}
        rows.append(
            {
                "question": item["question"],
                "question_type": item.get("question_type"),
                "expected_behavior": expected_behavior(item),
                **{metric: ragas_by_index.get(i, {}).get(metric) for metric in RAGAS_METRIC_NAMES},
                **{metric: result.get(metric) for metric in CITATION_METRIC_NAMES},
                "refusal_correctness": _refusal_correctness(result, item),
                # Which path declined, when one did. low_retrieval_confidence
                # means the gate fired before generation and no LLM call was
                # spent; model_refused means retrieval looked fine and the
                # model still declined. They point at different fixes.
                "refusal_reason": (result.get("unanswered") or {}).get("reason"),
                "generated": result.get("generated"),
            }
        )
    return rows


def _print_stratified(per_question: list[dict]) -> None:
    """Break the aggregate down by question type and show the worst queries.

    A mean over 20 questions can't distinguish a broad regression from two
    queries collapsing, and those call for opposite responses.
    """
    if not per_question or not any(r.get("question_type") for r in per_question):
        return

    groups: dict[str, list[dict]] = {}
    for row in per_question:
        groups.setdefault(row.get("question_type") or "untyped", []).append(row)

    header = f"  {'stratum':14}{'n':>3}" + "".join(f"{m.split('_')[-1]:>12}" for m in METRIC_NAMES)
    print("\n" + "-" * len(header))
    print("BY QUESTION TYPE")
    print("-" * len(header))
    print(header)
    for qtype, rows in sorted(groups.items()):
        line = f"  {qtype:14}{len(rows):>3}"
        for metric in METRIC_NAMES:
            vals = [r[metric] for r in rows if r.get(metric) is not None]
            line += f"{(sum(vals) / len(vals)):>12.3f}" if vals else f"{'-':>12}"
        print(line)

    worst = sorted(
        (r for r in per_question if r.get("context_recall") is not None),
        key=lambda r: r["context_recall"],
    )[:3]
    if worst:
        print("\n  Lowest context_recall:")
        for row in worst:
            print(f"    {row['context_recall']:.2f}  {row['question'][:66]}")

    _print_abstention(per_question)


def _print_abstention(per_question: list[dict]) -> None:
    """Every question where the system answered the wrong kind of thing.

    Named separately from the metric because the aggregate hides which
    direction the errors run, and the two are opposite defects: declining a
    question the corpus answers is a retrieval or threshold problem, while
    answering one it does not is a grounding problem. A run can score 0.90
    refusal_correctness either way.
    """
    wrong = [r for r in per_question if r.get("refusal_correctness") == 0.0]
    if not wrong:
        return

    print("\n  Abstention errors:")
    for row in wrong:
        if row["expected_behavior"] == REFUSE:
            detail = "answered, should have declined"
        else:
            reason = row.get("refusal_reason") or "unknown"
            detail = f"declined ({reason}), should have answered"
        print(f"    {detail:48}  {row['question'][:58]}")


def main() -> None:
    print("=" * 60)
    print("RAG Enterprise - Evaluation Pipeline")
    print("=" * 60)

    # Load evaluation dataset
    eval_data = load_eval_dataset()
    print(f"\nLoaded {len(eval_data)} evaluation questions")

    # Run RAG pipeline for each question
    results: list[dict] = []
    for i, item in enumerate(eval_data):
        print(f"\n[{i + 1}/{len(eval_data)}] {item['question'][:60]}...")
        try:
            result = run_rag_pipeline(
                question=item["question"],
                user_roles=set(item["access_roles"]),
            )
            results.append(result)
            print(f"  Answer: {result['answer'][:80]}...")
            print(f"  Contexts retrieved: {len(result['contexts'])}")
        except Exception as e:
            print(f"  ERROR: {e}")
            # Appended so `results` stays index-aligned with `eval_data`; every
            # join downstream goes through that index.
            results.append(
                {
                    "question": item["question"],
                    "answer": f"Error: {e}",
                    "contexts": [],
                    "error": True,
                }
            )

    # Run evaluation
    print("\n" + "=" * 60)
    print("Running evaluation...")
    scores, ragas_by_index = evaluate_with_ragas(results, eval_data)
    per_question = _build_per_question(eval_data, results, ragas_by_index)

    # Citation metrics over the answerable questions only, on the same
    # reasoning as the RAGAS mask — see NO_ANSWER_SCORING.
    answerable = [row for row in per_question if row["expected_behavior"] == ANSWER]
    scores.update(_aggregate_citation_metrics(answerable))

    # Abstention is scored over everything: the two populations measure
    # opposite failures and both are real.
    behaviours = [
        row["refusal_correctness"]
        for row in per_question
        if row.get("refusal_correctness") is not None
    ]
    if behaviours:
        scores["refusal_correctness"] = sum(behaviours) / len(behaviours)
        scores["refusal_correctness_n"] = len(behaviours)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    _print_stratified(per_question)

    # Save results, tagged with the config that produced them — scores are
    # meaningless without knowing which retrieval stages were on and which
    # model judged them, and a results directory of untagged runs can't be
    # compared after the fact.
    from src.config import LLMProvider, settings
    from src.ingestion.embeddings import active_embedding_model_name

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {
                    # Every knob that changes what gets retrieved must appear
                    # here. Two runs recording the same config but different
                    # scores are worse than useless — they look like noise.
                    "hyde_enabled": settings.hyde_enabled,
                    "hyde_include_query": (
                        settings.hyde_include_query if settings.hyde_enabled else None
                    ),
                    "hybrid_search_enabled": settings.hybrid_search_enabled,
                    # Recorded as a ratio, not as the two raw values: only the
                    # ratio affects the ranking, so 0.7/0.3 and 7/3 are one
                    # configuration and must not fingerprint as two different
                    # runs. None when hybrid is off, where the weights are
                    # read but never reach a fusion call.
                    "hybrid_weight_ratio": (
                        _weight_ratio(settings.hybrid_dense_weight, settings.hybrid_sparse_weight)
                        if settings.hybrid_search_enabled
                        else None
                    ),
                    # Splits the retrieval budget across the companies a
                    # question names. Changes what gets retrieved for every
                    # comparative question and nothing else, so a run with it
                    # off is comparable to one with it on outside that stratum.
                    "multi_entity_retrieval_enabled": settings.multi_entity_retrieval_enabled,
                    "rerank_enabled": settings.rerank_enabled,
                    "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
                    "retrieval_top_k": settings.retrieval_top_k,
                    "rerank_candidate_k": (
                        settings.rerank_candidate_k if settings.rerank_enabled else None
                    ),
                    "chunking_strategy": settings.chunking_strategy.value,
                    "chroma_collection": settings.chroma_collection,
                    "generation_provider": settings.llm_provider.value,
                    "generation_model": (
                        settings.openai_model
                        if settings.llm_provider == LLMProvider.OPENAI
                        else None
                    ),
                    "judge_provider": settings.eval_judge_provider.value,
                    "judge_model": settings.eval_judge_model,
                    # Not a tuning knob — a coverage one. Below roughly 4096
                    # the judge overruns on long answers and RAGAS drops the
                    # row, so this changes which questions are in the mean.
                    "judge_max_tokens": settings.eval_judge_max_tokens,
                    "judge_embedding_model": EVAL_EMBEDDING_MODEL,
                    # How citations were judged. A separate judge from the
                    # RAGAS one, so it needs its own record — a citation
                    # accuracy figure means nothing without knowing what
                    # graded it, and this judge is free to move while the
                    # RAGAS one is pinned.
                    "citation_judge_provider": settings.citation_judge_provider.value,
                    "citation_judge_model": settings.citation_judge_model,
                    "insufficient_context_threshold": settings.insufficient_context_threshold,
                    "embedding_model": active_embedding_model_name(),
                    "eval_questions": EVAL_QUESTIONS_PATH.name,
                    # How questions expecting a refusal were scored. Recorded
                    # in full because a run that pooled them is not comparable
                    # to one that did not, and the difference is invisible in
                    # the numbers themselves.
                    "no_answer_scoring": NO_ANSWER_SCORING,
                },
                # Which corpus produced these scores — see _corpus_fingerprint.
                "corpus": _corpus_fingerprint(),
                "scores": {k: float(v) if isinstance(v, float) else v for k, v in scores.items()},
                "per_question_scores": per_question,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
