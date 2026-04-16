"""RAGAS evaluation pipeline for the RAG system.

Usage:
    python -m evaluation.run_evaluation
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluation.eval_config import DATASETS_DIR, EVAL_QUESTIONS_PATH, RESULTS_DIR


def load_eval_dataset() -> list[dict]:
    with open(EVAL_QUESTIONS_PATH) as f:
        return json.load(f)


def run_rag_pipeline(question: str, user_roles: set[str]) -> dict:
    """Run the RAG pipeline for a single question and return results."""
    from src.retrieval.retriever import RBACRetriever
    from src.generation.chains import query_with_context

    retriever = RBACRetriever(user_roles=user_roles)
    documents = retriever.retrieve(question)

    contexts = [doc.page_content for doc in documents]

    if not documents:
        return {
            "question": question,
            "answer": "No relevant documents found.",
            "contexts": [],
        }

    answer = query_with_context(question, documents)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }


def evaluate_with_ragas(results: list[dict], eval_data: list[dict]) -> dict:
    """Run RAGAS evaluation on the collected results."""
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        from src.config import settings

        # Use OpenAI gpt-4o-mini for RAGAS judging (reliable, supports n>1,
        # industry-standard judge model). Cost ~$0.30 per full 20-question run.
        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.openai_api_key,
                temperature=0.0,
            )
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        )

        # Build RAGAS dataset
        ragas_data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [e["ground_truth"] for e in eval_data[: len(results)]],
        }

        dataset = Dataset.from_dict(ragas_data)

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        score = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        # RAGAS 0.4.x: use to_pandas() and take column means
        try:
            df = score.to_pandas()
            metric_cols = [
                c for c in df.columns
                if c in {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
            ]
            return {c: float(df[c].mean()) for c in metric_cols}
        except Exception:
            # Fallback: try attribute access on Result object
            return {
                "faithfulness": float(getattr(score, "faithfulness", 0.0) or 0.0),
                "answer_relevancy": float(getattr(score, "answer_relevancy", 0.0) or 0.0),
                "context_precision": float(getattr(score, "context_precision", 0.0) or 0.0),
                "context_recall": float(getattr(score, "context_recall", 0.0) or 0.0),
            }

    except ImportError:
        print("RAGAS not installed. Install with: pip install 'rag-enterprise[eval]'")
        print("Running basic evaluation instead...")
        return run_basic_evaluation(results, eval_data)


def run_basic_evaluation(results: list[dict], eval_data: list[dict]) -> dict:
    """Basic evaluation without RAGAS — checks if answers contain key terms."""
    scores = {
        "questions_evaluated": len(results),
        "answers_with_context": sum(1 for r in results if r["contexts"]),
        "non_empty_answers": sum(1 for r in results if r["answer"] and len(r["answer"]) > 20),
    }

    # Simple keyword overlap scoring
    keyword_scores = []
    for result, eval_item in zip(results, eval_data):
        gt_words = set(eval_item["ground_truth"].lower().split())
        answer_words = set(result["answer"].lower().split())
        if gt_words:
            overlap = len(gt_words & answer_words) / len(gt_words)
            keyword_scores.append(overlap)

    if keyword_scores:
        scores["avg_keyword_overlap"] = sum(keyword_scores) / len(keyword_scores)

    return scores


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
        print(f"\n[{i+1}/{len(eval_data)}] {item['question'][:60]}...")
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
            results.append({
                "question": item["question"],
                "answer": f"Error: {e}",
                "contexts": [],
            })

    # Run evaluation
    print("\n" + "=" * 60)
    print("Running evaluation...")
    scores = evaluate_with_ragas(results, eval_data)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "scores": {k: float(v) if isinstance(v, float) else v for k, v in scores.items()},
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
