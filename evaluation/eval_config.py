from pathlib import Path

# Paths
EVAL_DIR = Path(__file__).parent
DATASETS_DIR = EVAL_DIR / "datasets"
# v4 is v3's 20 questions unchanged, plus 34 written for Phase 4. It is the
# first version where the answer to a question can correctly be *no answer*:
# 16 of the 54 carry expected_behavior="refuse" and are scored on
# refusal_correctness instead of the RAGAS four, because RAGAS scores a
# correct refusal 0.000 answer relevancy. See run_evaluation.NO_ANSWER_SCORING.
#
# It is also the first version that exercises the RBAC layer. Every v3 question
# lists "admin", for which build_role_filter() returns None, so the ChromaDB
# where-clause was never applied on any eval question — the information barrier
# was covered by unit tests and by nothing in the eval. The 34 new questions
# name real roles.
#
# v3 ground truths are written from full filing text by
# scripts/generate_ground_truths_from_filings.py. v2 answered each question
# from this project's own top-20 retrieval, so the reference used to score
# retrieval was itself produced by retrieval — any section the retriever
# missed was missing from the reference too, and therefore uncounted. The v3
# rows are carried into v4 byte-for-byte so their scores stay comparable.
EVAL_QUESTIONS_PATH = DATASETS_DIR / "eval_questions_v4.json"
RESULTS_DIR = EVAL_DIR / "results"

# RAGAS metrics to compute
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# Judge embeddings for answer_relevancy. Deliberately NOT the corpus
# embedding model: scoring an OpenAI-embedded retrieval with the same OpenAI
# embeddings would let the judge share the retriever's blind spots, and it
# would move every historical score the moment the corpus model changed.
# A fixed local model keeps the measuring stick independent and comparable
# across runs. The judge LLM is settings.eval_judge_provider /
# settings.eval_judge_model.
EVAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
