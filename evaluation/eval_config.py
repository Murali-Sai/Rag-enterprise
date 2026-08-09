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

# RAGAS metrics to compute.
#
# answer_correctness is the fourth metric Project 6 Phase 4.2 asks for
# ("answer correctness — LLM-as-judge against golden answer"); the other three
# it names map to faithfulness, context precision/recall and citation
# accuracy. It was missing until 2026-08-09, so every result file before then
# has no column for it — a gap, not a zero.
#
# It is the only metric here that grades the answer against the ground truth
# rather than against the retrieved context, which makes it the one that can
# catch an answer that is faithful to the wrong passages: internally grounded,
# correctly cited, and not what the filing says.
#
# It can go slightly negative. RAGAS blends a factual-claim F1 with a cosine
# similarity between answer and ground truth, and the cosine term can be a
# small negative number, so a wholly wrong answer lands near -0.01 rather than
# at 0.000 (measured: a fabricated figure scored -0.007). Left unclamped —
# squashing it to zero would hide that the floor is empirical, not defined.
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

# Judge embeddings for answer_relevancy. Deliberately NOT the corpus
# embedding model: scoring an OpenAI-embedded retrieval with the same OpenAI
# embeddings would let the judge share the retriever's blind spots, and it
# would move every historical score the moment the corpus model changed.
# A fixed local model keeps the measuring stick independent and comparable
# across runs. The judge LLM is settings.eval_judge_provider /
# settings.eval_judge_model.
EVAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
