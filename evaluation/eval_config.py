from pathlib import Path

# Paths
EVAL_DIR = Path(__file__).parent
DATASETS_DIR = EVAL_DIR / "datasets"
EVAL_QUESTIONS_PATH = DATASETS_DIR / "eval_questions_v2.json"
RESULTS_DIR = EVAL_DIR / "results"

# RAGAS metrics to compute
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# Evaluation settings
EVAL_LLM_PROVIDER = "groq"  # Use free LLM for evaluation too
EVAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
