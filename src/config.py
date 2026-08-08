from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    OPENSEARCH = "opensearch"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    llm_provider: LLMProvider = LLMProvider.GROQ
    groq_api_key: str = ""
    google_api_key: str = ""
    huggingface_api_key: str = ""
    openai_api_key: str = ""

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector Store
    vector_store_type: VectorStoreType = VectorStoreType.CHROMA
    chroma_persist_dir: str = "./chroma_data"
    opensearch_url: str = ""
    opensearch_index: str = "rag-enterprise"

    # Auth
    jwt_secret_key: str = "change-this-to-a-random-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Database
    database_url: str = "sqlite+aiosqlite:///./rag_enterprise.db"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    environment: Environment = Environment.DEVELOPMENT

    # Rate Limiting
    rate_limit: str = "20/minute"

    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    retrieval_top_k: int = 5

    # Re-ranking — a cross-encoder scores (query, doc) pairs jointly, which is
    # slower but substantially more precise than the bi-encoder cosine
    # similarity ChromaDB returns. Cheap because it only runs on a small
    # candidate set, not the whole corpus.
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_k: int = 20

    # Hybrid search — pairs dense embedding search with a BM25 lexical index
    # and fuses the rankings (RRF). Dense handles paraphrase; BM25 handles the
    # exact-match terms filings are full of (tickers, "Item 7A", dollar
    # figures) that a 384-dim embedding blurs together.
    #
    # Off by default: the ablation in README ("Retrieval Pipeline Ablation")
    # measured it lifting Faithfulness (+0.09) and Precision, but costing
    # Context Recall -0.23 — five times the run-to-run noise floor. Lexical
    # hits crowd semantically-relevant chunks out of a fixed fusion budget.
    # Turn on once the candidate budget or fusion weighting is tuned.
    hybrid_search_enabled: bool = False

    # RAGAS eval judge — separate from LLM_PROVIDER (the runtime generation
    # model) so switching the app's provider doesn't silently change what
    # judges eval scores. Defaults to OpenAI gpt-4o-mini, matching the
    # historical baseline in evaluation/results/. Override for a run where
    # OpenAI isn't available (e.g. EVAL_JUDGE_PROVIDER=gemini) — just note
    # scores from a different judge aren't directly comparable to the baseline.
    eval_judge_provider: LLMProvider = LLMProvider.OPENAI

    # SEC EDGAR
    edgar_user_agent: str = "RAGEnterprise murali140824@gmail.com"
    edgar_rate_limit: int = 10
    edgar_data_dir: str = "./data/edgar"

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent


settings = Settings()
