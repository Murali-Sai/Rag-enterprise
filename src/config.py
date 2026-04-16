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

    # SEC EDGAR
    edgar_user_agent: str = "RAGEnterprise research@example.com"
    edgar_rate_limit: int = 10
    edgar_data_dir: str = "./data/edgar"

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent


settings = Settings()
