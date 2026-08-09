"""Embedding model factory.

Two providers, selected by EMBEDDING_PROVIDER:

  openai       text-embedding-3-small (default) — 1536-dim, 8k-token window.
  huggingface  sentence-transformers/all-MiniLM-L6-v2 — 384-dim, 256-token
               window, runs locally with no API key.

The vector dimensionality differs between them, so a Chroma collection built
under one provider cannot be queried under the other. Switching means
rebuilding the index; get_embedding_model() caches per-process, so a switch
also needs reset_embedding_model() (or a fresh process) to take effect.
"""

from langchain_core.embeddings import Embeddings

from src.common.logging import get_logger
from src.config import EmbeddingProvider, settings

logger = get_logger(__name__)

_embeddings_instance: Embeddings | None = None


def create_embedding_model() -> Embeddings:
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY. Set the key, or set "
                "EMBEDDING_PROVIDER=huggingface to embed locally (note: a collection "
                "built with one provider must be rebuilt to use the other)."
            )

        logger.info(
            "loading_embedding_model",
            provider="openai",
            model=settings.openai_embedding_model,
        )
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("loading_embedding_model", provider="huggingface", model=settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        # Normalising here means a dot product is cosine similarity, which
        # semantic_chunking relies on. OpenAI returns unit vectors already.
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embedding_model() -> Embeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = create_embedding_model()
    return _embeddings_instance


def reset_embedding_model() -> None:
    """Drop the cached model — for tests and for scripts that switch provider."""
    global _embeddings_instance
    _embeddings_instance = None


def active_embedding_model_name() -> str:
    """Provider-qualified model id, for tagging eval results and logs."""
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        return f"openai/{settings.openai_embedding_model}"
    return f"huggingface/{settings.embedding_model}"
