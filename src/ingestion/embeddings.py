from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.common.logging import get_logger

logger = get_logger(__name__)

_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info("loading_embedding_model", model=settings.embedding_model)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance
