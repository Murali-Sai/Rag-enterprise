from abc import ABC, abstractmethod

from langchain_core.documents import Document

from src.common.logging import get_logger
from src.config import VectorStoreType, settings

logger = get_logger(__name__)


class VectorStoreBase(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]: ...

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict | None = None,
    ) -> list[Document]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def get_all_documents(self, filter_dict: dict | None = None) -> list[Document]: ...


class ChromaVectorStore(VectorStoreBase):
    def __init__(self) -> None:
        from langchain_chroma import Chroma

        from src.ingestion.embeddings import get_embedding_model

        self._store = Chroma(
            collection_name="rag_enterprise",
            embedding_function=get_embedding_model(),
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info("chroma_initialized", persist_dir=settings.chroma_persist_dir)

    def add_documents(self, documents: list[Document]) -> list[str]:
        # Imported lazily — bm25 imports this module for its type hints.
        from src.retrieval.bm25 import reset_bm25_cache

        ids = self._store.add_documents(documents)
        # BM25 indexes are in-memory snapshots of the corpus; without this the
        # lexical half of hybrid search would keep serving a stale corpus while
        # dense search already sees the new documents.
        reset_bm25_cache()
        logger.info("documents_added", count=len(ids))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict | None = None,
    ) -> list[Document]:
        kwargs: dict = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        return self._store.similarity_search(query, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: dict | None = None,
    ) -> list[tuple[Document, float]]:
        kwargs: dict = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        return self._store.similarity_search_with_relevance_scores(query, **kwargs)

    def delete(self, ids: list[str]) -> None:
        self._store.delete(ids)

    def get_all_documents(self, filter_dict: dict | None = None) -> list[Document]:
        """Fetch whole documents (no vector search) for building a BM25 index.

        Takes the same where-filter dense retrieval uses, so the lexical index
        is built over exactly the documents the caller is allowed to see —
        RBAC stays enforced at the database level rather than being re-applied
        to a global index afterwards.
        """
        kwargs: dict = {"include": ["documents", "metadatas"]}
        if filter_dict:
            kwargs["where"] = filter_dict
        raw = self._store.get(**kwargs)

        contents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        documents = [
            Document(page_content=content, metadata=dict(metadata or {}))
            for content, metadata in zip(contents, metadatas, strict=False)
        ]
        logger.info("corpus_fetched", count=len(documents), filtered=filter_dict is not None)
        return documents

    @property
    def store(self):  # noqa: ANN201
        return self._store


_vector_store_instance: VectorStoreBase | None = None


def get_vector_store() -> VectorStoreBase:
    global _vector_store_instance
    if _vector_store_instance is None:
        if settings.vector_store_type == VectorStoreType.CHROMA:
            _vector_store_instance = ChromaVectorStore()
        else:
            raise ValueError(f"Unsupported vector store: {settings.vector_store_type}")
    return _vector_store_instance


def reset_vector_store() -> None:
    global _vector_store_instance
    _vector_store_instance = None
