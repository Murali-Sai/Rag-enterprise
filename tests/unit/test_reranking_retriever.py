"""Tests for RerankingRetriever wiring — RBAC filtering still happens first,
at the vector-store query level, and reranking only narrows that already
-authorized candidate set. No real Chroma or cross-encoder model is loaded.
"""

from unittest.mock import patch

from langchain_core.documents import Document

from src.retrieval.retriever import (
    HybridRetriever,
    RBACRetriever,
    RerankingRetriever,
    get_retriever,
)


class FakeVectorStore:
    """Stand-in for ChromaVectorStore — records the k it was asked for."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.last_k: int | None = None
        self.last_filter: dict | None = None

    def similarity_search(self, query: str, k: int = 5, filter_dict: dict | None = None):
        self.last_k = k
        self.last_filter = filter_dict
        return self.docs[:k]

    def get_all_documents(self, filter_dict: dict | None = None):
        return self.docs

    def add_documents(self, documents):  # pragma: no cover - unused in these tests
        raise NotImplementedError

    def delete(self, ids):  # pragma: no cover - unused in these tests
        raise NotImplementedError


class FakeCrossEncoder:
    def __init__(self, score_by_content: dict[str, float]):
        self.score_by_content = score_by_content

    def predict(self, pairs):
        return [self.score_by_content[doc_text] for _, doc_text in pairs]


class TestRerankingRetriever:
    def test_retrieves_candidate_k_then_cuts_to_final_top_k(self):
        docs = [Document(page_content=f"doc{i}") for i in range(20)]
        store = FakeVectorStore(docs)
        fake_encoder = FakeCrossEncoder({f"doc{i}": float(i) for i in range(20)})

        with patch("src.retrieval.reranker.get_reranker", return_value=fake_encoder):
            retriever = RerankingRetriever(
                user_roles={"admin"},
                vector_store=store,
                final_top_k=5,
                candidate_k=20,
            )
            result = retriever.retrieve("q")

        assert store.last_k == 20  # asked the vector store for the wide candidate set
        assert len(result) == 5  # but only returns the final top_k
        assert [d.page_content for d in result] == ["doc19", "doc18", "doc17", "doc16", "doc15"]

    def test_rbac_filter_still_applied_before_reranking(self):
        """Reranking must not be able to see documents outside the RBAC filter —
        it only reorders what RBACRetriever already returned."""
        docs = [Document(page_content="research doc")]
        store = FakeVectorStore(docs)
        fake_encoder = FakeCrossEncoder({"research doc": 1.0})

        with patch("src.retrieval.reranker.get_reranker", return_value=fake_encoder):
            retriever = RerankingRetriever(user_roles={"research"}, vector_store=store)
            retriever.retrieve("q")

        # research role -> department filter should have been built and passed through
        assert store.last_filter is not None

    def test_admin_gets_no_filter_but_reranking_still_applies(self):
        docs = [Document(page_content="a"), Document(page_content="b")]
        store = FakeVectorStore(docs)
        fake_encoder = FakeCrossEncoder({"a": 0.1, "b": 0.9})

        with patch("src.retrieval.reranker.get_reranker", return_value=fake_encoder):
            retriever = RerankingRetriever(user_roles={"admin"}, vector_store=store, final_top_k=1)
            result = retriever.retrieve("q")

        assert store.last_filter is None
        assert result[0].page_content == "b"


class TestGetRetrieverFactory:
    """Each stage toggles independently, so the eval harness can isolate them."""

    def test_rerank_only(self):
        with (
            patch("src.retrieval.retriever.settings.rerank_enabled", True),
            patch("src.retrieval.retriever.settings.hybrid_search_enabled", False),
        ):
            retriever = get_retriever(user_roles={"admin"}, vector_store=FakeVectorStore([]))

        assert isinstance(retriever, RerankingRetriever)
        assert isinstance(retriever._base, RBACRetriever)

    def test_neither_stage_gives_plain_dense_retriever(self):
        with (
            patch("src.retrieval.retriever.settings.rerank_enabled", False),
            patch("src.retrieval.retriever.settings.hybrid_search_enabled", False),
        ):
            retriever = get_retriever(user_roles={"admin"}, vector_store=FakeVectorStore([]))

        assert isinstance(retriever, RBACRetriever)
        assert not isinstance(retriever, RerankingRetriever)

    def test_hybrid_only(self):
        with (
            patch("src.retrieval.retriever.settings.rerank_enabled", False),
            patch("src.retrieval.retriever.settings.hybrid_search_enabled", True),
        ):
            retriever = get_retriever(user_roles={"admin"}, vector_store=FakeVectorStore([]))

        assert isinstance(retriever, HybridRetriever)

    def test_hybrid_then_rerank_is_the_full_pipeline(self):
        with (
            patch("src.retrieval.retriever.settings.rerank_enabled", True),
            patch("src.retrieval.retriever.settings.hybrid_search_enabled", True),
        ):
            retriever = get_retriever(user_roles={"admin"}, vector_store=FakeVectorStore([]))

        assert isinstance(retriever, RerankingRetriever)
        assert isinstance(retriever._base, HybridRetriever)

    def test_base_widens_to_candidate_k_only_when_reranking_follows(self):
        """Without a reranker there's nothing to narrow the candidates back
        down, so the first stage must return the final top_k directly."""
        with (
            patch("src.retrieval.retriever.settings.rerank_enabled", False),
            patch("src.retrieval.retriever.settings.hybrid_search_enabled", False),
            patch("src.retrieval.retriever.settings.retrieval_top_k", 5),
            patch("src.retrieval.retriever.settings.rerank_candidate_k", 20),
        ):
            retriever = get_retriever(user_roles={"admin"}, vector_store=FakeVectorStore([]))
        assert retriever.top_k == 5

        with (
            patch("src.retrieval.retriever.settings.rerank_enabled", True),
            patch("src.retrieval.retriever.settings.hybrid_search_enabled", False),
            patch("src.retrieval.retriever.settings.retrieval_top_k", 5),
            patch("src.retrieval.retriever.settings.rerank_candidate_k", 20),
        ):
            retriever = get_retriever(user_roles={"admin"}, vector_store=FakeVectorStore([]))
        assert retriever._base.top_k == 20
        assert retriever.final_top_k == 5
