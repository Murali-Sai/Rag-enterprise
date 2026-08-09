"""Tests for HybridRetriever — dense + BM25 fused by RRF.

The security-critical property: adding a lexical retrieval path must not
open a way around the Chinese Wall. BM25 is built from a department-filtered
corpus fetch, so it can only ever surface documents the dense path was also
allowed to see.
"""

from langchain_core.documents import Document

from src.config import settings
from src.retrieval.bm25 import reset_bm25_cache
from src.retrieval.retriever import HybridRetriever


class FakeVectorStore:
    """Records both the vector-search filter and the corpus-fetch filter."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.search_filter: dict | None = None
        self.corpus_filter: dict | None = None
        self.corpus_fetched = False

    def similarity_search(self, query: str, k: int = 5, filter_dict: dict | None = None):
        self.search_filter = filter_dict
        return self.docs[:k]

    def get_all_documents(self, filter_dict: dict | None = None):
        self.corpus_filter = filter_dict
        self.corpus_fetched = True
        return self.docs


class TestHybridRetrieverRBAC:
    def setup_method(self):
        reset_bm25_cache()

    def teardown_method(self):
        reset_bm25_cache()

    def test_bm25_corpus_uses_the_same_rbac_filter_as_dense_search(self):
        """If these two filters ever diverge, the lexical path becomes a
        Chinese Wall bypass."""
        store = FakeVectorStore([Document(page_content="revenue", metadata={})])

        retriever = HybridRetriever(user_roles={"research"}, vector_store=store)
        retriever.retrieve("revenue")

        assert store.corpus_filter == store.search_filter
        assert store.corpus_filter is not None  # research is a restricted role

    def test_research_role_filter_excludes_walled_departments(self):
        store = FakeVectorStore([Document(page_content="revenue", metadata={})])

        retriever = HybridRetriever(user_roles={"research"}, vector_store=store)
        retriever.retrieve("revenue")

        departments = store.corpus_filter["department"]["$in"]
        assert "trading" not in departments
        assert "compliance" not in departments
        assert "sec_filings" in departments

    def test_admin_gets_unfiltered_corpus_on_both_paths(self):
        store = FakeVectorStore([Document(page_content="revenue", metadata={})])

        retriever = HybridRetriever(user_roles={"admin"}, vector_store=store)
        retriever.retrieve("revenue")

        assert store.search_filter is None
        assert store.corpus_filter is None


class TestHybridRetrieverFusion:
    def setup_method(self):
        reset_bm25_cache()

    def teardown_method(self):
        reset_bm25_cache()

    def test_exact_term_document_surfaces_even_when_dense_ranks_it_last(self):
        """The reason hybrid exists: an exact-match chunk that dense retrieval
        buries should be pulled up by the lexical half."""
        exact = Document(page_content="Item 7A quantitative disclosures", metadata={})
        filler = [Document(page_content=f"unrelated prose {i}", metadata={}) for i in range(4)]
        # Dense returns the exact-match doc dead last.
        store = FakeVectorStore([*filler, exact])

        retriever = HybridRetriever(user_roles={"admin"}, vector_store=store, top_k=5)
        result = retriever.retrieve("Item 7A")

        assert result[0].page_content == "Item 7A quantitative disclosures"

    def test_returns_at_most_top_k(self):
        docs = [Document(page_content=f"revenue doc {i}", metadata={}) for i in range(10)]
        store = FakeVectorStore(docs)

        retriever = HybridRetriever(user_roles={"admin"}, vector_store=store, top_k=3)

        assert len(retriever.retrieve("revenue")) == 3

    def test_empty_corpus_returns_empty(self):
        store = FakeVectorStore([])

        retriever = HybridRetriever(user_roles={"admin"}, vector_store=store)

        assert retriever.retrieve("revenue") == []

    def test_results_are_deduplicated_across_both_paths(self):
        """Dense and BM25 will often return the same chunk; it must appear once."""
        docs = [Document(page_content="apple revenue grew", metadata={})]
        store = FakeVectorStore(docs)

        retriever = HybridRetriever(user_roles={"admin"}, vector_store=store, top_k=5)
        result = retriever.retrieve("apple revenue")

        assert len(result) == 1


class TestFusionWeighting:
    """The settings actually reaching the fusion call.

    `test_fusion.py` covers the weighting arithmetic. This covers the wiring,
    which is the half that fails silently: a knob that is read nowhere still
    accepts values, logs them, and changes nothing.
    """

    def setup_method(self):
        reset_bm25_cache()

    def teardown_method(self):
        reset_bm25_cache()

    def _retrieve_with(self, monkeypatch, dense_weight: float, sparse_weight: float):
        # Dense ranks the prose first; BM25 ranks the exact-term chunk first.
        exact = Document(page_content="Item 7A quantitative disclosures", metadata={})
        prose = Document(page_content="general discussion of market conditions", metadata={})
        store = FakeVectorStore([prose, exact])

        monkeypatch.setattr(settings, "hybrid_dense_weight", dense_weight)
        monkeypatch.setattr(settings, "hybrid_sparse_weight", sparse_weight)

        retriever = HybridRetriever(user_roles={"admin"}, vector_store=store, top_k=2)
        return retriever.retrieve("Item 7A")

    def test_favouring_dense_puts_the_dense_leader_first(self, monkeypatch):
        result = self._retrieve_with(monkeypatch, dense_weight=1.0, sparse_weight=0.0)

        assert result[0].page_content == "general discussion of market conditions"

    def test_favouring_sparse_puts_the_lexical_leader_first(self, monkeypatch):
        """Same corpus, same query, opposite order — so the setting is read."""
        result = self._retrieve_with(monkeypatch, dense_weight=0.0, sparse_weight=1.0)

        assert result[0].page_content == "Item 7A quantitative disclosures"

    def test_weights_are_read_per_call_not_captured_at_construction(self, monkeypatch):
        """Retrievers are built per request and the eval sweeps this between
        runs, so a value captured in __init__ would go stale mid-sweep."""
        store = FakeVectorStore(
            [
                Document(page_content="general discussion of market conditions", metadata={}),
                Document(page_content="Item 7A quantitative disclosures", metadata={}),
            ]
        )
        retriever = HybridRetriever(user_roles={"admin"}, vector_store=store, top_k=2)

        monkeypatch.setattr(settings, "hybrid_dense_weight", 0.0)
        monkeypatch.setattr(settings, "hybrid_sparse_weight", 1.0)
        after = retriever.retrieve("Item 7A")

        assert after[0].page_content == "Item 7A quantitative disclosures"
