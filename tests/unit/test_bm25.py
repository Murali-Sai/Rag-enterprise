"""Tests for the BM25 lexical index.

These use the real rank_bm25 implementation over a tiny corpus — it's pure
Python with no model download, so there's nothing worth mocking.
"""

from langchain_core.documents import Document

from src.retrieval.bm25 import BM25Index, get_bm25_index, reset_bm25_cache, tokenize


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Apple REVENUE") == ["apple", "revenue"]

    def test_keeps_section_labels_as_tokens(self):
        """'Item 7A' must survive as a matchable token — it's one of the
        exact-match terms dense embeddings handle badly."""
        assert tokenize("Item 7A") == ["item", "7a"]

    def test_keeps_decimal_figures_intact(self):
        """$391.0 must not fragment into '391' and '0'."""
        assert "391.0" in tokenize("revenue was $391.0 billion")

    def test_strips_punctuation(self):
        assert tokenize("net-sales, (2024)") == ["net", "sales", "2024"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestBM25Index:
    def test_exact_ticker_match_ranks_first(self):
        docs = [
            Document(page_content="General discussion of market conditions."),
            Document(page_content="AAPL reported strong iPhone sales."),
        ]
        index = BM25Index(docs)

        result = index.search("AAPL", k=2)

        assert result[0].page_content.startswith("AAPL")

    def test_matches_ticker_from_metadata_not_just_body(self):
        """A table-row chunk names the company only in metadata; BM25 indexes
        the same ticker-prefixed rendering the reranker scores.

        Uses a five-company corpus so the ticker is a rare term with real IDF —
        on a two-document corpus every term is in half the corpus and BM25 has
        no discriminative power at all.
        """
        docs = [
            Document(
                page_content="Net revenue was $151 million.",
                metadata={"ticker": ticker, "section_name": "MD&A"},
            )
            for ticker in ("GS", "JPM", "MSFT", "TSLA")
        ]
        docs.append(
            Document(
                page_content="Net revenue was $391 million.",
                metadata={"ticker": "AAPL", "section_name": "MD&A"},
            )
        )
        index = BM25Index(docs)

        result = index.search("AAPL revenue", k=5)

        assert result[0].metadata["ticker"] == "AAPL"

    def test_excludes_documents_sharing_no_query_term(self):
        docs = [
            Document(page_content="completely unrelated content"),
            Document(page_content="revenue figures for the year"),
        ]
        index = BM25Index(docs)

        result = index.search("revenue", k=10)

        assert len(result) == 1
        assert result[0].page_content == "revenue figures for the year"

    def test_matching_document_survives_zero_or_negative_bm25_score(self):
        """BM25Okapi's IDF is log((N-n+0.5)/(n+0.5)) — exactly 0 when a term
        appears in half the corpus, negative when more common. Filtering on
        score sign would silently drop real matches on small corpora."""
        docs = [
            Document(page_content="revenue was strong"),
            Document(page_content="revenue was weak"),
        ]
        index = BM25Index(docs)

        # "revenue" is in 2 of 2 docs -> negative IDF for every candidate.
        result = index.search("revenue", k=10)

        assert len(result) == 2

    def test_respects_k(self):
        docs = [Document(page_content=f"revenue item {i}") for i in range(10)]
        index = BM25Index(docs)

        assert len(index.search("revenue", k=3)) == 3

    def test_empty_corpus_returns_empty(self):
        assert BM25Index([]).search("anything", k=5) == []

    def test_query_with_no_usable_tokens_returns_empty(self):
        index = BM25Index([Document(page_content="revenue")])

        assert index.search("!!!", k=5) == []


class FakeVectorStore:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.get_all_calls: list[dict | None] = []

    def get_all_documents(self, filter_dict: dict | None = None) -> list[Document]:
        self.get_all_calls.append(filter_dict)
        return self.docs


class TestIndexCaching:
    def setup_method(self):
        reset_bm25_cache()

    def teardown_method(self):
        reset_bm25_cache()

    def test_index_is_cached_per_access_level(self):
        store = FakeVectorStore([Document(page_content="revenue")])
        departments = frozenset({"sec_filings"})

        get_bm25_index(store, departments, {"department": {"$eq": "sec_filings"}})
        get_bm25_index(store, departments, {"department": {"$eq": "sec_filings"}})

        assert len(store.get_all_calls) == 1  # second call served from cache

    def test_different_access_levels_get_separate_indexes(self):
        """A trader's index must not be reused for a research analyst — that
        would leak trading documents through the lexical path."""
        store = FakeVectorStore([Document(page_content="revenue")])

        get_bm25_index(store, frozenset({"sec_filings"}), {"department": {"$eq": "sec_filings"}})
        get_bm25_index(store, frozenset({"trading"}), {"department": {"$eq": "trading"}})

        assert len(store.get_all_calls) == 2

    def test_corpus_fetch_passes_rbac_filter_through(self):
        store = FakeVectorStore([Document(page_content="revenue")])
        role_filter = {"department": {"$in": ["research", "sec_filings"]}}

        get_bm25_index(store, frozenset({"research", "sec_filings"}), role_filter)

        assert store.get_all_calls == [role_filter]

    def test_admin_cache_key_is_none_and_fetches_unfiltered(self):
        store = FakeVectorStore([Document(page_content="revenue")])

        get_bm25_index(store, None, None)

        assert store.get_all_calls == [None]

    def test_reset_clears_cache(self):
        store = FakeVectorStore([Document(page_content="revenue")])
        departments = frozenset({"sec_filings"})

        get_bm25_index(store, departments, None)
        reset_bm25_cache()
        get_bm25_index(store, departments, None)

        assert len(store.get_all_calls) == 2
