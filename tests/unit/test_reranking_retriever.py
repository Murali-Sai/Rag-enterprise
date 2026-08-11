"""Tests for RerankingRetriever wiring — RBAC filtering still happens first,
at the vector-store query level, and reranking only narrows that already
-authorized candidate set. No real Chroma or cross-encoder model is loaded.
"""

import contextlib
from unittest.mock import patch

from langchain_core.documents import Document

from src.retrieval.bm25 import reset_bm25_cache
from src.retrieval.retriever import (
    HybridRetriever,
    MultiEntityRetriever,
    RBACRetriever,
    RerankingRetriever,
    get_retriever,
    retrieve_scored,
)
from src.retrieval.scores import ScoreType


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


class FilterRecordingStore(FakeVectorStore):
    """Serves documents by the ticker in the where-clause, and records which
    tickers were asked for — the per-entity split is only doing its job if
    each named company gets its own trip to the store."""

    def __init__(self, docs_by_ticker: dict[str, list[Document]]):
        super().__init__([doc for docs in docs_by_ticker.values() for doc in docs])
        self.docs_by_ticker = docs_by_ticker
        self.tickers_queried: list[str] = []

    def similarity_search(self, query: str, k: int = 5, filter_dict: dict | None = None):
        self.last_k = k
        self.last_filter = filter_dict
        ticker = (filter_dict or {}).get("ticker", {}).get("$eq")
        if ticker is None:
            return self.docs[:k]
        self.tickers_queried.append(ticker)
        return self.docs_by_ticker.get(ticker, [])[:k]


class FakeCrossEncoder:
    def __init__(self, score_by_content: dict[str, float]):
        self.score_by_content = score_by_content

    def predict(self, pairs):
        return [self._score(doc_text) for _, doc_text in pairs]

    def _score(self, doc_text: str) -> float:
        # Falls back to substring lookup because the reranker scores
        # passage_text(), which prefixes the chunk with its provenance
        # ("JPM: jpm credit") rather than passing page_content through raw.
        if doc_text in self.score_by_content:
            return self.score_by_content[doc_text]
        for content, score in self.score_by_content.items():
            if content in doc_text:
                return score
        raise KeyError(doc_text)


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


class TestScoreChannel:
    """The scores the reranker computes have to reach the confidence layer.

    They used to be discarded inside rerank_documents, which left the system
    with no way to say how sure it was — and so no threshold below which the
    honest answer is "I don't know".
    """

    def test_reranking_retriever_hands_back_cross_encoder_scores(self):
        docs = [Document(page_content="a"), Document(page_content="b")]
        store = FakeVectorStore(docs)
        fake_encoder = FakeCrossEncoder({"a": -4.0, "b": 7.5})

        with patch("src.retrieval.reranker.get_reranker", return_value=fake_encoder):
            retriever = RerankingRetriever(user_roles={"admin"}, vector_store=store, final_top_k=2)
            scored = retriever.retrieve_scored("q")

        assert [item.score for item in scored] == [7.5, -4.0]
        assert all(item.score_type is ScoreType.CROSS_ENCODER for item in scored)
        # A logit squashes to a relevance the confidence layer can threshold.
        assert scored[0].relevance > 0.99
        assert scored[1].relevance < 0.02

    def test_documents_only_view_is_unchanged(self):
        docs = [Document(page_content="a"), Document(page_content="b")]
        store = FakeVectorStore(docs)
        fake_encoder = FakeCrossEncoder({"a": 0.1, "b": 0.9})

        with patch("src.retrieval.reranker.get_reranker", return_value=fake_encoder):
            retriever = RerankingRetriever(user_roles={"admin"}, vector_store=store, final_top_k=1)
            result = retriever.retrieve("q")

        assert [d.page_content for d in result] == ["b"]

    def test_a_retriever_without_the_channel_degrades_to_unscored(self):
        """Not to zero — a foreign retriever that cannot score must read as
        'unknown', or the confidence layer would treat it as certain junk."""

        class ScorelessRetriever:
            def retrieve(self, query: str):  # noqa: ANN201, ARG002
                return [Document(page_content="a")]

        scored = retrieve_scored(ScorelessRetriever(), "q")

        assert len(scored) == 1
        assert scored[0].score is None
        assert scored[0].relevance is None

    def test_hybrid_results_are_tagged_ordinal(self):
        docs = [Document(page_content="a"), Document(page_content="b")]
        retriever = HybridRetriever(user_roles={"admin"}, vector_store=FakeVectorStore(docs))

        reset_bm25_cache()
        try:
            scored = retrieve_scored(retriever, "q")
        finally:
            reset_bm25_cache()

        assert all(item.score_type is ScoreType.RRF for item in scored)
        assert all(item.relevance is None for item in scored)


def single_entity_pipeline(**overrides):
    """Build get_retriever's pipeline with the per-entity split switched off.

    The split wraps the whole pipeline and decides per query, so the stage
    composition it wraps is not reachable from the returned object. These
    tests are about that composition; the split has its own class below.
    """
    patches = {"multi_entity_retrieval_enabled": False, **overrides}
    with contextlib.ExitStack() as stack:
        for name, value in patches.items():
            stack.enter_context(patch(f"src.retrieval.retriever.settings.{name}", value))
        return get_retriever(user_roles={"admin"}, vector_store=FakeVectorStore([]))


class TestGetRetrieverFactory:
    """Each stage toggles independently, so the eval harness can isolate them."""

    def test_rerank_only(self):
        retriever = single_entity_pipeline(rerank_enabled=True, hybrid_search_enabled=False)

        assert isinstance(retriever, RerankingRetriever)
        assert isinstance(retriever._base, RBACRetriever)

    def test_neither_stage_gives_plain_dense_retriever(self):
        retriever = single_entity_pipeline(rerank_enabled=False, hybrid_search_enabled=False)

        assert isinstance(retriever, RBACRetriever)
        assert not isinstance(retriever, RerankingRetriever)

    def test_hybrid_only(self):
        retriever = single_entity_pipeline(rerank_enabled=False, hybrid_search_enabled=True)

        assert isinstance(retriever, HybridRetriever)

    def test_hybrid_then_rerank_is_the_full_pipeline(self):
        retriever = single_entity_pipeline(rerank_enabled=True, hybrid_search_enabled=True)

        assert isinstance(retriever, RerankingRetriever)
        assert isinstance(retriever._base, HybridRetriever)

    def test_base_widens_to_candidate_k_only_when_reranking_follows(self):
        """Without a reranker there's nothing to narrow the candidates back
        down, so the first stage must return the final top_k directly."""
        retriever = single_entity_pipeline(
            rerank_enabled=False,
            hybrid_search_enabled=False,
            retrieval_top_k=5,
            rerank_candidate_k=20,
        )
        assert retriever.top_k == 5

        retriever = single_entity_pipeline(
            rerank_enabled=True,
            hybrid_search_enabled=False,
            retrieval_top_k=5,
            rerank_candidate_k=20,
        )
        assert retriever._base.top_k == 20
        assert retriever.final_top_k == 5


class TestMultiEntityRetrieval:
    """A comparison is two questions and needs two questions' worth of budget.

    Under a global top-k the ranker filled every slot from whichever filing
    scored better overall, so the model could only report that it was unable
    to compare — and a refusal scores 0.000 answer relevancy, which reads as
    a generation failure rather than the retrieval-budget bug it is.
    """

    def test_one_company_named_is_filtered_to_that_company(self):
        """This used to assert the opposite — that naming one company added no
        entity clause — on the reasoning that a single-company question needs
        no budget split. It needs no split; it needs the filter.

        Unfiltered, "What was Goldman Sachs' total net revenues and return on
        average common shareholders' equity" retrieved two GS chunks, one
        Tesla, one JPMorgan, and one from a fabricated sample document reading
        "Total Net Revenue: $38.2B / Return on Equity (ROE): 14.8%". Nothing in
        a ground truth about Goldman Sachs can be supported by those, which is
        how GS scored a hard 0.000 context recall while every other company
        scored between 0.167 and 0.506.
        """
        store = FakeVectorStore([Document(page_content="a")])
        retriever = get_retriever(user_roles={"admin"}, vector_store=store)

        with patch(
            "src.retrieval.reranker.get_reranker", return_value=FakeCrossEncoder({"a": 1.0})
        ):
            retriever.retrieve("What was Apple's total net revenue?")

        assert store.last_filter == {"ticker": {"$eq": "AAPL"}}

    def test_the_entity_filter_is_added_to_the_barrier_not_instead_of_it(self):
        """The one way this change could do real damage.

        A ticker clause that replaced the department clause would drop the
        information barrier — the control this whole system exists to
        demonstrate — while every retrieval test still passed, because the
        documents come back either way.
        """
        store = FakeVectorStore([Document(page_content="a")])
        retriever = get_retriever(user_roles={"research"}, vector_store=store)

        with patch(
            "src.retrieval.reranker.get_reranker", return_value=FakeCrossEncoder({"a": 1.0})
        ):
            retriever.retrieve("What was Apple's total net revenue?")

        assert store.last_filter == {
            "$and": [
                {"department": {"$in": ["general", "research", "sec_filings"]}},
                {"ticker": {"$eq": "AAPL"}},
            ]
        }

    def test_a_question_naming_nobody_still_searches_everything(self):
        """The only case where an unfiltered search is what was asked for."""
        store = FakeVectorStore([Document(page_content="a")])
        retriever = get_retriever(user_roles={"admin"}, vector_store=store)

        with patch(
            "src.retrieval.reranker.get_reranker", return_value=FakeCrossEncoder({"a": 1.0})
        ):
            retriever.retrieve("What are the main risks disclosed in these filings?")

        assert store.last_filter is None

    def test_two_companies_each_get_their_own_filtered_retrieval(self):
        store = FilterRecordingStore(
            {
                "JPM": [Document(page_content="jpm credit", metadata={"ticker": "JPM"})],
                "GS": [Document(page_content="gs credit", metadata={"ticker": "GS"})],
            }
        )
        encoder = FakeCrossEncoder({"jpm credit": 2.0, "gs credit": 6.0})

        with patch("src.retrieval.reranker.get_reranker", return_value=encoder):
            result = get_retriever(user_roles={"admin"}, vector_store=store).retrieve(
                "Compare JPMorgan and Goldman Sachs on credit risk."
            )

        assert store.tickers_queried == ["JPM", "GS"]
        # Both companies present — the failure this exists to fix is one of
        # them being absent entirely.
        assert {d.metadata["ticker"] for d in result} == {"JPM", "GS"}
        # Ordered by cross-encoder relevance across companies, so citation [1]
        # is still the best-matching passage overall.
        assert [d.page_content for d in result] == ["gs credit", "jpm credit"]

    def test_entity_clause_is_intersected_with_the_rbac_filter(self):
        """The per-entity clause narrows; it must never widen. A user who
        cannot see a department must not reach it by naming a company."""
        retriever = RBACRetriever(
            user_roles={"research"},
            vector_store=FakeVectorStore([]),
            extra_filter={"ticker": {"$eq": "GS"}},
        )

        where = retriever.build_role_filter()

        assert "$and" in where
        assert {"ticker": {"$eq": "GS"}} in where["$and"]
        assert any("department" in clause for clause in where["$and"])

    def test_disabled_setting_restores_the_global_budget(self):
        retriever = single_entity_pipeline(rerank_enabled=True, hybrid_search_enabled=False)

        assert not isinstance(retriever, MultiEntityRetriever)
