"""Tests for cross-encoder re-ranking.

Uses a fake CrossEncoder (no model download) so these stay fast and offline,
consistent with the rest of the unit suite.
"""

from unittest.mock import patch

from langchain_core.documents import Document

from src.retrieval.reranker import rerank_documents


class FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder — scores by lookup."""

    def __init__(self, score_by_content: dict[str, float]):
        self.score_by_content = score_by_content

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.score_by_content[doc_text] for _, doc_text in pairs]


class TestRerankDocuments:
    def test_reorders_by_cross_encoder_score(self):
        docs = [
            Document(page_content="low relevance"),
            Document(page_content="high relevance"),
            Document(page_content="medium relevance"),
        ]
        fake = FakeCrossEncoder(
            {"low relevance": 0.1, "high relevance": 0.9, "medium relevance": 0.5}
        )
        with patch("src.retrieval.reranker.get_reranker", return_value=fake):
            result = rerank_documents("query", docs, top_k=3)

        assert [d.page_content for d in result] == [
            "high relevance",
            "medium relevance",
            "low relevance",
        ]

    def test_cuts_to_top_k(self):
        docs = [Document(page_content=str(i)) for i in range(5)]
        fake = FakeCrossEncoder({str(i): float(i) for i in range(5)})
        with patch("src.retrieval.reranker.get_reranker", return_value=fake):
            result = rerank_documents("q", docs, top_k=2)

        assert len(result) == 2
        assert [d.page_content for d in result] == ["4", "3"]

    def test_empty_documents_returns_empty(self):
        result = rerank_documents("q", [], top_k=5)
        assert result == []

    def test_top_k_larger_than_candidates_returns_all(self):
        docs = [Document(page_content="a"), Document(page_content="b")]
        fake = FakeCrossEncoder({"a": 0.2, "b": 0.8})
        with patch("src.retrieval.reranker.get_reranker", return_value=fake):
            result = rerank_documents("q", docs, top_k=10)

        assert len(result) == 2
        assert result[0].page_content == "b"

    def test_preserves_document_metadata(self):
        docs = [Document(page_content="x", metadata={"ticker": "AAPL"})]
        fake = FakeCrossEncoder({"AAPL: x": 1.0})
        with patch("src.retrieval.reranker.get_reranker", return_value=fake):
            result = rerank_documents("q", docs, top_k=1)

        assert result[0].metadata == {"ticker": "AAPL"}

    def test_scores_ticker_and_section_prefixed_text_not_raw_content(self):
        """The corpus spans multiple companies — the cross-encoder needs the
        ticker/section in the scored text, since a chunk's own body doesn't
        always mention the company (e.g. a table row). Catches the real
        cross-company contamination bug this reranker shipped with initially:
        an off-topic GS chunk outranking on-topic AAPL chunks for an Apple
        question, because the raw chunk text never said "Apple"."""
        docs = [
            Document(
                page_content="Net revenue was $151 million for 2025.",
                metadata={"ticker": "GS", "section_name": "MD&A"},
            ),
            Document(
                page_content="iPhone revenue was $209,586 million.",
                metadata={"ticker": "AAPL", "section_name": "Financial Statements"},
            ),
        ]
        # A cross-encoder that only rewards passages naming the right company —
        # the raw page_content alone can't distinguish these for an Apple query.
        fake = FakeCrossEncoder(
            {
                "GS MD&A: Net revenue was $151 million for 2025.": 0.1,
                "AAPL Financial Statements: iPhone revenue was $209,586 million.": 0.9,
            }
        )
        with patch("src.retrieval.reranker.get_reranker", return_value=fake):
            result = rerank_documents("What was Apple's revenue?", docs, top_k=2)

        assert result[0].metadata["ticker"] == "AAPL"

    def test_passage_without_metadata_scores_raw_content(self):
        docs = [Document(page_content="plain text", metadata={})]
        fake = FakeCrossEncoder({"plain text": 1.0})
        with patch("src.retrieval.reranker.get_reranker", return_value=fake):
            result = rerank_documents("q", docs, top_k=1)

        assert len(result) == 1
