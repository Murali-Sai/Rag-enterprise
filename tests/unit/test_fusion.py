"""Tests for Reciprocal Rank Fusion."""

from langchain_core.documents import Document

from src.retrieval.fusion import reciprocal_rank_fusion


def _doc(text: str) -> Document:
    return Document(page_content=text)


class TestReciprocalRankFusion:
    def test_document_in_both_rankings_outranks_single_ranking_leader(self):
        """The core reason to fuse: agreement across retrievers beats being
        #1 in only one of them."""
        dense = [_doc("agreed"), _doc("dense only")]
        sparse = [_doc("sparse only"), _doc("agreed")]

        result = reciprocal_rank_fusion([dense, sparse], top_k=3)

        assert result[0].page_content == "agreed"

    def test_deduplicates_same_document_across_rankings(self):
        dense = [_doc("a"), _doc("b")]
        sparse = [_doc("a"), _doc("b")]

        result = reciprocal_rank_fusion([dense, sparse], top_k=10)

        assert [d.page_content for d in result] == ["a", "b"]

    def test_respects_top_k(self):
        dense = [_doc(str(i)) for i in range(10)]

        result = reciprocal_rank_fusion([dense], top_k=3)

        assert len(result) == 3

    def test_single_ranking_preserves_order(self):
        dense = [_doc("first"), _doc("second"), _doc("third")]

        result = reciprocal_rank_fusion([dense], top_k=3)

        assert [d.page_content for d in result] == ["first", "second", "third"]

    def test_empty_rankings_returns_empty(self):
        assert reciprocal_rank_fusion([], top_k=5) == []
        assert reciprocal_rank_fusion([[], []], top_k=5) == []

    def test_documents_found_by_only_one_retriever_still_included(self):
        """Hybrid recall depends on keeping single-retriever finds, not just
        the intersection."""
        dense = [_doc("dense only")]
        sparse = [_doc("sparse only")]

        result = reciprocal_rank_fusion([dense, sparse], top_k=5)

        assert {d.page_content for d in result} == {"dense only", "sparse only"}

    def test_preserves_metadata_of_fused_documents(self):
        dense = [Document(page_content="x", metadata={"ticker": "AAPL"})]

        result = reciprocal_rank_fusion([dense], top_k=1)

        assert result[0].metadata["ticker"] == "AAPL"
