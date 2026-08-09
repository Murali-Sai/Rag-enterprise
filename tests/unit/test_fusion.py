"""Tests for Reciprocal Rank Fusion."""

import pytest
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


class TestWeighting:
    """Per-retriever weights (Project 6 Phase 2.3).

    The weights are matched to rankings by position, which is the kind of
    coupling that fails quietly — a mismatch reorders results rather than
    raising, and the output still looks like a plausible ranking.
    """

    def test_weighting_can_flip_which_retriever_wins(self):
        """The point of the knob: each retriever's #1 wins when it is favoured.

        Unweighted these tie exactly — both are rank 0 in one list — so any
        change in the winner is the weighting and nothing else.
        """
        dense = [_doc("dense top")]
        sparse = [_doc("sparse top")]

        favour_dense = reciprocal_rank_fusion([dense, sparse], top_k=2, weights=[0.7, 0.3])
        favour_sparse = reciprocal_rank_fusion([dense, sparse], top_k=2, weights=[0.3, 0.7])

        assert favour_dense[0].page_content == "dense top"
        assert favour_sparse[0].page_content == "sparse top"

    def test_only_the_ratio_matters(self):
        """Scaling every weight scales every score, so the ranking is identical.

        Worth pinning: it is why nothing normalises the weights, and why
        0.7/0.3 and 7/3 are the same configuration.
        """
        dense = [_doc("a"), _doc("b")]
        sparse = [_doc("b"), _doc("c")]

        small = reciprocal_rank_fusion([dense, sparse], top_k=3, weights=[0.7, 0.3])
        large = reciprocal_rank_fusion([dense, sparse], top_k=3, weights=[7.0, 3.0])

        assert [d.page_content for d in small] == [d.page_content for d in large]

    def test_default_is_equal_weighting(self):
        """The measured hybrid baseline is an equal-weighted one. If the
        default ever moves, the README's comparison stops describing the
        shipped pipeline."""
        dense = [_doc("dense top")]
        sparse = [_doc("sparse top")]

        unweighted = reciprocal_rank_fusion([dense, sparse], top_k=2)
        explicit = reciprocal_rank_fusion([dense, sparse], top_k=2, weights=[1.0, 1.0])

        assert [d.page_content for d in unweighted] == [d.page_content for d in explicit]

    def test_zero_weight_silences_a_retriever_without_dropping_its_finds(self):
        """A zeroed retriever stops influencing the order, but documents only
        it found are still returned — they just sort last."""
        dense = [_doc("dense only")]
        sparse = [_doc("sparse only")]

        result = reciprocal_rank_fusion([dense, sparse], top_k=5, weights=[1.0, 0.0])

        assert [d.page_content for d in result] == ["dense only", "sparse only"]

    def test_length_mismatch_raises(self):
        dense = [_doc("a")]
        sparse = [_doc("b")]

        with pytest.raises(ValueError, match="same length"):
            reciprocal_rank_fusion([dense, sparse], top_k=2, weights=[0.7])

    def test_negative_weight_raises(self):
        """A negative weight demotes a document for being well ranked."""
        dense = [_doc("a")]

        with pytest.raises(ValueError, match="non-negative"):
            reciprocal_rank_fusion([dense], top_k=1, weights=[-1.0])
