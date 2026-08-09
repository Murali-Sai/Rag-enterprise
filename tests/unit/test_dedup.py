"""Tests for near-duplicate suppression.

A duplicate chunk costs a top_k slot at query time, so the cost of missing
one is a context slot; the cost of over-matching is a disclosure silently
dropped from the index. Both directions are pinned here.
"""

from unittest.mock import patch

import numpy as np
import pytest
from langchain_core.documents import Document

from src.ingestion import dedup as dedup_module
from src.ingestion.dedup import _unit, deduplicate


class StubEmbeddings:
    """Maps a text to a preset vector by a marker it contains.

    Real embeddings would make "is 0.95 the boundary" untestable — the point
    here is the comparison logic, not the model.
    """

    VECTORS = {
        "alpha": [1.0, 0.0, 0.0],
        "alpha-near": [0.999, 0.045, 0.0],  # cosine ~0.999 vs alpha
        "beta": [0.0, 1.0, 0.0],
        "gamma": [0.0, 0.0, 1.0],
        # cosine 0.9 vs alpha — similar topic, distinct text
        "alpha-related": [0.9, 0.4359, 0.0],
    }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Longest marker first, so "alpha-related" doesn't prefix-match "alpha"
        # and silently get the identical vector — which would make these tests
        # assert on a similarity the stub invented.
        markers = sorted(self.VECTORS, key=len, reverse=True)
        vectors = []
        for text in texts:
            for marker in markers:
                if text.startswith(marker):
                    vectors.append(self.VECTORS[marker])
                    break
            else:
                raise AssertionError(f"unmapped test text: {text!r}")
        return vectors


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={})


@pytest.fixture(autouse=True)
def _stub_embeddings():
    with patch("src.ingestion.embeddings.get_embedding_model", return_value=StubEmbeddings()):
        yield


class TestWithinBatch:
    def test_near_identical_chunk_is_skipped(self):
        result = deduplicate([_doc("alpha one"), _doc("alpha-near two")], threshold=0.95)

        assert len(result.kept) == 1
        assert result.kept[0].page_content == "alpha one"
        assert result.skipped_count == 1

    def test_first_occurrence_is_the_one_kept(self):
        """Order matters: the survivor should be the chunk already accepted,
        not whichever happened to be compared last."""
        result = deduplicate([_doc("alpha first"), _doc("alpha-near second")], threshold=0.95)

        assert [d.page_content for d in result.kept] == ["alpha first"]

    def test_distinct_chunks_all_survive(self):
        docs = [_doc("alpha"), _doc("beta"), _doc("gamma")]
        result = deduplicate(docs, threshold=0.95)

        assert len(result.kept) == 3
        assert result.skipped_count == 0

    def test_related_but_distinct_text_is_not_suppressed(self):
        """0.9 similarity is two chunks on the same topic. Dropping those is a
        silent recall loss, which is the failure mode that matters."""
        result = deduplicate([_doc("alpha"), _doc("alpha-related")], threshold=0.95)

        assert len(result.kept) == 2

    def test_threshold_is_respected(self):
        docs = [_doc("alpha"), _doc("alpha-related")]

        assert len(deduplicate(docs, threshold=0.85).kept) == 1
        assert len(deduplicate(docs, threshold=0.95).kept) == 2


class TestAgainstExistingCorpus:
    def test_chunk_matching_the_index_is_skipped(self):
        existing = _unit(np.array([StubEmbeddings.VECTORS["alpha"]], dtype=np.float32))

        result = deduplicate([_doc("alpha-near again")], existing, threshold=0.95)

        assert result.kept == []
        assert result.skipped_count == 1

    def test_new_chunk_survives_a_populated_index(self):
        existing = _unit(np.array([StubEmbeddings.VECTORS["alpha"]], dtype=np.float32))

        result = deduplicate([_doc("beta new")], existing, threshold=0.95)

        assert len(result.kept) == 1

    def test_empty_index_falls_back_to_batch_comparison(self):
        result = deduplicate([_doc("alpha"), _doc("alpha-near")], None, threshold=0.95)

        assert len(result.kept) == 1


class TestReporting:
    def test_match_score_is_recorded_for_each_skip(self):
        """Suppressions have to be auditable — a threshold that is too low
        fails silently otherwise."""
        result = deduplicate([_doc("alpha"), _doc("alpha-near")], threshold=0.95)

        assert len(result.matches) == 1
        score, matched_text = result.matches[0]
        assert score >= 0.95
        assert "alpha" in matched_text

    def test_empty_input_is_handled(self):
        assert deduplicate([]).kept == []


class TestCorpusVectorFetch:
    def test_missing_vectors_degrade_to_none(self):
        """A store that cannot return embeddings must not break ingestion —
        batch-level dedup still applies."""

        class BrokenStore:
            def get(self, **_):
                raise RuntimeError("no embeddings available")

        class Wrapper:
            store = BrokenStore()

        assert dedup_module.existing_corpus_vectors(Wrapper()) is None
