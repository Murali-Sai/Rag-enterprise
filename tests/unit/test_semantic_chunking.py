"""Tests for semantic chunking.

Uses a stub embedding model with hand-placed vectors so boundary behaviour is
deterministic — the real MiniLM would make these assertions about "where does
the topic shift" untestable.
"""

from unittest.mock import patch

import numpy as np
from langchain_core.documents import Document

from src.ingestion.semantic_chunking import (
    _breakpoint_indices,
    _combine_with_context,
    _split_sentences,
    split_documents_semantically,
)


class StubEmbeddings:
    """Returns a preset unit vector per text, keyed by a substring marker.

    Texts containing 'A' embed near one pole, 'B' near another, so the
    distance between an A-run and a B-run is large and the breakpoint is
    predictable.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            # Context-widening means a boundary sentence contains both markers;
            # count decides which pole it leans toward.
            leans_b = text.count("topic B") > text.count("topic A")
            vectors.append([0.0, 1.0] if leans_b else [1.0, 0.0])
        return vectors


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self):
        assert _split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_keeps_decimal_figures_intact(self):
        """'$391.0 billion' must not split at the decimal point."""
        result = _split_sentences("Revenue was $391.0 billion in 2024. It grew.")
        assert result[0] == "Revenue was $391.0 billion in 2024."

    def test_drops_empty_fragments(self):
        assert _split_sentences("   ") == []


class TestCombineWithContext:
    def test_widens_each_sentence_with_neighbours(self):
        result = _combine_with_context(["a", "b", "c"], buffer_size=1)
        assert result == ["a b", "a b c", "b c"]

    def test_buffer_zero_leaves_sentences_alone(self):
        assert _combine_with_context(["a", "b"], buffer_size=0) == ["a", "b"]


class TestBreakpointIndices:
    def test_no_breakpoints_for_single_embedding(self):
        assert _breakpoint_indices(np.array([[1.0, 0.0]]), percentile=95) == []

    def test_finds_the_distance_spike(self):
        # Three similar vectors then a sharp turn: the gap is at index 2.
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert _breakpoint_indices(embeddings, percentile=50) == [2]

    def test_uniform_text_yields_no_cut_above_threshold(self):
        embeddings = np.array([[1.0, 0.0]] * 5)
        assert _breakpoint_indices(embeddings, percentile=95) == []


class TestSplitDocumentsSemantically:
    def test_cuts_where_the_topic_changes(self):
        text = (
            "This is topic A one. This is topic A two. This is topic A three. "
            "Now topic B one. Now topic B two. Now topic B three."
        )
        docs = [Document(page_content=text, metadata={"ticker": "AAPL"})]

        with (
            patch("src.ingestion.embeddings.get_embedding_model", return_value=StubEmbeddings()),
            patch("src.ingestion.semantic_chunking.settings.semantic_breakpoint_percentile", 50),
            patch("src.ingestion.semantic_chunking.settings.semantic_max_chunk_chars", 10_000),
        ):
            chunks = split_documents_semantically(docs)

        assert len(chunks) > 1
        # The A material and the B material must not end up in one chunk.
        assert not any("topic A" in c.page_content and "topic B" in c.page_content for c in chunks)

    def test_preserves_metadata_on_every_chunk(self):
        text = "Topic A one. Topic A two. Now topic B one. Now topic B two."
        docs = [Document(page_content=text, metadata={"ticker": "AAPL", "department": "sec"})]

        with (
            patch("src.ingestion.embeddings.get_embedding_model", return_value=StubEmbeddings()),
            patch("src.ingestion.semantic_chunking.settings.semantic_breakpoint_percentile", 50),
            patch("src.ingestion.semantic_chunking.settings.semantic_max_chunk_chars", 10_000),
        ):
            chunks = split_documents_semantically(docs)

        assert all(c.metadata["ticker"] == "AAPL" for c in chunks)
        assert all(c.metadata["department"] == "sec" for c in chunks)

    def test_oversized_chunk_is_resplit_by_the_fixed_size_splitter(self):
        """Uniform prose produces no distance spike, so semantic chunking
        alone would emit one unretrievable mega-chunk."""
        text = " ".join(f"This is topic A sentence {i}." for i in range(60))
        docs = [Document(page_content=text, metadata={})]

        with (
            patch("src.ingestion.embeddings.get_embedding_model", return_value=StubEmbeddings()),
            patch("src.ingestion.semantic_chunking.settings.semantic_breakpoint_percentile", 99),
            patch("src.ingestion.semantic_chunking.settings.semantic_max_chunk_chars", 200),
        ):
            chunks = split_documents_semantically(docs)

        assert len(chunks) > 1
        # Fixed-size fallback honours chunk_size (512 default), not max_chars.
        assert all(len(c.page_content) <= 512 for c in chunks)

    def test_single_sentence_document_survives(self):
        docs = [Document(page_content="Only one sentence here.", metadata={})]

        with patch("src.ingestion.embeddings.get_embedding_model", return_value=StubEmbeddings()):
            chunks = split_documents_semantically(docs)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Only one sentence here."

    def test_empty_document_produces_nothing(self):
        with patch("src.ingestion.embeddings.get_embedding_model", return_value=StubEmbeddings()):
            chunks = split_documents_semantically([Document(page_content="   ", metadata={})])

        assert chunks == []
