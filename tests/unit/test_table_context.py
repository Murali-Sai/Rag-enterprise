"""Tests for the table-context chunking post-pass.

The transform is index-side repair for a measured failure: ground-truth
figures sitting in caption-less table chunks that no embedder can rank
against a thematic question. The properties worth pinning are the ones the
comparison against `recursive` depends on: prose chunks pass through
byte-identical, captions come from the right neighbour and never cross a
document or section boundary, and the carry survives a table split across
several chunks.
"""

from langchain_core.documents import Document

from src.config import ChunkingStrategy, settings
from src.ingestion.chunking import split_documents
from src.ingestion.table_context import add_table_context, looks_like_table

TABLE = (
    "CET1 capital | $ | 104,297 | $ | 104,297\n"
    "Tier 1 capital | $ | 118,943 | $ | 118,943\n"
    "Total capital | $ | 130,665 | $ | 128,470\n"
    "RWAs | $ | 727,338 | $ | 691,470"
)

PROSE = (
    "The firm is subject to consolidated regulatory capital requirements which "
    "are calculated in accordance with the regulations of the Federal Reserve. "
    "The table below presents information about risk-based capital ratios."
)


def chunk(
    text: str, source: str = "GS_10-K.html", section: str = "Financial Statements"
) -> Document:
    return Document(
        page_content=text,
        metadata={"ticker": "GS", "source_file": source, "section_name": section},
    )


class TestDetection:
    def test_pipe_rows_are_a_table(self):
        assert looks_like_table(TABLE)

    def test_prose_is_not(self):
        assert not looks_like_table(PROSE)

    def test_one_quoted_row_inside_prose_is_not(self):
        text = PROSE + "\nTotal capital | $ | 130,665 | $ | 128,470\n" + PROSE
        assert not looks_like_table(text)

    def test_a_short_string_with_digits_in_it_is_not_a_table(self):
        """The digit-ratio fallback needs enough text to be evidence. "doc19"
        is two digits in five alphanumerics — 0.4 exactly, over the threshold,
        and obviously not a table. Found when placeholder documents in the
        retrieval tests started being treated as tables and dragged a vector
        store into a suite that runs without one."""
        assert not looks_like_table("doc19")
        assert not looks_like_table("2025 | 2024")


class TestCaptionCarry:
    def test_table_gets_the_previous_chunks_caption(self):
        table = chunk(TABLE)
        add_table_context([chunk(PROSE), table])

        assert table.page_content.endswith(TABLE)
        header = table.page_content[: -len(TABLE)]
        assert "risk-based capital ratios" in header
        assert table.metadata["table_context"] is True

    def test_header_carries_nothing_shared_across_tables(self):
        """The v1 header led with company and section — identical across
        every table in a filing — and the screen measured the consequence:
        all of a company's tables pulled toward its thematic questions
        equally, crowding out the prose that held the answers. Only the
        caption discriminates, so only the caption ships."""
        table = chunk(TABLE)
        add_table_context([chunk(PROSE), table])

        header = table.page_content[: -len(TABLE)]
        assert "Goldman Sachs" not in header
        assert "Financial Statements" not in header

    def test_prose_chunks_pass_through_byte_identical(self):
        prose = chunk(PROSE)
        add_table_context([prose, chunk(TABLE)])

        assert prose.page_content == PROSE
        assert "table_context" not in prose.metadata

    def test_a_table_with_no_known_caption_is_left_untouched(self):
        """A header with nothing discriminating in it is the crowding
        failure again, so no caption means no header at all."""
        table = chunk(TABLE)
        add_table_context([table])

        assert table.page_content == TABLE
        assert "table_context" not in table.metadata

    def test_caption_survives_a_table_split_across_chunks(self):
        """A wide table becomes several chunks; only the first neighbours the
        caption. The carry state must persist until new prose replaces it."""
        first, second = chunk(TABLE), chunk(TABLE)
        add_table_context([chunk(PROSE), first, second])

        assert "risk-based capital ratios" in second.page_content

    def test_caption_never_crosses_a_section_boundary(self):
        table = chunk(TABLE, section="Item 8")
        add_table_context([chunk(PROSE, section="Item 7A"), table])

        assert table.page_content == TABLE

    def test_caption_never_crosses_a_document_boundary(self):
        table = chunk(TABLE, source="JPM_10-K.html")
        add_table_context([chunk(PROSE, source="GS_10-K.html"), table])

        assert table.page_content == TABLE


class TestStrategyWiring:
    def test_split_documents_applies_the_pass_and_stamps_the_strategy(self, monkeypatch):
        monkeypatch.setattr(settings, "chunking_strategy", ChunkingStrategy.TABLE_CONTEXT)
        doc = chunk(PROSE + "\n\n" + TABLE + "\n" + TABLE)

        chunks = split_documents([doc])

        assert chunks, "splitter produced nothing"
        assert all(c.metadata["chunking_strategy"] == "table_context" for c in chunks)
        tagged = [c for c in chunks if c.metadata.get("table_context")]
        assert tagged, "no chunk was recognised as a table"
        # char_count is stamped after the pass, so it describes the final text.
        for c in chunks:
            assert c.metadata["char_count"] == len(c.page_content)
