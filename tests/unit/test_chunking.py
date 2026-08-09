from unittest.mock import patch

from langchain_core.documents import Document

from src.config import ChunkingStrategy
from src.ingestion import chunking as chunking_module
from src.ingestion.chunking import create_fixed_splitter, create_text_splitter, split_documents


class TestChunking:
    def test_split_creates_chunks(self):
        docs = [Document(page_content="word " * 200)]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_short_doc_no_split(self):
        # Well inside the 512 budget but above MIN_CHUNK_CHARS, so this
        # exercises the splitter rather than the stub floor.
        docs = [
            Document(
                page_content=(
                    "Short enough to fit in one chunk, long enough to clear the stub floor "
                    "that discards content-free fragments."
                )
            )
        ]
        chunks = split_documents(docs, chunk_size=512, chunk_overlap=50)
        assert len(chunks) == 1

    def test_metadata_preserved(self):
        docs = [Document(page_content="word " * 200, metadata={"source": "test.pdf"})]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"

    def test_custom_chunk_size(self):
        docs = [Document(page_content="word " * 500)]
        small = split_documents(docs, chunk_size=100, chunk_overlap=10)
        large = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(small) > len(large)


class TestStubFloor:
    """The floor beneath the parser's page-furniture stripping.

    The recursive splitter cannot merge a short line into a neighbour already
    near the budget, so bare headings and leftover markers survive as chunks
    of their own and then compete for a top-k slot against real paragraphs.
    """

    def test_stub_chunks_are_dropped(self):
        docs = [
            Document(page_content="Competition"),
            Document(page_content="THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES"),
            Document(page_content="PART II\nItem 8"),
            Document(page_content="million and $"),
        ]
        assert split_documents(docs, chunk_size=512, chunk_overlap=50) == []

    def test_substantive_short_chunk_survives(self):
        """The floor sits below the shortest text that still says something —
        this sentence is 97 characters and a 200-char floor would delete it."""
        sentence = (
            ". As of September 27, 2025, the Company had approximately "
            "166,000 full-time equivalent employees."
        )
        chunks = split_documents([Document(page_content=sentence)], chunk_size=512)

        assert [c.page_content for c in chunks] == [sentence]

    def test_whitespace_does_not_count_toward_the_floor(self):
        docs = [Document(page_content="Competition" + " " * 200)]
        assert split_documents(docs, chunk_size=512, chunk_overlap=50) == []

    def test_chunk_index_stays_contiguous_after_drops(self):
        """Dropping happens before provenance is stamped. A gap in chunk_index
        would read as a lost chunk rather than a rejected one."""
        docs = [
            Document(page_content="Competition", metadata={"source_file": "a.txt"}),
            Document(page_content="word " * 200, metadata={"source_file": "a.txt"}),
        ]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)

        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_floor_can_be_disabled(self):
        docs = [Document(page_content="Competition")]
        with patch.object(chunking_module.settings, "min_chunk_chars", 0):
            chunks = split_documents(docs, chunk_size=512, chunk_overlap=50)

        assert len(chunks) == 1


class TestStrategySelection:
    """The three strategies must actually differ. A switchable setting that
    produces identical output would make the chunking comparison meaningless
    while still appearing to work."""

    def test_fixed_splitter_ignores_boundaries(self):
        # Sentence boundaries land well inside the budget; a structure-blind
        # splitter cuts on the character count anyway.
        text = "Revenue rose. " * 40
        pieces = create_fixed_splitter(chunk_size=100, chunk_overlap=0).split_text(text)

        assert any(not piece.endswith(" ") and not piece.endswith(".") for piece in pieces)

    def test_recursive_splitter_prefers_boundaries(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        pieces = create_text_splitter(chunk_size=40, chunk_overlap=0).split_text(text)

        assert all(piece == piece.strip() for piece in pieces)
        assert any("paragraph" in piece for piece in pieces)

    def test_fixed_strategy_is_selected_from_config(self):
        docs = [Document(page_content="word " * 200)]
        with (
            patch.object(chunking_module.settings, "chunking_strategy", ChunkingStrategy.FIXED),
            patch.object(chunking_module.settings, "chunk_size", 100),
            patch.object(chunking_module.settings, "chunk_overlap", 0),
        ):
            chunks = split_documents(docs)

        assert all(c.metadata["chunking_strategy"] == "fixed" for c in chunks)

    def test_forced_chunk_size_is_not_recorded_as_semantic(self):
        """The semantic path has no chunk_size. A caller passing one gets
        size-based splitting, and labelling that 'semantic' would misattribute
        the whole collection."""
        docs = [Document(page_content="word " * 200)]
        with patch.object(chunking_module.settings, "chunking_strategy", ChunkingStrategy.SEMANTIC):
            chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)

        assert all(c.metadata["chunking_strategy"] == "recursive" for c in chunks)


class TestChunkProvenance:
    def test_every_chunk_records_strategy_and_size(self):
        docs = [Document(page_content="word " * 200, metadata={"source_file": "a.txt"})]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)

        for chunk in chunks:
            assert chunk.metadata["chunking_strategy"] == "recursive"
            assert chunk.metadata["char_count"] == len(chunk.page_content)

    def test_chunk_index_restarts_per_source_document(self):
        """Indexes are per source, so ingesting files in a different order
        doesn't renumber chunks that didn't change."""
        docs = [
            Document(page_content="word " * 200, metadata={"source_file": "a.txt"}),
            Document(page_content="word " * 200, metadata={"source_file": "b.txt"}),
        ]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)

        for source in ("a.txt", "b.txt"):
            indexes = [
                c.metadata["chunk_index"] for c in chunks if c.metadata["source_file"] == source
            ]
            assert indexes == list(range(len(indexes)))
