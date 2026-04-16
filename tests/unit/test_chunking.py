from langchain_core.documents import Document

from src.ingestion.chunking import split_documents


class TestChunking:
    def test_split_creates_chunks(self):
        docs = [Document(page_content="word " * 200)]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_short_doc_no_split(self):
        docs = [Document(page_content="Short text.")]
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
