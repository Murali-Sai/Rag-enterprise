"""Rolling chunk metadata up into a document listing.

`GET /documents` answers "what is in the index" from chunk provenance, since
ingestion writes chunks and there is no document table. The grouping is the
part worth pinning: a listing that counted wrong, or that named a document a
role cannot retrieve, would be a disclosure the information barrier exists to
prevent.
"""

from langchain_core.documents import Document

from src.api.routes.documents import _group_by_source
from src.retrieval.vector_store import VectorStoreBase


def meta(source, department="sec_filings", **extra):
    return {"source_file": source, "department": department, **extra}


class TestGrouping:
    def test_chunks_are_counted_per_source(self):
        documents = _group_by_source([meta("AAPL_10-K"), meta("AAPL_10-K"), meta("MSFT_10-K")])

        assert [(d.source, d.chunks) for d in documents] == [("AAPL_10-K", 2), ("MSFT_10-K", 1)]

    def test_sections_are_collected_across_a_filing(self):
        """Which Items the parser actually recovered is only answerable by
        unioning the sections its chunks carry."""
        documents = _group_by_source(
            [
                meta("GS_10-K", ticker="GS", section_name="Business"),
                meta("GS_10-K", ticker="GS", section_name="Risk Factors"),
                meta("GS_10-K", ticker="GS", section_name="Business"),
            ]
        )

        assert documents[0].sections == ["Business", "Risk Factors"]
        assert documents[0].chunks == 3
        assert documents[0].ticker == "GS"

    def test_documents_are_ordered(self):
        """They come out of a dict keyed by source; an arbitrary order would
        make two identical listings differ."""
        documents = _group_by_source([meta("TSLA"), meta("AAPL"), meta("MSFT")])

        assert [d.source for d in documents] == sorted(d.source for d in documents)

    def test_a_chunk_with_no_provenance_still_lists(self):
        documents = _group_by_source([{}])

        assert documents[0].source == "unknown"
        assert documents[0].department == "unknown"
        assert documents[0].sections == []

    def test_sections_are_absent_rather_than_blank_for_plain_documents(self):
        documents = _group_by_source([meta("policy.txt", department="trading")])

        assert documents[0].sections == []
        assert documents[0].filing_type is None


class TestMetadataProjection:
    def test_a_backend_without_the_projection_still_works(self):
        """`get_all_metadata` is concrete on the base class, not abstract, so
        an implementation that predates it — or a test fake — keeps working."""

        class OldBackend(VectorStoreBase):
            def add_documents(self, documents):
                return []

            def similarity_search(self, query, k=5, filter_dict=None):
                return []

            def delete(self, ids):
                return None

            def get_all_documents(self, filter_dict=None):
                return [Document(page_content="body", metadata=meta("AAPL_10-K"))]

        assert OldBackend().get_all_metadata() == [meta("AAPL_10-K")]
