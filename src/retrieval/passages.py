"""Shared passage representation for retrieval scorers.

Both the BM25 index and the cross-encoder reranker score a *rendering* of a
chunk rather than its raw text. The corpus spans five companies, and a chunk's
own body often never names the filer — a table row reading "Net revenue |
$151 million" carries the company only in its metadata. Any scorer that sees
just page_content will happily rank that against "What was Apple's revenue?".
Prefixing ticker + section gives lexical and neural scorers the same signal
the LLM already gets from format_documents().
"""

from langchain_core.documents import Document


def passage_text(doc: Document) -> str:
    """Render a document as the text retrieval scorers should match against."""
    ticker = doc.metadata.get("ticker")
    section = doc.metadata.get("section_name")
    prefix = " ".join(p for p in (ticker, section) if p)
    return f"{prefix}: {doc.page_content}" if prefix else doc.page_content
