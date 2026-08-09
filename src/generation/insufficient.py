"""Structured "I don't know" (Project 6, §3.4).

The system already refuses correctly — rule 2 of the prompt reliably
produces "I don't have enough information in the available documents to
answer this question." What that sentence does not tell an analyst is what
the search *did* find, which means the only way to learn whether the gap is
in the corpus or in the retrieval is to go and look. This module turns the
refusal into something actionable: what was searched, what came back, and
which filings are worth opening by hand.

Two paths reach here, and they are different failures:

- **low_retrieval_confidence** — the retrieved chunks scored badly, so
  nothing is generated at all. Caught before the generation call, which also
  means the cheapest failure is the one that costs nothing.
- **model_refused** — retrieval looked fine and the model still declined.
  The chunks were on topic but did not contain the specific fact. The
  generated refusal is kept; the structure is added alongside it.

The distinction is worth preserving because they point at different fixes:
the first is a retrieval problem, the second a corpus or chunking one.
"""

from dataclasses import dataclass

from langchain_core.documents import Document

LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"
MODEL_REFUSED = "model_refused"

_REASON_SUMMARY = {
    LOW_RETRIEVAL_CONFIDENCE: (
        "No sufficiently relevant passage was found in the filings you have "
        "access to, so no answer was generated."
    ),
    MODEL_REFUSED: (
        "Relevant filings were retrieved, but they do not state what the question asks for."
    ),
}


@dataclass(frozen=True)
class UnansweredReport:
    reason: str
    summary: str
    searched: tuple[str, ...]
    """The chunks that were actually consulted, best match first."""

    suggested_documents: tuple[str, ...]
    """Distinct filings behind those chunks — what to open manually."""

    def as_dict(self) -> dict:
        return {
            "reason": self.reason,
            "summary": self.summary,
            "searched": list(self.searched),
            "suggested_documents": list(self.suggested_documents),
        }


def describe_chunk(doc: Document) -> str:
    """One-line identity of a chunk from its provenance metadata.

    Chunk ids mean nothing to an analyst. Every chunk carries ticker,
    filing_type, filing_date and section_name, which name the thing a person
    would actually go and read.
    """
    ticker = doc.metadata.get("ticker")
    filing = doc.metadata.get("filing_type")
    date = doc.metadata.get("filing_date")
    section = doc.metadata.get("section_name")

    head = " ".join(str(p) for p in (ticker, filing) if p)
    if date:
        head = f"{head} ({date})" if head else str(date)
    if section:
        head = f"{head} — {section}" if head else str(section)
    return head or str(doc.metadata.get("source_file", "unknown source"))


def describe_filing(doc: Document) -> str:
    """Same, minus the section — the document rather than the passage."""
    ticker = doc.metadata.get("ticker")
    filing = doc.metadata.get("filing_type")
    date = doc.metadata.get("filing_date")

    head = " ".join(str(p) for p in (ticker, filing) if p)
    if date:
        head = f"{head} ({date})" if head else str(date)
    return head or str(doc.metadata.get("source_file", "unknown source"))


def _unique(values: list[str]) -> tuple[str, ...]:
    """Deduplicate while keeping retrieval order — rank carries information."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return tuple(seen)


def build_unanswered_report(
    reason: str,
    documents: list[Document],
    detail: str | None = None,
) -> UnansweredReport:
    summary = _REASON_SUMMARY.get(reason, "The question could not be answered from the filings.")
    if detail:
        summary = f"{summary} {detail}"

    return UnansweredReport(
        reason=reason,
        summary=summary,
        searched=_unique([describe_chunk(doc) for doc in documents]),
        suggested_documents=_unique([describe_filing(doc) for doc in documents]),
    )


def render_unanswered(report: UnansweredReport) -> str:
    """Plain-text rendering, for the surfaces that only return a string.

    The MCP server and the low-confidence path both need the structure as
    prose; the REST API returns the object itself.
    """
    lines = [report.summary]
    if report.searched:
        lines.append("\nPassages consulted:")
        lines.extend(f"  - {item}" for item in report.searched)
    if report.suggested_documents:
        lines.append("\nWorth checking manually:")
        lines.extend(f"  - {item}" for item in report.suggested_documents)
    return "\n".join(lines)
