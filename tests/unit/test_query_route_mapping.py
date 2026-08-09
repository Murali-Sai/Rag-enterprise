"""Tests for the API's translation of a GroundedAnswer into QueryResponse.

The join is the risky part: verdicts are keyed by (claim text, block) because
a claim can cite several blocks and each pairing is judged separately, so a
naive join by claim alone would attach one block's verdict to another's
citation.

Also pinned here: the new response fields are optional. QueryResponse is
built on four paths — injection-blocked, nothing retrieved, generation
failed, and success — and three of them have no answer to analyse. A
required field would break those at runtime rather than at import.
"""

from langchain_core.documents import Document

from src.api.routes.query import _to_claim_citations, _to_source_documents
from src.common.schemas import QueryResponse
from src.generation.answer import GroundedAnswer
from src.generation.citations import parse_citations
from src.generation.confidence import score_confidence
from src.generation.verification import (
    CitationReport,
    CitationVerdict,
    unverified_report,
)


def grounded(answer: str, report: CitationReport | None = None) -> GroundedAnswer:
    parsed = parse_citations(answer, 3)
    citations = report or unverified_report(parsed)
    return GroundedAnswer(
        answer=answer,
        documents=[Document(page_content=f"chunk {i}") for i in range(3)],
        parsed=parsed,
        citations=citations,
        confidence=score_confidence(answer, parsed, citations, 0.9),
    )


class TestClaimMapping:
    def test_claims_carry_their_block_numbers(self):
        result = _to_claim_citations(grounded("Net sales rose sharply [2]."))

        assert result[0].cited_documents == [2]
        assert result[0].invalid_citations == []

    def test_broken_references_are_reported_separately(self):
        result = _to_claim_citations(grounded("Net sales rose sharply [9]."))

        assert result[0].cited_documents == []
        assert result[0].invalid_citations == [9]

    def test_verdicts_attach_to_the_right_claim(self):
        answer = "Net sales rose sharply [1]. Margins held steady [2]."
        parsed = parse_citations(answer, 3)
        report = CitationReport(
            verdicts=(
                CitationVerdict(parsed.claims[0].text, 1, "supported", "yes"),
                CitationVerdict(parsed.claims[1].text, 2, "unsupported", "no"),
            ),
            verified=True,
            total_claims=2,
            cited_claims=2,
        )

        result = _to_claim_citations(grounded(answer, report))

        assert [c.verdict for c in result] == ["supported", "unsupported"]
        assert result[1].reason == "no"

    def test_unverified_answers_report_no_verdict(self):
        result = _to_claim_citations(grounded("Net sales rose sharply [2]."))

        assert result[0].verdict is None


class TestResponseDefaults:
    def test_the_new_fields_are_optional(self):
        """The three paths that never generate an answer still construct."""
        response = QueryResponse(answer="blocked", sources=[], query="q")

        assert response.confidence is None
        assert response.claims == []
        assert response.unanswered is None


class TestSourceMapping:
    def test_sources_keep_retrieval_order(self):
        """Bracketed citations index into this list, so reordering it would
        point every citation at the wrong filing."""
        docs = [
            Document(page_content="a", metadata={"ticker": "MSFT"}),
            Document(page_content="b", metadata={"ticker": "AAPL"}),
        ]

        assert [s.ticker for s in _to_source_documents(docs)] == ["MSFT", "AAPL"]
