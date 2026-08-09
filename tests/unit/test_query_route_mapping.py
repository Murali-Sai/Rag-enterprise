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

import pytest
from langchain_core.documents import Document

from src.api.routes.access import to_information_barriers
from src.api.routes.query import _to_claim_citations, _to_source_documents
from src.auth.rbac import get_information_barriers_for_user
from src.common.schemas import QueryResponse
from src.generation.answer import GroundedAnswer
from src.generation.citations import parse_citations
from src.generation.confidence import score_confidence
from src.generation.verification import (
    CitationReport,
    CitationVerdict,
    unverified_report,
)
from src.retrieval.scores import ScoredDocument, ScoreType


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
        assert response.accessible_departments == []
        assert response.information_barriers == []


class TestSourceMapping:
    def test_sources_keep_retrieval_order(self):
        """Bracketed citations index into this list, so reordering it would
        point every citation at the wrong filing."""
        docs = [
            ScoredDocument(Document(page_content="a", metadata={"ticker": "MSFT"})),
            ScoredDocument(Document(page_content="b", metadata={"ticker": "AAPL"})),
        ]

        assert [s.ticker for s in _to_source_documents(docs)] == ["MSFT", "AAPL"]

    def test_scores_reach_the_response_normalised_and_raw(self):
        """`relevance_score` was declared on the schema and null on every
        response ever served, because the route mapped documents rather than
        the scored channel behind them."""
        docs = [
            ScoredDocument(
                Document(page_content="a"),
                score=-2.33,
                score_type=ScoreType.CROSS_ENCODER,
            )
        ]

        source = _to_source_documents(docs)[0]

        assert source.raw_score == -2.33
        assert source.score_type == "cross_encoder"
        assert source.relevance_score == pytest.approx(0.0887, abs=1e-4)

    def test_an_ordinal_stage_reports_no_relevance(self):
        """RRF fuses ranks and discards scores, so it carries no relevance.
        Null rather than 0.0 — the latter reads as 'measured, and bad'."""
        docs = [ScoredDocument(Document(page_content="a"), score=0.03, score_type=ScoreType.RRF)]

        source = _to_source_documents(docs)[0]

        assert source.relevance_score is None
        assert source.raw_score == 0.03
        assert source.score_type == "rrf"

    def test_an_unscored_retriever_degrades_to_nulls(self):
        source = _to_source_documents([ScoredDocument(Document(page_content="a"))])[0]

        assert source.relevance_score is None
        assert source.raw_score is None
        assert source.score_type is None


class TestBarrierMapping:
    def test_barriers_arrive_as_data_not_as_the_audit_string(self):
        """`guardrail_flags` flattens the barriers into one string and drops
        the descriptions. The structured field is the same fact, unparsed."""
        barriers = to_information_barriers(get_information_barriers_for_user({"research"}))

        assert [b.name for b in barriers] == [
            "Research-Trading Wall",
            "Research-Compliance Wall",
        ]
        assert barriers[0].blocked_departments == ["trading"]
        assert "non-public trading positions" in barriers[0].description

    def test_blocked_departments_are_ordered(self):
        """The rule holds them in a set, which has no stable iteration order
        and no JSON representation."""
        barriers = to_information_barriers(
            [
                {
                    "name": "W",
                    "description": "d",
                    "blocked_departments": {"trading", "compliance"},
                }
            ]
        )

        assert barriers[0].blocked_departments == ["compliance", "trading"]

    def test_a_role_behind_no_wall_reports_none(self):
        assert to_information_barriers(get_information_barriers_for_user({"trading"})) == []
