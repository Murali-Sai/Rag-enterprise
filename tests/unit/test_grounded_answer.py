"""Tests for the grounded-generation entry point.

This is the module that makes Phase 3 measurable: the REST API, the MCP
server and evaluation/run_evaluation.py all call it, so anything tested here
is true of the code path the citation-accuracy number grades. The tests
therefore care mostly about ordering and about what happens on the paths
where there is no answer:

- citations parsed from the raw text, before the route rewrites it
- the low-confidence gate firing *before* the generation call, not after
- a model refusal keeping its own wording and gaining structure alongside it
"""

from unittest.mock import patch

from langchain_core.documents import Document

from src.config import settings
from src.generation.answer import generate_grounded_answer
from src.generation.insufficient import LOW_RETRIEVAL_CONFIDENCE, MODEL_REFUSED
from src.generation.verification import CitationReport
from src.retrieval.scores import ScoredDocument, ScoreType

ANSWER = "Net sales were $391.0 billion in fiscal 2024 [1]. Services grew 13% [2]."
REFUSAL = "I don't have enough information in the available documents to answer this question."

# Declines one half of a two-part question and cites the other. Recorded in
# evaluation/results/eval_20260808_212927.json, where it was scored as a full
# refusal despite two 'supported' citation verdicts.
PARTIAL_REFUSAL = (
    "I don't have enough information in the available documents to answer the question about "
    "Tesla's total operating expenses for 2025. However, Tesla's net income attributable to "
    "common stockholders for 2025 was $3.79 billion [1][2]."
)


def filing_chunk(text: str, ticker: str = "AAPL", section: str = "Item 7 - MD&A") -> Document:
    return Document(
        page_content=text,
        metadata={
            "ticker": ticker,
            "filing_type": "10-K",
            "filing_date": "2024-11-01",
            "section_name": section,
            "source_file": f"{ticker}_10-K.txt",
        },
    )


def retrieved(*logits: float) -> list[ScoredDocument]:
    return [
        ScoredDocument(
            document=filing_chunk(f"chunk {i}"),
            score=logit,
            score_type=ScoreType.CROSS_ENCODER,
        )
        for i, logit in enumerate(logits)
    ]


def generating(answer: str):  # noqa: ANN201
    return patch("src.generation.answer.query_with_context", return_value=answer)


class TestSuccessPath:
    def test_parses_citations_out_of_the_generated_answer(self):
        with generating(ANSWER):
            result = generate_grounded_answer("q", retrieved(8.0, 6.0), verify=False)

        assert len(result.parsed.claims) == 2
        assert result.citations.coverage == 1.0
        assert result.unanswered is None
        assert result.generated is True

    def test_the_returned_answer_is_the_raw_generation(self):
        """The route appends disclaimers and redacts PII on top of this; the
        citation analysis has to have happened against the untouched text."""
        with generating(ANSWER):
            result = generate_grounded_answer("q", retrieved(8.0, 6.0), verify=False)

        assert result.answer == ANSWER

    def test_confidence_combines_retrieval_and_citations(self):
        with generating(ANSWER):
            result = generate_grounded_answer("q", retrieved(8.0, 6.0), verify=False)

        assert result.confidence.retrieval is not None
        assert result.confidence.coverage == 1.0
        assert result.confidence.label == "high"

    def test_plain_documents_work_and_leave_retrieval_unknown(self):
        """Callers without a score channel still get citations and coverage —
        they just lose the one signal their retriever cannot provide."""
        with generating(ANSWER):
            result = generate_grounded_answer("q", [filing_chunk("a"), filing_chunk("b")])

        assert result.confidence.retrieval is None
        assert result.confidence.coverage == 1.0

    def test_verification_is_off_by_default_and_reports_none(self):
        with generating(ANSWER), patch("src.generation.answer.verify_citations") as judge:
            result = generate_grounded_answer("q", retrieved(8.0, 6.0))

        judge.assert_not_called()
        assert result.citations.accuracy is None
        assert result.citations.verified is False

    def test_verification_runs_when_asked(self):
        report = CitationReport(verified=True, total_claims=2, cited_claims=2)
        with (
            generating(ANSWER),
            patch("src.generation.answer.verify_citations", return_value=report) as judge,
        ):
            result = generate_grounded_answer("q", retrieved(8.0, 6.0), verify=True)

        judge.assert_called_once()
        assert result.citations is report


class TestLowRetrievalConfidence:
    def test_gate_fires_before_the_generation_call(self):
        """The cheapest failure is the one that costs nothing — refusing on
        bad chunks after paying for generation gets the economics backwards."""
        with generating(ANSWER) as generate:
            result = generate_grounded_answer("q", retrieved(-8.0, -9.0), verify=False)

        generate.assert_not_called()
        assert result.generated is False
        assert result.unanswered is not None
        assert result.unanswered.reason == LOW_RETRIEVAL_CONFIDENCE

    def test_reports_what_was_searched_and_what_to_read(self):
        scored = [
            ScoredDocument(
                filing_chunk("a", "AAPL", "Item 1A - Risk Factors"), -8.0, ScoreType.CROSS_ENCODER
            ),
            ScoredDocument(
                filing_chunk("b", "AAPL", "Item 7 - MD&A"), -9.0, ScoreType.CROSS_ENCODER
            ),
        ]
        with generating(ANSWER):
            result = generate_grounded_answer("q", scored, verify=False)

        report = result.unanswered
        assert report.searched == (
            "AAPL 10-K (2024-11-01) — Item 1A - Risk Factors",
            "AAPL 10-K (2024-11-01) — Item 7 - MD&A",
        )
        # The two chunks come from the same filing, so there is one document
        # worth opening, not two.
        assert report.suggested_documents == ("AAPL 10-K (2024-11-01)",)

    def test_the_answer_text_carries_the_structure(self):
        with generating(ANSWER):
            result = generate_grounded_answer("q", retrieved(-8.0), verify=False)

        assert "Passages consulted:" in result.answer
        assert "Worth checking manually:" in result.answer

    def test_threshold_is_configurable(self):
        with (
            patch("src.generation.answer.settings.insufficient_context_threshold", 0.99),
            generating(ANSWER) as generate,
        ):
            result = generate_grounded_answer("q", retrieved(2.0), verify=False)

        generate.assert_not_called()
        assert result.unanswered.reason == LOW_RETRIEVAL_CONFIDENCE

    def test_the_calibrated_floor_admits_every_answerable_probe(self):
        """The gate must not be why an answerable question is declined.

        0.0031 is the lowest score any in-corpus question reached across the
        49-item held-out probe set in gate_calibration_v1.json — Microsoft's
        dividend, which the corpus does disclose. The shipped threshold has to
        sit below it, and the previous value of 0.15 did not: it refused that
        question, and in the eval suite it refused one retrieving context
        scoring 1.000 context recall.

        Pinned as a test because the number looks like a knob and reads as far
        too low to anyone who has not seen the measurement. Raising it back
        toward 0.15 buys nothing — with the gate open the model refused 24 of
        24 out-of-corpus questions by itself — and costs answerable ones.
        """
        lowest_answerable_probe = 0.0031
        assert settings.insufficient_context_threshold < lowest_answerable_probe

        with generating(ANSWER) as generate:
            # 0.6 * logistic(-5.0) + 0.4 * logistic(-6.0) ~= 0.006, which is
            # roughly what that Microsoft question looks like at the gate.
            result = generate_grounded_answer("q", retrieved(-5.0, -6.0), verify=False)

        generate.assert_called_once()
        assert result.unanswered is None

    def test_gate_cannot_fire_without_a_retrieval_signal(self):
        """RRF-scored results carry no relevance, so there is nothing to
        threshold — the system must generate rather than refuse on a number
        it does not have."""
        rrf = [ScoredDocument(filing_chunk("a"), None, ScoreType.RRF)]
        with generating(ANSWER) as generate:
            result = generate_grounded_answer("q", rrf, verify=False)

        generate.assert_called_once()
        assert result.unanswered is None


class TestModelRefusal:
    def test_refusal_keeps_its_wording_and_gains_structure(self):
        """The model's sentence says which part of the question it could not
        cover; the structure cannot infer that, so it is added alongside."""
        with generating(REFUSAL):
            result = generate_grounded_answer("q", retrieved(8.0), verify=False)

        assert result.answer == REFUSAL
        assert result.unanswered is not None
        assert result.unanswered.reason == MODEL_REFUSED
        assert result.generated is True

    def test_refusal_over_good_retrieval_is_a_different_failure(self):
        """Retrieval scored well and the model still declined — a corpus or
        chunking gap, not a retrieval one, and the two want opposite fixes."""
        with generating(REFUSAL):
            result = generate_grounded_answer("q", retrieved(8.0), verify=False)

        assert result.confidence.retrieval > 0.9
        assert result.confidence.label == "low"

    def test_partial_refusal_is_not_reported_as_unanswered(self):
        """The answer cites a figure verification scores as supported. Attaching
        the "I don't know" structure would have the REST route return it beside
        a sourced number, and tells the eval harness the row went unanswered."""
        with generating(PARTIAL_REFUSAL):
            result = generate_grounded_answer("q", retrieved(8.0, 6.0), verify=False)

        assert result.unanswered is None
        assert result.parsed.cited_claims
        assert result.confidence.completeness == 0.5

    def test_refusal_with_nothing_cited_is_still_unanswered(self):
        """The narrowing is to *cited* content — this path must not move."""
        with generating(f"{REFUSAL} Apple is a large issuer."):
            result = generate_grounded_answer("q", retrieved(8.0), verify=False)

        assert result.unanswered is not None
        assert result.unanswered.reason == MODEL_REFUSED


class TestBrokenCitations:
    def test_out_of_range_reference_is_surfaced_not_swallowed(self):
        with generating("Net sales were $391.0 billion [4]."):
            result = generate_grounded_answer("q", retrieved(8.0, 6.0), verify=False)

        assert result.parsed.out_of_range_count == 1
        assert result.citations.out_of_range == 1
        # It is a broken reference, not a missing one.
        assert result.citations.uncited_claims == 0
