"""Tests for citation verification.

The judge is stubbed throughout — the suite makes no network calls, matching
the pattern in test_dedup.py and test_hyde.py. What is being tested is the
apparatus around the judge: that each (claim, block) pair is judged against
that block alone, that a broken reference is scored without asking a judge
at all, and that a judge failure is recorded as a hole in the measurement
rather than as a hallucinating model.
"""

from langchain_core.documents import Document

from src.generation.citations import parse_citations
from src.generation.verification import (
    OUT_OF_RANGE,
    PARTIAL,
    SUPPORTED,
    UNJUDGED,
    UNSUPPORTED,
    unverified_report,
    verify_citations,
)


class StubJudge:
    """Returns queued verdicts and records the prompts it was given."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str):  # noqa: ANN201
        self.prompts.append(prompt)
        return type("Response", (), {"content": self.responses.pop(0)})()


class FailingJudge:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt: str):  # noqa: ANN201, ARG002
        self.calls += 1
        raise RuntimeError("provider down")


def docs(*contents: str) -> list[Document]:
    return [
        Document(page_content=text, metadata={"ticker": "AAPL", "filing_type": "10-K"})
        for text in contents
    ]


class TestVerdicts:
    def test_supported_citation_scores_one(self):
        parsed = parse_citations("Net sales were $391.0 billion [1].", 1)
        judge = StubJudge(["SUPPORTED\nThe source states the figure."])

        report = verify_citations(parsed, docs("Total net sales were $391.0 billion."), judge)

        assert report.accuracy == 1.0
        assert report.verdicts[0].verdict == SUPPORTED

    def test_unsupported_scores_zero_and_is_not_read_as_supported(self):
        """'UNSUPPORTED' contains 'SUPPORTED' — substring order matters."""
        parsed = parse_citations("Net sales were $500 billion [1].", 1)
        judge = StubJudge(["UNSUPPORTED\nThe source gives a different figure."])

        report = verify_citations(parsed, docs("Total net sales were $391.0 billion."), judge)

        assert report.verdicts[0].verdict == UNSUPPORTED
        assert report.accuracy == 0.0

    def test_partial_earns_half_credit(self):
        parsed = parse_citations("Net sales rose 2% to $391.0 billion [1].", 1)
        judge = StubJudge(["PARTIAL\nThe growth rate is not in the source."])

        report = verify_citations(parsed, docs("Total net sales were $391.0 billion."), judge)

        assert report.verdicts[0].verdict == PARTIAL
        assert report.accuracy == 0.5


class TestJudgeInputs:
    def test_one_call_per_citation_not_per_claim(self):
        """A claim citing two blocks makes two assertions that can fail
        independently; one merged verdict would hide which was wrong."""
        parsed = parse_citations("Revenue rose and margin held [1][2].", 2)
        judge = StubJudge(["SUPPORTED\nyes", "UNSUPPORTED\nno"])

        report = verify_citations(parsed, docs("Revenue rose.", "Something else."), judge)

        assert len(judge.prompts) == 2
        assert report.accuracy == 0.5

    def test_judge_sees_only_the_cited_block(self):
        """Showing the whole context turns this into a faithfulness check —
        the citation would pass because the corpus supports the claim, not
        because the cited chunk does."""
        parsed = parse_citations("Revenue rose year over year [1].", 2)
        judge = StubJudge(["SUPPORTED\nyes"])

        verify_citations(parsed, docs("Revenue rose.", "UNRELATED MARKER TEXT"), judge)

        assert "UNRELATED MARKER TEXT" not in judge.prompts[0]
        assert "Revenue rose." in judge.prompts[0]

    def test_prompt_names_the_filing_the_chunk_came_from(self):
        parsed = parse_citations("Revenue rose year over year [1].", 1)
        judge = StubJudge(["SUPPORTED\nyes"])

        verify_citations(parsed, docs("Revenue rose."), judge)

        assert "AAPL 10-K" in judge.prompts[0]

    def test_uncited_claims_cost_nothing_to_judge(self):
        parsed = parse_citations("Revenue rose year over year.", 1)
        judge = StubJudge([])

        report = verify_citations(parsed, docs("Revenue rose."), judge)

        assert judge.prompts == []
        assert report.accuracy is None  # nothing was cited, so nothing was judged
        assert report.coverage == 0.0


class TestOutOfRange:
    def test_broken_reference_needs_no_judge(self):
        parsed = parse_citations("Revenue rose year over year [7].", 2)
        judge = StubJudge([])

        report = verify_citations(parsed, docs("a", "b"), judge)

        assert judge.prompts == []
        assert report.verdicts[0].verdict == OUT_OF_RANGE
        assert report.out_of_range == 1

    def test_broken_reference_counts_against_accuracy(self):
        """It is a wrong citation, not an unmeasurable one — but it stays
        separately counted so the two failures can still be told apart."""
        parsed = parse_citations("Revenue rose [1]. Margin held [9].", 2)
        judge = StubJudge(["SUPPORTED\nyes"])

        report = verify_citations(parsed, docs("Revenue rose.", "b"), judge)

        assert report.accuracy == 0.5
        assert report.out_of_range == 1
        assert report.total_citations == 2


class TestFailureModes:
    def test_judge_error_is_excluded_rather_than_scored_zero(self):
        """A flaky judge must not look like a hallucinating model."""
        parsed = parse_citations("Revenue rose year over year [1].", 1)
        judge = FailingJudge()

        report = verify_citations(parsed, docs("Revenue rose."), judge)

        assert judge.calls == 1
        assert report.accuracy is None
        assert report.counts[UNJUDGED] == 1

    def test_unparseable_verdict_is_treated_as_unjudged(self):
        parsed = parse_citations("Revenue rose year over year [1].", 1)
        judge = StubJudge(["It depends on how you look at it."])

        report = verify_citations(parsed, docs("Revenue rose."), judge)

        assert report.counts[UNJUDGED] == 1
        assert report.accuracy is None

    def test_one_failed_call_does_not_sink_the_others(self):
        parsed = parse_citations("Revenue rose [1]. Margin held [2].", 2)
        judge = StubJudge(["SUPPORTED\nyes", "not a verdict"])

        report = verify_citations(parsed, docs("Revenue rose.", "Margin held."), judge)

        assert report.accuracy == 1.0
        assert report.total_citations == 1


class TestUnverifiedReport:
    def test_skipping_verification_reports_none_not_zero(self):
        """The runtime default. A missing measurement is not a bad score."""
        parsed = parse_citations("Revenue rose [1]. Margin held steady.", 2)

        report = unverified_report(parsed)

        assert report.accuracy is None
        assert report.verified is False
        assert report.coverage == 0.5
        assert report.total_claims == 2
        assert report.uncited_claims == 1
