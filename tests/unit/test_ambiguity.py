"""Tests for the underspecified-question detector and its refusal path.

The stakes are asymmetric, and the tests mirror that. A missed ambiguous
question falls through to today's behaviour — a fluent answer about an
arbitrarily chosen company — while a false positive refuses a question the
corpus answers, the exact failure `_refusal_correctness` scores against the
answerable population. So the suite-level test here pins *both* directions
against the real evaluation dataset: the three questions that must fire, and
the fifty-one that must not.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.generation.answer import generate_grounded_answer
from src.generation.insufficient import AMBIGUOUS_ENTITY
from src.retrieval.ambiguity import clarification_detail, is_underspecified

EVAL_DATASET = Path(__file__).parents[2] / "evaluation" / "datasets" / "eval_questions_v4.json"


class TestDetector:
    @pytest.mark.parametrize(
        "question",
        [
            # The three ambiguous-refuse questions from the eval suite, verbatim.
            "What was total revenue last year?",
            "How much did the bank set aside for credit losses?",
            "What are the company's main risks?",
            # Possessive and plural variants of the definite reference.
            "What is the firm's dividend policy?",
            "How do the banks compare on credit losses?",
        ],
    )
    def test_fires_on_questions_that_need_a_company_and_name_none(self, question):
        assert is_underspecified(question)

    @pytest.mark.parametrize(
        "question",
        [
            # A named company always wins, even alongside 'the company'.
            "What does Apple say about the company's supply chain?",
            "What was Apple's revenue last year?",
            # Out-of-corpus questions name a company this corpus lacks. The
            # model refuses these unaided, 24 of 24 measured — this module
            # must not reroute them into a differently-worded refusal.
            "What was Netflix's total streaming revenue for its most recent fiscal year?",
            "How does Berkshire Hathaway describe its insurance float?",
            # Internal policy documents answer these without any company named.
            "What pre-trade controls does the internal compliance manual specify?",
            "What are the position limits for the trading desk?",
        ],
    )
    def test_stays_silent_when_a_subject_is_present_or_no_metric_is_asked(self, question):
        assert not is_underspecified(question)

    def test_the_eval_suite_splits_exactly_as_classified(self):
        """Both failure directions at once, against the dataset itself.

        Exactly the three ambiguous-refuse questions fire; every other
        question in the suite — the 38 answerable ones most of all — stays
        untouched. If a rewording of the dataset breaks this, the detector
        and the dataset have genuinely diverged and one of them is wrong.
        """
        items = json.loads(EVAL_DATASET.read_text(encoding="utf-8"))
        fired = {item["question"] for item in items if is_underspecified(item["question"])}
        expected = {
            item["question"]
            for item in items
            if item["question_type"] == "ambiguous" and item["expected_behavior"] == "refuse"
        }
        assert len(expected) == 3
        assert fired == expected

    def test_clarification_names_every_registry_company(self):
        from src.edgar.client import COMPANY_REGISTRY

        detail = clarification_detail()
        for company in COMPANY_REGISTRY.values():
            assert company["name"] in detail


class TestRefusalPath:
    def test_refuses_before_the_generation_call(self):
        with patch("src.generation.answer.query_with_context") as generate:
            result = generate_grounded_answer(
                "How much did the bank set aside for credit losses?", [], verify=False
            )

        generate.assert_not_called()
        assert result.generated is False
        assert result.unanswered is not None
        assert result.unanswered.reason == AMBIGUOUS_ENTITY

    def test_refusal_is_labelled_low_confidence(self):
        """`_refusal_correctness` counts a refusal only when the label agrees
        with the structure — a report that still claims confidence is a bug,
        not a refusal."""
        result = generate_grounded_answer("What was total revenue last year?", [], verify=False)

        assert result.confidence.label == "low"

    def test_refusal_tells_the_asker_which_companies_are_available(self):
        result = generate_grounded_answer("What are the company's main risks?", [], verify=False)

        assert result.unanswered is not None
        assert "Apple" in result.unanswered.summary
        assert "Goldman Sachs" in result.unanswered.summary

    def test_a_named_company_bypasses_the_check_entirely(self):
        answer = "Apple's revenue was $416,161 million [1]."
        with patch("src.generation.answer.query_with_context", return_value=answer):
            result = generate_grounded_answer(
                "What was Apple's revenue last year?",
                [],
                verify=False,
            )

        assert result.unanswered is None
        assert result.generated is True
