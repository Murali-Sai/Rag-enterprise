"""Tests for how the eval harness scores questions that have no answer.

The system can now be right by refusing, and the four RAGAS metrics cannot
express that: a refusal genuinely has no relevance to the question, so RAGAS
scores a correct one 0.000 answer relevancy. Pooling no-answer questions into
the aggregates would therefore lower every number precisely because the
system behaved correctly, and adding ten of them would look like a
regression. These tests pin the two halves of the alternative — the mask that
keeps them out of the RAGAS means, and the axis they are scored on instead.
"""

from evaluation.run_evaluation import (
    _aggregate_citation_metrics,
    _build_per_question,
    _refusal_correctness,
    expected_behavior,
    scored_by_ragas,
)


def question(behaviour: str | None = None, **extra) -> dict:
    item = {"question": "q?", "ground_truth": "gt", **extra}
    if behaviour is not None:
        item["expected_behavior"] = behaviour
    return item


def answered(**extra) -> dict:
    return {
        "question": "q?",
        "answer": "An answer [1].",
        "contexts": ["c"],
        "citation_accuracy": 1.0,
        "citation_coverage": 1.0,
        "confidence": 0.9,
        "confidence_label": "high",
        "unanswered": None,
        "generated": True,
        **extra,
    }


def refused(reason: str = "model_refused", label: str = "low", **extra) -> dict:
    return {
        "question": "q?",
        "answer": "I don't have enough information in the available documents.",
        "contexts": ["c"],
        "citation_accuracy": None,
        "citation_coverage": 0.0,
        "confidence": 0.1,
        "confidence_label": label,
        "unanswered": {"reason": reason},
        "generated": reason != "low_retrieval_confidence",
        **extra,
    }


class TestExpectedBehavior:
    def test_defaults_to_answer(self):
        """Every question written before v4 is answerable by construction, so
        an absent field must not read as 'no expectation'."""
        assert expected_behavior(question()) == "answer"
        assert scored_by_ragas(question()) is True

    def test_refusal_questions_are_kept_out_of_ragas(self):
        assert scored_by_ragas(question("refuse")) is False


class TestRefusalCorrectness:
    def test_correct_refusal_scores_one(self):
        assert _refusal_correctness(refused(), question("refuse")) == 1.0

    def test_answering_a_question_the_corpus_cannot_answer_scores_zero(self):
        assert _refusal_correctness(answered(), question("refuse")) == 0.0

    def test_refusal_must_also_be_labelled_low_confidence(self):
        """Returning the structured 'I don't know' while still claiming
        confidence is a different bug from failing to refuse, and the label is
        what a reader acts on."""
        assert _refusal_correctness(refused(label="medium"), question("refuse")) == 0.0

    def test_answering_an_answerable_question_scores_one(self):
        assert _refusal_correctness(answered(), question()) == 1.0

    def test_declining_an_answerable_question_scores_zero(self):
        """Over-refusal is the live failure: the first measured run declined
        Apple's total net revenue, a figure that is in the index, because the
        cross-encoder scored every candidate below the gate. Nothing in the
        RAGAS four distinguishes that from a genuine miss."""
        assert _refusal_correctness(refused("low_retrieval_confidence"), question()) == 0.0

    def test_a_harness_error_is_not_scored(self):
        """No behaviour was observed. Scoring it zero would blame the system
        for the harness falling over."""
        assert _refusal_correctness({"error": True}, question()) is None


class TestPerQuestionRows:
    def test_no_answer_rows_still_appear_with_null_ragas_columns(self):
        """A no-answer question that vanished from the report would take the
        whole refusal path with it — the gap Phase 4 exists to close."""
        eval_data = [question("refuse", question_type="no_answer")]
        rows = _build_per_question(eval_data, [refused()], ragas_by_index={})

        assert len(rows) == 1
        assert rows[0]["faithfulness"] is None
        assert rows[0]["answer_relevancy"] is None
        assert rows[0]["refusal_correctness"] == 1.0
        assert rows[0]["refusal_reason"] == "model_refused"

    def test_ragas_scores_join_by_index_not_position(self):
        """The index survives the no-answer rows being filtered out of RAGAS.
        Joining by position through that filter would attach every score to
        the wrong question and still produce a plausible-looking file."""
        eval_data = [
            question("refuse", question_type="no_answer"),
            question(question_type="exact_figure"),
        ]
        results = [refused(), answered()]
        # Only index 1 went to RAGAS.
        rows = _build_per_question(eval_data, results, ragas_by_index={1: {"faithfulness": 0.8}})

        assert rows[0]["faithfulness"] is None
        assert rows[1]["faithfulness"] == 0.8

    def test_error_rows_keep_the_alignment(self):
        eval_data = [question(), question()]
        results = [{"question": "q?", "answer": "Error: boom", "contexts": [], "error": True}]
        results.append(answered())

        rows = _build_per_question(eval_data, results, ragas_by_index={1: {"faithfulness": 1.0}})

        assert rows[0]["refusal_correctness"] is None
        assert rows[1]["faithfulness"] == 1.0


class TestCitationAggregates:
    def test_a_correct_refusal_does_not_depress_citation_coverage(self):
        """A refusal cites nothing. Folding its 0.0 coverage into the mean
        would make the citation numbers fall as the no-answer stratum grows,
        which describes the dataset rather than the system."""
        rows = _build_per_question(
            [question(), question("refuse")],
            [answered(), refused()],
            ragas_by_index={},
        )
        answerable = [r for r in rows if r["expected_behavior"] == "answer"]

        aggregate = _aggregate_citation_metrics(answerable)

        assert aggregate["citation_coverage"] == 1.0
        assert aggregate["citation_coverage_n"] == 1

    def test_missing_citation_accuracy_is_skipped_not_zeroed(self):
        """An answer that cited nothing has no citation accuracy — the failure
        belongs in coverage."""
        rows = _build_per_question(
            [question(), question()],
            [answered(), answered(citation_accuracy=None, citation_coverage=0.0)],
            ragas_by_index={},
        )

        aggregate = _aggregate_citation_metrics(rows)

        assert aggregate["citation_accuracy"] == 1.0
        assert aggregate["citation_accuracy_n"] == 1
        assert aggregate["citation_coverage"] == 0.5
