"""Tests for the bounded agentic retry.

Two properties matter more than whether it ever recovers anything, and both
are failure-shaped: **it must stop**, and **it must fail closed**. An agent
that loops is a billing incident, and one that talks itself into answering a
correctly-refused question converts the strongest number in the project into
a regression. Everything here is a fake LLM — the loop's control flow is
deterministic and worth pinning exactly, and none of these tests spends money.
"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.config import settings
from src.generation.agent import (
    ANSWERED,
    GAVE_UP,
    ITERATION_CAP,
    NO_SEARCH,
    TOOL_FAILURES,
    UNSUPPORTED,
    attempt_recovery,
)

CHUNK = Document(
    page_content="Goldman Sachs operates through segments including Global Banking & Markets.",
    metadata={"ticker": "GS", "section_name": "Business", "source_file": "GS_10-K.html"},
)


def call(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "c1"}],
    )


class FakeLLM:
    """Replays a scripted sequence of model replies, and counts invocations."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.invocations = 0

    def bind_tools(self, _specs):
        return self

    def invoke(self, _messages):
        self.invocations += 1
        if not self.replies:
            return AIMessage(content="NOTFOUND")
        return self.replies.pop(0)


class ExplodingLLM:
    def bind_tools(self, _specs):
        raise NotImplementedError("this provider does not support tool calling")


def with_llm(llm):
    return patch("src.generation.agent.get_llm", return_value=llm)


def search_returning(*docs):
    return lambda _query: list(docs)


class TestTermination:
    def test_stops_at_the_iteration_cap_when_the_model_keeps_calling_tools(self):
        """The property that makes this safe to run on a paid API: a model
        that never stops is stopped for it."""
        llm = FakeLLM(*[call("search_filings", {"query": f"attempt {i}"}) for i in range(10)])

        with with_llm(llm):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK), max_iterations=2)

        assert outcome.stopped_because == ITERATION_CAP
        assert llm.invocations == 2, "the cap bounds model turns, not just tool calls"
        assert len(outcome.steps) == 2
        assert not outcome.recovered

    def test_a_cap_of_zero_runs_nothing(self):
        llm = FakeLLM(call("search_filings", {"query": "x"}))

        with with_llm(llm):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK), max_iterations=0)

        assert outcome.stopped_because == ITERATION_CAP
        assert llm.invocations == 0

    def test_the_configured_cap_is_used_when_none_is_passed(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_max_iterations", 1)
        llm = FakeLLM(*[call("search_filings", {"query": "x"}) for _ in range(5)])

        with with_llm(llm):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK))

        assert llm.invocations == 1
        assert outcome.stopped_because == ITERATION_CAP


class TestFailingClosed:
    def test_notfound_keeps_the_original_refusal(self):
        with with_llm(FakeLLM(AIMessage(content="NOTFOUND"))):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK))

        assert outcome.stopped_because == GAVE_UP
        assert outcome.answer is None
        assert not outcome.recovered

    def test_an_empty_reply_is_not_treated_as_an_answer(self):
        with with_llm(FakeLLM(AIMessage(content="   "))):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK))

        assert outcome.stopped_because == GAVE_UP
        assert not outcome.recovered

    def test_no_search_callable_means_no_recovery_attempted(self):
        """Without an injected search the agent would have to build its own
        retriever, which would answer to nobody's roles."""
        outcome = attempt_recovery("q", [CHUNK], None)

        assert outcome.stopped_because == NO_SEARCH
        assert not outcome.recovered

    def test_a_provider_without_tool_calling_degrades_instead_of_raising(self):
        with with_llm(ExplodingLLM()):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK))

        assert outcome.stopped_because == UNSUPPORTED
        assert not outcome.recovered

    def test_a_model_call_that_raises_keeps_the_refusal(self):
        class Failing(FakeLLM):
            def invoke(self, _messages):
                raise RuntimeError("provider exploded")

        with with_llm(Failing()):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK))

        assert not outcome.recovered


class TestToolErrorHandling:
    def test_a_tool_that_raises_becomes_an_observation_rather_than_a_crash(self):
        def exploding_search(_query):
            raise ConnectionError("vector store unreachable")

        llm = FakeLLM(call("search_filings", {"query": "x"}), AIMessage(content="NOTFOUND"))

        with with_llm(llm):
            outcome = attempt_recovery("q", [CHUNK], exploding_search, max_iterations=3)

        assert outcome.steps[0].failed
        assert "ConnectionError" in outcome.steps[0].observation
        assert outcome.stopped_because == GAVE_UP, "the loop continued after the failure"

    def test_repeated_tool_failures_abort_the_loop(self):
        def exploding_search(_query):
            raise ConnectionError("still down")

        llm = FakeLLM(*[call("search_filings", {"query": "x"}) for _ in range(5)])

        with with_llm(llm), patch.object(settings, "agent_max_tool_errors", 2):
            outcome = attempt_recovery("q", [CHUNK], exploding_search, max_iterations=5)

        assert outcome.stopped_because == TOOL_FAILURES
        assert len(outcome.steps) == 2, "aborted rather than retrying into a wall"

    def test_an_unknown_tool_is_reported_not_raised(self):
        llm = FakeLLM(call("delete_everything", {}), AIMessage(content="NOTFOUND"))

        with with_llm(llm):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK), max_iterations=3)

        assert outcome.steps[0].failed
        assert "Unknown tool" in outcome.steps[0].observation

    def test_an_empty_query_is_rejected_without_searching(self):
        called = []
        llm = FakeLLM(call("search_filings", {"query": "  "}), AIMessage(content="NOTFOUND"))

        with with_llm(llm):
            attempt_recovery("q", [CHUNK], lambda q: called.append(q) or [CHUNK], max_iterations=3)

        assert not called

    def test_a_search_returning_nothing_is_not_an_error(self):
        llm = FakeLLM(call("search_filings", {"query": "x"}), AIMessage(content="NOTFOUND"))

        with with_llm(llm):
            outcome = attempt_recovery("q", [CHUNK], lambda _q: [], max_iterations=3)

        assert not outcome.steps[0].failed


class TestRecovery:
    def test_an_answer_after_a_search_is_returned_with_that_search_s_documents(self):
        """Citations in the recovered answer are numbered against the passages
        the agent's own search returned — carrying the first attempt's
        documents forward would verify claims against the wrong text."""
        found = Document(
            page_content="Segments: Global Banking & Markets.", metadata={"ticker": "GS"}
        )
        llm = FakeLLM(
            call("search_filings", {"query": "Goldman business segments"}),
            AIMessage(content="Goldman Sachs reports three segments [1]."),
        )

        with with_llm(llm):
            outcome = attempt_recovery("q", [CHUNK], search_returning(found), max_iterations=3)

        assert outcome.stopped_because == ANSWERED
        assert outcome.recovered
        assert outcome.documents == [found]

    def test_an_answer_without_any_search_keeps_the_original_documents(self):
        with with_llm(FakeLLM(AIMessage(content="It is disclosed in Item 1 [1]."))):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK))

        assert outcome.recovered
        assert outcome.documents == [CHUNK]

    def test_every_step_is_recorded_for_the_audit_trail(self):
        llm = FakeLLM(
            call("list_sections", {"ticker": "GS"}),
            AIMessage(content="NOTFOUND"),
        )

        with (
            with_llm(llm),
            patch("src.generation.agent.list_sections", return_value="GS: 2,888 passages"),
        ):
            outcome = attempt_recovery("q", [CHUNK], search_returning(CHUNK), max_iterations=3)

        assert len(outcome.steps) == 1
        assert outcome.steps[0].tool == "list_sections"
        assert outcome.steps[0].arguments == {"ticker": "GS"}
        assert outcome.as_dict()["iterations"] == 1


class TestWiring:
    def test_generation_does_not_invoke_the_agent_when_the_setting_is_off(self, monkeypatch):
        from src.generation.answer import generate_grounded_answer

        monkeypatch.setattr(settings, "agentic_recovery_enabled", False)
        refusal = "I don't have enough information in the available documents to answer this."

        with (
            patch("src.generation.answer.query_with_context", return_value=refusal),
            patch("src.generation.answer.attempt_recovery") as agent,
        ):
            result = generate_grounded_answer("q", [CHUNK], verify=False, search=lambda _q: [CHUNK])

        agent.assert_not_called()
        assert result.agent is None
        assert result.unanswered is not None

    def test_an_answered_question_never_reaches_the_agent(self, monkeypatch):
        from src.generation.answer import generate_grounded_answer

        monkeypatch.setattr(settings, "agentic_recovery_enabled", True)

        with (
            patch("src.generation.answer.query_with_context", return_value="Revenue was $1 [1]."),
            patch("src.generation.answer.attempt_recovery") as agent,
        ):
            generate_grounded_answer("q", [CHUNK], verify=False, search=lambda _q: [CHUNK])

        agent.assert_not_called()

    @pytest.mark.parametrize("enabled", [True, False])
    def test_a_failed_recovery_leaves_the_refusal_untouched(self, monkeypatch, enabled):
        from src.generation.agent import AgentOutcome
        from src.generation.answer import generate_grounded_answer

        monkeypatch.setattr(settings, "agentic_recovery_enabled", enabled)
        refusal = "I don't have enough information in the available documents to answer this."

        with (
            patch("src.generation.answer.query_with_context", return_value=refusal),
            patch(
                "src.generation.answer.attempt_recovery",
                return_value=AgentOutcome(stopped_because=GAVE_UP),
            ),
        ):
            result = generate_grounded_answer("q", [CHUNK], verify=False, search=lambda _q: [CHUNK])

        assert result.answer == refusal
        assert result.unanswered is not None
        assert result.confidence.label == "low"
