"""Tests for HyDE query transformation.

The behaviour that matters: the rewritten query reaches the *search* stage
and nothing else. Downstream scorers and the generation prompt must still
see the user's real question, or the hypothetical passage's invented figures
start influencing what gets shown.
"""

from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.retrieval.hyde import build_retrieval_query, generate_hypothetical_document
from src.retrieval.retriever import HyDERetriever
from src.retrieval.scores import ScoredDocument

HYPOTHETICAL = "The Company reported total net sales of $391.0 billion in fiscal 2024."


def fake_llm(output: str = HYPOTHETICAL) -> FakeListChatModel:
    """A real Runnable, so the LCEL chain in hyde.py accepts it."""
    return FakeListChatModel(responses=[output])


class TestGenerateHypotheticalDocument:
    def test_returns_generated_passage(self):
        with patch("src.generation.llm_factory.get_llm", return_value=fake_llm()):
            result = generate_hypothetical_document("What was Apple's revenue?")

        assert result == HYPOTHETICAL

    def test_falls_back_to_original_query_when_llm_fails(self):
        """A dead LLM should degrade retrieval, not fail the request."""
        with patch("src.generation.llm_factory.get_llm", side_effect=RuntimeError("provider down")):
            result = generate_hypothetical_document("What was Apple's revenue?")

        assert result == "What was Apple's revenue?"

    def test_falls_back_when_generation_is_empty(self):
        with patch("src.generation.llm_factory.get_llm", return_value=fake_llm("   ")):
            result = generate_hypothetical_document("What was Apple's revenue?")

        assert result == "What was Apple's revenue?"


class TestBuildRetrievalQuery:
    def test_pure_hyde_discards_the_question(self):
        with (
            patch("src.generation.llm_factory.get_llm", return_value=fake_llm()),
            patch("src.retrieval.hyde.settings.hyde_include_query", False),
        ):
            result = build_retrieval_query("What was Apple's revenue?")

        assert result == HYPOTHETICAL
        assert "What was" not in result

    def test_include_query_keeps_both(self):
        with (
            patch("src.generation.llm_factory.get_llm", return_value=fake_llm()),
            patch("src.retrieval.hyde.settings.hyde_include_query", True),
        ):
            result = build_retrieval_query("What was Apple's revenue?")

        assert HYPOTHETICAL in result
        assert "What was Apple's revenue?" in result

    def test_no_duplication_when_generation_failed(self):
        """Fallback returns the query; concatenating it with itself would
        double-weight the question for no reason."""
        with (
            patch("src.generation.llm_factory.get_llm", side_effect=RuntimeError("boom")),
            patch("src.retrieval.hyde.settings.hyde_include_query", True),
        ):
            result = build_retrieval_query("What was Apple's revenue?")

        assert result == "What was Apple's revenue?"


class RecordingRetriever:
    def __init__(self):
        self.received: list[str] = []

    def retrieve(self, query: str) -> list[Document]:
        self.received.append(query)
        return [Document(page_content="retrieved", metadata={})]


class TestHyDERetriever:
    def test_search_stage_receives_the_hypothetical_not_the_question(self):
        base = RecordingRetriever()
        with (
            patch("src.generation.llm_factory.get_llm", return_value=fake_llm()),
            patch("src.retrieval.hyde.settings.hyde_include_query", False),
        ):
            HyDERetriever(base=base).retrieve("What was Apple's revenue?")

        assert base.received == [HYPOTHETICAL]

    def test_reranker_still_scores_against_the_original_question(self):
        """HyDE's invented figures must not leak into reranking — the cross
        encoder judges relevance to what the user actually asked."""
        from src.retrieval.retriever import RerankingRetriever

        base = RecordingRetriever()
        hyde = HyDERetriever(base=base)
        scored_queries: list[str] = []

        def fake_rerank(query, documents, top_k):  # noqa: ANN001
            scored_queries.append(query)
            return [ScoredDocument(document=doc) for doc in documents[:top_k]]

        with (
            patch("src.generation.llm_factory.get_llm", return_value=fake_llm()),
            patch("src.retrieval.hyde.settings.hyde_include_query", False),
            patch("src.retrieval.retriever.rerank_with_scores", side_effect=fake_rerank),
        ):
            RerankingRetriever(user_roles={"admin"}, base=hyde, final_top_k=1).retrieve(
                "What was Apple's revenue?"
            )

        assert base.received == [HYPOTHETICAL]  # search got the passage
        assert scored_queries == ["What was Apple's revenue?"]  # reranker got the question

    def test_retrieval_failure_of_llm_still_returns_documents(self):
        base = RecordingRetriever()
        with patch("src.generation.llm_factory.get_llm", side_effect=RuntimeError("provider down")):
            result = HyDERetriever(base=base).retrieve("What was Apple's revenue?")

        assert base.received == ["What was Apple's revenue?"]
        assert len(result) == 1
