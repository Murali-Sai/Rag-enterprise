"""Per-request choice of search stage.

The dashboard's hybrid-vs-dense comparison needs the same question answered
both ways in one session, which a global setting cannot express. The override
threads through `get_retriever(hybrid=...)`; everything else about the
pipeline stays configuration, so "which pipeline produced this run" remains a
property of the run for the eval harness.
"""

import pytest

from src.api.routes.query import _resolve_retriever, _retrieval_config
from src.auth.models import Role, User
from src.common.schemas import RetrievalMode
from src.config import settings
from src.retrieval.bm25 import reset_bm25_cache
from src.retrieval.retriever import (
    HybridRetriever,
    MultiEntityRetriever,
    RBACRetriever,
    get_retriever,
)


class FakeVectorStore:
    def __init__(self, docs=None):
        self.docs = docs or []

    def similarity_search(self, query, k=5, filter_dict=None):
        return self.docs[:k]

    def similarity_search_with_score(self, query, k=5, filter_dict=None):
        return [(d, 0.5) for d in self.docs[:k]]

    def get_all_documents(self, filter_dict=None):
        return self.docs


def base_stage(retriever):
    """Unwrap the pipeline down to the stage that actually searches."""
    if isinstance(retriever, MultiEntityRetriever):
        retriever = retriever._build(None)
    while getattr(retriever, "_base", None) is not None:
        retriever = retriever._base
    return retriever


@pytest.fixture(autouse=True)
def _clean_bm25():
    reset_bm25_cache()
    yield
    reset_bm25_cache()


def user_with(*roles):
    user = User(id=1, username="t", password_hash="unused")  # noqa: S106
    user.roles = [Role(id=i, name=r) for i, r in enumerate(roles, start=1)]
    return user


class TestOverride:
    def test_hybrid_true_selects_the_fused_stage(self):
        retriever = get_retriever(
            user_roles={"research"}, vector_store=FakeVectorStore(), hybrid=True
        )

        assert isinstance(base_stage(retriever), HybridRetriever)

    def test_hybrid_false_selects_dense_only(self):
        retriever = get_retriever(
            user_roles={"research"}, vector_store=FakeVectorStore(), hybrid=False
        )

        assert isinstance(base_stage(retriever), RBACRetriever)

    def test_none_follows_the_configured_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "hybrid_search_enabled", True)

        retriever = get_retriever(user_roles={"research"}, vector_store=FakeVectorStore())

        assert isinstance(base_stage(retriever), HybridRetriever)

    def test_the_override_does_not_mutate_the_setting(self, monkeypatch):
        """A per-request choice that changed the process's configuration would
        leak into the next request, and into the eval harness."""
        monkeypatch.setattr(settings, "hybrid_search_enabled", False)

        get_retriever(user_roles={"research"}, vector_store=FakeVectorStore(), hybrid=True)

        assert settings.hybrid_search_enabled is False


class TestRouteResolution:
    def test_the_default_mode_reuses_the_injected_retriever(self):
        """Not merely equivalent — the same object. The dependency is what
        tests override, and rebuilding would quietly bypass them."""
        configured = RBACRetriever(user_roles={"research"}, vector_store=FakeVectorStore())

        resolved = _resolve_retriever(RetrievalMode.DEFAULT, user_with("research"), configured)

        assert resolved is configured

    def test_a_pinned_mode_builds_a_fresh_pipeline(self):
        configured = RBACRetriever(user_roles={"research"}, vector_store=FakeVectorStore())

        resolved = _resolve_retriever(RetrievalMode.HYBRID, user_with("research"), configured)

        assert resolved is not configured
        assert isinstance(base_stage(resolved), HybridRetriever)


class TestReportedConfig:
    def test_a_pinned_mode_is_reported_as_itself(self):
        assert _retrieval_config(RetrievalMode.HYBRID).mode == "hybrid"
        assert _retrieval_config(RetrievalMode.DENSE).mode == "dense"

    def test_default_reports_what_the_server_resolved(self, monkeypatch):
        """`default` on the request says nothing; a client comparing two runs
        needs the stage that actually ran."""
        monkeypatch.setattr(settings, "hybrid_search_enabled", True)

        assert _retrieval_config(RetrievalMode.DEFAULT).mode == "hybrid"

    def test_the_other_stages_are_reported_too(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_enabled", True)
        monkeypatch.setattr(settings, "hyde_enabled", False)

        config = _retrieval_config(RetrievalMode.DENSE)

        assert config.reranked is True
        assert config.hyde is False
        assert config.top_k == settings.retrieval_top_k


class TestRequestSchema:
    def test_the_mode_defaults_to_the_server_setting(self):
        from src.common.schemas import QueryRequest

        assert QueryRequest(question="q").retrieval_mode is RetrievalMode.DEFAULT

    def test_an_unknown_mode_is_rejected(self):
        from pydantic import ValidationError

        from src.common.schemas import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(question="q", retrieval_mode="bm25-only")
