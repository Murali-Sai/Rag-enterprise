from typing import Protocol

from langchain_core.documents import Document

from src.auth.rbac import get_accessible_departments
from src.common.logging import get_logger
from src.config import settings
from src.retrieval.bm25 import get_bm25_index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank_documents
from src.retrieval.vector_store import VectorStoreBase, get_vector_store

logger = get_logger(__name__)


class Retriever(Protocol):
    """What every retrieval stage exposes, so they can be composed."""

    def retrieve(self, query: str) -> list[Document]: ...


class RBACRetriever:
    """Retriever that filters documents based on user roles.

    Uses the RBAC department mapping to build a ChromaDB where-filter
    so only documents from accessible departments are returned.
    Information barriers (Chinese Walls) are enforced via get_accessible_departments().
    """

    def __init__(
        self,
        user_roles: set[str],
        vector_store: VectorStoreBase | None = None,
        top_k: int | None = None,
    ):
        self.user_roles = user_roles
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k or settings.retrieval_top_k

    def accessible_departments(self) -> frozenset[str] | None:
        """Departments this user may read; None means unrestricted (admin)."""
        if "admin" in self.user_roles:
            return None
        return frozenset(get_accessible_departments(self.user_roles))

    def build_role_filter(self) -> dict | None:
        if "admin" in self.user_roles:
            return None  # Admin sees everything

        # Get departments this user can access (with Chinese Wall enforcement)
        accessible = get_accessible_departments(self.user_roles)

        if not accessible:
            # No accessible departments — return impossible filter
            return {"department": {"$eq": "__none__"}}

        # ChromaDB $in filter on the department metadata field
        dept_list = sorted(accessible)
        if len(dept_list) == 1:
            return {"department": {"$eq": dept_list[0]}}
        return {"department": {"$in": dept_list}}

    def retrieve(self, query: str) -> list[Document]:
        role_filter = self.build_role_filter()
        logger.info(
            "rbac_retrieval",
            query_preview=query[:50],
            user_roles=list(self.user_roles),
            filter_applied=role_filter is not None,
        )

        results = self.vector_store.similarity_search(
            query=query,
            k=self.top_k,
            filter_dict=role_filter,
        )

        logger.info("retrieval_complete", results_count=len(results))
        return results

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        role_filter = self.build_role_filter()

        from src.retrieval.vector_store import ChromaVectorStore

        if isinstance(self.vector_store, ChromaVectorStore):
            return self.vector_store.similarity_search_with_score(
                query=query,
                k=self.top_k,
                filter_dict=role_filter,
            )

        docs = self.retrieve(query)
        return [(doc, 0.0) for doc in docs]


class HybridRetriever:
    """Dense + BM25 retrieval fused with Reciprocal Rank Fusion.

    Runs the same query through ChromaDB's embedding search (good at
    paraphrase) and a BM25 index (good at exact terms — tickers, "Item 7A",
    dollar figures), then fuses the two rankings by rank rather than score.

    Both halves are built from the same RBAC where-filter, so the lexical
    path is scoped to the same departments as the dense path. Fusion can
    only reorder the union of two already-authorized result sets.
    """

    def __init__(
        self,
        user_roles: set[str],
        vector_store: VectorStoreBase | None = None,
        top_k: int | None = None,
    ):
        self.user_roles = user_roles
        self.top_k = top_k or settings.retrieval_top_k
        self.vector_store = vector_store or get_vector_store()
        self._dense = RBACRetriever(
            user_roles=user_roles,
            vector_store=self.vector_store,
            top_k=self.top_k,
        )

    def retrieve(self, query: str) -> list[Document]:
        dense_results = self._dense.retrieve(query)

        index = get_bm25_index(
            vector_store=self.vector_store,
            accessible_departments=self._dense.accessible_departments(),
            filter_dict=self._dense.build_role_filter(),
        )
        sparse_results = index.search(query, k=self.top_k)

        logger.info(
            "hybrid_retrieval",
            dense_results=len(dense_results),
            sparse_results=len(sparse_results),
        )
        return reciprocal_rank_fusion([dense_results, sparse_results], top_k=self.top_k)


class RerankingRetriever:
    """Wraps a base retriever with a cross-encoder re-ranking stage.

    The base pulls a wide candidate set (default 20), then the cross-encoder
    re-scores those candidates and cuts to the final top_k passed to the LLM.
    Because the base already applied the RBAC/Chinese-Wall filter at the
    ChromaDB query level, re-ranking only ever narrows an already-authorized
    candidate set — it can't surface a document the filter excluded.
    """

    def __init__(
        self,
        user_roles: set[str],
        vector_store: VectorStoreBase | None = None,
        final_top_k: int | None = None,
        candidate_k: int | None = None,
        base: Retriever | None = None,
    ):
        self.user_roles = user_roles
        self.final_top_k = final_top_k or settings.retrieval_top_k
        self.candidate_k = candidate_k or settings.rerank_candidate_k
        self._base = base or RBACRetriever(
            user_roles=user_roles,
            vector_store=vector_store,
            top_k=self.candidate_k,
        )

    def retrieve(self, query: str) -> list[Document]:
        candidates = self._base.retrieve(query)
        return rerank_documents(query, candidates, self.final_top_k)


def get_retriever(
    user_roles: set[str],
    vector_store: VectorStoreBase | None = None,
    top_k: int | None = None,
) -> Retriever:
    """Build the retriever the app should actually use for a query.

    Composes the enabled stages into the standard pipeline:

        dense (+ BM25, fused by RRF) -> cross-encoder rerank -> top_k

    Each stage is independently toggleable so the eval harness can measure
    them in isolation. Centralizing the choice here means the REST API, MCP
    server, and eval harness all resolve the same pipeline.
    """
    final_top_k = top_k or settings.retrieval_top_k
    # When reranking follows, the first stage retrieves a wide candidate set
    # for it to re-score; otherwise it returns the final result directly.
    candidate_k = settings.rerank_candidate_k if settings.rerank_enabled else final_top_k

    base: Retriever
    if settings.hybrid_search_enabled:
        base = HybridRetriever(user_roles=user_roles, vector_store=vector_store, top_k=candidate_k)
    else:
        base = RBACRetriever(user_roles=user_roles, vector_store=vector_store, top_k=candidate_k)

    if settings.rerank_enabled:
        return RerankingRetriever(
            user_roles=user_roles,
            vector_store=vector_store,
            final_top_k=final_top_k,
            candidate_k=candidate_k,
            base=base,
        )
    return base
