from langchain_core.documents import Document

from src.auth.rbac import get_accessible_departments
from src.common.logging import get_logger
from src.config import settings
from src.retrieval.vector_store import VectorStoreBase, get_vector_store

logger = get_logger(__name__)


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

    def _build_role_filter(self) -> dict | None:
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
        role_filter = self._build_role_filter()
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
        role_filter = self._build_role_filter()

        if not isinstance(self.vector_store, type(self.vector_store)):
            # Fallback for stores without score support
            docs = self.retrieve(query)
            return [(doc, 0.0) for doc in docs]

        from src.retrieval.vector_store import ChromaVectorStore

        if isinstance(self.vector_store, ChromaVectorStore):
            return self.vector_store.similarity_search_with_score(
                query=query,
                k=self.top_k,
                filter_dict=role_filter,
            )

        docs = self.retrieve(query)
        return [(doc, 0.0) for doc in docs]
