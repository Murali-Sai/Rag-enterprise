from collections.abc import Callable
from typing import Protocol

from langchain_core.documents import Document

from src.auth.rbac import get_accessible_departments
from src.common.logging import get_logger
from src.config import settings
from src.retrieval.bm25 import get_bm25_index
from src.retrieval.entities import detect_entities, entity_filter, scope_query
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hyde import build_retrieval_query
from src.retrieval.reranker import rerank_with_scores
from src.retrieval.scores import (
    ScoredDocument,
    ScoreType,
    as_scored,
    documents_of,
)
from src.retrieval.vector_store import VectorStoreBase, get_vector_store

logger = get_logger(__name__)


class Retriever(Protocol):
    """What every retrieval stage exposes, so they can be composed."""

    def retrieve(self, query: str) -> list[Document]: ...


class ScoringRetriever(Protocol):
    """A retriever that can also hand back the scores it ranked by.

    Deliberately a second protocol rather than a method on `Retriever`: the
    score channel is additive, so a retriever from outside this module (a
    test double, a future stage) stays usable by everything that only needs
    documents. `retrieve_scored()` below bridges the two.
    """

    def retrieve(self, query: str) -> list[Document]: ...

    def retrieve_scored(self, query: str) -> list[ScoredDocument]: ...


def retrieve_scored(retriever: Retriever, query: str) -> list[ScoredDocument]:
    """Retrieve with scores from any retriever, scored or not.

    A retriever that doesn't implement the score channel yields documents
    with `score=None`, which the confidence layer reads as "no retrieval
    signal available" rather than as a score of zero. That distinction is the
    whole point: a zero would read as a confident "nothing is relevant".
    """
    scored = getattr(retriever, "retrieve_scored", None)
    if callable(scored):
        return scored(query)
    return as_scored(retriever.retrieve(query))


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
        extra_filter: dict | None = None,
    ):
        self.user_roles = user_roles
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k or settings.retrieval_top_k
        self.extra_filter = extra_filter

    def accessible_departments(self) -> frozenset[str] | None:
        """Departments this user may read; None means unrestricted (admin)."""
        if "admin" in self.user_roles:
            return None
        return frozenset(get_accessible_departments(self.user_roles))

    def build_role_filter(self) -> dict | None:
        """The where-clause for this query: RBAC, plus any caller restriction.

        `extra_filter` narrows retrieval further (per-entity retrieval uses it
        to pin a company). It is intersected with the role filter, never
        substituted for it, so a caller-supplied clause can only ever remove
        documents from what the user is already allowed to see.
        """
        role_filter = self._role_filter()
        if self.extra_filter is None:
            return role_filter
        if role_filter is None:
            return self.extra_filter
        return {"$and": [role_filter, self.extra_filter]}

    def _role_filter(self) -> dict | None:
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

    def retrieve_scored(self, query: str) -> list[ScoredDocument]:
        """Same retrieval, with Chroma's relevance scores attached.

        Stores that can't produce scores fall through to unscored documents
        rather than to 0.0 — the previous behaviour, which was indistinguishable
        from "every result is maximally irrelevant".
        """
        role_filter = self.build_role_filter()

        from src.retrieval.vector_store import ChromaVectorStore

        if isinstance(self.vector_store, ChromaVectorStore):
            return [
                ScoredDocument(
                    document=doc,
                    score=float(score),
                    score_type=ScoreType.COSINE_RELEVANCE,
                )
                for doc, score in self.vector_store.similarity_search_with_score(
                    query=query,
                    k=self.top_k,
                    filter_dict=role_filter,
                )
            ]

        return as_scored(self.retrieve(query))


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
        extra_filter: dict | None = None,
    ):
        self.user_roles = user_roles
        self.top_k = top_k or settings.retrieval_top_k
        self.vector_store = vector_store or get_vector_store()
        self._dense = RBACRetriever(
            user_roles=user_roles,
            vector_store=self.vector_store,
            top_k=self.top_k,
            extra_filter=extra_filter,
        )

    def retrieve(self, query: str) -> list[Document]:
        dense_results = self._dense.retrieve(query)

        role_filter = self._dense.build_role_filter()
        index = get_bm25_index(
            vector_store=self.vector_store,
            accessible_departments=self._dense.accessible_departments(),
            filter_dict=role_filter,
        )
        sparse_results = index.search(query, k=self.top_k)

        logger.info(
            "hybrid_retrieval",
            dense_results=len(dense_results),
            sparse_results=len(sparse_results),
        )
        # Read per call rather than captured at construction, so changing the
        # weighting does not need the retriever rebuilt — the eval harness
        # sweeps this by setting the values between runs.
        return reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=self.top_k,
            weights=[settings.hybrid_dense_weight, settings.hybrid_sparse_weight],
        )

    def retrieve_scored(self, query: str) -> list[ScoredDocument]:
        """Fused results, tagged RRF so downstream reads them as ordinal.

        RRF throws the input scores away by design and fuses ranks, so what
        comes out ranks documents without measuring them. Tagging the type
        keeps that honest: the confidence layer maps RRF to "no retrieval
        signal" instead of treating a fusion score as a relevance."""
        return [
            ScoredDocument(document=doc, score=None, score_type=ScoreType.RRF)
            for doc in self.retrieve(query)
        ]


class HyDERetriever:
    """Wraps a base retriever, searching with a generated passage instead of
    the raw question.

    Sits between the caller and the search stage: the base retriever never
    sees the original question, only the hypothetical passage. Anything
    downstream that needs the real question — the cross-encoder reranker,
    the generation prompt — still gets it, because those stages wrap *this*
    one and pass the untransformed query to their own scorers.
    """

    def __init__(self, base: Retriever):
        self._base = base

    def retrieve(self, query: str) -> list[Document]:
        retrieval_query = build_retrieval_query(query)
        return self._base.retrieve(retrieval_query)

    def retrieve_scored(self, query: str) -> list[ScoredDocument]:
        return retrieve_scored(self._base, build_retrieval_query(query))


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
        extra_filter: dict | None = None,
    ):
        self.user_roles = user_roles
        self.final_top_k = final_top_k or settings.retrieval_top_k
        self.candidate_k = candidate_k or settings.rerank_candidate_k
        self._base = base or RBACRetriever(
            user_roles=user_roles,
            vector_store=vector_store,
            top_k=self.candidate_k,
            extra_filter=extra_filter,
        )

    def retrieve(self, query: str) -> list[Document]:
        return documents_of(self.retrieve_scored(query))

    def retrieve_scored(self, query: str) -> list[ScoredDocument]:
        candidates = self._base.retrieve(query)
        return rerank_with_scores(query, candidates, self.final_top_k)


class MultiEntityRetriever:
    """Keeps a question's answer inside the filings it asked about.

    A question naming one company is restricted to that company's filing. A
    question naming two or more runs the pipeline once per company, each
    restricted the same way, and merges the results. A question naming none
    searches everything, which is the only case where that is what was asked
    for.

    Each company gets a full `top_k` slots rather than a share of one. The
    alternative — splitting five slots two ways — starves both halves of the
    comparison to keep the context the size it would be for a simpler
    question, which is the wrong thing to hold fixed. A comparison is two
    questions and costs two questions' worth of context.

    Each leg is scored against the question scoped to its own company rather
    than against the whole comparison, because a cross-encoder asked whether a
    passage answers "compare A and B" correctly judges any single chunk to be
    half an answer and scores it accordingly. See `scope_query()` for the
    measurement that motivated it.

    The merged list is ordered by relevance across companies, so citation [1]
    is still the best-matching passage overall. That ordering rests on a
    weaker assumption than it used to: the legs share a model but no longer
    share a query, so their logits come from different (query, passage)
    distributions and are comparable by construction of the model rather than
    by identity of the input. In exchange the numbers now mean "how well does
    this passage answer the question about *its* company", which is the thing
    worth ranking across a comparison. Under RRF, which is ordinal, `relevance`
    is None for every document and the merge falls back to interleaving by
    rank.

    The split is decided per query, at retrieve() time, because the retriever
    is constructed before the question is known.
    """

    def __init__(
        self,
        build: Callable[[dict | None], Retriever],
        top_k: int | None = None,
    ):
        self._build = build
        self.top_k = top_k or settings.retrieval_top_k
        self._single: Retriever | None = None

    def _plain(self) -> Retriever:
        if self._single is None:
            self._single = self._build(None)
        return self._single

    def retrieve(self, query: str) -> list[Document]:
        return documents_of(self.retrieve_scored(query))

    def retrieve_scored(self, query: str) -> list[ScoredDocument]:
        entities = detect_entities(query)
        if not entities:
            return retrieve_scored(self._plain(), query)

        # One company named: filter to its filing, but leave the query alone —
        # there is nothing to scope and the cross-encoder already has the
        # question it wants.
        #
        # This branch used to fall through to the unfiltered pipeline, on the
        # reasoning that a single-company question needs no budget split. It
        # does not need a split; it does need the filter. Asked for Goldman
        # Sachs' net revenues, retrieval returned two GS chunks, one Tesla, one
        # JPMorgan, and one from `annual_report_10k.txt` — a fabricated sample
        # document reading "Total Net Revenue: $38.2B / Return on Equity (ROE):
        # 14.8%", which is a perfect-looking answer about a company that is not
        # Goldman Sachs. No ground-truth claim about GS can be supported by
        # those, which is how GS scored a hard 0.000 context recall while every
        # other company scored 0.167 to 0.506.
        #
        # Banks suffer worst because their tables are structurally identical:
        # every one of them reports total net revenues and a return on equity,
        # so the wrong company's figures are not merely retrievable but
        # *convincing*.
        if len(entities) == 1:
            return retrieve_scored(self._build(entity_filter(entities[0])), query)

        # Each leg runs on the question scoped to its own company. The cross
        # encoder scores "does this passage answer this query?", and no single
        # chunk answers "compare A and B", so scoring A's chunks against the
        # whole comparison marked every one of them down — see scope_query().
        scoped = {ticker: scope_query(query, ticker, entities) for ticker in entities}
        per_entity = [
            retrieve_scored(self._build(entity_filter(ticker)), scoped[ticker])
            for ticker in entities
        ]
        merged = _merge_by_relevance(per_entity)

        logger.info(
            "multi_entity_retrieval",
            entities=list(entities),
            per_entity=[len(group) for group in per_entity],
            merged=len(merged),
            scoped_queries=[scoped[ticker] for ticker in entities],
        )
        return merged


def _merge_by_relevance(groups: list[list[ScoredDocument]]) -> list[ScoredDocument]:
    """Interleave per-entity results into one list, best first.

    Sorted by relevance when every document carries one. When any does not —
    RRF tags its output ordinal precisely so it cannot be read as a
    measurement — the sort would silently rank the scored documents above the
    unscored ones regardless of merit, so those fall back to round-robin,
    which preserves each entity's ranking and gives neither company the top
    slot by default.
    """
    if all(item.relevance is not None for group in groups for item in group):
        return sorted(
            (item for group in groups for item in group),
            key=lambda item: item.relevance,  # type: ignore[arg-type,return-value]
            reverse=True,
        )

    merged: list[ScoredDocument] = []
    for rank in range(max((len(group) for group in groups), default=0)):
        merged.extend(group[rank] for group in groups if rank < len(group))
    return merged


def get_retriever(
    user_roles: set[str],
    vector_store: VectorStoreBase | None = None,
    top_k: int | None = None,
    hybrid: bool | None = None,
) -> Retriever:
    """Build the retriever the app should actually use for a query.

    Composes the enabled stages into the standard pipeline:

        [per-entity split] -> [HyDE query rewrite]
                           -> dense (+ BM25, fused by RRF)
                           -> cross-encoder rerank -> top_k

    Stage order matters. HyDE wraps the search stage so only the search sees
    the rewritten query; reranking wraps HyDE so the cross-encoder still
    scores against the user's real question. The per-entity split wraps all of
    it, because each branch needs its own metadata filter at the database
    level — a split applied after retrieval could only re-rank chunks that a
    global top-k had already excluded, which is the failure it exists to fix.

    Each stage is independently toggleable so the eval harness can measure
    them in isolation. Centralizing the choice here means the REST API, MCP
    server, and eval harness all resolve the same pipeline.

    `hybrid` overrides `HYBRID_SEARCH_ENABLED` for one call. `None` means "use
    the configured value", which is what every existing caller gets — the
    override exists so a client can ask the same question both ways and
    compare, without a setting change that would apply to everyone. It is
    deliberately the only stage exposed this way: the eval harness measures
    stages by running under a configuration, and a per-request knob for each
    one would make "which pipeline produced this result" a property of the
    request rather than of the run.
    """
    use_hybrid = settings.hybrid_search_enabled if hybrid is None else hybrid
    final_top_k = top_k or settings.retrieval_top_k
    # When reranking follows, the first stage retrieves a wide candidate set
    # for it to re-score; otherwise it returns the final result directly.
    candidate_k = settings.rerank_candidate_k if settings.rerank_enabled else final_top_k

    def build(extra_filter: dict | None) -> Retriever:
        base: Retriever
        if use_hybrid:
            base = HybridRetriever(
                user_roles=user_roles,
                vector_store=vector_store,
                top_k=candidate_k,
                extra_filter=extra_filter,
            )
        else:
            base = RBACRetriever(
                user_roles=user_roles,
                vector_store=vector_store,
                top_k=candidate_k,
                extra_filter=extra_filter,
            )

        if settings.hyde_enabled:
            base = HyDERetriever(base=base)

        if settings.rerank_enabled:
            return RerankingRetriever(
                user_roles=user_roles,
                vector_store=vector_store,
                final_top_k=final_top_k,
                candidate_k=candidate_k,
                base=base,
            )
        return base

    if settings.multi_entity_retrieval_enabled:
        return MultiEntityRetriever(build=build, top_k=final_top_k)
    return build(None)
