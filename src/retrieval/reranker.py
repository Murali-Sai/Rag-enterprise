"""Cross-encoder re-ranking for retrieved documents.

ChromaDB's similarity_search scores documents with a bi-encoder (embed query,
embed doc, compare vectors independently) — fast enough to search the whole
corpus, but the query and document never actually attend to each other. A
cross-encoder scores each (query, document) pair jointly, which is far more
precise at judging relevance — at the cost of being too slow to run over an
entire collection. The fix is the usual two-stage retrieval pattern: let the
bi-encoder pull a wide candidate set, then let the cross-encoder re-order
just that small set before it reaches the LLM.
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.common.logging import get_logger
from src.config import settings
from src.retrieval.passages import passage_text
from src.retrieval.scores import ScoredDocument, ScoreType, documents_of

logger = get_logger(__name__)

_reranker_instance: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        logger.info("loading_reranker_model", model=settings.rerank_model)
        _reranker_instance = CrossEncoder(settings.rerank_model)
    return _reranker_instance


def rerank_with_scores(query: str, documents: list[Document], top_k: int) -> list[ScoredDocument]:
    """Re-score candidate documents with a cross-encoder and return the top_k.

    The scores come back with the documents rather than being dropped on the
    floor. A cross-encoder logit is the best relevance signal the pipeline
    produces — the query and the document were scored jointly, unlike the
    bi-encoder cosine — so it is what the confidence layer reads to decide
    whether an answer is worth generating at all.

    Args:
        query: The user's question.
        documents: Candidate documents from a wider bi-encoder retrieval pass.
        top_k: How many documents to keep after re-scoring.

    Returns:
        The top_k documents with their raw cross-encoder logits, ordered by
        relevance (descending).
    """
    if not documents:
        return []

    model = get_reranker()
    pairs = [(query, passage_text(doc)) for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(zip(documents, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    top = [
        ScoredDocument(
            document=doc,
            score=float(score),
            score_type=ScoreType.CROSS_ENCODER,
        )
        for doc, score in ranked[:top_k]
    ]

    logger.info(
        "reranking_complete",
        candidates=len(documents),
        returned=len(top),
        top_score=float(ranked[0][1]) if ranked else None,
    )
    return top


def rerank_documents(query: str, documents: list[Document], top_k: int) -> list[Document]:
    """Documents-only view of rerank_with_scores, for callers that don't score."""
    return documents_of(rerank_with_scores(query, documents, top_k))
