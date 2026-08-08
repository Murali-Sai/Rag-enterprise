"""Reciprocal Rank Fusion for combining retrieval rankings.

Dense cosine similarity and BM25 produce scores on incompatible scales — a
0.82 cosine and a 14.3 BM25 score can't be averaged, and normalizing them
requires per-query calibration that shifts as the corpus changes. RRF sidesteps
the problem by discarding scores and fusing *ranks*:

    score(d) = sum over rankings of 1 / (k + rank(d))

The k constant (60 by convention, from Cormack et al. 2009) damps the
contribution of top ranks so a single retriever's #1 hit can't dominate a
document that both retrievers ranked respectably. Documents found by both
paths accumulate contributions and rise; documents found by only one still
place, which is what makes hybrid recall better than either input.
"""

from hashlib import sha256

from langchain_core.documents import Document

from src.common.logging import get_logger

logger = get_logger(__name__)

RRF_K = 60


def _document_key(doc: Document) -> str:
    """Identity for deduplication across rankings.

    Chunk text is the identity — the same chunk retrieved by both the dense
    and sparse paths must collapse to one entry, and Document objects coming
    back from separate queries are distinct Python objects.
    """
    return sha256(doc.page_content.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    top_k: int,
    k: int = RRF_K,
) -> list[Document]:
    """Fuse several ranked document lists into one.

    Args:
        rankings: Ranked lists, each ordered best-first.
        top_k: How many documents to return.
        k: RRF damping constant.

    Returns:
        Up to top_k documents ordered by fused score.
    """
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = _document_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            documents.setdefault(key, doc)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    result = [documents[key] for key, _ in fused[:top_k]]

    logger.info(
        "rrf_fusion_complete",
        input_rankings=len(rankings),
        unique_documents=len(documents),
        returned=len(result),
    )
    return result
