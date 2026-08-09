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

Rankings can be weighted, so the sum becomes

    score(d) = sum over rankings of weight_i / (k + rank_i(d))

which is what lets one retriever count for more than the other on a corpus
where it is known to be the better signal. Only the *ratio* between weights
changes anything: the output is a ranking, and scaling every weight by the
same factor scales every score by it too. So 0.7/0.3 and 7/3 are the same
configuration, and there is nothing to normalise.
"""

from collections.abc import Sequence
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
    weights: Sequence[float] | None = None,
) -> list[Document]:
    """Fuse several ranked document lists into one.

    Args:
        rankings: Ranked lists, each ordered best-first.
        top_k: How many documents to return.
        k: RRF damping constant.
        weights: Per-ranking multipliers, positionally matched to `rankings`.
            None weights every ranking equally. Only relative magnitude
            matters — see the module docstring.

    Returns:
        Up to top_k documents ordered by fused score.

    Raises:
        ValueError: If `weights` is a different length than `rankings`, or
            contains a negative value.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        # Silently zipping to the shorter of the two would drop a whole
        # retriever from the fusion, and the result would still look like a
        # plausible ranking.
        raise ValueError(
            f"weights has {len(weights)} entries for {len(rankings)} rankings; "
            "they are matched by position and must be the same length"
        )
    elif any(weight < 0 for weight in weights):
        # A negative weight demotes documents for being *well* ranked, which
        # is not a tuning choice anyone means to make.
        raise ValueError(f"weights must be non-negative, got {list(weights)}")

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc in enumerate(ranking):
            key = _document_key(doc)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            documents.setdefault(key, doc)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    result = [documents[key] for key, _ in fused[:top_k]]

    logger.info(
        "rrf_fusion_complete",
        input_rankings=len(rankings),
        weights=list(weights),
        unique_documents=len(documents),
        returned=len(result),
    )
    return result
