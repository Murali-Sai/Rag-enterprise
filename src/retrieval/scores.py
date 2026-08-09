"""A score channel that survives the retrieval pipeline.

Every stage in the pipeline produces a relevance number and every stage used
to throw it away — `retrieve()` returns `list[Document]`, and
`rerank_documents()` computed cross-encoder scores only to drop them on the
way out. That is fine while the only consumer is the LLM, which sees a
ranked list and nothing else. It stops being fine the moment the system has
to say *how sure it is*: without a score there is no retrieval confidence,
and without retrieval confidence there is no threshold below which the
honest answer is "I don't know" (Project 6, §3.4).

Rather than change the shape of `Retriever.retrieve()` — which every stage,
the API, the MCP server and the eval harness all depend on — stages grow a
parallel `retrieve_scored()`. Callers that want scores ask for them; callers
that don't are untouched.

Scores from different stages are not on the same scale, so a raw float is
useless on its own. Each one is tagged with the stage that produced it and
normalised to a common 0–1 relevance through `ScoredDocument.relevance`.
"""

import math
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document


class ScoreType(str, Enum):
    """Which stage produced a score, and therefore how to read it."""

    # Cross-encoder logit from ms-marco-MiniLM. Unbounded either side of
    # zero (roughly -11..+11 in practice), and strongly bimodal: a clearly
    # relevant pair lands around +5 and up, a clearly irrelevant one around
    # -8. Squashed with a logistic, which is what the model was trained
    # against, so the result reads as a relevance probability.
    CROSS_ENCODER = "cross_encoder"
    # Chroma's similarity_search_with_relevance_scores: already 0–1, higher
    # is better, but far less separated than the cross-encoder — a good match
    # sits near 0.5, not near 1.0. Passed through unchanged and clamped.
    COSINE_RELEVANCE = "cosine_relevance"
    # Reciprocal Rank Fusion. Deliberately ordinal: RRF discards the
    # underlying scores and fuses ranks, so its output says "this came first"
    # and nothing at all about how relevant first was. Normalising it would
    # invent a confidence that the number does not carry, so it maps to None
    # and the confidence layer treats retrieval confidence as unavailable.
    RRF = "rrf"


@dataclass(frozen=True)
class ScoredDocument:
    """A retrieved document alongside the score that retrieved it."""

    document: Document
    score: float | None = None
    score_type: ScoreType | None = None

    @property
    def relevance(self) -> float | None:
        """The score as a 0–1 relevance, or None if it doesn't carry one."""
        if self.score is None or self.score_type is None:
            return None
        if self.score_type is ScoreType.CROSS_ENCODER:
            return _logistic(self.score)
        if self.score_type is ScoreType.COSINE_RELEVANCE:
            return min(1.0, max(0.0, self.score))
        return None


def _logistic(x: float) -> float:
    # Written out rather than 1/(1+exp(-x)) because a large negative logit
    # overflows exp() on the way to a value that is simply 0.0.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def as_scored(documents: list[Document]) -> list[ScoredDocument]:
    """Wrap plain documents so a scoreless retriever still fits the channel."""
    return [ScoredDocument(document=doc) for doc in documents]


def documents_of(scored: list[ScoredDocument]) -> list[Document]:
    return [item.document for item in scored]
