"""Near-duplicate suppression at ingestion time.

A duplicate chunk is not merely wasted disk. Retrieval returns a fixed top_k,
so an indexed duplicate competes for one of those slots and wins it — the
same text scores the same similarity twice, and the chunk it displaces is by
definition the next most relevant thing the LLM would have seen. The cost of
duplication is paid in context the model never gets.

Filings duplicate heavily by construction: boilerplate risk language repeats
across companies, and a figure stated in MD&A is restated in the notes and
again in a summary table.

Cosine similarity is computed here from raw vectors rather than read off a
vector-store distance score. Chroma's default metric is L2, and LangChain's
"relevance score" is a metric-dependent rescaling of distance — comparing
that against a cosine threshold silently means something different depending
on how the collection was created. An explicit dot product over L2-normalised
vectors does not have that ambiguity.
"""

from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document

from src.common.logging import get_logger
from src.config import settings

logger = get_logger(__name__)


@dataclass
class DedupResult:
    kept: list[Document]
    skipped: list[Document] = field(default_factory=list)
    # Parallel to `skipped`: what each duplicate matched, and how closely.
    # Kept so a suppression can be audited rather than taken on trust — the
    # failure mode of a too-low threshold is silently discarding distinct
    # disclosures, which is invisible unless the matches are inspectable.
    matches: list[tuple[float, str]] = field(default_factory=list)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


def _unit(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise so a dot product is cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def existing_corpus_vectors(vector_store) -> np.ndarray | None:  # noqa: ANN001
    """Fetch the embeddings already in the collection.

    Returns None when the collection is empty or does not expose vectors, in
    which case dedup still runs within the incoming batch.
    """
    try:
        raw = vector_store.store.get(include=["embeddings"])
    except Exception as exc:
        logger.warning("dedup_existing_vectors_unavailable", error=str(exc))
        return None

    embeddings = raw.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None
    return _unit(np.asarray(embeddings, dtype=np.float32))


def deduplicate(
    documents: list[Document],
    existing: np.ndarray | None = None,
    threshold: float | None = None,
) -> DedupResult:
    """Drop chunks that near-duplicate the corpus or each other.

    Compares against two things, because either alone leaves a hole: already
    indexed chunks (re-ingesting a filing that overlaps last year's), and
    chunks earlier in this same batch (boilerplate repeated within one
    document, which no check against the existing index would catch on a
    first ingest).
    """
    if not documents:
        return DedupResult(kept=[])

    threshold = settings.dedup_threshold if threshold is None else threshold

    from src.ingestion.embeddings import get_embedding_model

    candidates = _unit(
        np.asarray(
            get_embedding_model().embed_documents([d.page_content for d in documents]),
            dtype=np.float32,
        )
    )

    kept: list[Document] = []
    kept_vectors: list[np.ndarray] = []
    result = DedupResult(kept=kept)

    for doc, vector in zip(documents, candidates, strict=True):
        best_score = 0.0
        best_text = ""

        if existing is not None and len(existing):
            similarities = existing @ vector
            index = int(np.argmax(similarities))
            best_score = float(similarities[index])
            best_text = "<indexed chunk>"

        if kept_vectors:
            similarities = np.asarray(kept_vectors) @ vector
            index = int(np.argmax(similarities))
            if float(similarities[index]) > best_score:
                best_score = float(similarities[index])
                best_text = kept[index].page_content[:120]

        if best_score >= threshold:
            result.skipped.append(doc)
            result.matches.append((best_score, best_text))
            continue

        kept.append(doc)
        kept_vectors.append(vector)

    if result.skipped:
        logger.info(
            "duplicates_skipped",
            skipped=result.skipped_count,
            kept=len(kept),
            threshold=threshold,
            top_similarity=max(score for score, _ in result.matches),
        )

    return result
