"""Semantic chunking — split on meaning shifts rather than character counts.

RecursiveCharacterTextSplitter cuts every 512 characters at the nearest
separator. It has no idea what the text says, so it will happily split a
revenue table from its header, or the middle of a risk factor's argument.
The retrieved chunk then carries half a thought.

Semantic chunking instead embeds each sentence, measures the distance
between consecutive sentences, and cuts where that distance spikes — the
points where the text changes subject. Chunks come out variable-length and
(in principle) self-contained.

Implemented here rather than taken from langchain_experimental for two
reasons: that package requires langchain-core 1.x while this project pins
<1.0 for API stability, and it has been officially sunset. The algorithm is
short enough that vendoring it costs less than the dependency conflict.
"""

import numpy as np
from langchain_core.documents import Document

from src.common.logging import get_logger
from src.common.text import split_sentences as _split_sentences
from src.config import settings

logger = get_logger(__name__)


def _combine_with_context(sentences: list[str], buffer_size: int) -> list[str]:
    """Widen each sentence with its neighbours before embedding.

    A single sentence is a weak embedding signal — too short, too sensitive
    to individual words. Embedding a sentence plus its neighbours measures
    whether the local *topic* shifted, not whether the wording did.
    """
    combined = []
    for i in range(len(sentences)):
        start = max(0, i - buffer_size)
        end = min(len(sentences), i + buffer_size + 1)
        combined.append(" ".join(sentences[start:end]))
    return combined


def _breakpoint_indices(embeddings: np.ndarray, percentile: float) -> list[int]:
    """Indices after which the text changes subject enough to cut."""
    if len(embeddings) < 2:
        return []

    # Normalise here rather than relying on the provider. Both current ones
    # return unit vectors (HuggingFace via normalize_embeddings, OpenAI by
    # construction), but if that ever stopped holding, the dot products would
    # scale with magnitude and the percentile threshold would silently cut in
    # the wrong places — chunking would degrade without erroring.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.where(norms == 0, 1.0, norms)

    # With unit vectors the dot product is cosine similarity, so 1 - it is
    # cosine distance.
    similarities = np.sum(unit[:-1] * unit[1:], axis=1)
    distances = 1.0 - similarities

    threshold = float(np.percentile(distances, percentile))
    return [i for i, d in enumerate(distances) if d > threshold]


def split_text_semantically(text: str) -> list[str]:
    """Split one document's text at semantic boundaries."""
    from src.ingestion.embeddings import get_embedding_model

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return [text] if text.strip() else []

    contextualized = _combine_with_context(sentences, settings.semantic_buffer_size)
    embeddings = np.array(get_embedding_model().embed_documents(contextualized))

    breakpoints = _breakpoint_indices(embeddings, settings.semantic_breakpoint_percentile)

    chunks: list[str] = []
    start = 0
    for cut in [*breakpoints, len(sentences) - 1]:
        chunk = " ".join(sentences[start : cut + 1]).strip()
        if chunk:
            chunks.append(chunk)
        start = cut + 1

    return chunks


def split_documents_semantically(documents: list[Document]) -> list[Document]:
    """Semantic-chunk documents, capping runaway chunks.

    A section of uniform prose produces no distance spikes, so semantic
    chunking can emit one enormous chunk covering an entire Item 1A. That
    wastes the LLM's context budget, dilutes the chunk's embedding across
    several subjects, and — on the huggingface provider, whose 256-token
    window truncates at ~1300 characters — makes the tail unretrievable
    outright. Oversized chunks fall back to the fixed-size splitter, which is
    a real hybrid — reported honestly rather than presenting this as pure
    semantic chunking.
    """
    from src.ingestion.chunking import create_text_splitter

    fallback_splitter = create_text_splitter()
    max_chars = settings.semantic_max_chunk_chars

    output: list[Document] = []
    oversized = 0

    for doc in documents:
        for chunk_text in split_text_semantically(doc.page_content):
            if len(chunk_text) <= max_chars:
                output.append(Document(page_content=chunk_text, metadata=dict(doc.metadata)))
                continue

            oversized += 1
            for piece in fallback_splitter.split_text(chunk_text):
                output.append(Document(page_content=piece, metadata=dict(doc.metadata)))

    logger.info(
        "documents_split_semantically",
        input_docs=len(documents),
        output_chunks=len(output),
        oversized_chunks_resplit=oversized,
        percentile=settings.semantic_breakpoint_percentile,
    )
    return output
