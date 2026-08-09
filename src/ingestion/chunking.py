"""Chunking strategies, and the provenance stamped on what they produce.

Three strategies, switchable by CHUNKING_STRATEGY:

    fixed      character windows with overlap, structure-blind
    recursive  splits on paragraph/line/sentence boundaries first
    semantic   splits where consecutive sentence embeddings diverge

Chunking is baked into an index at ingestion time — an existing collection
keeps whatever chunks it was built with — so comparing strategies means
building a collection per strategy and pointing the eval at each, not
flipping the setting on a live index. Every chunk carries the strategy that
produced it so a collection can never be misattributed after the fact.
"""

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from src.common.logging import get_logger
from src.config import ChunkingStrategy, settings

logger = get_logger(__name__)


def create_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """The structure-aware splitter.

    Separators are ordered widest-boundary-first: paragraph, line, sentence,
    word, and only then raw characters. RecursiveCharacterTextSplitter tries
    each in turn and only falls through when a run of text has no instance of
    the current one, so a chunk breaks at a paragraph break where it can and
    mid-word only where it must.
    """
    # `is None`, not `or`: chunk_overlap=0 is a legitimate request for no
    # overlap, and `or` would silently substitute the configured default —
    # producing a splitter that quietly disagrees with its own arguments.
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size if chunk_size is None else chunk_size,
        chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def create_fixed_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> CharacterTextSplitter:
    """The structure-blind baseline.

    An empty separator means no boundary is preferred over any other, so this
    cuts strictly on the character budget — through a sentence, a table row,
    or a dollar figure. That is the point: it isolates what structure
    awareness is worth by removing it, and any claim that the recursive
    splitter helps has to beat this to mean anything.
    """
    return CharacterTextSplitter(
        separator="",
        chunk_size=settings.chunk_size if chunk_size is None else chunk_size,
        chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
        length_function=len,
    )


def _drop_stubs(chunks: list[Document]) -> list[Document]:
    """Discard chunks too short to answer anything.

    Runs before _tag_chunks so chunk_index stays contiguous — a gap in it
    would read as a lost chunk rather than a rejected one.

    The parser removes page furniture at the source, which is the better place
    for anything with a recognisable shape. This is the catch-all beneath it,
    for the residue no pattern describes: bare section headings, a stray
    bullet, the tail of a word left behind by a split. See MIN_CHUNK_CHARS in
    config for where the threshold comes from.
    """
    if settings.min_chunk_chars <= 0:
        return chunks

    kept = [c for c in chunks if len(c.page_content.strip()) >= settings.min_chunk_chars]

    dropped = len(chunks) - len(kept)
    if dropped:
        logger.info(
            "stub_chunks_dropped",
            dropped=dropped,
            kept=len(kept),
            min_chunk_chars=settings.min_chunk_chars,
        )
    return kept


def _tag_chunks(chunks: list[Document], strategy: ChunkingStrategy) -> list[Document]:
    """Stamp provenance on each chunk.

    chunk_index is per source document, not global, so it survives ingesting
    documents in a different order. char_count is stored rather than derived
    because metadata is queryable in Chroma and the text is not — it makes
    "what did this strategy actually produce" answerable with a filter
    instead of a full corpus fetch.
    """
    counters: dict[str, int] = {}

    for chunk in chunks:
        source = str(
            chunk.metadata.get("source_file")
            or chunk.metadata.get("source")
            or chunk.metadata.get("section_name")
            or ""
        )
        index = counters.get(source, 0)
        counters[source] = index + 1

        chunk.metadata.update(
            {
                "chunking_strategy": strategy.value,
                "chunk_index": index,
                "char_count": len(chunk.page_content),
            }
        )

    return chunks


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    # Explicit chunk_size/overlap means the caller wants size-based splitting
    # regardless of the configured strategy (the semantic path has no use for
    # either parameter).
    strategy = settings.chunking_strategy
    if strategy == ChunkingStrategy.SEMANTIC and chunk_size is None:
        from src.ingestion.semantic_chunking import split_documents_semantically

        return _tag_chunks(_drop_stubs(split_documents_semantically(documents)), strategy)

    if strategy == ChunkingStrategy.SEMANTIC:
        # Caller forced a size; that is the fixed-size path by definition, and
        # recording it as "semantic" would misattribute the collection.
        strategy = ChunkingStrategy.RECURSIVE

    splitter = (
        create_fixed_splitter(chunk_size, chunk_overlap)
        if strategy == ChunkingStrategy.FIXED
        else create_text_splitter(chunk_size, chunk_overlap)
    )
    chunks = _drop_stubs(splitter.split_documents(documents))

    logger.info(
        "documents_split",
        strategy=strategy.value,
        input_docs=len(documents),
        output_chunks=len(chunks),
        chunk_size=chunk_size or settings.chunk_size,
    )
    return _tag_chunks(chunks, strategy)
