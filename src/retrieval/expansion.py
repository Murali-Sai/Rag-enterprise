"""Give a retrieved table row back the rest of its table.

The measured failure. Asked for Goldman's total net revenues and return on
average common equity, the pipeline retrieved a chunk ending:

    Return on average common equity | 2.1 | % | (17.4) | % | (39.4) | %
    Total
    Net revenues | $ | 58,283 | $ | 53,512 | $ | 46,254

and answered "total net revenues were $58,283 million … return on average
common equity was 2.1%". Both figures are real, both are in that chunk, and
pairing them is wrong: 2.1% is the *Platform Solutions segment*, the word
"Total" is where the firmwide table starts, and the firmwide ROE — 15.0%,
over $108,726 million of average common equity — is the first row of the
**next chunk**, which was never retrieved.

The segments sum to the firmwide row exactly (13,117 + 3,093 + 90 = 16,300),
so the corpus is correct and complete. The split is the whole problem: a
table crosses a chunk boundary, one side is retrieved, and the numbers on
that side read as though they answer the question.

Nothing at query time fixes this by ranking, because the chunk that was
retrieved *is* relevant — it holds one of the two figures asked for. What is
missing is its continuation. So this stage does what a person reading the
filing would: having found the table, read the next row.

**Why not fix it in chunking.** Because chunking is baked into an index at
ingestion time, and re-chunking means re-embedding, a new corpus digest, and
the invalidation of every published figure. This runs at query time against
metadata every chunk already carries (`source_file`, `section_name`,
`chunk_index`), so it is measurable today against the shipped index — and
`table_context`, the chunking-side attempt at the same family of problem, was
measured and refuted (see src/ingestion/table_context.py).

**Why only table chunks.** Prose is written to be read in one piece; a
paragraph that continues into the next chunk still says what it is about.
A table row does not. Expanding everything would double the context on every
query to fix a failure that only tables have, and diluting five good chunks
with five mediocre continuations is how a precision problem gets made.

**Why forward only.** The failure is a table whose *later rows* are missing —
totals, and the labels that introduce them, come after the rows they
summarize. A chunk starting mid-table also exists, but its header sits in a
neighbour that dense retrieval usually returns anyway, since the header is
the part with words in it.
"""

from langchain_core.documents import Document

from src.common.logging import get_logger
from src.config import settings
from src.ingestion.table_context import looks_like_table
from src.retrieval.scores import ScoredDocument
from src.retrieval.vector_store import VectorStoreBase, get_vector_store

logger = get_logger(__name__)

# Marks where the retrieved chunk ends and its continuation begins. Visible in
# the dashboard's chunk view and in the prompt, because a model reading a
# stitched table should be able to tell it was stitched.
CONTINUATION_MARKER = "\n[continues]\n"


def _neighbour_filter(metadata: dict) -> dict | None:
    """Where-clause selecting the chunk immediately after this one.

    Keyed on source *and* section as well as index: `chunk_index` counts per
    source document, so index+1 alone would happily match a different
    filing's chunk — the same class of bug as an entity filter that forgets
    which company it is scoped to.
    """
    source = metadata.get("source_file") or metadata.get("source")
    section = metadata.get("section_name")
    index = metadata.get("chunk_index")
    if source is None or index is None:
        return None

    clauses: list[dict] = [
        {"source_file": {"$eq": source}},
        {"chunk_index": {"$eq": int(index) + 1}},
    ]
    if section is not None:
        clauses.append({"section_name": {"$eq": section}})
    return {"$and": clauses}


def expand_table_chunks(
    scored: list[ScoredDocument],
    vector_store: VectorStoreBase | None = None,
) -> list[ScoredDocument]:
    """Append each table chunk's following chunk to its text.

    Returns the same list, same length, same order, same scores — only the
    text of table-shaped documents grows. Keeping the list shape fixed is
    deliberate: citation indices are positional, so adding or reordering
    documents here would renumber the citations the model is about to emit.

    Skips a neighbour that is already in the retrieved set, which would
    otherwise put the same text in the context twice and let one passage win
    two of the five citation slots.
    """
    if not scored or not settings.table_expansion_enabled:
        return scored

    # Resolved on first use, not here. A retrieved set with no table in it
    # needs no store, and building one is not free: get_vector_store() opens
    # Chroma and constructs the embedding function, which raises without an
    # API key. Eagerly resolving it made this stage a hidden dependency of
    # every caller that injects its own retriever — the suite runs with no
    # keys set, by design, and ten tests failed the moment it did.
    store = vector_store
    present = {
        (
            item.document.metadata.get("source_file"),
            item.document.metadata.get("section_name"),
            item.document.metadata.get("chunk_index"),
        )
        for item in scored
    }

    expanded = 0
    for item in scored:
        document = item.document
        if not looks_like_table(document.page_content):
            continue

        where = _neighbour_filter(document.metadata)
        if where is None:
            continue

        metadata = document.metadata
        neighbour_key = (
            metadata.get("source_file"),
            metadata.get("section_name"),
            int(metadata["chunk_index"]) + 1,
        )
        if neighbour_key in present:
            continue

        if store is None:
            store = get_vector_store()
        matches = store.get_all_documents(where)
        if not matches:
            continue

        document.page_content = (
            document.page_content + CONTINUATION_MARKER + matches[0].page_content
        )
        expanded += 1

    if expanded:
        logger.info("table_chunks_expanded", expanded=expanded, retrieved=len(scored))
    return scored


def expanded_documents(documents: list[Document]) -> list[Document]:
    """Document-only entry point, for callers without a score channel."""
    from src.retrieval.scores import as_scored, documents_of

    return documents_of(expand_table_chunks(as_scored(documents)))
