"""Give bare tables back the words that say what they are.

The measured failure this exists for: Goldman's quantitative market-risk
question scored 0.000 context recall in every run, and at rerank_candidate_k
= 100 six of its seven ground-truth figures still never entered the candidate
set. All seven are indexed. They sit in chunks like

    CET1 capital | $ | 104,297 | $ | 104,297
    Total capital | $ | 130,665 | $ | 128,470
    RWAs | $ | 727,338 | $ | 691,470

— row after row of labels and numbers, with no sentence anywhere in the chunk
saying what the table is about. The sentence exists; the splitter severed it.
The preceding chunk ends "…increased the firm's total risk-based capital
requirements for each of the Standardized and Advanced capital ratios by
0.5%. The table below presents inf" — cut mid-word, stranding the caption on
the wrong side of the boundary. A bi-encoder embeds the table chunk somewhere
near "list of numbers" and the question ("quantitative disclosures about
market risk") somewhere near prose, and they never meet. This is the corpus
half of the defect handoff §7.1 splits with the reranker; the reranker half
was screened separately (bge-reranker-base recovers dropped candidates but
cannot rank what retrieval never surfaces).

The repair is done at index time, because it cannot be done at query time: no
rewriting of the question puts vocabulary inside a chunk that has none. Each
chunk that looks like a table gets one prepended context line carrying the
trailing prose of the nearest preceding chunk that had any — which is where
captions live, by construction of the split. Only the caption: the shared
provenance the first version added is measured crowding, see _context_line.

The caption state persists across consecutive table chunks, because a wide
table splits into several chunks and only the first sits next to the prose
that introduces it. A table with no known caption is left untouched.

Detection is lexical and biased toward not firing, matching this project's
standing argument (entities.py, ambiguity.py) that anything cleverer puts an
unmeasured model inside the pipeline. A chunk is a table when at least three
of its lines are pipe-joined rows and those rows are at least half of its
non-empty lines, or when its text is overwhelmingly numeric. A prose chunk
mentioning one figure is left alone: the failure being repaired is the chunk
with no prose at all, and prepending headers to everything would just teach
every chunk the same words.

This runs inside the `table_context` chunking strategy, a fourth peer of
fixed/recursive/semantic, so it is baked into a collection at ingestion time
and stamped on every chunk it produced. The measured corpus (8,232 chunks at
c2f8c13673cf5ca5) is recursive and stays untouched; comparing the strategies
means building a collection per strategy, same as it always has.
"""

from langchain_core.documents import Document

from src.common.logging import get_logger

logger = get_logger(__name__)

# A pipe-joined row, the shape the EDGAR parser flattens table rows into.
_ROW_MARKER = " | "

# At least this many pipe rows, making up at least this share of non-empty
# lines. Three rows keeps a lone "Assets | Liabilities" header line from
# flagging a prose chunk; the share keeps a paragraph quoting one table row
# from counting as a table.
_MIN_TABLE_ROWS = 3
_MIN_TABLE_ROW_SHARE = 0.5

# Fallback for tables flattened without pipes: digits dominate the
# alphanumerics. 0.4 is far above any prose observed in the filings —
# ordinary MD&A paragraphs with several dollar figures sit under 0.2.
_MIN_DIGIT_RATIO = 0.4

# …but a ratio over a handful of characters is noise, not evidence. "doc19"
# is two digits in five alphanumerics — 0.4 exactly, and not a table. Real
# table chunks in this corpus run 300-500 characters, and config's
# MIN_CHUNK_CHARS already drops anything under 80, so this only excludes text
# far too short to be a table of anything.
_MIN_DIGIT_RATIO_CHARS = 120

# How much trailing prose to carry. Enough for a caption sentence and the
# clause before it; short enough that the table stays most of its own chunk.
_CAPTION_TAIL_CHARS = 240


def _digit_ratio(text: str) -> float:
    digits = sum(ch.isdigit() for ch in text)
    alnum = sum(ch.isalnum() for ch in text)
    return digits / alnum if alnum else 0.0


def looks_like_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    rows = sum(1 for line in lines if _ROW_MARKER in line)
    if rows >= _MIN_TABLE_ROWS and rows / len(lines) >= _MIN_TABLE_ROW_SHARE:
        return True
    if len(text) < _MIN_DIGIT_RATIO_CHARS:
        return False
    return _digit_ratio(text) >= _MIN_DIGIT_RATIO


def _prose_tail(text: str) -> str:
    """The last stretch of caption-bearing prose in a chunk, if any.

    Table rows and numeric lines are excluded so that a chunk which is half
    prose, half table contributes its prose — the caption for the *next*
    table is the sentence after the previous one ends, not the previous
    table's own rows.
    """
    prose_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and _ROW_MARKER not in line and _digit_ratio(line) < 0.3
    ]
    if not prose_lines:
        return ""
    tail = " ".join(prose_lines)[-_CAPTION_TAIL_CHARS:]
    # Cut the leading fragment of a word the slice landed inside.
    cut = tail.find(" ")
    return tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail


def _context_line(metadata: dict, caption: str) -> str:
    """The caption, and nothing shared.

    The first version prepended provenance too — company name and section —
    and the free-eval screen measured the consequence: every table in a
    filing began with the same ~70 characters, which pulled all of them
    toward any thematic question about that company equally. Non-answer
    tables crowded the k=20 candidate set and displaced the prose chunks
    that held the figures; the combination screen scored *below* the same
    reranker on the unenriched corpus (0.339 vs 0.376 overall, GS 0.095 vs
    0.133). The caption is the only part of the header that discriminates
    between one table and the next, so it is the only part that survives.
    Company and section scoping belong to the entity filter and the metadata,
    which already do that job without touching the embedding.
    """
    return f"[Table — preceding text: …{caption}]"


def add_table_context(chunks: list[Document]) -> list[Document]:
    """Prepend a context line to every table chunk, in document order.

    Chunks arrive in the order the splitter produced them, so "the previous
    chunk" is a positional fact — but only within one source document and
    section. The carry state is keyed on (source, section) so a table at the
    top of Goldman's Item 8 can never inherit a caption from the end of its
    Item 7A, or from another company entirely.

    Mutates and returns the same Documents, matching `_tag_chunks`.
    """
    tail_by_group: dict[tuple, str] = {}
    tagged = 0

    for chunk in chunks:
        key = (
            chunk.metadata.get("source_file") or chunk.metadata.get("source"),
            chunk.metadata.get("section_name"),
        )
        original = chunk.page_content

        caption = tail_by_group.get(key, "")
        if caption and looks_like_table(original):
            # No caption, no header: a header with nothing discriminating in
            # it is exactly the crowding failure _context_line describes.
            chunk.page_content = _context_line(chunk.metadata, caption) + "\n\n" + original
            # Queryable marker: makes "how many chunks did this touch" a
            # metadata filter rather than a corpus scan, the same argument
            # as char_count.
            chunk.metadata["table_context"] = True
            tagged += 1

        # Carry state updates from the original text, never the header —
        # otherwise one header's words would leak into the next caption.
        tail = _prose_tail(original)
        if tail:
            tail_by_group[key] = tail

    logger.info("table_context_added", chunks=len(chunks), tables_tagged=tagged)
    return chunks
