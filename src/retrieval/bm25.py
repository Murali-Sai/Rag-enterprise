"""BM25 sparse retrieval over the filing corpus.

Dense embeddings are good at paraphrase ("revenue" ~ "net sales") and bad at
exactness. Financial filings are full of terms where exactness is the whole
point: ticker symbols (AAPL, GS), section labels ("Item 7A"), and specific
figures ($391.0 billion). A 384-dimension MiniLM vector blurs those together;
BM25 matches them literally. Running both and fusing the rankings covers the
failure modes neither handles alone.

RBAC note: the index is built from a department-filtered corpus fetch, and
cached per accessible-department set. A research analyst's BM25 index is
built only over research + sec_filings + general documents, so the lexical
path cannot surface a trading document the dense path would have excluded.
The Chinese Wall holds on both sides of the hybrid.
"""

import json
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.common.logging import get_logger
from src.retrieval.passages import passage_text
from src.retrieval.vector_store import VectorStoreBase

logger = get_logger(__name__)

# Keep alphanumeric runs and decimal numbers intact: "Item 7A" -> [item, 7a],
# "$391.0" -> [391.0], "AAPL" -> [aapl]. Splitting on whitespace alone would
# leave "$391.0" glued to punctuation and never match a query saying "391.0".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over a fixed set of documents."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        corpus_tokens = [tokenize(passage_text(doc)) for doc in documents]
        self._token_sets = [set(tokens) for tokens in corpus_tokens]
        # BM25Okapi divides by corpus stats, so an empty corpus is not valid.
        self._bm25 = BM25Okapi(corpus_tokens) if documents else None

    def search(self, query: str, k: int) -> list[Document]:
        if self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Drop documents sharing no query term — they're noise for the fusion
        # stage. Test overlap directly rather than filtering on score > 0:
        # BM25Okapi's IDF is log((N-n+0.5)/(n+0.5)), which is exactly 0 when a
        # term appears in half the corpus and negative when it's more common,
        # so a real match can legitimately score 0 or below.
        query_set = set(query_tokens)
        matches = [
            (index, score)
            for index, score in enumerate(scores)
            if query_set & self._token_sets[index]
        ]
        matches.sort(key=lambda pair: pair[1], reverse=True)
        return [self.documents[index] for index, _ in matches[:k]]


# Cached per corpus scope. There are only nine roles and five companies, so
# this settles at a handful of indexes rather than one per request.
_index_cache: dict[str, BM25Index] = {}


def _cache_key(accessible_departments: frozenset[str] | None, filter_dict: dict | None) -> str:
    """Identify the corpus slice an index was built over.

    Keyed on the where-filter as well as the department set, because they are
    no longer the same thing: per-entity retrieval narrows the filter to one
    company while leaving the user's departments untouched. Keying on
    departments alone would hand a JPMorgan-scoped query the index built for
    a Goldman-scoped one, and the lexical half of hybrid search would answer
    from the wrong filing.
    """
    departments = ",".join(sorted(accessible_departments)) if accessible_departments else "*"
    return f"{departments}|{json.dumps(filter_dict, sort_keys=True)}"


def get_bm25_index(
    vector_store: VectorStoreBase,
    accessible_departments: frozenset[str] | None,
    filter_dict: dict | None,
) -> BM25Index:
    """Build (or reuse) a BM25 index scoped to one access level.

    Args:
        vector_store: Store to pull the corpus from.
        accessible_departments: Departments the caller may read; None means
            unrestricted (admin).
        filter_dict: The where-filter restricting the corpus fetch.
    """
    key = _cache_key(accessible_departments, filter_dict)
    cached = _index_cache.get(key)
    if cached is not None:
        return cached

    documents = vector_store.get_all_documents(filter_dict=filter_dict)
    logger.info(
        "building_bm25_index",
        documents=len(documents),
        departments=sorted(accessible_departments) if accessible_departments else "all",
        filtered=filter_dict is not None,
    )
    index = BM25Index(documents)
    _index_cache[key] = index
    return index


def reset_bm25_cache() -> None:
    """Drop cached indexes — call after ingesting new documents."""
    _index_cache.clear()
