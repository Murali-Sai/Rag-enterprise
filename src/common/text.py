"""Sentence segmentation shared by chunking and citation analysis.

Both halves of the system need the same notion of "a sentence": semantic
chunking measures where consecutive sentences diverge, and citation
verification pairs each sentence-level claim with the chunk it cites. Two
different splitters would mean a claim could straddle a boundary that
chunking treated as a unit, so the regex lives here and both import it.
"""

import re

# Split on sentence-ending punctuation followed by whitespace. Filings are
# full of "$1.5 billion" and "Item 7A." — the lookbehind keeps the decimal
# points attached, though abbreviations still split occasionally. That noise
# is tolerable: a stray boundary shifts one sentence between chunks, and in
# citation analysis it splits one claim into two rather than losing it.
SENTENCE_RE = re.compile(r"(?<=[.?!])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
