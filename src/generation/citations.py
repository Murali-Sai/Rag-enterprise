"""Parsing bracketed citations out of a generated answer.

This is the half of citation checking that needs no LLM: split the answer
into claims, read the `[n]` markers attached to each, and separate the three
failure modes that a single "unverified" bucket would flatten together.

    cited + in range      -> a claim that can be checked (verification.py)
    cited + out of range  -> "[7]" when five blocks were supplied. The model
                             invented a source. Nothing to check it against,
                             and it is a different defect from a claim the
                             source doesn't support — one is a broken
                             reference, the other a wrong one.
    uncited               -> not a failed citation at all. It is missing
                             coverage: the claim may well be true and
                             supported, it just doesn't say by what. Counting
                             it as a citation failure would make an answer
                             that cites nothing score perfectly.

Claims are sentences. A citation attaches to an assertion, not to a whole
answer, and the sentence is the smallest unit the model reliably ends with a
bracket. The splitter is the one from src/common/text.py, shared with
semantic chunking, so "$391.0 billion" and "Item 7A." stay intact.

Everything here runs on the *raw* generated answer. The API route rewrites
the text afterwards — appending compliance disclaimers and redacting PII —
so a parser run on the response body would scan sentences the model never
wrote, and could see a claim whose interior was altered between generation
and verification.
"""

import re
from dataclasses import dataclass

from src.common.text import split_sentences

# [2] and [2, 5] are what the prompt asks for. "Document" is tolerated
# because models echo context-block labels, and the label used to read
# "[Document 2]" — answers generated against the old format still parse.
_CITATION_RE = re.compile(r"\[\s*(?:Document\s+)?(\d+(?:\s*,\s*\d+)*)\s*\]")

# Leading list markers, stripped so a bulleted answer segments into claims
# rather than into decorations.
_BULLET_RE = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")


@dataclass(frozen=True)
class Claim:
    """One sentence of an answer, with the blocks it points at."""

    text: str
    """The sentence with its citation markers removed — what gets judged."""

    raw_text: str
    """The sentence as generated, markers included."""

    citations: tuple[int, ...]
    """1-based block numbers that exist in the supplied context."""

    out_of_range: tuple[int, ...]
    """1-based block numbers the model invented."""

    @property
    def is_cited(self) -> bool:
        return bool(self.citations)


@dataclass(frozen=True)
class ParsedAnswer:
    claims: tuple[Claim, ...]
    document_count: int

    @property
    def cited_claims(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if c.is_cited)

    @property
    def uncited_claims(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if not c.is_cited and not c.out_of_range)

    @property
    def citation_pairs(self) -> tuple[tuple[Claim, int], ...]:
        """Every (claim, block) pair worth sending to a judge."""
        return tuple((c, n) for c in self.claims for n in c.citations)

    @property
    def out_of_range_count(self) -> int:
        return sum(len(c.out_of_range) for c in self.claims)

    @property
    def total_citations(self) -> int:
        """In-range and out-of-range together — every reference the model made."""
        return len(self.citation_pairs) + self.out_of_range_count

    @property
    def coverage(self) -> float:
        """Share of claims carrying at least one citation.

        The answer's own claim about itself: how much of what it asserts it
        is willing to attribute. An answer with no claims scores 0.0 — there
        is nothing attributed, so nothing to be confident about.
        """
        if not self.claims:
            return 0.0
        return len(self.cited_claims) / len(self.claims)


def _is_claim(text: str, *, has_citation: bool) -> bool:
    """Filter out fragments that assert nothing.

    Headings ("Key risks:"), stray markers and one-word lines are structure,
    not claims. Counting them would deflate coverage for an answer that is
    in fact fully cited.
    """
    if not re.search(r"[A-Za-z]", text):
        return False
    if has_citation:
        # The model attached a source to it, so it is asserting something —
        # including short bullet fragments like "Currency exposure [2]",
        # which a word-count floor would silently discard along with the
        # citation attached to them.
        return True
    # A heading introduces claims rather than making one. Uncited two-word
    # fragments are structure or stray text, not assertions; counting them
    # would deflate the coverage of an answer that is in fact fully cited.
    return not text.endswith(":") and len(text.split()) >= 3


def _read_markers(sentence: str, document_count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a sentence's block references into in-range and out-of-range."""
    numbers: list[int] = []
    for match in _CITATION_RE.finditer(sentence):
        for part in match.group(1).split(","):
            numbers.append(int(part.strip()))

    in_range: list[int] = []
    out_of_range: list[int] = []
    for n in numbers:
        bucket = in_range if 1 <= n <= document_count else out_of_range
        if n not in bucket:
            bucket.append(n)

    return tuple(in_range), tuple(out_of_range)


def parse_citations(answer: str, document_count: int) -> ParsedAnswer:
    """Segment an answer into claims and read the citations on each.

    Args:
        answer: The raw generated answer, before disclaimers or redaction.
        document_count: How many context blocks were supplied. A reference
            above this is out of range by definition.
    """
    claims: list[Claim] = []

    for line in answer.splitlines():
        stripped = _BULLET_RE.sub("", line).strip()
        if not stripped:
            continue
        for sentence in split_sentences(stripped):
            without_markers = _CITATION_RE.sub("", sentence).strip()
            # Collapse the double spaces a removed mid-sentence marker leaves,
            # and the space stranded before a trailing full stop.
            without_markers = re.sub(r"\s+", " ", without_markers)
            without_markers = re.sub(r"\s+([.,;:!?])", r"\1", without_markers)
            in_range, out_of_range = _read_markers(sentence, document_count)
            if not _is_claim(without_markers, has_citation=bool(in_range or out_of_range)):
                continue
            claims.append(
                Claim(
                    text=without_markers,
                    raw_text=sentence,
                    citations=in_range,
                    out_of_range=out_of_range,
                )
            )

    return ParsedAnswer(claims=tuple(claims), document_count=document_count)
