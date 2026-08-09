"""Composite confidence: how much should a reader trust this answer?

Project 6 asks for three signals, and they fail independently — which is the
argument for combining them rather than picking one:

- **Retrieval confidence.** Did the search find anything relevant? Computed
  from the cross-encoder scores now carried through by
  src/retrieval/scores.py. This is the only signal available *before*
  generation, so it is also the gate for the structured "I don't know".
- **Citation coverage.** How much of the answer is attributed at all. Comes
  free from parsing — no LLM call — so it is available on every query.
- **Answer completeness.** Whether the answer actually answers. A model that
  correctly refuses produces a *good* response with *low* confidence, and
  nothing else in the composite would notice: retrieval can score well while
  the chunks miss the specific figure asked for, and a refusal that cites
  nothing has undefined coverage rather than bad coverage.

Weighted 0.5 / 0.3 / 0.2. Retrieval leads because a grounded answer over bad
chunks is the failure mode that matters here — the known reranker regression
(-0.230 context precision) is precisely that failure, and it is invisible in
the answer text. Coverage outweighs completeness because it is measured off
the answer's own structure rather than off phrase matching.

A deliberate consequence: nothing in this module calls an LLM. Confidence is
computed on every query, on the critical path, so it has to be free.
"""

from dataclasses import dataclass

from src.generation.citations import ParsedAnswer
from src.generation.verification import CitationReport
from src.retrieval.scores import ScoredDocument

# Weights over (retrieval, coverage, completeness) — see module docstring.
RETRIEVAL_WEIGHT = 0.5
COVERAGE_WEIGHT = 0.3
COMPLETENESS_WEIGHT = 0.2

# The refusal wording rule 2 of the system prompt asks for, plus the
# paraphrases models reach for anyway. Matched on a lowercased answer.
REFUSAL_MARKERS = (
    "i don't have enough information",
    "i do not have enough information",
    "not enough information in the available documents",
    "the provided context does not contain",
    "the context does not contain",
    "the provided documents do not contain",
    "no relevant documents",
    "unable to answer",
    "cannot answer this question",
)

# Hedges that qualify part of an otherwise real answer. Distinct from the
# above: the answer said something, then admitted a gap.
PARTIAL_MARKERS = (
    "does not specify",
    "do not specify",
    "does not provide",
    "do not provide",
    "is not disclosed",
    "not addressed in",
    "however, the context",
    "however, the documents",
)

HIGH_THRESHOLD = 0.66
MEDIUM_THRESHOLD = 0.40


@dataclass(frozen=True)
class ConfidenceReport:
    overall: float
    retrieval: float | None
    coverage: float
    completeness: float
    label: str

    def as_dict(self) -> dict:
        return {
            "overall": self.overall,
            "retrieval": self.retrieval,
            "citation_coverage": self.coverage,
            "answer_completeness": self.completeness,
            "label": self.label,
        }


def retrieval_confidence(scored: list[ScoredDocument]) -> float | None:
    """Aggregate a retrieved set's relevance into one number, or None.

    None when the stage produced no usable score (RRF is ordinal; a retriever
    outside this package may produce nothing at all). None propagates as
    "unknown" rather than collapsing to 0.0, which would read as a confident
    verdict that nothing relevant exists.

    Weighted toward the single best document (0.6) over the set mean (0.4).
    A question answered squarely by one chunk should score high even when the
    other four slots are filler — which is the normal shape of a top-5 over
    a corpus of 10-K prose. Using the mean alone would mark those answers
    down for the padding around the chunk that actually answered.
    """
    relevances = [r for r in (item.relevance for item in scored) if r is not None]
    if not relevances:
        return None
    best = max(relevances)
    mean = sum(relevances) / len(relevances)
    return 0.6 * best + 0.4 * mean


def is_refusal(answer: str) -> bool:
    """True when the answer's *wording* declines.

    A marker test only. Models routinely open with the refusal sentence rule 2
    asks for and then answer the part they can support, so this says the answer
    contains a refusal — not that the answer is one. Use `is_full_refusal()`
    for that.
    """
    return any(marker in answer.lower() for marker in REFUSAL_MARKERS)


def has_partial_refusal(answer: str) -> bool:
    return any(marker in answer.lower() for marker in PARTIAL_MARKERS)


def is_full_refusal(answer: str, parsed: ParsedAnswer) -> bool:
    """True when the answer declines *and* asserts nothing attributable.

    The distinction the marker test alone cannot make. A question naming two
    figures where the filing discloses one produces:

        "I don't have enough information ... to answer this question. However,
        Tesla's net income attributable to common stockholders for 2025 was
        $3.79 billion [1][3]."

    which opens with the refusal wording and then answers half the question
    with a citation that verification scores as supported. Treating that as a
    full refusal throws away a correct, sourced figure.

    Cited claims rather than claims: the qualifier is that the answer attributed
    something. A refusal followed by uncited prose has asserted nothing the
    system is willing to stand behind, and stays a full refusal.
    """
    return is_refusal(answer) and not parsed.cited_claims


def answer_completeness(answer: str, parsed: ParsedAnswer) -> float:
    """How fully the answer engages with the question, on the text alone.

    Three states rather than a continuous score, because that is as much as
    phrase matching can honestly distinguish:

        0.0  refused, or asserted nothing
        0.5  answered, then flagged something it could not cover
        1.0  answered

    A full refusal scores 0.0 and drags the composite down — the intended
    behaviour, since a refusal is a correct response with no content to be
    confident about.

    An answer carrying refusal wording *and* cited claims is the 0.5 state, not
    the 0.0 one: it answered, then flagged the part it could not cover, which
    is what the middle state describes. Scoring it 0.0 was measurably wrong —
    two rows of eval_20260808_212927.json (JPMorgan's CET1 ratio, Tesla's net
    income) carried verified citations and were scored as though they had said
    nothing at all.
    """
    if is_full_refusal(answer, parsed):
        return 0.0
    if is_refusal(answer):
        # Refused in part. The gap it flagged is explicit rather than hedged,
        # so it never reaches the PARTIAL_MARKERS test below.
        return 0.5
    if not parsed.claims:
        # Text with no assertions in it: an empty answer, or nothing but
        # headings and boilerplate.
        return 0.0
    if has_partial_refusal(answer):
        return 0.5
    return 1.0


def _label(overall: float, completeness: float) -> str:
    """Bucket the composite, with one override.

    Completeness of zero means the answer asserted nothing — a full refusal,
    or empty text. The composite can still land in the middle there, because
    retrieval carries half the weight and retrieval may well have gone fine;
    that number is meaningful (it says the chunks looked relevant and the
    model still declined) but the *label* is a claim about the answer, and
    there is no answer to be moderately confident in.
    """
    if completeness == 0.0:
        return "low"
    if overall >= HIGH_THRESHOLD:
        return "high"
    if overall >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def score_confidence(
    answer: str,
    parsed: ParsedAnswer,
    report: CitationReport,
    retrieval: float | None,
) -> ConfidenceReport:
    """Combine the three signals into one 0–1 score plus a label.

    When retrieval confidence is unavailable its weight is redistributed
    across the other two rather than counted as zero — an unmeasurable signal
    must not masquerade as a bad one.
    """
    coverage = report.coverage
    completeness = answer_completeness(answer, parsed)

    if retrieval is None:
        total = COVERAGE_WEIGHT + COMPLETENESS_WEIGHT
        overall = (COVERAGE_WEIGHT * coverage + COMPLETENESS_WEIGHT * completeness) / total
    else:
        overall = (
            RETRIEVAL_WEIGHT * retrieval
            + COVERAGE_WEIGHT * coverage
            + COMPLETENESS_WEIGHT * completeness
        )

    overall = min(1.0, max(0.0, overall))
    return ConfidenceReport(
        overall=overall,
        retrieval=retrieval,
        coverage=coverage,
        completeness=completeness,
        label=_label(overall, completeness),
    )
