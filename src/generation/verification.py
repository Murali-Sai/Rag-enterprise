"""Citation verification — does the cited chunk actually say that?

Parsing (citations.py) proves a citation is *well-formed*. It cannot prove
it is *right*: "[2]" attached to a fabricated figure parses exactly as
cleanly as "[2]" attached to a true one. Checking the pairing needs a reader,
so each (claim, cited block) pair goes to an LLM judge with nothing but the
claim and that one block, and the judge says whether the block supports it.

Design notes:

- **One call per pair, not per claim.** A claim citing "[2][5]" is making
  two assertions about two sources, and they can be right and wrong
  independently. Judging them together produces one verdict that hides which
  reference was the bad one — and the bad reference is the thing worth
  reporting.
- **The judge sees one block at a time.** Handing it the whole context lets
  it mark a citation supported because the *corpus* supports the claim, which
  is a faithfulness check, not a citation check. RAGAS already measures
  faithfulness; this measures whether the pointer points at the right place.
- **gpt-4o-mini, and freely changeable.** This judge is new apparatus, so
  nothing in evaluation/results/ depends on it — unlike the RAGAS judge,
  which is pinned precisely because moving it would invalidate every
  historical comparison.
- **Partial credit.** A block that is on-topic and supports half the
  sentence is not the same failure as one that supports none of it; filings
  answers routinely combine a figure from one chunk with a qualifier from
  its neighbour. PARTIAL scores 0.5 and the raw counts are kept, so anyone
  who disagrees with that weighting can recompute from the verdicts.
"""

from collections import Counter
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from src.common.logging import get_logger
from src.config import settings
from src.generation.citations import ParsedAnswer

logger = get_logger(__name__)

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
OUT_OF_RANGE = "out_of_range"
UNJUDGED = "unjudged"

_VERDICT_SCORES = {
    SUPPORTED: 1.0,
    PARTIAL: 0.5,
    UNSUPPORTED: 0.0,
    OUT_OF_RANGE: 0.0,
}

CITATION_JUDGE_PROMPT = """You are auditing one citation in an answer written from SEC filings.

CLAIM:
{claim}

CITED SOURCE (context block [{index}]{descriptor}):
{source}

Does this source, on its own, support the claim?

SUPPORTED — every fact asserted in the claim appears in the source, or follows directly from it.
PARTIAL — the source is on topic and supports part of the claim, but at least
one asserted fact is absent from it.
UNSUPPORTED — the source does not support the claim, or contradicts it.

Judge only against the source shown. Do not use outside knowledge, and do not
credit the claim because it sounds plausible.

Reply with the verdict word on the first line and one short sentence of
reasoning on the second."""


@dataclass(frozen=True)
class CitationVerdict:
    claim: str
    document_index: int
    verdict: str
    reason: str

    @property
    def score(self) -> float:
        return _VERDICT_SCORES.get(self.verdict, 0.0)


@dataclass(frozen=True)
class CitationReport:
    """What the answer's citations are worth, and how much it cited at all."""

    verdicts: tuple[CitationVerdict, ...] = ()
    total_claims: int = 0
    cited_claims: int = 0
    uncited_claims: int = 0
    out_of_range: int = 0
    verified: bool = False
    """False when verification was skipped — accuracy is then None, not 0."""

    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_citations(self) -> int:
        """Citations that carry a verdict — the accuracy denominator."""
        return len(self.verdicts)

    @property
    def accuracy(self) -> float | None:
        """Mean citation score, or None if there was nothing to judge.

        Out-of-range references are in the denominator and score zero: a
        pointer to a block that was never supplied is a wrong citation, not
        an unmeasurable one. They stay separately counted in `out_of_range`
        so the two failure modes can still be told apart.
        """
        if not self.verified or not self.verdicts:
            return None
        return sum(v.score for v in self.verdicts) / len(self.verdicts)

    @property
    def coverage(self) -> float:
        if not self.total_claims:
            return 0.0
        return self.cited_claims / self.total_claims

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "total_claims": self.total_claims,
            "cited_claims": self.cited_claims,
            "uncited_claims": self.uncited_claims,
            "out_of_range": self.out_of_range,
            "verified": self.verified,
            "verdict_counts": dict(self.counts),
        }


def _skeleton(parsed: ParsedAnswer) -> dict:
    return {
        "total_claims": len(parsed.claims),
        "cited_claims": len(parsed.cited_claims),
        "uncited_claims": len(parsed.uncited_claims),
        "out_of_range": parsed.out_of_range_count,
    }


def unverified_report(parsed: ParsedAnswer) -> CitationReport:
    """Structural counts only — what parsing alone can establish.

    The runtime default: verification costs one LLM call per citation on the
    critical path, which the coverage and out-of-range signals do not.
    """
    return CitationReport(verdicts=(), verified=False, **_skeleton(parsed))


def _parse_verdict(response: str) -> tuple[str, str]:
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    if not lines:
        return UNJUDGED, "judge returned nothing"

    head = lines[0].upper()
    reason = lines[1] if len(lines) > 1 else ""
    # Checked before SUPPORTED because "UNSUPPORTED" contains it.
    if "UNSUPPORTED" in head:
        return UNSUPPORTED, reason
    if "PARTIAL" in head:
        return PARTIAL, reason
    if "SUPPORTED" in head:
        return SUPPORTED, reason
    return UNJUDGED, f"unrecognized verdict: {lines[0][:80]}"


def _descriptor(doc: Document) -> str:
    """Human-readable identity of a chunk, from its provenance metadata."""
    parts = [
        str(doc.metadata.get(key))
        for key in ("ticker", "filing_type", "section_name")
        if doc.metadata.get(key)
    ]
    return f", {' '.join(parts)}" if parts else ""


def get_citation_judge() -> BaseChatModel:
    from src.generation.llm_factory import create_llm

    return create_llm(settings.citation_judge_provider, model=settings.citation_judge_model)


def verify_citations(
    parsed: ParsedAnswer,
    documents: list[Document],
    llm: BaseChatModel | None = None,
) -> CitationReport:
    """Judge every citation in a parsed answer against the chunk it points at.

    Args:
        parsed: Output of parse_citations over the raw generated answer.
        documents: The context blocks, in the order they were numbered.
        llm: Judge model; defaults to the configured citation judge. Injected
            in tests so the suite never makes a network call.
    """
    verdicts: list[CitationVerdict] = []

    # Broken references need no judge: there is no block to read.
    for claim in parsed.claims:
        for n in claim.out_of_range:
            verdicts.append(
                CitationVerdict(
                    claim=claim.text,
                    document_index=n,
                    verdict=OUT_OF_RANGE,
                    reason=f"cited block [{n}] but only {parsed.document_count} were supplied",
                )
            )

    pairs = parsed.citation_pairs
    if pairs:
        judge = llm or get_citation_judge()
        for claim, n in pairs:
            document = documents[n - 1]
            prompt = CITATION_JUDGE_PROMPT.format(
                claim=claim.text,
                index=n,
                descriptor=_descriptor(document),
                source=document.page_content,
            )
            try:
                response = judge.invoke(prompt)
                verdict, reason = _parse_verdict(
                    response.content if hasattr(response, "content") else str(response)
                )
            except Exception as e:
                logger.error("citation_judge_failed", error=str(e), error_type=type(e).__name__)
                verdict, reason = UNJUDGED, f"judge call failed: {type(e).__name__}"
            verdicts.append(
                CitationVerdict(claim=claim.text, document_index=n, verdict=verdict, reason=reason)
            )

    # An unjudged pair is a hole in the measurement, not a failed citation —
    # scoring it zero would let a flaky judge look like a hallucinating model.
    scored = tuple(v for v in verdicts if v.verdict != UNJUDGED)
    counts = Counter(v.verdict for v in verdicts)

    report = CitationReport(
        verdicts=scored,
        verified=True,
        counts=dict(counts),
        **_skeleton(parsed),
    )
    logger.info(
        "citation_verification_complete",
        claims=report.total_claims,
        citations=report.total_citations,
        accuracy=report.accuracy,
        coverage=report.coverage,
        out_of_range=report.out_of_range,
        unjudged=counts.get(UNJUDGED, 0),
    )
    return report
