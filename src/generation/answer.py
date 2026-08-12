"""The grounded-generation entry point.

Everything Phase 3 adds — bracketed citations, verification, confidence,
the structured "I don't know" — lives behind this one function, because of
where the callers are. `evaluation/run_evaluation.py` calls the generation
layer directly and never touches `src/api/routes/query.py`; so does the MCP
server. Anything implemented in the route is invisible to the measurement
that is supposed to prove it works, and a citation-accuracy number that
grades a different code path than the one users hit is worse than no number.

So the route, the MCP server and the eval harness all call
`generate_grounded_answer()`, and the route's remaining job is to translate
the result into HTTP and apply the compliance layer on top.

Order of operations matters here:

0. Underspecified questions are refused before anything else — a question
   that needs a company and names none has no correct answer for retrieval
   to find, so neither the gate nor the model gets a say. See
   src/retrieval/ambiguity.py for the measurement.
1. Retrieval confidence is checked *before* generating. It is the only
   retrieval signal available pre-generation, and refusing early skips the
   expensive call entirely rather than paying for an answer built on bad
   chunks.

   That check is a cost guard, not the refusal mechanism. It reads as one —
   it is the first thing in the function and it returns the same structured
   report the model's own refusal produces — so it is worth being explicit
   that the correctness of this system's refusals does not rest on it. With
   the gate held open, the model declined 24 of 24 out-of-corpus questions
   unprompted (scripts/probe_refusal.py), including one the gate scored 0.975
   and would have admitted anyway. The threshold is set low enough that it
   cannot be the reason an answerable question is declined; see
   `settings.insufficient_context_threshold` for the measurement that fixes
   the value.
2. Citations are parsed from the raw answer, before the route appends
   disclaimers or redacts PII. Parsing afterwards would scan text the model
   never wrote, and redaction can rewrite the interior of a claim between
   generation and verification.
3. Confidence is computed last, because two of its three inputs come from
   the parsed answer.
"""

from dataclasses import dataclass

from langchain_core.documents import Document

from src.common.logging import get_logger
from src.config import settings
from src.generation.chains import query_with_context
from src.generation.citations import ParsedAnswer, parse_citations
from src.generation.confidence import (
    ConfidenceReport,
    is_full_refusal,
    retrieval_confidence,
    score_confidence,
)
from src.generation.insufficient import (
    AMBIGUOUS_ENTITY,
    LOW_RETRIEVAL_CONFIDENCE,
    MODEL_REFUSED,
    UnansweredReport,
    build_unanswered_report,
    render_unanswered,
)
from src.generation.verification import CitationReport, unverified_report, verify_citations
from src.retrieval.ambiguity import clarification_detail, is_underspecified
from src.retrieval.scores import ScoredDocument, as_scored, documents_of

logger = get_logger(__name__)


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    """The raw generated text, citations intact and nothing appended."""

    documents: list[Document]
    parsed: ParsedAnswer
    citations: CitationReport
    confidence: ConfidenceReport
    unanswered: UnansweredReport | None = None
    """Set when the system declined, on either path. None on a real answer."""

    generated: bool = True
    """False when the low-confidence gate fired and no LLM call was made."""


def _normalize(retrieved: list[ScoredDocument] | list[Document]) -> list[ScoredDocument]:
    if retrieved and isinstance(retrieved[0], Document):
        return as_scored(retrieved)  # type: ignore[arg-type]
    return list(retrieved)  # type: ignore[arg-type]


def generate_grounded_answer(
    question: str,
    retrieved: list[ScoredDocument] | list[Document],
    *,
    verify: bool | None = None,
) -> GroundedAnswer:
    """Generate an answer and everything needed to judge it.

    Args:
        question: The user's question, already through input guardrails.
        retrieved: Retrieved context. Pass `ScoredDocument`s (from
            `retrieve_scored()`) to get retrieval confidence; plain
            `Document`s work too and simply leave that signal unavailable.
        verify: Run the citation judge. Defaults to
            `settings.citation_verification_enabled` — off at runtime,
            because it costs one LLM call per citation on the critical path,
            and on in the eval harness, where that cost buys the metric.
    """
    scored = _normalize(retrieved)
    documents = documents_of(scored)
    verify = settings.citation_verification_enabled if verify is None else verify

    retrieval = retrieval_confidence(scored)
    threshold = settings.insufficient_context_threshold

    # A property of the question, not of what was retrieved — checked first
    # because no retrieval result can repair an underspecified question, only
    # pick a company on the asker's behalf. Like the gate below, this skips
    # the LLM call; unlike the gate, it is a correctness mechanism, and the
    # only one this pipeline has for the ambiguous stratum (0.667 in every
    # baseline before it existed). The retrieved documents still ride along
    # in the report: what a vague search returned is exactly the evidence
    # that the question admits several answers.
    if is_underspecified(question):
        report = build_unanswered_report(
            AMBIGUOUS_ENTITY,
            documents,
            detail=clarification_detail(),
        )
        parsed = parse_citations("", len(documents))
        empty = unverified_report(parsed)
        logger.info(
            "ambiguous_entity",
            reason=AMBIGUOUS_ENTITY,
            documents=len(documents),
        )
        return GroundedAnswer(
            answer=render_unanswered(report),
            documents=documents,
            parsed=parsed,
            citations=empty,
            confidence=score_confidence("", parsed, empty, retrieval),
            unanswered=report,
            generated=False,
        )

    if retrieval is not None and retrieval < threshold:
        report = build_unanswered_report(
            LOW_RETRIEVAL_CONFIDENCE,
            documents,
            detail=(f"Best passage scored {retrieval:.2f} against a {threshold:.2f} threshold."),
        )
        parsed = parse_citations("", len(documents))
        empty = unverified_report(parsed)
        logger.info(
            "insufficient_context",
            reason=LOW_RETRIEVAL_CONFIDENCE,
            retrieval_confidence=retrieval,
            threshold=threshold,
            documents=len(documents),
        )
        return GroundedAnswer(
            answer=render_unanswered(report),
            documents=documents,
            parsed=parsed,
            citations=empty,
            confidence=score_confidence("", parsed, empty, retrieval),
            unanswered=report,
            generated=False,
        )

    answer = query_with_context(question, documents)
    parsed = parse_citations(answer, len(documents))

    citations = verify_citations(parsed, documents) if verify else unverified_report(parsed)
    confidence = score_confidence(answer, parsed, citations, retrieval)

    unanswered = None
    if is_full_refusal(answer, parsed):
        # The model's own refusal wording is kept — it says which part of the
        # question it could not cover, which the structure cannot infer — and
        # the retrieval detail is attached alongside it.
        #
        # Full refusal, not the marker test: an answer that declines one half
        # of a two-part question and cites the other half has an answer in it.
        # Attaching the report there tells every caller the question went
        # unanswered while the response body visibly answers part of it, and
        # the REST route would return the "I don't know" structure alongside a
        # sourced figure.
        unanswered = build_unanswered_report(MODEL_REFUSED, documents)

    logger.info(
        "grounded_answer_complete",
        claims=len(parsed.claims),
        cited_claims=len(parsed.cited_claims),
        out_of_range=parsed.out_of_range_count,
        citation_accuracy=citations.accuracy,
        confidence=confidence.overall,
        confidence_label=confidence.label,
        refused=unanswered is not None,
    )

    return GroundedAnswer(
        answer=answer,
        documents=documents,
        parsed=parsed,
        citations=citations,
        confidence=confidence,
        unanswered=unanswered,
    )
