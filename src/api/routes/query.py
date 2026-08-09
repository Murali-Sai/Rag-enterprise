from fastapi import APIRouter, Depends

from src.api.audit import log_query_audit
from src.api.deps import get_current_user, get_rbac_retriever
from src.api.routes.access import to_information_barriers
from src.auth.models import User
from src.auth.rbac import get_accessible_departments, get_information_barriers_for_user
from src.common.logging import get_logger
from src.common.schemas import (
    ClaimCitation,
    ConfidenceScore,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    UnansweredReport,
)
from src.generation.answer import GroundedAnswer, generate_grounded_answer
from src.guardrails.financial_compliance import (
    apply_financial_disclaimers,
    check_financial_compliance,
)
from src.guardrails.input_validator import validate_input
from src.guardrails.output_safety import check_output_safety
from src.guardrails.pii_detector import redact_pii
from src.guardrails.prompt_injection import detect_prompt_injection
from src.retrieval.retriever import Retriever, retrieve_scored
from src.retrieval.scores import ScoredDocument

router = APIRouter(tags=["Query"])
logger = get_logger(__name__)


def _to_source_documents(scored: list[ScoredDocument]) -> list[SourceDocument]:
    """The retrieved set as the response sees it, scores included.

    `retrieve_scored()` already carries the number each stage produced, and
    dropping it here meant `relevance_score` was declared on the schema and
    null on every response ever served. Confidence reads the same channel, so
    a client showing `confidence.retrieval` without the per-chunk scores can
    see that retrieval scored 0.077 and not which chunk dragged it there.
    """
    return [
        SourceDocument(
            content=item.document.page_content[:200],
            source=item.document.metadata.get("source_file", "Unknown"),
            department=item.document.metadata.get("department", "Unknown"),
            relevance_score=item.relevance,
            raw_score=item.score,
            score_type=item.score_type.value if item.score_type else None,
            ticker=item.document.metadata.get("ticker"),
            filing_type=item.document.metadata.get("filing_type"),
            filing_date=item.document.metadata.get("filing_date"),
            section_name=item.document.metadata.get("section_name"),
        )
        for item in scored
    ]


def _to_claim_citations(grounded: GroundedAnswer) -> list[ClaimCitation]:
    """Flatten the parsed answer and any verdicts into the response shape.

    Verdicts are keyed by (claim text, block) because a claim can cite more
    than one block and each pairing is judged separately.
    """
    verdicts = {(v.claim, v.document_index): v for v in grounded.citations.verdicts}
    claims = []
    for claim in grounded.parsed.claims:
        judged = [verdicts.get((claim.text, n)) for n in claim.citations]
        first = next((v for v in judged if v is not None), None)
        claims.append(
            ClaimCitation(
                claim=claim.text,
                cited_documents=list(claim.citations),
                invalid_citations=list(claim.out_of_range),
                verdict=first.verdict if first else None,
                reason=first.reason if first else None,
            )
        )
    return claims


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    user: User = Depends(get_current_user),
    retriever: Retriever = Depends(get_rbac_retriever),
) -> QueryResponse:
    guardrail_flags: list[str] = []

    # The access state this query runs under. Computed before the guardrails
    # rather than after retrieval so that every return path carries it —
    # a query blocked at the door was still gated by a role, and a client
    # showing which wall was in force should not have to guess on the paths
    # where nothing was retrieved.
    barriers = get_information_barriers_for_user(user.role_names)
    barrier_names = [b["name"] for b in barriers]
    departments = sorted(get_accessible_departments(user.role_names))
    structured_barriers = to_information_barriers(barriers)

    # Input guardrails
    validate_input(request.question)

    injection_result = detect_prompt_injection(request.question)
    if injection_result.is_blocked:
        guardrail_flags.append(f"injection_blocked: {injection_result.reason}")
        return QueryResponse(
            answer="Your query was blocked by our safety system. Please rephrase your question.",
            sources=[],
            query=request.question,
            guardrail_flags=guardrail_flags,
            accessible_departments=departments,
            information_barriers=structured_barriers,
        )

    if injection_result.risk_score > 0.3:
        guardrail_flags.append(f"injection_warning: score={injection_result.risk_score:.2f}")

    # Clean PII from input
    clean_question = redact_pii(request.question)
    if clean_question != request.question:
        guardrail_flags.append("pii_redacted_from_input")

    # Retrieve relevant documents (RBAC-filtered with information barriers).
    # retrieve_scored keeps the relevance scores attached, which is what the
    # confidence layer reads; a retriever without a score channel degrades to
    # unscored documents rather than erroring.
    scored = retrieve_scored(retriever, clean_question)
    documents = [item.document for item in scored]

    # Record the barriers in the audit trail's own flattened form. Kept
    # alongside `information_barriers` rather than replaced by it: this string
    # is what 17a-4 log lines already contain, and rewriting it would break
    # the comparability of the existing trail.
    if barrier_names:
        guardrail_flags.append(f"information_barriers: {', '.join(barrier_names)}")

    if not documents:
        return QueryResponse(
            answer="No relevant documents found for your query within your access level.",
            sources=[],
            query=request.question,
            guardrail_flags=guardrail_flags,
            accessible_departments=departments,
            information_barriers=structured_barriers,
        )

    # Generate answer, parse and score its citations. All of it happens in the
    # generation layer so the eval harness measures the same code path.
    try:
        grounded = generate_grounded_answer(clean_question, scored)
    except Exception as e:
        logger.error("llm_generation_failed", error=str(e), error_type=type(e).__name__)
        return QueryResponse(
            answer=(
                "Retrieved relevant filings, but answer generation failed "
                f"({type(e).__name__}). The LLM provider may be misconfigured — "
                "check the LLM_PROVIDER and API key environment variables."
            ),
            sources=_to_source_documents(scored),
            query=request.question,
            guardrail_flags=[*guardrail_flags, "llm_generation_error"],
            accessible_departments=departments,
            information_barriers=structured_barriers,
        )

    # Citations were parsed from grounded.answer above — i.e. before the
    # rewrites below. Disclaimers append sentences the model never wrote, and
    # PII redaction can alter the interior of a claim, so anything derived
    # from the answer text has to be derived first.
    answer = grounded.answer

    if grounded.unanswered:
        guardrail_flags.append(f"insufficient_context: {grounded.unanswered.reason}")
    if grounded.parsed.out_of_range_count:
        guardrail_flags.append(f"invalid_citations: {grounded.parsed.out_of_range_count}")
    guardrail_flags.append(f"confidence: {grounded.confidence.label}")

    # Output guardrails
    output_check = check_output_safety(answer)
    if output_check.flags:
        guardrail_flags.extend(output_check.flags)

    # Financial compliance guardrails
    fin_compliance = check_financial_compliance(
        query=request.question,
        response=answer,
        user_roles=user.role_names,
    )
    if fin_compliance.flags:
        guardrail_flags.extend(fin_compliance.flags)

    # Apply financial disclaimers (investment advice, MNPI, forward-looking)
    answer = apply_financial_disclaimers(answer, fin_compliance)

    # Redact PII from output
    clean_answer = redact_pii(answer)
    if clean_answer != answer:
        guardrail_flags.append("pii_redacted_from_output")

    # Compliance audit trail
    log_query_audit(
        user_id=user.id,
        username=user.username,
        user_roles=list(user.role_names),
        query=request.question,
        retrieved_departments=list({doc.metadata.get("department", "") for doc in documents}),
        documents_accessed=len(documents),
        guardrail_flags=guardrail_flags,
        information_barriers_applied=barrier_names,
        response_length=len(clean_answer),
    )

    return QueryResponse(
        answer=clean_answer,
        sources=_to_source_documents(scored),
        query=request.question,
        guardrail_flags=guardrail_flags,
        accessible_departments=departments,
        information_barriers=structured_barriers,
        confidence=ConfidenceScore(**grounded.confidence.as_dict()),
        claims=_to_claim_citations(grounded),
        unanswered=(
            UnansweredReport(**grounded.unanswered.as_dict()) if grounded.unanswered else None
        ),
    )
