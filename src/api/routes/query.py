from fastapi import APIRouter, Depends, Request, Response

from src.api.audit import log_query_audit
from src.api.deps import get_current_user, get_rbac_retriever
from src.api.middleware import limiter
from src.api.routes.access import to_information_barriers
from src.auth.models import User
from src.auth.rbac import get_accessible_departments, get_information_barriers_for_user
from src.common.logging import get_logger
from src.common.schemas import (
    ClaimCitation,
    ConfidenceScore,
    QueryRequest,
    QueryResponse,
    RetrievalConfig,
    RetrievalMode,
    SourceDocument,
    UnansweredReport,
)
from src.config import settings
from src.generation.answer import GroundedAnswer, generate_grounded_answer
from src.guardrails.financial_compliance import (
    apply_financial_disclaimers,
    check_financial_compliance,
)
from src.guardrails.input_validator import validate_input
from src.guardrails.output_safety import check_output_safety
from src.guardrails.pii_detector import redact_pii
from src.guardrails.prompt_injection import detect_prompt_injection
from src.retrieval.retriever import Retriever, get_retriever, retrieve_scored
from src.retrieval.scores import ScoredDocument

router = APIRouter(tags=["Query"])
logger = get_logger(__name__)


# One rate-limit bucket for every path that reaches the generation pipeline.
# slowapi keys on the request path by default, so `/query` and `/v1/ask` would
# otherwise hold a budget each and a client alternating between them would get
# twice the limit — silently, since nothing errors and nothing logs.
QUERY_LIMIT_SCOPE = "query"


def _resolve_retriever(mode: RetrievalMode, user: User, configured: Retriever) -> Retriever:
    """The pipeline for this request, honouring a per-request mode override.

    The configured retriever arrives as a dependency and is returned
    unchanged for the default mode — which keeps it overridable in tests and
    means the common path builds nothing extra. A pinned mode constructs a
    fresh pipeline instead, because the stage choice is baked in at
    construction.
    """
    if mode is RetrievalMode.DEFAULT:
        return configured
    return get_retriever(
        user_roles=user.role_names,
        hybrid=mode is RetrievalMode.HYBRID,
    )


# Substrings that identify a provider refusing on billing or rate grounds
# rather than on anything the caller did. Matched on the lowercased exception
# text because the alternative — importing each provider's exception classes —
# would make this module depend on every embedding backend the project can be
# configured to use, including ones not installed.
_QUOTA_SIGNALS = (
    "insufficient_quota",
    "credit_balance",
    "no credits remaining",
    "exceeded your current quota",
    "rate limit",
    "429",
)
_AUTH_SIGNALS = ("invalid_api_key", "incorrect api key", "unauthorized", "401")


def _retrieval_failure(exc: Exception) -> tuple[str, str]:
    """A truthful sentence for the reader, and a flag for the audit trail.

    Three cases, because they mean different things to whoever is reading. A
    quota failure is the demo being out of credit and is nobody's fault but
    the owner's; an auth failure is a misconfigured deployment; anything else
    is a real defect worth reporting as one. None of them is the user's
    question being wrong, and the message says so rather than leaving a
    reader to guess whether the system rejected what they asked.
    """
    text = f"{type(exc).__name__}: {exc}".lower()

    if any(signal in text for signal in _QUOTA_SIGNALS):
        return (
            "Search is temporarily unavailable: this demo's embedding provider has no "
            "quota remaining, and a question has to be embedded before the filings can "
            "be searched. Nothing is wrong with the question or the index — the corpus "
            "and every published measurement are unaffected. Please try again later.",
            "retrieval_unavailable: embedding_quota_exhausted",
        )

    if any(signal in text for signal in _AUTH_SIGNALS):
        return (
            "Search is unavailable: the embedding provider rejected this deployment's "
            "credentials. This is a configuration problem on the server, not a problem "
            "with your question.",
            "retrieval_unavailable: embedding_auth_failed",
        )

    return (
        f"Search failed before any filings could be retrieved ({type(exc).__name__}). "
        "This is a server-side error rather than a problem with your question.",
        f"retrieval_unavailable: {type(exc).__name__}",
    )


def _retrieval_config(mode: RetrievalMode) -> RetrievalConfig:
    hybrid = (
        settings.hybrid_search_enabled
        if mode is RetrievalMode.DEFAULT
        else (mode is RetrievalMode.HYBRID)
    )
    return RetrievalConfig(
        mode="hybrid" if hybrid else "dense",
        reranked=settings.rerank_enabled,
        hyde=settings.hyde_enabled,
        top_k=settings.retrieval_top_k,
    )


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
@limiter.shared_limit(settings.rate_limit, scope=QUERY_LIMIT_SCOPE)
async def query_documents(
    request: Request,
    response: Response,
    payload: QueryRequest,
    user: User = Depends(get_current_user),
    retriever: Retriever = Depends(get_rbac_retriever),
) -> QueryResponse:
    """The expensive route, and the only one that reaches an LLM.

    `request` and `response` are the rate limiter's, which requires parameters
    of exactly those names and types; the body moved to `payload` to make room.
    FastAPI treats a lone Pydantic parameter as the whole body regardless of
    its name, so the wire format is unchanged.

    A thin shell over `answer_query` so that `/v1/ask`, the name Project 6
    §5.1 gives this endpoint, can be a second decorated route over the same
    implementation. It cannot simply call this function: slowapi keys a limit
    on the **request path**, so a delegating alias gets its own budget and
    doubles what a client can spend. `shared_limit` with an explicit scope is
    what puts both paths in one bucket, and a test pins it.
    """
    return await answer_query(request, response, payload, user, retriever)


async def answer_query(
    request: Request,
    response: Response,
    payload: QueryRequest,
    user: User,
    retriever: Retriever,
) -> QueryResponse:
    """Everything the query endpoint does, minus the routing and the limit.

    Undecorated on purpose: each *route* is rate limited exactly once, and this
    is called after that has already happened. Adding a limit here would charge
    a request twice.
    """
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
    validate_input(payload.question)

    injection_result = detect_prompt_injection(payload.question)
    if injection_result.is_blocked:
        guardrail_flags.append(f"injection_blocked: {injection_result.reason}")
        return QueryResponse(
            answer="Your query was blocked by our safety system. Please rephrase your question.",
            sources=[],
            query=payload.question,
            guardrail_flags=guardrail_flags,
            accessible_departments=departments,
            information_barriers=structured_barriers,
        )

    if injection_result.risk_score > 0.3:
        guardrail_flags.append(f"injection_warning: score={injection_result.risk_score:.2f}")

    # Clean PII from input
    clean_question = redact_pii(payload.question)
    if clean_question != payload.question:
        guardrail_flags.append("pii_redacted_from_input")

    # Retrieve relevant documents (RBAC-filtered with information barriers).
    # retrieve_scored keeps the relevance scores attached, which is what the
    # confidence layer reads; a retriever without a score channel degrades to
    # unscored documents rather than erroring.
    retriever = _resolve_retriever(payload.retrieval_mode, user, retriever)
    retrieval_config = _retrieval_config(payload.retrieval_mode)

    # Retrieval reaches the embedding provider before it reaches the index, and
    # that call is the one external dependency on the serving path with no
    # fallback: the shipped index is 1536-dimensional, so a deployment cannot
    # quietly switch to the local 384-dimensional embedder without rebuilding
    # the corpus. Generation has had a handler since the beginning; this one was
    # missing, and the gap surfaced the way gaps do — the OpenAI balance hit
    # zero and every query on a public demo returned a bare
    # "Internal Server Error" with no indication of what had failed or whether
    # the system was broken. Costing $0.000005 to embed a question is not a
    # reason to skip the error path around it.
    try:
        scored = retrieve_scored(retriever, clean_question)
    except Exception as e:
        reason, flag = _retrieval_failure(e)
        logger.error("retrieval_failed", error=str(e), error_type=type(e).__name__, flag=flag)
        # A query that failed is still a query that was asked, and 17a-4 has no
        # exception for the ones that errored. Recorded with zero documents
        # accessed, which is the true statement about what this query reached.
        log_query_audit(
            user_id=user.id,
            username=user.username,
            user_roles=list(user.role_names),
            query=payload.question,
            retrieved_departments=[],
            documents_accessed=0,
            guardrail_flags=[*guardrail_flags, flag],
            information_barriers_applied=barrier_names,
            response_length=len(reason),
        )
        return QueryResponse(
            answer=reason,
            sources=[],
            query=payload.question,
            guardrail_flags=[*guardrail_flags, flag],
            accessible_departments=departments,
            information_barriers=structured_barriers,
            retrieval=retrieval_config,
        )

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
            query=payload.question,
            guardrail_flags=guardrail_flags,
            accessible_departments=departments,
            information_barriers=structured_barriers,
            retrieval=retrieval_config,
        )

    # Generate answer, parse and score its citations. All of it happens in the
    # generation layer so the eval harness measures the same code path.
    try:
        # The bounded retry searches through this closure rather than building
        # a retriever of its own, so it inherits the RBAC and information-barrier
        # scope already resolved for this caller and cannot widen it.
        grounded = generate_grounded_answer(
            clean_question,
            scored,
            search=lambda query: retriever.retrieve(query),
        )
    except Exception as e:
        logger.error("llm_generation_failed", error=str(e), error_type=type(e).__name__)
        return QueryResponse(
            answer=(
                "Retrieved relevant filings, but answer generation failed "
                f"({type(e).__name__}). The LLM provider may be misconfigured — "
                "check the LLM_PROVIDER and API key environment variables."
            ),
            sources=_to_source_documents(scored),
            query=payload.question,
            guardrail_flags=[*guardrail_flags, "llm_generation_error"],
            accessible_departments=departments,
            information_barriers=structured_barriers,
            retrieval=retrieval_config,
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
        query=payload.question,
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
        query=payload.question,
        retrieved_departments=list({doc.metadata.get("department", "") for doc in documents}),
        documents_accessed=len(documents),
        guardrail_flags=guardrail_flags,
        information_barriers_applied=barrier_names,
        response_length=len(clean_answer),
    )

    return QueryResponse(
        answer=clean_answer,
        sources=_to_source_documents(scored),
        query=payload.question,
        guardrail_flags=guardrail_flags,
        accessible_departments=departments,
        information_barriers=structured_barriers,
        retrieval=retrieval_config,
        confidence=ConfidenceScore(**grounded.confidence.as_dict()),
        claims=_to_claim_citations(grounded),
        unanswered=(
            UnansweredReport(**grounded.unanswered.as_dict()) if grounded.unanswered else None
        ),
    )
