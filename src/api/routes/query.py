from fastapi import APIRouter, Depends

from src.api.audit import log_query_audit
from src.api.deps import get_current_user, get_rbac_retriever
from src.auth.models import User
from src.auth.rbac import get_information_barriers_for_user
from src.common.schemas import QueryRequest, QueryResponse, SourceDocument
from src.generation.chains import query_with_context
from src.guardrails.financial_compliance import (
    apply_financial_disclaimers,
    check_financial_compliance,
)
from src.guardrails.input_validator import validate_input
from src.guardrails.output_safety import check_output_safety
from src.guardrails.pii_detector import redact_pii
from src.guardrails.prompt_injection import detect_prompt_injection
from src.retrieval.retriever import RBACRetriever

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    user: User = Depends(get_current_user),
    retriever: RBACRetriever = Depends(get_rbac_retriever),
) -> QueryResponse:
    guardrail_flags: list[str] = []

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
        )

    if injection_result.risk_score > 0.3:
        guardrail_flags.append(f"injection_warning: score={injection_result.risk_score:.2f}")

    # Clean PII from input
    clean_question = redact_pii(request.question)
    if clean_question != request.question:
        guardrail_flags.append("pii_redacted_from_input")

    # Retrieve relevant documents (RBAC-filtered with information barriers)
    documents = retriever.retrieve(clean_question)

    # Track active information barriers for audit
    barriers = get_information_barriers_for_user(user.role_names)
    barrier_names = [b["name"] for b in barriers]
    if barrier_names:
        guardrail_flags.append(f"information_barriers: {', '.join(barrier_names)}")

    if not documents:
        return QueryResponse(
            answer="No relevant documents found for your query within your access level.",
            sources=[],
            query=request.question,
            guardrail_flags=guardrail_flags,
        )

    # Generate answer
    answer = query_with_context(clean_question, documents)

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

    # Build source documents
    sources = [
        SourceDocument(
            content=doc.page_content[:200],
            source=doc.metadata.get("source_file", "Unknown"),
            department=doc.metadata.get("department", "Unknown"),
            ticker=doc.metadata.get("ticker"),
            filing_type=doc.metadata.get("filing_type"),
            filing_date=doc.metadata.get("filing_date"),
            section_name=doc.metadata.get("section_name"),
        )
        for doc in documents
    ]

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
        sources=sources,
        query=request.question,
        guardrail_flags=guardrail_flags,
    )
