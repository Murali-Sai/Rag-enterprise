from datetime import datetime

from pydantic import BaseModel, Field

# --- Auth Schemas ---


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50, examples=["research_analyst"])
    password: str = Field(min_length=8, examples=["research1!"])
    roles: list[str] = Field(
        default_factory=lambda: ["viewer"],
        examples=[["research"]],
        description="Roles: admin, trading, risk, compliance, research, wealth_management, operations, auditor, viewer",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"username": "new_analyst", "password": "secure123!", "roles": ["research"]}
            ]
        }
    }


class UserResponse(BaseModel):
    id: int
    username: str
    roles: list[str]
    created_at: datetime


class TokenRequest(BaseModel):
    username: str = Field(examples=["research_analyst"])
    password: str = Field(examples=["research1!"])

    model_config = {
        "json_schema_extra": {
            "examples": [{"username": "research_analyst", "password": "research1!"}]
        }
    }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# --- Query Schemas ---


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Natural language question about SEC 10-K filings (AAPL, JPM, TSLA, MSFT, GS)",
        examples=["What was Apple's total net revenue for fiscal year 2024?"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What was Apple's total net revenue for fiscal year 2024?"},
                {"question": "Compare JPMorgan and Goldman Sachs credit risk disclosures"},
                {"question": "What are Tesla's key risk factors from their latest 10-K?"},
            ]
        }
    }


class SourceDocument(BaseModel):
    content: str = Field(description="First 200 chars of the retrieved chunk")
    source: str = Field(description="Source file path")
    department: str = Field(description="Document department (sec_filings, trading, etc.)")
    relevance_score: float | None = Field(
        default=None,
        description="Normalised 0-1 relevance. Null when the retrieval stage produced no comparable score (RRF is ordinal).",
    )
    # The raw number and the stage that produced it, alongside the normalised
    # one. A cross-encoder logit of -2.33 and a cosine relevance of 0.42 both
    # normalise into the same 0-1 channel, and only the raw pair says which
    # scale a reader is looking at. `relevance_score` on its own invites the
    # reading that every retrieval was scored the same way.
    raw_score: float | None = Field(
        default=None, description="The stage's own score, before normalisation"
    )
    score_type: str | None = Field(
        default=None, description="cross_encoder | cosine_relevance | rrf — how to read raw_score"
    )
    ticker: str | None = Field(
        default=None, description="Company ticker (AAPL, JPM, TSLA, MSFT, GS)"
    )
    filing_type: str | None = Field(default=None, description="Filing type (e.g. 10-K)")
    filing_date: str | None = Field(default=None, description="Filing date")
    section_name: str | None = Field(default=None, description="10-K section (e.g. Item 7 - MD&A)")


class ClaimCitation(BaseModel):
    """One sentence of the answer and the context blocks it cited."""

    claim: str
    cited_documents: list[int] = Field(
        default_factory=list, description="1-based indices into `sources`"
    )
    invalid_citations: list[int] = Field(
        default_factory=list,
        description="Cited block numbers that were never supplied — a fabricated reference",
    )
    verdict: str | None = Field(
        default=None,
        description="supported | partial | unsupported | out_of_range, when verification ran",
    )
    reason: str | None = Field(default=None, description="Judge's one-line reasoning")


class ConfidenceScore(BaseModel):
    overall: float = Field(description="Weighted composite, 0-1")
    retrieval: float | None = Field(
        default=None,
        description="Relevance of the retrieved passages; null when the retrieval stage produced no comparable score",
    )
    citation_coverage: float = Field(description="Share of claims carrying a citation")
    answer_completeness: float = Field(description="0 refused, 0.5 partial, 1 answered")
    label: str = Field(description="high | medium | low")


class InformationBarrier(BaseModel):
    """One Chinese Wall that was in force for this query.

    `guardrail_flags` already carries the barrier names, but flattened into a
    single string — `"information_barriers: Research-Trading Wall, ..."` —
    which drops the descriptions and the departments each wall removed. That
    string is what the audit trail records and it stays as it is; this is the
    same fact in a shape a caller can read without parsing prose.
    """

    name: str = Field(examples=["Research-Trading Wall"])
    description: str = Field(
        examples=["Research analysts cannot access non-public trading positions or strategies"]
    )
    blocked_departments: list[str] = Field(
        default_factory=list,
        description="Departments this barrier removed from the search, regardless of role grants",
    )


class AccessProfile(BaseModel):
    """What the current token may search, before any query is run.

    The same access facts `QueryResponse` reports, available without spending
    an LLM call to see them. A client that wants to show the effect of
    switching roles otherwise has to run a real query per role.
    """

    username: str
    roles: list[str]
    accessible_departments: list[str] = Field(
        description="Departments the roles grant, after barriers have removed what they remove"
    )
    information_barriers: list[InformationBarrier] = Field(default_factory=list)
    unrestricted: bool = Field(
        description=(
            "True for admin, where no where-clause is applied at all. The department list is "
            "still populated — it says what the roles grant, not what was filtered on."
        )
    )


class UnansweredReport(BaseModel):
    """What was searched, when the system declines to answer."""

    reason: str = Field(description="low_retrieval_confidence | model_refused")
    summary: str
    searched: list[str] = Field(
        default_factory=list, description="Passages consulted, best match first"
    )
    suggested_documents: list[str] = Field(
        default_factory=list, description="Filings worth checking manually"
    )


class QueryResponse(BaseModel):
    answer: str = Field(description="LLM-generated answer with financial disclaimers if applicable")
    sources: list[SourceDocument] = Field(
        description="Retrieved document chunks used to generate the answer"
    )
    query: str = Field(description="Original user query")
    guardrail_flags: list[str] = Field(
        default_factory=list,
        description="Triggered guardrails: MNPI detection, investment advice, information barriers, PII redaction",
    )
    accessible_departments: list[str] = Field(
        default_factory=list,
        description="Departments this user's roles could search, after information barriers were applied. The RBAC where-clause, as data.",
    )
    information_barriers: list[InformationBarrier] = Field(
        default_factory=list,
        description="Chinese Walls in force for this user. Empty for roles no barrier applies to.",
    )
    # Defaulted, and not by accident. QueryResponse is constructed on four
    # paths — injection-blocked, no documents retrieved, generation failed,
    # and success — and three of them have no answer to analyse. A required
    # field here would break them at runtime rather than at import.
    confidence: ConfidenceScore | None = Field(
        default=None,
        description="Composite of retrieval relevance, citation coverage and answer completeness. Null on paths where nothing was generated.",
    )
    claims: list[ClaimCitation] = Field(
        default_factory=list,
        description="The answer segmented into claims, each with the context blocks it cited",
    )
    unanswered: UnansweredReport | None = Field(
        default=None,
        description="Present when the system declined to answer: what it searched and what to check manually",
    )


# --- Document Schemas ---


class DocumentIngestRequest(BaseModel):
    department: str
    access_roles: list[str]


class DocumentIngestResponse(BaseModel):
    filename: str
    chunks_created: int
    department: str
    access_roles: list[str]


# --- Health Schemas ---


class HealthResponse(BaseModel):
    status: str = Field(examples=["healthy"])
    environment: str = Field(examples=["production"])
    version: str = "0.1.0"


class ReadinessResponse(BaseModel):
    status: str = Field(description="Overall readiness: ready | not_ready")
    vector_store: str = Field(description="ChromaDB connection status")
    database: str = Field(description="SQLite connection status")
    llm: str = Field(description="LLM provider configuration status")
