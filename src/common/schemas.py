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

    model_config = {"json_schema_extra": {
        "examples": [{"username": "new_analyst", "password": "secure123!", "roles": ["research"]}]
    }}


class UserResponse(BaseModel):
    id: int
    username: str
    roles: list[str]
    created_at: datetime


class TokenRequest(BaseModel):
    username: str = Field(examples=["research_analyst"])
    password: str = Field(examples=["research1!"])

    model_config = {"json_schema_extra": {
        "examples": [{"username": "research_analyst", "password": "research1!"}]
    }}


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

    model_config = {"json_schema_extra": {
        "examples": [
            {"question": "What was Apple's total net revenue for fiscal year 2024?"},
            {"question": "Compare JPMorgan and Goldman Sachs credit risk disclosures"},
            {"question": "What are Tesla's key risk factors from their latest 10-K?"},
        ]
    }}


class SourceDocument(BaseModel):
    content: str = Field(description="First 200 chars of the retrieved chunk")
    source: str = Field(description="Source file path")
    department: str = Field(description="Document department (sec_filings, trading, etc.)")
    relevance_score: float | None = None
    ticker: str | None = Field(default=None, description="Company ticker (AAPL, JPM, TSLA, MSFT, GS)")
    filing_type: str | None = Field(default=None, description="Filing type (e.g. 10-K)")
    filing_date: str | None = Field(default=None, description="Filing date")
    section_name: str | None = Field(default=None, description="10-K section (e.g. Item 7 - MD&A)")


class QueryResponse(BaseModel):
    answer: str = Field(description="LLM-generated answer with financial disclaimers if applicable")
    sources: list[SourceDocument] = Field(description="Retrieved document chunks used to generate the answer")
    query: str = Field(description="Original user query")
    guardrail_flags: list[str] = Field(
        default_factory=list,
        description="Triggered guardrails: MNPI detection, investment advice, information barriers, PII redaction",
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
