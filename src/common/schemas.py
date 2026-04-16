from datetime import datetime

from pydantic import BaseModel, Field


# --- Auth Schemas ---

class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)
    roles: list[str] = Field(default_factory=lambda: ["viewer"])


class UserResponse(BaseModel):
    id: int
    username: str
    roles: list[str]
    created_at: datetime


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# --- Query Schemas ---

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class SourceDocument(BaseModel):
    content: str
    source: str
    department: str
    relevance_score: float | None = None
    ticker: str | None = None
    filing_type: str | None = None
    filing_date: str | None = None
    section_name: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str
    guardrail_flags: list[str] = Field(default_factory=list)


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
    status: str
    environment: str
    version: str = "0.1.0"


class ReadinessResponse(BaseModel):
    status: str
    vector_store: str
    database: str
    llm: str
