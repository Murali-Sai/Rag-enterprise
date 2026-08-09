from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api.middleware import setup_middleware
from src.api.router import api_router
from src.auth.repository import init_db
from src.common.exceptions import GuardrailViolation, RAGEnterpriseError
from src.common.logging import get_logger, setup_logging
from src.config import settings
from src.web import STATIC_DIR, templates

logger = get_logger(__name__)

# On-brand favicon: terminal-black square with an amber ascending bar chart
# (matches the filing/terminal landing-page palette). Inline SVG — no binary asset.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#07090b"/>'
    '<rect x="7.5" y="17" width="4" height="8" rx="1" fill="#f5a623"/>'
    '<rect x="14" y="12" width="4" height="13" rx="1" fill="#f5a623"/>'
    '<rect x="20.5" y="8" width="4" height="17" rx="1" fill="#f5a623"/>'
    "</svg>"
)

SWAGGER_UI_PARAMETERS = {
    "docExpansion": "list",
    "defaultModelsExpandDepth": 0,
    "persistAuthorization": True,
    "filter": True,
    "syntaxHighlight.theme": "monokai",
    "tryItOutEnabled": True,
}

# ── Tag metadata for Swagger UI grouping ──────────────────────
tags_metadata = [
    {
        "name": "Health",
        "description": "Service health and readiness probes.",
    },
    {
        "name": "Authentication",
        "description": "JWT-based auth. Login to get a Bearer token, then use it in the **Authorize** button above.",
    },
    {
        "name": "Access",
        "description": (
            "What the current token may search — accessible departments and the information "
            "barriers in force — without running a query. Read-only, no LLM call."
        ),
    },
    {
        "name": "Query",
        "description": (
            "Query real SEC 10-K filings with RAG. Documents are filtered by your role's access level "
            "(Chinese Wall enforcement). Responses include financial compliance guardrails."
        ),
    },
    {
        "name": "Documents",
        "description": "Document ingestion (admin only). Upload files to specific departments with role-based access.",
    },
    {
        "name": "Admin",
        "description": "Administrative endpoints. Requires admin role.",
    },
]

DESCRIPTION = """
## SEC EDGAR RAG System with RBAC & Financial Compliance

A production-grade Retrieval-Augmented Generation system that queries **real SEC 10-K filings**
from Apple, JPMorgan, Tesla, Microsoft, and Goldman Sachs.

### Key Features

| Feature | Description |
|---|---|
| **Real SEC Filings** | Queries actual 10-K annual reports downloaded from SEC EDGAR |
| **Information Barriers** | Chinese Walls between Research and Trading (SEC Rule 15g-1) |
| **Financial Guardrails** | MNPI detection, investment advice blocking, forward-looking statement filters |
| **Audit Trail** | SEC Rule 17a-4 / FINRA 4511 compliant append-only query logs |
| **Evaluated** | 54 questions across 7 strata — Faithfulness 0.70, Citation Accuracy 0.94, Refusal Correctness 0.85 |

### Quick Start

1. **Login** &rarr; `POST /auth/token` with `research_analyst` / `research1!`
2. **Copy the token** from the response
3. **Click Authorize** (lock icon above) &rarr; paste: `Bearer <your_token>`
4. **Query** &rarr; `POST /query` with a question about SEC filings

### Demo Accounts

| Username | Password | Role | Access |
|---|---|---|---|
| `research_analyst` | `research1!` | research | SEC filings (Chinese Wall) |
| `trader_desk` | `trade1234!` | trading | Trading, Risk, SEC |
| `admin_user` | `admin1234!` | admin | All departments |
| `compliance_officer` | `compl1234!` | compliance | Compliance, SEC, Risk |

### Example Questions

- *"What was Apple's total net revenue for fiscal year 2024?"*
- *"Compare JPMorgan and Goldman Sachs credit risk disclosures"*
- *"What are Tesla's key risk factors from their latest 10-K?"*
- *"What is Microsoft's cloud revenue growth?"*

---
**GitHub**: [Murali-Sai/Rag-enterprise](https://github.com/Murali-Sai/Rag-enterprise)
"""


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    # Startup
    setup_logging()
    logger.info("starting_app", environment=settings.environment.value)

    await init_db()
    logger.info("database_ready")

    yield

    # Shutdown
    logger.info("shutting_down")


app = FastAPI(
    title="RAG Enterprise",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,  # served via custom route below (to set a branded favicon)
    redoc_url=None,
    openapi_tags=tags_metadata,
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    contact={
        "name": "Murali Sai",
        "url": "https://github.com/Murali-Sai",
    },
)

setup_middleware(app)
app.include_router(api_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Favicon (inline SVG, no static files) ────────────────────
@app.get("/favicon.svg", include_in_schema=False)
@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


# ── Branded API docs (Swagger UI + ReDoc with custom favicon) ─
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} — API Docs",
        swagger_favicon_url="/favicon.svg",
        init_oauth=app.swagger_ui_init_oauth,
        swagger_ui_parameters=SWAGGER_UI_PARAMETERS,
    )


@app.get("/redoc", include_in_schema=False)
async def custom_redoc() -> HTMLResponse:
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} — API Reference",
        redoc_favicon_url="/favicon.svg",
    )


# ── Server-rendered pages ────────────────────────────────────
#
# Templates, not a string literal. This module used to carry the landing page
# as one 780-line inline HTML document — style, script and all — which was
# tolerable while it was the only screen and stopped being so the moment a
# second one existed. See src/web/.
@app.get("/", include_in_schema=False)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "landing.html")


@app.get("/dashboard", include_in_schema=False)
async def dashboard(request: Request) -> HTMLResponse:
    """The query dashboard.

    Renders the shell only. Every number on the screen arrives from
    `POST /query` and `GET /access` at run time, because a template that
    baked in a score would be inventing one — and the scores this system
    reports are the thing it is trying to be honest about.
    """
    return templates.TemplateResponse(request, "dashboard.html")


@app.exception_handler(GuardrailViolation)
async def guardrail_handler(request: Request, exc: GuardrailViolation) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": "guardrail_violation",
            "violation_type": exc.violation_type,
            "message": str(exc),
        },
    )


@app.exception_handler(RAGEnterpriseError)
async def app_error_handler(request: Request, exc: RAGEnterpriseError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": str(exc)},
    )
