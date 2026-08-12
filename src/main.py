import asyncio
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


async def _warm_models() -> None:
    """Load the cross-encoder so the first real query does not pay for it.

    `get_reranker()` is a lazy singleton, so without this the model loads on
    whichever request happens to be first. Measured against the live
    deployment: a warm landing page serves in 0.10s and a steady-state query
    in 3.4s, but the *first* query after a cold start took 23.1s — essentially
    all of it this load.

    Run in a thread rather than on the event loop because it is blocking CPU
    and disk work, and started as a background task rather than awaited in
    `lifespan` because awaiting it would move those seconds into container
    startup, where Cloud Run holds the request that triggered the cold start.
    The landing page needs neither the model nor the index, so it should not
    wait for them. This way the container reports ready immediately and the
    model loads while the visitor is reading the page and logging in.
    """
    try:
        from src.retrieval.reranker import get_reranker

        await asyncio.get_running_loop().run_in_executor(None, get_reranker)
        logger.info("models_warm")
    except Exception as exc:  # noqa: BLE001
        # A warmup is an optimisation, never a reason to fail startup. If it
        # breaks, the lazy path still works and the first query is merely slow.
        logger.warning("model_warmup_failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    # Startup
    setup_logging()
    logger.info("starting_app", environment=settings.environment.value)

    await init_db()
    logger.info("database_ready")

    # Off by default so the test suite and local runs do not pay to load a
    # cross-encoder they may never use; the Dockerfile turns it on, the same
    # way it pins ALLOW_RUNTIME_INGEST.
    warmup = asyncio.create_task(_warm_models()) if settings.warm_models_on_startup else None

    yield

    # Shutdown
    if warmup is not None and not warmup.done():
        warmup.cancel()
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
# as one 780-line inline HTML document — style, script and all. See src/web/.
#
# The query dashboard is not here: it is a Streamlit app in dashboard/, a
# separate process and a separate container, reached at settings.dashboard_url.
@app.get("/", include_in_schema=False)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "landing.html", {"dashboard_url": settings.dashboard_url}
    )


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
