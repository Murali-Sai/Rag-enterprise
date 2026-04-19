from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.middleware import setup_middleware
from src.api.router import api_router
from src.auth.repository import init_db
from src.common.exceptions import GuardrailViolation, RAGEnterpriseError
from src.common.logging import get_logger, setup_logging
from src.config import settings

logger = get_logger(__name__)

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
| **RAGAS Evaluated** | Faithfulness 0.65, Answer Relevancy 0.68, Context Recall 0.70 |

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
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "docExpansion": "list",
        "defaultModelsExpandDepth": 0,
        "persistAuthorization": True,
        "filter": True,
        "syntaxHighlight.theme": "monokai",
        "tryItOutEnabled": True,
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    contact={
        "name": "Murali Sai",
        "url": "https://github.com/Murali-Sai",
    },
)

setup_middleware(app)
app.include_router(api_router)


# ── Landing page (root path) ─────────────────────────────────
@app.get("/", include_in_schema=False)
async def root() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Enterprise — SEC EDGAR Filing Analyzer</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
                color: #e2e8f0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                max-width: 720px;
                padding: 48px;
                text-align: center;
            }
            .badge {
                display: inline-block;
                background: #22c55e;
                color: #052e16;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                padding: 4px 12px;
                border-radius: 100px;
                margin-bottom: 24px;
            }
            h1 {
                font-size: 2.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 12px;
            }
            .subtitle {
                font-size: 1.1rem;
                color: #94a3b8;
                margin-bottom: 40px;
                line-height: 1.6;
            }
            .features {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                margin-bottom: 40px;
                text-align: left;
            }
            .feature {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 20px;
            }
            .feature-icon { font-size: 1.5rem; margin-bottom: 8px; }
            .feature h3 { font-size: 0.85rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px; }
            .feature p { font-size: 0.75rem; color: #94a3b8; line-height: 1.5; }
            .cta-group { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
            .btn {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 28px;
                border-radius: 10px;
                font-size: 0.95rem;
                font-weight: 600;
                text-decoration: none;
                transition: all 0.2s;
            }
            .btn-primary {
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                color: white;
            }
            .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(59,130,246,0.3); }
            .btn-secondary {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.15);
                color: #e2e8f0;
            }
            .btn-secondary:hover { background: rgba(255,255,255,0.1); }
            .companies {
                margin-top: 40px;
                font-size: 0.8rem;
                color: #64748b;
            }
            .tickers {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-top: 8px;
                font-size: 0.85rem;
                font-weight: 700;
                color: #475569;
                letter-spacing: 2px;
            }
            @media (max-width: 600px) {
                .features { grid-template-columns: 1fr; }
                h1 { font-size: 1.8rem; }
                .container { padding: 24px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <span class="badge">Live</span>
            <h1>RAG Enterprise</h1>
            <p class="subtitle">
                Production-grade RAG system querying real SEC 10-K filings<br>
                with RBAC, Chinese Walls, and financial compliance guardrails
            </p>

            <div class="features">
                <div class="feature">
                    <div class="feature-icon">&#128200;</div>
                    <h3>Real SEC Filings</h3>
                    <p>Queries actual 10-K annual reports from EDGAR — not synthetic data</p>
                </div>
                <div class="feature">
                    <div class="feature-icon">&#128737;</div>
                    <h3>Information Barriers</h3>
                    <p>Chinese Walls enforced at the vector store layer (SEC Rule 15g-1)</p>
                </div>
                <div class="feature">
                    <div class="feature-icon">&#9888;&#65039;</div>
                    <h3>Financial Guardrails</h3>
                    <p>MNPI detection, investment advice blocking, PII redaction</p>
                </div>
                <div class="feature">
                    <div class="feature-icon">&#128202;</div>
                    <h3>RAGAS Evaluated</h3>
                    <p>Faithfulness 0.65 &bull; Relevancy 0.68 &bull; Recall 0.70</p>
                </div>
            </div>

            <div class="cta-group">
                <a href="/docs" class="btn btn-primary">
                    &#9889; API Docs
                </a>
                <a href="https://github.com/Murali-Sai/Rag-enterprise" class="btn btn-secondary" target="_blank">
                    &#128187; GitHub
                </a>
            </div>

            <div class="companies">
                <div>10-K filings indexed from</div>
                <div class="tickers">
                    <span>AAPL</span>
                    <span>JPM</span>
                    <span>TSLA</span>
                    <span>MSFT</span>
                    <span>GS</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


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
