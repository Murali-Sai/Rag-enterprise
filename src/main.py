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
            }
            .hero {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 60vh;
                padding: 48px 24px 0;
            }
            .hero-content {
                max-width: 720px;
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
                cursor: pointer;
                border: none;
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

            /* ── Try It section ── */
            .try-section {
                max-width: 800px;
                margin: 0 auto;
                padding: 0 24px 60px;
            }
            .try-section h2 {
                text-align: center;
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 8px;
                color: #f1f5f9;
            }
            .try-subtitle {
                text-align: center;
                color: #64748b;
                font-size: 0.85rem;
                margin-bottom: 32px;
            }
            .demo-card {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                padding: 32px;
            }
            .step { margin-bottom: 24px; }
            .step-label {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }
            .step-num {
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                color: white;
                width: 28px;
                height: 28px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.8rem;
                font-weight: 700;
                flex-shrink: 0;
            }
            .step-label span:last-child {
                font-size: 0.9rem;
                font-weight: 600;
                color: #cbd5e1;
            }
            .role-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }
            .role-btn {
                background: rgba(255,255,255,0.06);
                border: 2px solid rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 12px;
                cursor: pointer;
                transition: all 0.2s;
                text-align: left;
                color: #e2e8f0;
            }
            .role-btn:hover { border-color: #3b82f6; background: rgba(59,130,246,0.1); }
            .role-btn.active { border-color: #3b82f6; background: rgba(59,130,246,0.15); }
            .role-btn .role-name { font-weight: 600; font-size: 0.85rem; }
            .role-btn .role-desc { font-size: 0.7rem; color: #64748b; margin-top: 2px; }
            .role-btn .role-access { font-size: 0.65rem; color: #475569; margin-top: 4px; }
            .login-status {
                margin-top: 8px;
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 0.8rem;
                display: none;
            }
            .login-status.success { display: block; background: rgba(34,197,94,0.15); color: #4ade80; }
            .login-status.error { display: block; background: rgba(239,68,68,0.15); color: #f87171; }
            .example-questions {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 12px;
            }
            .example-q {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                padding: 8px 14px;
                font-size: 0.75rem;
                color: #94a3b8;
                cursor: pointer;
                transition: all 0.15s;
            }
            .example-q:hover { background: rgba(59,130,246,0.15); color: #93c5fd; border-color: #3b82f6; }
            .query-input-row {
                display: flex;
                gap: 8px;
            }
            .query-input {
                flex: 1;
                background: rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 10px;
                padding: 14px 16px;
                color: #e2e8f0;
                font-size: 0.9rem;
                outline: none;
                transition: border-color 0.2s;
            }
            .query-input:focus { border-color: #3b82f6; }
            .query-input::placeholder { color: #475569; }
            .send-btn {
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 14px 24px;
                font-weight: 600;
                font-size: 0.9rem;
                cursor: pointer;
                transition: all 0.2s;
                white-space: nowrap;
            }
            .send-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(59,130,246,0.3); }
            .send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

            /* ── Response area ── */
            .response-area {
                margin-top: 20px;
                display: none;
            }
            .response-area.visible { display: block; }
            .response-box {
                background: rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 20px;
            }
            .response-answer {
                font-size: 0.9rem;
                line-height: 1.7;
                color: #e2e8f0;
                white-space: pre-wrap;
            }
            .response-meta {
                margin-top: 16px;
                padding-top: 16px;
                border-top: 1px solid rgba(255,255,255,0.08);
            }
            .guardrail-flags {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                margin-bottom: 12px;
            }
            .flag-chip {
                background: rgba(251,191,36,0.15);
                color: #fbbf24;
                font-size: 0.7rem;
                padding: 4px 10px;
                border-radius: 100px;
                font-weight: 600;
            }
            .sources-toggle {
                background: none;
                border: none;
                color: #64748b;
                font-size: 0.8rem;
                cursor: pointer;
                padding: 4px 0;
            }
            .sources-toggle:hover { color: #94a3b8; }
            .sources-list {
                display: none;
                margin-top: 8px;
            }
            .sources-list.open { display: block; }
            .source-item {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 8px;
                font-size: 0.75rem;
            }
            .source-item .source-header {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                margin-bottom: 6px;
            }
            .source-tag {
                background: rgba(99,102,241,0.2);
                color: #a5b4fc;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.65rem;
                font-weight: 600;
            }
            .source-content {
                color: #64748b;
                line-height: 1.5;
            }
            .loading-spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid rgba(255,255,255,0.3);
                border-top-color: white;
                border-radius: 50%;
                animation: spin 0.6s linear infinite;
                vertical-align: middle;
                margin-right: 8px;
            }
            @keyframes spin { to { transform: rotate(360deg); } }

            .companies {
                text-align: center;
                padding: 20px 24px 40px;
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
                .role-grid { grid-template-columns: 1fr; }
                h1 { font-size: 1.8rem; }
                .hero { padding: 32px 16px 0; }
                .query-input-row { flex-direction: column; }
            }
        </style>
    </head>
    <body>
        <!-- ── Hero ── -->
        <div class="hero">
            <div class="hero-content">
                <span class="badge">Live</span>
                <h1>RAG Enterprise</h1>
                <p class="subtitle">
                    Production-grade RAG system querying real SEC 10-K filings<br>
                    with RBAC, Chinese Walls, and financial compliance guardrails
                </p>

                <div class="features">
                    <div class="feature">
                        <h3>&#128200; Real SEC Filings</h3>
                        <p>Queries actual 10-K annual reports from EDGAR</p>
                    </div>
                    <div class="feature">
                        <h3>&#128737; Information Barriers</h3>
                        <p>Chinese Walls between Research and Trading</p>
                    </div>
                    <div class="feature">
                        <h3>&#9888;&#65039; Financial Guardrails</h3>
                        <p>MNPI detection, investment advice blocking</p>
                    </div>
                    <div class="feature">
                        <h3>&#128202; RAGAS Evaluated</h3>
                        <p>Faithfulness 0.65 &bull; Relevancy 0.68 &bull; Recall 0.70</p>
                    </div>
                </div>

                <div class="cta-group">
                    <a href="#try-it" class="btn btn-primary">&#9889; Try It Live</a>
                    <a href="/docs" class="btn btn-secondary">API Docs</a>
                    <a href="https://github.com/Murali-Sai/Rag-enterprise" class="btn btn-secondary" target="_blank">
                        GitHub
                    </a>
                </div>
            </div>
        </div>

        <!-- ── Try It Live ── -->
        <div class="try-section" id="try-it">
            <h2>Try It Live</h2>
            <p class="try-subtitle">Query real SEC 10-K filings from Apple, JPMorgan, Tesla, Microsoft, and Goldman Sachs</p>

            <div class="demo-card">
                <!-- Step 1: Pick a role -->
                <div class="step">
                    <div class="step-label">
                        <div class="step-num">1</div>
                        <span>Choose a demo account</span>
                    </div>
                    <div class="role-grid">
                        <button class="role-btn" onclick="selectRole(this, 'research_analyst', 'research1!')"
                                data-user="research_analyst">
                            <div class="role-name">Research Analyst</div>
                            <div class="role-desc">Equity research team</div>
                            <div class="role-access">Access: SEC filings (Chinese Wall enforced)</div>
                        </button>
                        <button class="role-btn" onclick="selectRole(this, 'trader_desk', 'trade1234!')"
                                data-user="trader_desk">
                            <div class="role-name">Trading Desk</div>
                            <div class="role-desc">Front office trading</div>
                            <div class="role-access">Access: Trading, Risk, SEC filings</div>
                        </button>
                        <button class="role-btn" onclick="selectRole(this, 'compliance_officer', 'compl1234!')"
                                data-user="compliance_officer">
                            <div class="role-name">Compliance Officer</div>
                            <div class="role-desc">Regulatory compliance</div>
                            <div class="role-access">Access: Compliance, SEC, Risk</div>
                        </button>
                        <button class="role-btn" onclick="selectRole(this, 'admin_user', 'admin1234!')"
                                data-user="admin_user">
                            <div class="role-name">Admin</div>
                            <div class="role-desc">System administrator</div>
                            <div class="role-access">Access: All departments</div>
                        </button>
                    </div>
                    <div class="login-status" id="loginStatus"></div>
                </div>

                <!-- Step 2: Ask a question -->
                <div class="step">
                    <div class="step-label">
                        <div class="step-num">2</div>
                        <span>Ask a question about SEC filings</span>
                    </div>
                    <div class="example-questions">
                        <button class="example-q" onclick="setQuestion(this)">What was Apple's total revenue in fiscal year 2024?</button>
                        <button class="example-q" onclick="setQuestion(this)">What are Tesla's key risk factors?</button>
                        <button class="example-q" onclick="setQuestion(this)">Compare JPMorgan and Goldman Sachs credit risk</button>
                        <button class="example-q" onclick="setQuestion(this)">What is Microsoft's cloud revenue growth?</button>
                    </div>
                    <div class="query-input-row">
                        <input type="text" class="query-input" id="questionInput"
                               placeholder="Ask anything about SEC 10-K filings..."
                               onkeydown="if(event.key==='Enter') sendQuery()">
                        <button class="send-btn" id="sendBtn" onclick="sendQuery()" disabled>
                            Ask
                        </button>
                    </div>
                </div>

                <!-- Response -->
                <div class="response-area" id="responseArea">
                    <div class="response-box">
                        <div class="response-answer" id="responseAnswer"></div>
                        <div class="response-meta" id="responseMeta"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ── Footer ── -->
        <div class="companies">
            <div>10-K filings indexed from</div>
            <div class="tickers">
                <span>AAPL</span> <span>JPM</span> <span>TSLA</span> <span>MSFT</span> <span>GS</span>
            </div>
        </div>

        <script>
            let token = null;

            async function selectRole(el, username, password) {
                // Highlight selected
                document.querySelectorAll('.role-btn').forEach(b => b.classList.remove('active'));
                el.classList.add('active');

                const status = document.getElementById('loginStatus');
                status.className = 'login-status';
                status.textContent = 'Logging in...';
                status.style.display = 'block';

                try {
                    const res = await fetch('/auth/token', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ username, password })
                    });
                    if (!res.ok) throw new Error('Login failed');
                    const data = await res.json();
                    token = data.access_token;
                    status.className = 'login-status success';
                    status.textContent = 'Logged in as ' + username + ' — ready to query';
                    document.getElementById('sendBtn').disabled = false;
                } catch (e) {
                    token = null;
                    status.className = 'login-status error';
                    status.textContent = 'Login failed — try again';
                    document.getElementById('sendBtn').disabled = true;
                }
            }

            function setQuestion(el) {
                document.getElementById('questionInput').value = el.textContent;
            }

            async function sendQuery() {
                const input = document.getElementById('questionInput');
                const question = input.value.trim();
                if (!question || !token) return;

                const btn = document.getElementById('sendBtn');
                const area = document.getElementById('responseArea');
                const answer = document.getElementById('responseAnswer');
                const meta = document.getElementById('responseMeta');

                btn.disabled = true;
                btn.innerHTML = '<span class="loading-spinner"></span> Thinking...';
                area.classList.add('visible');
                answer.textContent = 'Querying SEC filings...';
                meta.innerHTML = '';

                try {
                    const res = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + token
                        },
                        body: JSON.stringify({ question })
                    });

                    if (res.status === 401) {
                        answer.textContent = 'Session expired — please select a role again.';
                        token = null;
                        return;
                    }

                    const data = await res.json();

                    if (!res.ok) {
                        answer.textContent = data.message || data.detail || 'Error processing query';
                        return;
                    }

                    // Show answer
                    answer.textContent = data.answer;

                    // Build meta section
                    let metaHtml = '';

                    // Guardrail flags
                    if (data.guardrail_flags && data.guardrail_flags.length > 0) {
                        metaHtml += '<div class="guardrail-flags">';
                        data.guardrail_flags.forEach(f => {
                            metaHtml += '<span class="flag-chip">' + f + '</span>';
                        });
                        metaHtml += '</div>';
                    }

                    // Sources
                    if (data.sources && data.sources.length > 0) {
                        const id = 'src-' + Date.now();
                        metaHtml += '<button class="sources-toggle" onclick="toggleSources(\\''+id+'\\')">&#9660; ' + data.sources.length + ' source(s)</button>';
                        metaHtml += '<div class="sources-list" id="'+id+'">';
                        data.sources.forEach(s => {
                            metaHtml += '<div class="source-item"><div class="source-header">';
                            if (s.ticker) metaHtml += '<span class="source-tag">' + s.ticker + '</span>';
                            if (s.section_name) metaHtml += '<span class="source-tag">' + s.section_name + '</span>';
                            if (s.filing_type) metaHtml += '<span class="source-tag">' + s.filing_type + '</span>';
                            metaHtml += '</div><div class="source-content">' + (s.content || '').substring(0, 200) + '...</div></div>';
                        });
                        metaHtml += '</div>';
                    }

                    meta.innerHTML = metaHtml;

                } catch (e) {
                    answer.textContent = 'Network error — is the server running?';
                } finally {
                    btn.disabled = false;
                    btn.textContent = 'Ask';
                }
            }

            function toggleSources(id) {
                document.getElementById(id).classList.toggle('open');
            }
        </script>
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
