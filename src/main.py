from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, Response

from src.api.middleware import setup_middleware
from src.api.router import api_router
from src.auth.repository import init_db
from src.common.exceptions import GuardrailViolation, RAGEnterpriseError
from src.common.logging import get_logger, setup_logging
from src.config import settings

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
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,340;0,9..144,480;0,9..144,600;0,9..144,700;1,9..144,480&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#16181d;
    --ink-soft:rgba(22,24,29,.66);
    --ink-faint:rgba(22,24,29,.4);
    --paper:#efe9da;
    --paper-panel:#e6dfcb;
    --paper-line:rgba(22,24,29,.15);
    --terminal:#07090b;
    --terminal-panel:#0d1113;
    --terminal-line:rgba(255,255,255,.09);
    --terminal-text:rgba(232,236,232,.86);
    --amber:#f5a623;
    --wire:#4ade80;
    --barrier:#c0392b;
    --font-display:'Fraunces',serif;
    --font-body:'IBM Plex Sans',sans-serif;
    --font-mono:'IBM Plex Mono',monospace;
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  html{scroll-behavior:smooth;}
  body{
    background:var(--paper);
    color:var(--ink);
    font-family:var(--font-body);
    -webkit-font-smoothing:antialiased;
    display:flex;
    flex-direction:column;
    min-height:100vh;
  }
  a{color:inherit;}
  ::selection{background:var(--amber);color:var(--ink);}
  :focus-visible{outline:2px solid var(--ink);outline-offset:3px;}
  .terminal :focus-visible{outline:2px solid var(--amber);}

  /* Terminal is the last section — let it absorb any leftover viewport height
     on short pages / tall screens instead of exposing paper-colored body
     background below it. */
  .terminal{flex:1 0 auto;}

  /* ============ PAPER (filing) ============ */
  .paper{
    flex:0 0 auto;
    max-width:840px;
    margin:0 auto;
    padding:56px 28px 0;
  }
  .kicker-row{
    display:flex;
    justify-content:space-between;
    align-items:baseline;
    font-family:var(--font-mono);
    font-size:11px;
    letter-spacing:.14em;
    text-transform:uppercase;
    color:var(--ink-soft);
    padding-bottom:14px;
    border-bottom:1px solid var(--paper-line);
    flex-wrap:wrap;
    gap:8px;
  }
  .kicker-row .status-dot{color:var(--wire);}
  .kicker-nav{display:flex;gap:18px;}
  .kicker-nav a{text-decoration:none;border-bottom:1px solid transparent;transition:border-color .15s;}
  .kicker-nav a:hover{border-color:var(--ink-soft);}

  .masthead{
    padding:44px 0 36px;
    opacity:0;
    animation:rise .7s ease forwards;
  }
  .masthead .form-tag{
    font-family:var(--font-mono);
    font-size:12px;
    letter-spacing:.12em;
    color:var(--ink-soft);
    text-transform:uppercase;
    margin-bottom:18px;
  }
  h1.title{
    font-family:var(--font-display);
    font-weight:600;
    font-size:clamp(2.6rem,7vw,4.4rem);
    line-height:1.02;
    letter-spacing:-.01em;
    margin-bottom:22px;
  }
  h1.title em{
    font-style:italic;
    font-weight:480;
    color:var(--ink-soft);
  }
  .dek{
    font-size:1.08rem;
    line-height:1.6;
    color:var(--ink-soft);
    max-width:52ch;
    margin-bottom:34px;
  }
  .filing-meta{
    display:grid;
    grid-template-columns:auto 1fr;
    gap:9px 22px;
    font-family:var(--font-mono);
    font-size:12.5px;
    padding:20px 0;
    border-top:1px solid var(--paper-line);
    border-bottom:1px solid var(--paper-line);
    margin-bottom:32px;
  }
  .filing-meta dt{color:var(--ink-faint);text-transform:uppercase;letter-spacing:.08em;}
  .filing-meta dd{color:var(--ink);}
  .filing-meta .dot{color:var(--wire);}

  .cta-row{display:flex;gap:14px;flex-wrap:wrap;animation-delay:.15s;}
  .btn{
    font-family:var(--font-mono);
    font-size:13px;
    letter-spacing:.03em;
    padding:13px 22px;
    border-radius:3px;
    text-decoration:none;
    display:inline-flex;
    align-items:center;
    gap:8px;
    cursor:pointer;
    transition:transform .15s ease, box-shadow .15s ease;
    border:1px solid transparent;
  }
  .btn:hover{transform:translateY(-1px);}
  .btn-primary{background:var(--ink);color:var(--paper);}
  .btn-primary:hover{box-shadow:0 6px 18px rgba(22,24,29,.25);}
  .btn-ghost{border-color:var(--ink-soft);color:var(--ink);background:transparent;}
  .btn-ghost:hover{background:var(--paper-panel);}

  /* ============ TOC / Items ============ */
  .toc{padding:8px 0 8px;}
  .part-label{
    font-family:var(--font-mono);
    font-size:11px;
    letter-spacing:.16em;
    text-transform:uppercase;
    color:var(--ink-faint);
    padding:26px 0 10px;
  }
  details.item{
    border-top:1px solid var(--paper-line);
  }
  details.item:last-of-type{border-bottom:1px solid var(--paper-line);}
  details.item summary{
    list-style:none;
    display:grid;
    grid-template-columns:76px 1fr 20px;
    align-items:baseline;
    gap:18px;
    padding:18px 4px;
    cursor:pointer;
  }
  details.item summary::-webkit-details-marker{display:none;}
  .item-num{
    font-family:var(--font-mono);
    font-size:13px;
    color:var(--ink-faint);
    padding-top:2px;
  }
  .item-title{
    font-family:var(--font-display);
    font-size:1.18rem;
    font-weight:500;
  }
  .item-chevron{
    font-family:var(--font-mono);
    color:var(--ink-faint);
    transition:transform .2s ease;
    padding-top:2px;
  }
  details.item[open] .item-chevron{transform:rotate(45deg);}
  .item-body{
    padding:0 4px 24px 94px;
    max-width:56ch;
    color:var(--ink-soft);
    font-size:.98rem;
    line-height:1.65;
  }
  .item-body .translated{
    display:block;
    margin-top:10px;
    font-family:var(--font-mono);
    font-size:12px;
    color:var(--ink-faint);
    border-left:2px solid var(--paper-line);
    padding-left:10px;
  }
  .item-body .metric-row{
    display:flex;
    gap:22px;
    flex-wrap:wrap;
    margin-top:14px;
    font-family:var(--font-mono);
  }
  .metric{}
  .metric .val{font-size:1.3rem;color:var(--ink);display:block;}
  .metric .lbl{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-faint);}

  .paper-foot-space{height:52px;}

  /* ============ BARRIER (signature) ============ */
  .barrier{
    position:relative;
    background:
      repeating-linear-gradient(
        135deg,
        rgba(192,57,43,.13) 0 14px,
        rgba(192,57,43,0) 14px 28px
      ),
      #0f1113;
    padding:46px 28px;
    text-align:center;
    overflow:hidden;
  }
  .barrier-inner{max-width:640px;margin:0 auto;}
  .barrier-label{
    font-family:var(--font-mono);
    font-size:11px;
    letter-spacing:.22em;
    text-transform:uppercase;
    color:var(--barrier);
    margin-bottom:12px;
  }
  .barrier h2{
    font-family:var(--font-display);
    font-style:italic;
    font-weight:480;
    color:#efe9da;
    font-size:1.5rem;
    margin-bottom:12px;
  }
  .barrier p{
    font-family:var(--font-mono);
    font-size:12.5px;
    line-height:1.7;
    color:rgba(239,233,218,.6);
  }

  /* ============ TERMINAL ============ */
  .terminal{
    background:var(--terminal);
    color:var(--terminal-text);
    padding:0 0 70px;
  }
  .ticker-wrap{
    border-bottom:1px solid var(--terminal-line);
    overflow:hidden;
    white-space:nowrap;
    padding:11px 0;
  }
  .ticker-track{
    display:inline-block;
    font-family:var(--font-mono);
    font-size:12px;
    letter-spacing:.03em;
    color:var(--amber);
    animation:scroll-left 34s linear infinite;
  }
  .ticker-track span{padding:0 26px;color:rgba(245,166,35,.85);}
  .ticker-track span b{color:var(--terminal-text);font-weight:500;}

  .term-inner{max-width:840px;margin:0 auto;padding:0 28px;}
  .term-head{padding:40px 0 26px;}
  .term-kicker{
    font-family:var(--font-mono);
    font-size:12px;
    letter-spacing:.1em;
    color:var(--amber);
    margin-bottom:10px;
  }
  .term-head h2{
    font-family:var(--font-display);
    font-weight:500;
    font-size:1.9rem;
    margin-bottom:8px;
  }
  .term-head p{color:rgba(232,236,232,.5);font-size:.92rem;max-width:56ch;}

  .term-panel{
    background:var(--terminal-panel);
    border:1px solid var(--terminal-line);
    border-radius:6px;
    overflow:hidden;
  }
  .term-titlebar{
    display:flex;align-items:center;gap:8px;
    padding:11px 16px;
    border-bottom:1px solid var(--terminal-line);
    font-family:var(--font-mono);
    font-size:11px;
    color:rgba(232,236,232,.4);
  }
  .term-titlebar .dots{display:flex;gap:6px;margin-right:6px;}
  .term-titlebar .dots i{width:9px;height:9px;border-radius:50%;background:rgba(232,236,232,.15);display:block;}
  .term-body{padding:22px 22px 26px;}

  .term-step-label{
    font-family:var(--font-mono);
    font-size:11px;
    letter-spacing:.08em;
    text-transform:uppercase;
    color:rgba(232,236,232,.4);
    margin-bottom:12px;
  }
  .role-grid{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:8px;
    margin-bottom:8px;
  }
  .role-btn{
    background:transparent;
    border:1px solid var(--terminal-line);
    border-radius:4px;
    padding:12px 12px;
    text-align:left;
    color:var(--terminal-text);
    cursor:pointer;
    font-family:var(--font-mono);
    transition:border-color .15s, background .15s;
  }
  .role-btn:hover{border-color:rgba(245,166,35,.5);}
  .role-btn.active{border-color:var(--amber);background:rgba(245,166,35,.07);}
  .role-btn .fkey{font-size:10px;color:rgba(232,236,232,.35);}
  .role-btn.active .fkey{color:var(--amber);}
  .role-btn .rname{font-size:13px;margin:5px 0 3px;font-weight:500;}
  .role-btn .raccess{font-size:10.5px;color:rgba(232,236,232,.4);line-height:1.4;}

  .status-line{
    font-family:var(--font-mono);
    font-size:12px;
    padding:12px 2px 20px;
    color:rgba(232,236,232,.4);
  }
  .status-line.ok{color:var(--wire);}
  .status-line.err{color:var(--barrier);}
  .status-line .cursor{display:inline-block;width:7px;height:13px;background:currentColor;margin-left:2px;vertical-align:middle;animation:blink 1s step-end infinite;}

  .query-examples{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;}
  .qx{
    font-family:var(--font-mono);
    font-size:11.5px;
    color:rgba(232,236,232,.55);
    background:transparent;
    border:1px solid var(--terminal-line);
    border-radius:20px;
    padding:6px 13px;
    cursor:pointer;
    transition:border-color .15s,color .15s;
  }
  .qx:hover{border-color:var(--amber);color:var(--amber);}

  .query-row{display:flex;gap:10px;align-items:stretch;}
  .term-prompt{
    font-family:var(--font-mono);
    color:var(--amber);
    display:flex;align-items:center;padding-left:2px;font-size:15px;
  }
  .term-input{
    flex:1;
    background:transparent;
    border:1px solid var(--terminal-line);
    border-radius:4px;
    padding:13px 14px;
    color:var(--terminal-text);
    font-family:var(--font-mono);
    font-size:13.5px;
    outline:none;
  }
  .term-input:focus{border-color:var(--amber);}
  .term-input::placeholder{color:rgba(232,236,232,.28);}
  .term-go{
    background:var(--amber);
    color:var(--terminal);
    border:none;
    border-radius:4px;
    padding:0 22px;
    font-family:var(--font-mono);
    font-weight:600;
    font-size:12.5px;
    letter-spacing:.05em;
    cursor:pointer;
  }
  .term-go:disabled{opacity:.35;cursor:not-allowed;}
  .term-go:not(:disabled):hover{filter:brightness(1.08);}

  .resp{display:none;margin-top:22px;border-top:1px solid var(--terminal-line);padding-top:18px;}
  .resp.visible{display:block;}
  .resp-answer{
    font-family:var(--font-mono);
    font-size:13px;
    line-height:1.75;
    white-space:pre-wrap;
    color:var(--terminal-text);
  }
  .flags{display:flex;flex-wrap:wrap;gap:6px;margin-top:16px;}
  .flag{
    font-family:var(--font-mono);
    font-size:10.5px;
    padding:4px 9px;
    border-radius:3px;
    background:rgba(245,166,35,.1);
    color:var(--amber);
    border:1px solid rgba(245,166,35,.25);
  }
  .src-toggle{
    background:none;border:none;color:rgba(232,236,232,.4);
    font-family:var(--font-mono);font-size:11.5px;cursor:pointer;padding:10px 0 0;
  }
  .src-toggle:hover{color:var(--terminal-text);}
  .src-list{display:none;margin-top:10px;}
  .src-list.open{display:block;}
  .src-item{
    font-family:var(--font-mono);font-size:11.5px;
    border-left:2px solid var(--terminal-line);
    padding:8px 0 8px 12px;margin-bottom:8px;
    color:rgba(232,236,232,.55);
  }
  .src-item .tag{color:var(--amber);margin-right:8px;}
  .src-item .snippet{display:block;margin-top:4px;color:rgba(232,236,232,.35);}

  .term-foot{
    max-width:840px;margin:50px auto 0;padding:22px 28px 0;
    border-top:1px solid var(--terminal-line);
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;
  }
  .term-foot p{font-family:var(--font-mono);font-size:11px;color:rgba(232,236,232,.35);}
  .term-foot .links{display:flex;gap:16px;}
  .term-foot a{font-family:var(--font-mono);font-size:11px;text-decoration:none;color:rgba(232,236,232,.5);}
  .term-foot a:hover{color:var(--amber);}

  @keyframes rise{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
  @keyframes blink{50%{opacity:0;}}
  @keyframes scroll-left{from{transform:translateX(0);}to{transform:translateX(-50%);}}

  @media (max-width:680px){
    .filing-meta{grid-template-columns:1fr;gap:2px 0;}
    .filing-meta dt{margin-top:8px;}
    .role-grid{grid-template-columns:1fr 1fr;}
    details.item summary{grid-template-columns:44px 1fr 16px;}
    .item-body{padding-left:60px;}
    .query-row{flex-direction:column;}
    .term-prompt{display:none;}
  }
  @media (prefers-reduced-motion:reduce){
    .ticker-track{animation:none;}
    .masthead{animation:none;opacity:1;}
    .status-line .cursor{animation:none;}
    html{scroll-behavior:auto;}
  }
</style>
</head>
<body>

  <!-- ============ PAPER: filing cover + TOC ============ -->
  <div class="paper">
    <div class="kicker-row">
      <span><span class="status-dot">●</span> LIVE — RAG ENTERPRISE</span>
      <nav class="kicker-nav">
        <a href="#terminal">Terminal</a>
        <a href="/docs">API Docs</a>
        <a href="https://github.com/Murali-Sai/Rag-enterprise" target="_blank">GitHub</a>
      </nav>
    </div>

    <div class="masthead">
      <div class="form-tag">Form 10-K(RAG) · Filed with SEC EDGAR data</div>
      <h1 class="title">RAG Enterprise<br><em>an annual report on retrieval</em></h1>
      <p class="dek">A retrieval-augmented generation system that reads real SEC 10-K filings the way a compliance desk would — role-gated, disclaiming, and logged. Not a demo with synthetic PDFs: this indexes actual Apple, JPMorgan, Tesla, Microsoft, and Goldman Sachs filings.</p>

      <dl class="filing-meta">
        <dt>Filed by</dt><dd>Murali Sai</dd>
        <dt>Indexes</dt><dd>AAPL · JPM · TSLA · MSFT · GS (10-K, most recent)</dd>
        <dt>Evaluated</dt><dd>RAGAS — Faithfulness 0.65, Context Recall 0.70</dd>
        <dt>Status</dt><dd><span class="dot">●</span> Live — redeploys rebuild the index</dd>
      </dl>

      <div class="cta-row">
        <a href="#terminal" class="btn btn-primary">Enter the terminal →</a>
        <a href="/docs" class="btn btn-ghost">Read the API docs</a>
        <a href="https://github.com/Murali-Sai/Rag-enterprise" target="_blank" class="btn btn-ghost">Source on GitHub</a>
      </div>
    </div>

    <div class="toc">
      <div class="part-label">Part I — Overview</div>

      <details class="item" open>
        <summary>
          <span class="item-num">Item 1</span>
          <span class="item-title">Business</span>
          <span class="item-chevron">+</span>
        </summary>
        <div class="item-body">
          Downloads, parses, and indexes real 10-K filings straight from EDGAR — not synthetic demo text. Ask about revenue, margins, or risk factors and get an answer grounded in the actual filing, with the exact section it came from.
        </div>
      </details>

      <details class="item">
        <summary>
          <span class="item-num">Item 1A</span>
          <span class="item-title">Risk Factors</span>
          <span class="item-chevron">+</span>
        </summary>
        <div class="item-body">
          Every response passes through guardrails before it reaches you: MNPI detection, investment-advice blocking, forward-looking-statement flags, PII redaction. Violations are logged, not just hidden.
        </div>
      </details>

      <details class="item">
        <summary>
          <span class="item-num">Item 2</span>
          <span class="item-title">Properties</span>
          <span class="item-chevron">+</span>
        </summary>
        <div class="item-body">
          FastAPI, ChromaDB, LangChain, local embeddings. Deployed as a single Cloud Run container with the filing index baked in at build time — no ingestion step on cold start.
        </div>
      </details>

      <div class="part-label">Part II — Results &amp; Disclosures</div>

      <details class="item">
        <summary>
          <span class="item-num">Item 7</span>
          <span class="item-title">Management's Discussion &amp; Analysis</span>
          <span class="item-chevron">+</span>
        </summary>
        <div class="item-body">
          Measured with RAGAS across 20 filing-grounded questions, judged by GPT-4o-mini.
          <div class="metric-row">
            <div class="metric"><span class="val">0.65</span><span class="lbl">Faithfulness</span></div>
            <div class="metric"><span class="val">0.68</span><span class="lbl">Relevancy</span></div>
            <div class="metric"><span class="val">0.70</span><span class="lbl">Recall</span></div>
          </div>
        </div>
      </details>

      <details class="item">
        <summary>
          <span class="item-num">Item 7A</span>
          <span class="item-title">Disclosures About Access Risk</span>
          <span class="item-chevron">+</span>
        </summary>
        <div class="item-body">
          Who can see what is enforced as a ChromaDB filter, not a UI toggle — a research role never receives a trading-desk document, even in the raw retrieval results.
          <span class="translated">Real filings call this section "Market Risk." We're disclosing information-access risk instead — same idea, different exposure.</span>
        </div>
      </details>

      <details class="item">
        <summary>
          <span class="item-num">Item 8</span>
          <span class="item-title">Filings Currently Indexed</span>
          <span class="item-chevron">+</span>
        </summary>
        <div class="item-body">
          AAPL · JPM · TSLA · MSFT · GS — each company's latest 10-K, parsed into Item 1 (Business), 1A (Risk Factors), 7 (MD&amp;A), 7A (Market Risk), and 8 (Financials).
        </div>
      </details>
    </div>
    <div class="paper-foot-space"></div>
  </div>

  <!-- ============ BARRIER (signature) ============ -->
  <div class="barrier">
    <div class="barrier-inner">
      <div class="barrier-label">⟡ Information Barrier</div>
      <h2>Everything above this line, everyone sees.</h2>
      <p>Below it is the query path a research analyst or a trader would actually hit — gated by whichever role you pick next. That's the whole point of a Chinese Wall: what you're shown depends on who you are, enforced before the answer is generated, not after.</p>
    </div>
  </div>

  <!-- ============ TERMINAL ============ -->
  <div class="terminal" id="terminal">
    <div class="ticker-wrap">
      <div class="ticker-track" id="tickerTrack"></div>
    </div>

    <div class="term-inner">
      <div class="term-head">
        <div class="term-kicker">&gt; try it live</div>
        <h2>Query the filing index</h2>
        <p>Pick a demo account, ask a question, watch the access barrier apply in real time.</p>
      </div>

      <div class="term-panel">
        <div class="term-titlebar">
          <div class="dots"><i></i><i></i><i></i></div>
          sec-edgar-rag — role: none
        </div>
        <div class="term-body">
          <div class="term-step-label">01 — select role</div>
          <div class="role-grid">
            <button class="role-btn" data-fkey="F1" onclick="selectRole(this,'research_analyst','research1!')">
              <div class="fkey">F1</div>
              <div class="rname">Research Analyst</div>
              <div class="raccess">SEC filings — no trading, no compliance</div>
            </button>
            <button class="role-btn" data-fkey="F2" onclick="selectRole(this,'trader_desk','trade1234!')">
              <div class="fkey">F2</div>
              <div class="rname">Trading Desk</div>
              <div class="raccess">Trading, risk, SEC filings</div>
            </button>
            <button class="role-btn" data-fkey="F3" onclick="selectRole(this,'compliance_officer','compl1234!')">
              <div class="fkey">F3</div>
              <div class="rname">Compliance Officer</div>
              <div class="raccess">Compliance, SEC, risk</div>
            </button>
            <button class="role-btn" data-fkey="F4" onclick="selectRole(this,'admin_user','admin1234!')">
              <div class="fkey">F4</div>
              <div class="rname">Admin</div>
              <div class="raccess">All departments, no barriers</div>
            </button>
          </div>
          <div class="status-line" id="statusLine">&gt; status: not authenticated<span class="cursor"></span></div>

          <div class="term-step-label">02 — ask a question</div>
          <div class="query-examples">
            <button class="qx" onclick="setQuestion(this)">What was Apple's total net revenue in fiscal year 2024?</button>
            <button class="qx" onclick="setQuestion(this)">What are Tesla's key risk factors?</button>
            <button class="qx" onclick="setQuestion(this)">Compare JPMorgan and Goldman Sachs credit risk</button>
            <button class="qx" onclick="setQuestion(this)">What is Microsoft's cloud revenue growth?</button>
          </div>
          <div class="query-row">
            <div class="term-prompt">&gt;</div>
            <input type="text" class="term-input" id="questionInput" placeholder="query the filing index..." onkeydown="if(event.key==='Enter') sendQuery()">
            <button class="term-go" id="sendBtn" onclick="sendQuery()" disabled>RUN</button>
          </div>

          <div class="resp" id="responseArea">
            <div class="resp-answer" id="responseAnswer"></div>
            <div id="responseMeta"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="term-foot">
      <p>Personal portfolio project — not investment advice. Guardrail flags and disclaimers are real, generated live.</p>
      <div class="links"><a href="/redoc">ReDoc</a><a href="/docs">Swagger</a><a href="https://github.com/Murali-Sai/Rag-enterprise" target="_blank">GitHub</a></div>
    </div>
  </div>

<script>
  // Decorative ticker — indexed filings, not live market data.
  const tickerItems = [
    'AAPL <b>10-K FY2024</b>', 'JPM <b>10-K FY2024</b>', 'TSLA <b>10-K FY2024</b>',
    'MSFT <b>10-K FY2024</b>', 'GS <b>10-K FY2024</b>', 'ITEM 1 BUSINESS',
    'ITEM 1A RISK FACTORS', 'ITEM 7 MD&A', 'ITEM 7A MARKET RISK', 'ITEM 8 FINANCIALS'
  ];
  const track = document.getElementById('tickerTrack');
  const row = tickerItems.map(t => `<span>${t}</span>`).join('');
  track.innerHTML = row + row; // duplicated for seamless loop

  let token = null;

  async function selectRole(el, username, password) {
    document.querySelectorAll('.role-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');

    const status = document.getElementById('statusLine');
    status.className = 'status-line';
    status.innerHTML = '&gt; authenticating as ' + username + '...<span class="cursor"></span>';

    try {
      const res = await fetch('/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      if (!res.ok) throw new Error('Login failed');
      const data = await res.json();
      token = data.access_token;
      status.className = 'status-line ok';
      status.innerHTML = '&gt; authenticated as ' + username + ' — clearance: ' + el.querySelector('.raccess').textContent + '<span class="cursor"></span>';
      document.getElementById('sendBtn').disabled = false;
      document.querySelector('.term-titlebar').lastChild.textContent = ' sec-edgar-rag — role: ' + username;
    } catch (e) {
      token = null;
      status.className = 'status-line err';
      status.innerHTML = '&gt; authentication failed — try again<span class="cursor"></span>';
      document.getElementById('sendBtn').disabled = true;
    }
  }

  function setQuestion(el) {
    document.getElementById('questionInput').value = el.textContent;
    document.getElementById('questionInput').focus();
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
    btn.textContent = '...';
    area.classList.add('visible');
    answer.textContent = '> querying filing index...';
    meta.innerHTML = '';

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token },
        body: JSON.stringify({ question })
      });

      if (res.status === 401) {
        answer.textContent = '> session expired — select a role again';
        token = null;
        return;
      }

      const data = await res.json();
      if (!res.ok) {
        answer.textContent = '> error: ' + (data.message || data.detail || 'query failed');
        return;
      }

      answer.textContent = data.answer;

      let metaHtml = '';
      if (data.guardrail_flags && data.guardrail_flags.length > 0) {
        metaHtml += '<div class="flags">';
        data.guardrail_flags.forEach(f => { metaHtml += '<span class="flag">' + f + '</span>'; });
        metaHtml += '</div>';
      }
      if (data.sources && data.sources.length > 0) {
        const id = 'src-' + Date.now();
        metaHtml += '<button class="src-toggle" onclick="toggleSources(\\''+id+'\\')">▸ ' + data.sources.length + ' source(s)</button>';
        metaHtml += '<div class="src-list" id="'+id+'">';
        data.sources.forEach(s => {
          metaHtml += '<div class="src-item">';
          if (s.ticker) metaHtml += '<span class="tag">' + s.ticker + '</span>';
          if (s.section_name) metaHtml += s.section_name;
          if (s.filing_type) metaHtml += ' · ' + s.filing_type;
          metaHtml += '<span class="snippet">' + (s.content || '').substring(0, 180) + '…</span></div>';
        });
        metaHtml += '</div>';
      }
      meta.innerHTML = metaHtml;
    } catch (e) {
      answer.textContent = '> network error — is the server running?';
    } finally {
      btn.disabled = false;
      btn.textContent = 'RUN';
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
