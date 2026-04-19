# RAG Enterprise — SEC EDGAR Filing Analyzer

> **Live Demo**: [web-production-337e5.up.railway.app](https://web-production-337e5.up.railway.app/docs) &nbsp;|&nbsp; Try it: `POST /api/auth/login` then `POST /api/query`

Production-grade Retrieval Augmented Generation system that queries **real SEC 10-K filings** from the EDGAR API. Features role-based access control with information barriers (Chinese Walls), financial compliance guardrails, MNPI detection, regulatory audit trails, and RAGAS evaluation — built for investment banking workflows at firms like **JPMC, Morgan Stanley, and Goldman Sachs**.

Unlike typical RAG demos with synthetic documents, this system downloads, parses, and indexes **actual annual reports** from Apple, JPMorgan, Tesla, Microsoft, and Goldman Sachs.

## Architecture

```
                          ┌─────────────────────────────────────────────────┐
                          │              SEC EDGAR API                      │
                          │   (Real 10-K filings for AAPL, JPM, TSLA,      │
                          │    MSFT, GS — downloaded & parsed)              │
                          └────────────────────┬────────────────────────────┘
                                               │
                               Download & Parse (BeautifulSoup + regex)
                               Extract: Item 1, 1A, 7, 7A, 8
                                               │
                                               ▼
[User + JWT] → FastAPI → Input Guardrails → RBAC Retriever → LLM → Financial Compliance → Audit Trail → Response
                              │                   │                       │
                          Auth (SQLite)      ChromaDB                MNPI Detection
                                             (metadata              Investment Advice Check
                                              filtering)            Disclaimer Injection
```

### What Makes This Different

1. **Real SEC Filings**: Downloads actual 10-K annual reports from the SEC EDGAR API — not synthetic documents. Queries return real revenue figures, risk disclosures, and financial data.

2. **10-K Section Parsing**: Custom HTML parser extracts Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A), Item 7A (Market Risk), and Item 8 (Financial Statements) from raw EDGAR HTML, handling formatting variations across filers.

3. **Cross-Company Analysis**: "Compare JPMorgan and Goldman Sachs credit risk disclosures" retrieves from multiple filings and synthesizes a comparison with section citations.

4. **Information Barriers (Chinese Walls)**: Research analysts cannot access trading or compliance data — enforced at the vector store layer via ChromaDB `where` clauses (SEC Rule 15g-1, FINRA Rule 2241).

5. **Financial Compliance Guardrails**: Automatic detection of investment advice language, MNPI leakage, and forward-looking statements. Prohibited patterns are blocked and logged.

6. **Regulatory Audit Trail**: Every query, document access, and RBAC decision is logged to append-only JSONL (SEC Rule 17a-4, FINRA 4511).

7. **RAGAS Evaluation**: 20 filing-grounded questions with ground truth, measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

## Demo Companies

| Ticker | Company | Why |
|---|---|---|
| AAPL | Apple Inc. | Clean filings, well-known financial metrics |
| JPM | JPMorgan Chase | Target employer — demonstrates domain knowledge |
| TSLA | Tesla Inc. | Complex risk factors, high-profile filings |
| MSFT | Microsoft Corp. | AI/cloud narrative, enables AAPL comparison |
| GS | Goldman Sachs | Second IB — enables JPM vs GS comparison |

Each company's most recent 10-K is downloaded, parsed into 5-6 sections, and chunked into ~500-1000 total vectors.

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain + LCEL |
| Data Source | SEC EDGAR API (real 10-K filings) |
| Filing Parser | BeautifulSoup + regex section extraction |
| LLM (Primary) | Groq (llama-3.3-70b) — Free tier |
| LLM (Fallback) | Google Gemini 2.5 Flash / OpenAI gpt-4o-mini |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, free) |
| Vector Store | ChromaDB (prebuilt index committed for zero-build deploys) |
| API | FastAPI with async lifespan |
| Auth | JWT + SQLAlchemy + bcrypt |
| Financial Guardrails | MNPI detection, investment advice blocking, disclaimer injection |
| Audit Trail | Append-only JSONL (SEC 17a-4 / FINRA 4511) |
| Evaluation | RAGAS (Faithfulness, Relevancy, Precision, Recall) |
| Deployment | Railway (Nixpacks) — live demo |
| Infrastructure | Terraform + AWS ECS Fargate (reference architecture) |
| CI/CD | GitHub Actions |

## Quick Start

### Prerequisites

- Python 3.11+
- Free API key from [Groq](https://console.groq.com/) or [Google AI Studio](https://aistudio.google.com/)

### Setup

```bash
git clone https://github.com/Murali-Sai/Rag-enterprise.git
cd rag-enterprise
pip install -e ".[dev,eval]"

cp .env.example .env
# Edit .env — add your GROQ_API_KEY and optionally EDGAR_USER_AGENT
```

### Download Real SEC Filings & Run

```bash
make demo    # Seeds users + downloads 10-K filings from EDGAR + ingests into ChromaDB
make dev     # Starts FastAPI server at http://localhost:8000
```

Or run each step individually:

```bash
make seed               # Create demo users
make download-filings   # Download 10-K filings from SEC EDGAR API
make ingest-edgar       # Parse filings and ingest into ChromaDB
make dev                # Start the server
```

### Demo: Query Real SEC Filings

```bash
# 1. Login as a research analyst
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "research_analyst", "password": "research1!"}' | jq -r .access_token)

# 2. Ask about Apple's revenue (real data from 10-K Item 7 MD&A)
curl -s -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple total net revenue for fiscal year 2024?"}'
# -> Returns actual revenue figure from Apple's 10-K filing

# 3. Cross-company comparison
curl -s -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare JPMorgan and Goldman Sachs credit risk disclosures"}'
# -> Synthesizes from JPM and GS 10-K Item 1A Risk Factors
```

### Demo: Information Barrier in Action

```bash
# Research analyst CANNOT access trading desk procedures (Chinese Wall)
curl -s -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the daily P&L stop-loss limits?"}'
# -> "No relevant documents found for your query within your access level."

# Trader CAN access trading docs
TRADE_TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "trader_desk", "password": "trade1234!"}' | jq -r .access_token)

curl -s -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TRADE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the daily P&L stop-loss limits?"}'
# -> Returns trading desk procedures
```

### Demo Users

| Username | Password | Role | Access |
|---|---|---|---|
| admin_user | admin1234! | admin | All departments |
| trader_desk | trade1234! | trading | Trading, Risk, SEC |
| risk_analyst | risk12345! | risk | Risk, Trading, SEC, Compliance |
| compliance_officer | compl1234! | compliance | Compliance, SEC, Risk |
| research_analyst | research1! | research | Research, SEC **(Chinese Wall)** |
| wealth_advisor | wealth123! | wealth_management | Research, SEC |
| ops_manager | ops1234567! | operations | Trading, Compliance |
| external_auditor | audit1234! | auditor | Compliance, SEC |
| viewer_user | viewer123! | viewer | SEC Filings only |

## SEC EDGAR Integration

### How It Works

1. **Download**: `EdgarClient` fetches 10-K filings from `efts.sec.gov` using CIK numbers, respecting the SEC's 10 req/sec rate limit
2. **Parse**: `parse_10k_sections()` uses regex to find section boundaries (handles `Item 7`, `ITEM 7.`, `<b>Item 7` variants) and BeautifulSoup for HTML-to-text conversion with table preservation
3. **Load**: Each section becomes a LangChain `Document` with metadata (ticker, filing_date, section_id, section_name, department)
4. **Chunk**: Standard recursive text splitting with section-aware metadata propagation
5. **Index**: ChromaDB with metadata filtering — RBAC and section queries at the database level

### Parsed Sections

| Section | Content | Use Case |
|---|---|---|
| Item 1 — Business | Company overview, segments, products | Business analysis |
| Item 1A — Risk Factors | Risk disclosures, regulatory risks | Risk assessment |
| Item 7 — MD&A | Revenue, margins, segment performance | Financial analysis |
| Item 7A — Market Risk | Interest rate, FX, derivative exposures | Market risk |
| Item 8 — Financial Statements | Balance sheet, income statement, cash flow | Fundamentals |

## RBAC Model — Investment Bank Structure

```
                                admin (CRO / CEO)
                                |  Full access
                ┌───────────────┼───────────────┐
                |               |               |
          ┌─────┴─────┐   ┌────┴────┐   ┌──────┴──────┐
          |  Trading  |   |  Risk   |   | Compliance  |
          |  Desk     |   | Mgmt    |   | & Legal     |
          └─────┬─────┘   └────┬────┘   └──────┬──────┘
                |               |               |
          ══════╪═══════════════╪═══════════════╪══════════  <- Chinese Wall
                |               |               |
          ┌─────┴─────┐   ┌────┴────┐   ┌──────┴──────┐
          | Research  |   | Wealth  |   | Operations  |
          | (Blocked) |   | Mgmt    |   | (Back Off.) |
          └───────────┘   └─────────┘   └─────────────┘
```

## Guardrails

| Layer | Purpose | Financial Relevance |
|---|---|---|
| Input Validation | Block SQL/XSS, enforce length | Standard security |
| Prompt Injection | Detect "ignore instructions" attacks | Prevent social engineering |
| **MNPI Detection** | Flag potential insider information leakage | SEC Rule 10b-5 |
| **Investment Advice** | Detect buy/sell recommendations | SEC/FINRA suitability rules |
| **Forward-Looking** | Flag projections and forecasts | SEC safe harbor requirements |
| PII Redaction | Redact SSN, credit cards | GLBA / GDPR compliance |
| Output Safety | Flag hallucinations, unsafe content | Accuracy in financial context |

## Compliance Audit Trail

Every query generates an audit log entry:

```json
{
  "event_type": "rag_query",
  "timestamp": "2025-01-15T14:30:22.123Z",
  "user_id": 3,
  "username": "research_analyst",
  "user_roles": ["research"],
  "query": "What was Apple's total revenue in fiscal year 2024?",
  "retrieved_departments": ["sec_filings"],
  "documents_accessed": 5,
  "guardrail_flags": ["forward_looking_statement"],
  "information_barriers_applied": ["Research-Trading Wall", "Research-Compliance Wall"],
  "response_length": 342
}
```

## Project Structure

```
rag-enterprise/
├── src/
│   ├── main.py                          # FastAPI app
│   ├── config.py                        # Settings (inc. EDGAR config)
│   ├── api/
│   │   ├── routes/                      # auth, query, documents, health, admin
│   │   ├── audit.py                     # Compliance audit trail (SEC 17a-4)
│   │   └── middleware.py                # Rate limiting, request logging
│   ├── auth/
│   │   ├── rbac.py                      # Information barriers + role access map
│   │   ├── jwt_handler.py               # JWT creation/validation
│   │   └── repository.py               # User/Role CRUD
│   ├── edgar/                           # << NEW: SEC EDGAR integration
│   │   ├── client.py                    # Async EDGAR API client with rate limiting
│   │   ├── parser.py                    # 10-K HTML section extractor
│   │   └── loader.py                    # LangChain Document adapter
│   ├── ingestion/                       # Chunking, metadata, EDGAR pipeline
│   ├── retrieval/
│   │   ├── vector_store.py              # ChromaDB / OpenSearch abstraction
│   │   └── retriever.py                 # RBAC-filtered retriever
│   ├── generation/                      # LLM factory, RAG chain, finance prompts
│   ├── guardrails/                      # MNPI, prompt injection, PII, compliance
│   └── common/                          # Logging, exceptions, schemas
├── scripts/
│   ├── download_filings.py              # Download 10-K filings from EDGAR API
│   ├── ingest_edgar.py                  # Parse & ingest filings into ChromaDB
│   ├── seed_users.py                    # Create demo IB users
│   └── ingest_samples.py               # Ingest sample domain documents
├── tests/                               # Unit + integration tests
├── evaluation/                          # RAGAS pipeline + 20 filing-grounded Q&A
├── data/edgar/                          # Downloaded 10-K filings (committed for deploy)
├── data/sample/                         # Domain documents (risk, compliance, etc.)
├── infra/terraform/                     # AWS ECS + OpenSearch IaC
└── .github/workflows/                   # CI/CD pipelines
```

## Evaluation (RAGAS)

20 filing-grounded questions across all 5 companies. Judge model: OpenAI `gpt-4o-mini`. Generation: same (swappable to Groq / Gemini via `LLM_PROVIDER`).

### Results

| Metric | Baseline (`top_k=5`) | Tuned (`top_k=10`) | Δ |
|---|---|---|---|
| **Faithfulness** | 0.57 | **0.65** | +0.08 |
| **Answer Relevancy** | 0.58 | **0.68** | +0.10 |
| **Context Precision** | 0.49 | 0.49 | = |
| **Context Recall** | 0.57 | **0.70** | +0.13 |

Increasing retrieval breadth from 5 → 10 chunks improved Recall and Faithfulness substantially without hurting Precision — more relevant context was captured without diluting the signal passed to the LLM.

### Iteration Story (interview talking point)

First RAGAS run showed Context Recall of **0.08** — suspiciously low. Root cause: the shipped eval dataset had placeholder ground truths (e.g., *"Sourced from Apple's 10-K. The exact figure will be populated from the actual downloaded filing."*) rather than real answers. Fix: wrote `scripts/generate_ground_truths.py` to regenerate ground truths from the actual filings using `gpt-4o-mini` + wide retrieval (top_k=20). Recall jumped to 0.57 — now measuring the system, not the dataset. Tuning top_k to 10 then gave the scores above.

```bash
make eval                     # Runs RAGAS on eval_questions_v2.json
RETRIEVAL_TOP_K=10 make eval  # Tuned variant
```

## Testing

```bash
make test          # Run all tests
make test-cov      # Run with coverage
make lint          # Lint + type check
```

## Deployment

### Live Demo (Railway)

The app is deployed at **https://web-production-337e5.up.railway.app** via Railway with a Dockerfile builder. Try it:

```bash
# Health check
curl https://web-production-337e5.up.railway.app/health

# Login as research analyst
curl -X POST https://web-production-337e5.up.railway.app/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"research_analyst","password":"research1!"}'

# Query (use the token from login response)
curl -X POST https://web-production-337e5.up.railway.app/api/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"What was Apple total revenue in fiscal year 2024?","top_k":10}'
```

Or explore the interactive API docs: [/docs](https://web-production-337e5.up.railway.app/docs)

### Local (Docker)

```bash
docker-compose up -d --build
```

### AWS (Terraform — reference architecture)

```bash
cd infra/terraform
terraform init
terraform apply -var-file=environments/dev.tfvars
```

## Interview Talking Points

1. **"Why real SEC filings instead of synthetic data?"** — Synthetic documents are a solved tutorial problem. Real 10-K filings have inconsistent HTML formatting across filers, 200+ page documents that need section-level extraction, and actual financial figures that can be verified. This demonstrates production data handling, not just a LangChain wrapper.

2. **"How does the EDGAR parser handle different filing formats?"** — Regex pattern matching with fallback strategies for section boundary detection. The pattern handles `Item 7`, `ITEM 7.`, `<b>Item 7</b>`, and other HTML variants. BeautifulSoup converts HTML to clean text while preserving table structure for financial data.

3. **"Why information barriers at the retrieval layer?"** — UI-level filtering is insufficient; a compromised frontend could bypass it. ChromaDB `where` clauses enforce access at the database level, so unauthorized documents are never returned to the application layer.

4. **"How do you handle MNPI?"** — The financial compliance guardrail scans responses for patterns indicating non-public information. Flagged responses get MNPI warnings and are logged to the audit trail for compliance review.

5. **"How is this auditable?"** — Every query, including user identity, roles, documents accessed, guardrail flags, and information barriers applied, is written to an append-only JSONL log meeting SEC Rule 17a-4 and FINRA 4511 recordkeeping requirements.

6. **"How would you scale this?"** — Replace ChromaDB with OpenSearch Serverless (already abstracted), move auth to Cognito/Okta, add Redis caching for frequent queries, and deploy to ECS Fargate with ALB (Terraform modules included).

## License

MIT
