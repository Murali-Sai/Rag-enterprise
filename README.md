# RAG Enterprise — SEC EDGAR Filing Analyzer

> **Live Demo**: [rag-enterprise-laa65asupq-uc.a.run.app](https://rag-enterprise-laa65asupq-uc.a.run.app) &nbsp;|&nbsp; Interactive demo on the landing page, or explore the [API docs](https://rag-enterprise-laa65asupq-uc.a.run.app/docs)

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

7. **Verifiable Citations**: Answers cite numbered context blocks (`[2]`), so every claim can be paired back with the exact chunk it came from and judged against it. Prose citations read better and cannot be checked — which is why citation accuracy is a metric here rather than an aspiration.

8. **Confidence Scoring and a Structured "I don't know"**: Every answer carries a composite of retrieval relevance, citation coverage, and answer completeness. Below a retrieval-confidence threshold the system returns what it searched and which filings to read by hand, instead of an answer built on chunks it does not trust.

9. **RAGAS Evaluation**: 20 filing-grounded questions with ground truth, measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall — plus citation accuracy and coverage, computed outside RAGAS.

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
| LLM (Primary) | OpenAI `gpt-4o` |
| LLM (Fallback) | Google Gemini 2.5 Flash / Groq llama-3.3-70b — free tiers, set `LLM_PROVIDER` |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim, 8k-token window); `EMBEDDING_PROVIDER=huggingface` swaps in local all-MiniLM-L6-v2 |
| Chunking | Three switchable strategies: fixed-size (baseline), recursive/structure-aware (default), semantic (embedding-similarity breakpoints) |
| Deduplication | Cosine > 0.95 near-duplicate suppression at ingestion — 7.8% of this corpus |
| Query Transform | HyDE — optional hypothetical-document rewrite before retrieval |
| Hybrid Search | BM25 (`rank_bm25`) fused with dense retrieval via Reciprocal Rank Fusion |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 — retrieves top-20, reranks to top-5 |
| Vector Store | ChromaDB (index baked into the image at build time, so cold starts don't ingest) |
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
- An [OpenAI API key](https://platform.openai.com/api-keys) for the defaults (`gpt-4o` generation,
  `text-embedding-3-small` embeddings)
- Or run it free: set `LLM_PROVIDER=groq` (or `gemini`) with a key from
  [Groq](https://console.groq.com/) / [Google AI Studio](https://aistudio.google.com/), and
  `EMBEDDING_PROVIDER=huggingface` to embed locally. The embedding switch changes vector
  dimensionality, so re-ingest with `python scripts/ingest_edgar.py --from-disk --reset`.

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
4. **Chunk**: One of three switchable strategies, with the strategy, per-document chunk index, and character count recorded on every chunk
5. **Deduplicate**: Near-duplicates suppressed before insertion (cosine > 0.95)
6. **Index**: ChromaDB with metadata filtering — RBAC and section queries at the database level
7. **Retrieve**: Hybrid + rerank pipeline — dense embedding search and a BM25 lexical index each return candidates (both RBAC-filtered), fused by Reciprocal Rank Fusion, then rescored by a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker and cut to the top-5 sent to the LLM

### Chunking Strategies

Chunking is baked into an index at ingestion time, so these are not runtime switches — comparing them means building one collection per strategy and pointing the eval at each. Every chunk records which strategy produced it, so a collection can't be misattributed later.

| Strategy | Splits on | Why it exists |
|---|---|---|
| `fixed` | Character count only | The structure-blind baseline. Cuts through sentences, table rows, and dollar figures. Present to be beaten — it's what makes "structure awareness helps" a measured claim instead of an assumed one |
| `recursive` *(default)* | Paragraph → line → sentence → word → character | Stacks on the EDGAR parser, which has already split the filing at Item boundaries, so a chunk stays inside one section |
| `semantic` | Where consecutive sentence embeddings diverge | Cuts where the subject changes rather than where the budget runs out. Hand-implemented (~50 lines) — `langchain-experimental` requires `langchain-core` 1.x against this project's deliberate `<1.0` pin, and is sunset |

### Deduplication — 7.8% of this corpus is redundant

A duplicate chunk is not just wasted disk. Retrieval returns a fixed `top_k`, so an indexed duplicate competes for one of those slots and wins it — the chunk it displaces is by definition the next most relevant thing the model would have seen. Duplication is paid for in context the LLM never gets.

Measured on the live 9,572-chunk index with `scripts/analyze_duplicates.py` (read-only, reuses stored vectors, costs nothing):

| Cosine threshold | Chunks | Share |
|---|---|---|
| ≥ 0.90 | 1,043 | 10.90% |
| **≥ 0.95** (default) | **747** | **7.80%** |
| ≥ 0.98 | 567 | 5.92% |
| ≥ 0.999 (near-identical) | 242 | 2.53% |

Two distinct causes, and the examples make them legible. Page furniture becomes its own chunk — `"Apple Inc. | 2025 Form 10-K | 3"` against `"...| 2025 Form 10-K | 2"` at cosine 0.990, carrying no content whatsoever. And filings genuinely restate themselves: the same supply-chain risk paragraph appears twice at cosine 1.0000, because a disclosure made in Item 1A is repeated verbatim in the incorporated annual report.

The threshold is deliberately high. Two chunks discussing the same topic in different words sit near 0.90; dropping those would be a silent recall loss, which is a worse failure than the one being fixed. Every suppression records what it matched and how closely, so the choice is auditable rather than taken on trust.

Comparison is done on raw vectors rather than a vector-store distance score: Chroma's default metric is L2 and LangChain's "relevance score" is a metric-dependent rescaling of distance, so comparing that against a cosine threshold quietly means something different depending on how the collection was created.

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

## Generation, Citations, and Confidence

The quality layer most RAG systems skip. Four pieces, each of which exists because the one before it made the next measurable.

**Bracketed citations.** Context blocks are numbered `[1] (Source: …, Company: AAPL, Section: Item 1A …)` and the prompt requires every factual sentence to end with the block it came from. The previous prompt asked for prose citations — *"According to Apple's 2024 10-K, Item 7 MD&A…"* — which read better and could not be parsed, so no claim could be paired with its source and no citation metric could exist.

**Citation verification** (`src/generation/verification.py`). Each answer is split into sentence-level claims, and each `(claim, cited block)` pair goes to a judge with *only* that block. Showing the whole context would turn this into a faithfulness check — the citation would pass because the corpus supports the claim, not because the cited chunk does. Three outcomes are kept distinct rather than pooled as "unverified":

| | What it means | Where it shows up |
|---|---|---|
| Unsupported | The cited chunk does not say that | `citation_accuracy` |
| Out of range | `[7]` when 5 blocks were supplied — an invented source | `out_of_range`, and scored zero |
| Uncited | A claim naming no source at all | `citation_coverage`, not accuracy |

Off by default at runtime (one LLM call per citation, on the critical path); the eval harness forces it on, which is where the cost buys the metric.

**Confidence scoring** (`src/generation/confidence.py`) — a 0.5/0.3/0.2 composite of retrieval relevance, citation coverage, and answer completeness. Retrieval leads because a fluent, fully-cited answer over bad chunks is the failure mode that is invisible in the answer text — the measured reranker regression below is exactly that. Getting this signal out required a change underneath: `rerank_documents()` computed cross-encoder scores and then discarded them, so `src/retrieval/scores.py` now carries them through the pipeline tagged with which stage produced them. Scores that carry no relevance (RRF fuses ranks and throws scores away) report `None`, never `0.0` — an unmeasurable signal must not read as a bad one.

**Structured "I don't know"** (`src/generation/insufficient.py`). Below `INSUFFICIENT_CONTEXT_THRESHOLD` the system returns the passages it consulted and the filings worth opening manually, *before* spending a generation call. The same structure is attached when the model itself declines over good retrieval — a different failure, pointing at a corpus or chunking gap rather than a retrieval one.

All of it lives behind `generate_grounded_answer()` in `src/generation/answer.py`, which the REST API, the MCP server, and `evaluation/run_evaluation.py` all call. The eval harness bypasses the API route entirely, so anything implemented in the route would be invisible to the measurement meant to prove it works.

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

## MCP Server (Model Context Protocol)

The RAG pipeline is also exposed as an **MCP server**, so any MCP-compatible LLM client (Claude Desktop, Cursor, etc.) can invoke the tools natively — no REST calls, no copying tokens. The same RBAC department filtering, Chinese Wall information barriers, and financial-compliance guardrails used by the REST API are preserved.

### Tools exposed

| Tool | Description |
|---|---|
| `query_sec_filings(question, role)` | RAG query over real 10-K filings, RBAC-scoped to the role |
| `compare_companies(ticker_a, ticker_b, topic, role)` | Cross-company comparison (e.g. JPM vs GS credit risk) |
| `list_indexed_companies()` | List indexed companies and queryable 10-K sections |
| `describe_access(role)` | Show a role's accessible departments + active Chinese Walls |

### Run it

```bash
make mcp          # stdio transport (for Claude Desktop / Cursor)
make mcp-http     # streamable-HTTP transport on :8001 (for remote clients)
```

### Connect to Claude Desktop

Copy the `sec-edgar-rag` block from [`claude_desktop_config.example.json`](claude_desktop_config.example.json) into your Claude Desktop config (adjust the absolute paths and API key), then restart Claude Desktop. You can then ask Claude *"What was Apple's revenue in FY2024?"* and it will call `query_sec_filings` and answer from the real filing — with a `research` role walled off from trading/compliance documents.

Architecturally this turns the project into a **reusable AI capability**: the domain work (real SEC filings, RBAC, guardrails) is consumed by *any* agent via the open MCP standard, not just this app's own UI.

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
│   │   ├── retriever.py                 # RBAC-filtered retriever
│   │   └── scores.py                    # Relevance scores carried through the pipeline
│   ├── generation/
│   │   ├── answer.py                    # Entry point: API, MCP and eval all call this
│   │   ├── prompts.py                   # Finance prompts, bracketed-citation rules
│   │   ├── citations.py                 # Claim segmentation + [n] parsing
│   │   ├── verification.py              # LLM-as-judge citation checking
│   │   ├── confidence.py                # Composite confidence score
│   │   └── insufficient.py              # Structured "I don't know"
│   ├── guardrails/                      # MNPI, prompt injection, PII, compliance
│   ├── mcp_server/                      # << NEW: MCP server exposing RAG as tools
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

54 hand-written questions across all 5 companies, at `top_k=5`. Generation: OpenAI `gpt-4o`. Judge: `gpt-4o-mini` with local MiniLM judge embeddings — deliberately not the corpus embedding model, so the measuring stick doesn't move when the retriever does.

**16 of the 54 have no correct answer.** That is the point of them. A question can be unanswerable because the filer does not disclose the fact (`no_answer`), because no filing in the corpus covers the subject (`out_of_corpus`), because the asking role may not read the document that holds it (`rbac_blocked`), or because the question has no defensible single reading (`ambiguous`). Those rows are scored on `refusal_correctness` and excluded from the RAGAS and citation aggregates — RAGAS scores a correct refusal as 0.000 answer relevancy, and correctly so, since a refusal has no relevance to the question. Pooling them would lower every aggregate precisely because the system behaved correctly. The rule is written into each run's `config.no_answer_scoring`, because a run that pooled them is not comparable to one that did not and nothing in the numbers reveals which it was.

`refusal_correctness` is computed over all 54, not just the 16. On an answerable question it measures the opposite failure — declining something the corpus does answer — which turns out to be the live defect here, and which none of the RAGAS four can tell apart from a genuine miss.

**The eval exercises RBAC.** 34 of the 54 questions name a real role rather than `admin`, for which `build_role_filter()` returns `None`. Before that, the ChromaDB where-clause was never applied on any eval question and the information-barrier layer was covered by unit tests and by nothing in the eval.

Three more things about this dataset are worth stating before the numbers, because they make the scores *lower* and more honest:

**The ground truths do not come from retrieval.** The previous version answered each question from this project's own top-20 chunks, then scored that same retriever against the result. Anything retrieval systematically missed was missing from the reference too, so it could never be counted as a miss. `scripts/generate_ground_truths_from_filings.py` instead scans the complete parsed filing in overlapping windows — no retrieval in the loop. Recall dropped from 0.57 to ~0.33 when this changed. The system did not get worse; the measurement stopped grading itself.

**Questions are stratified.** Exact-figure lookups, interpretive questions, and cross-company comparatives behave differently enough that one mean over all three hides the mechanism, and the four unanswerable strata are not on the same axis at all. Every run reports per stratum. Classification is hand-assigned in `scripts/stratify_eval_questions.py`, not LLM-inferred — a classifier would put another model's judgment into a measurement chain the eval works to keep clean.

**Citation metrics are not RAGAS metrics.** `citation_accuracy`, `citation_coverage` and `confidence` are computed in `run_rag_pipeline()` and joined into the per-question rows by dataset index. RAGAS scores an answer against a ground truth; these score it against the chunks it claims to have used, which is a different question and needs its own judge — a judge that is free to change, unlike the RAGAS one, which is pinned to `gpt-4o-mini` because every result already in `evaluation/results/` was measured with it. A question that cited nothing reports `citation_accuracy: null` and is left out of that mean rather than counted as zero; its failure shows up in coverage, which is where it belongs.

### Current scores

> ⚠️ **Measured on the 9,572-chunk corpus (`e705ac4b47e9daf3`). The index on disk is now 8,232 chunks (`c2f8c13673cf5ca5`)** after page furniture was stripped from the parser — the fix for the Goldman Sachs recall failure described below. Every figure in this section, and the whole ablation table beneath it, predates that re-ingest. They are accurate records of what was measured and are not claims about the current build. Re-run `python -m evaluation.run_evaluation` for current numbers; `scripts/compare_eval_runs.py` will refuse to diff across the fingerprint change, which is the intended behaviour.

Mean of two identical runs on the 54-question set (`eval_20260808_212927`, `eval_20260808_213635`), shipped configuration: dense retrieval, cross-encoder rerank, per-entity retrieval for comparatives.

> **These two runs predate the `is_full_refusal` fix**, which narrowed `answer_completeness()` so that an answer carrying refusal wording *and* cited claims scores as partial (0.5) rather than as a full refusal (0.0). A third run on the fixed code — `eval_20260809_001003`, same config, same corpus — measures **refusal correctness 0.796 overall and 0.692 for `exact_figure`**, moving exactly two rows: JPMorgan's CET1 ratio and Tesla's net income, each of which had stated a verified, cited figure and been scored as though it had answered nothing. No row moved the other way, and the 16 `refuse` rows held at 14/16. The RAGAS and citation figures are unaffected in kind — the generated text is unchanged, only its classification — so the two-run mean below stands for everything except refusal correctness, where **0.796 is the current figure**.

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Citation Accuracy | Citation Coverage | Refusal Correctness |
|---|---|---|---|---|---|---|
| 0.737 | 0.639 | 0.419 | 0.383 | **0.945** | 0.607 | **0.796**¹ |

Per stratum, and this is where the system's actual shape shows:

| Stratum | n | Faithfulness | Answer Relevancy | Citation Accuracy | Refusal Correctness |
|---|---|---|---|---|---|
| `interpretive` | 14 | 0.897 | 0.822 | 0.968 | 0.929 |
| `exact_figure` | 13 | 0.730 | 0.498 | 0.921 | 0.692¹ |
| `comparative` | 8 | 0.493 | 0.565 | 0.905 | 0.625 |
| `ambiguous` | 6 | 0.667 | 0.582 | 0.875 | 0.500 |
| `no_answer` | 5 | — | — | — | **1.000** |
| `out_of_corpus` | 4 | — | — | — | **1.000** |
| `rbac_blocked` | 4 | — | — | — | **1.000** |

¹ Refusal correctness from `eval_20260809_001003`, the run on the current code; every other column is the two-run mean above.

**The refusal path is the strongest thing here.** All three unanswerable strata score 1.000, and identically across both runs — the system declines every question it should decline, structures the refusal, and labels it low confidence. Out-of-corpus questions are caught by the retrieval gate before any generation call is spent. RBAC-blocked questions never retrieve the document at all.

**Over-refusal is the weakest.** `exact_figure` scores 0.538 on refusal correctness — 0.692 after the `is_full_refusal` fix above. Either way the system declines a third to a half of the questions asking for a figure that is in the index. That is the reranker (below), and it is invisible in the RAGAS four — a false refusal and a genuine miss both read as 0.000 answer relevancy.

**Goldman Sachs scores 0.000 context recall on every question, in every run.** Broken down per company rather than per stratum, the aggregate hides a hard failure:

| | AAPL | MSFT | JPM | GS | TSLA |
|---|---|---|---|---|---|
| Faithfulness | 0.861 | 0.658 | 0.921 | 0.850 | 0.877 |
| Context Recall | 0.569 | 0.263 | 0.267 | **0.000** | 0.539 |
| Citation Accuracy | 0.938 | 1.000 | 0.917 | 0.917 | 1.000 |

The cause is page furniture in the index. About a fifth of every filing's chunks are under 200 characters; GS has 806 such chunks out of 3,402, of which 214 are bare running headers — `"Goldman Sachs 2025 Form 10-K | 123"`. On the GS revenue question, two of the five retrieved slots are page numbers. Deduplication cannot catch these because each header string differs by its page number; it needs a minimum-length filter in the parser. MSFT's low faithfulness is separate and expected — its filing text is only recovered at ~71%.

**Ambiguity is not handled.** Two of the three underspecified questions ("How much did the bank set aside for credit losses?", "What are the company's main risks?") were answered about one arbitrarily-chosen company, with no flag that the question admitted others. There is no ambiguity-detection mechanism; the stratum exists to say so.

### Retrieval Pipeline Ablation

All configs measured on the same 20 questions and the same corpus, one run each except the baseline. Results in `evaluation/results/` are tagged with both the config *and* a corpus fingerprint (chunk count + content digest), because a config tag alone cannot tell two runs apart when the index changed underneath them — which is exactly what happened to the previous version of this table.

**First, a per-metric noise floor.** The dense baseline was run twice, identical configuration:

| Dense-only baseline | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Run A | 0.576 | 0.458 | 0.695 | 0.308 |
| Run B | 0.714 | 0.502 | 0.696 | 0.346 |
| **Run-to-run spread (n=20)** | **0.138** | 0.044 | **0.001** | 0.038 |

The spread is not one number — it differs by metric, by two orders of magnitude. Faithfulness swings 0.138, because it compounds a non-deterministic generator with a non-deterministic judge. **A Faithfulness delta below ~0.14 in this table says nothing at all**, which retires several claims an earlier version of this README made.

**That floor was mostly an apparatus bug, not sampling.** Re-measured the same way on 54 questions, faithfulness spread is **0.014** — an order of magnitude lower, where sampling alone predicts a factor of 1.6. The cause is visible in the run files: `max_tokens=1024` was being applied to the judge as well as to generation, and RAGAS faithfulness emits one structured verdict per extracted statement, so it overran on 4 of 20 questions. RAGAS drops an overrun row rather than truncating it, *which* rows overran varied between runs, and the dropped rows were the long multi-claim answers — the ones most at risk of being unfaithful. At `eval_judge_max_tokens=4096` both 54-question runs score 38 of 38. The ablation table below still has to be read against its own 0.138 floor, because that is the floor those runs were measured under.

Context precision went the other way — 0.001 at n=20, 0.050 at n=54 — and it is not a retrieval effect: 53 of 54 questions retrieved byte-identical contexts across the two runs, and the largest single move has identical contexts *and* an identical answer. RAGAS context precision is an LLM judgment about chunk usefulness, so a deterministic retriever does not make it a deterministic metric. The 0.001 was a lucky draw.

| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Dense only (mean of 2) | 0.645 | 0.480 | 0.696 | 0.327 |
| **+ Cross-encoder rerank** | 0.770 | **0.719** | 0.466 | **0.350** |
| + BM25 hybrid (RRF) | 0.679 | 0.678 | 0.463 | 0.312 |
| + HyDE | 0.600 | 0.567 | 0.614 | 0.321 |
| + Semantic chunking¹ | 0.722 | 0.593 | 0.435 | 0.347 |

¹ Different index (8,936 chunks vs 9,572) — a chunking change rebuilds the corpus, so this row is not strictly same-corpus comparable.

> **These runs predate deduplication, and now also predate page-furniture removal.** All six were measured on the full 9,572-chunk index. Three corpus changes have landed since: `DEDUP_ENABLED` now defaults to true; the parser strips footers, page numbers, and running headers, the last of these keyed on position relative to a page break rather than on frequency alone; and chunking drops sub-`MIN_CHUNK_CHARS` stubs. A fresh ingest now produces **8,232 chunks** and a different corpus fingerprint, so every row above is measured against an index that no longer exists.
>
> The table is internally consistent and stays as measured. Re-running one row against the new corpus would be worse than re-running none — the comparison *between* rows is the whole point, and a single re-measured row would silently mix two corpora. Replacing it means re-running all six together.
>
> The honest expectation is that both changes help precision — 1,251 fewer chunks competing for 5 slots, and the ones removed are disproportionately the contentless kind — but that is a prediction, not a measurement.

**Reranking — one large real gain, and a cost that turns out to be worse than a metric.** Answer Relevancy +0.239 against a 0.044 floor is unambiguous. So is Context Precision **−0.230**, and it points the opposite way to the usual story about rerankers. `ms-marco-MiniLM` was trained on short web passages; asked to rank 512-token slabs of financial-statement prose it reorders confidently but not well. Faithfulness (+0.125) and Recall (+0.023) are both inside their floors and are not claimed.

The 54-question set showed what that precision loss actually does. The cross-encoder assigns *uniformly negative* logits to financial-table prose, and `ScoredDocument.relevance` squashes a logit through a logistic, so a whole retrieved set can land below the 0.15 insufficient-context threshold and the system declines a question it can answer. On "What was Apple's total net revenue for its most recent fiscal year?" it ranks a *Foreign pretax earnings* chunk first at −2.33 and pushes the actual net-sales table from rank 1 to rank 5, for a set relevance of 0.077 — under the gate, so no generation call is made at all. Eleven of the 38 answerable questions are declined this way or by the model that follows. Run the same five questions with `RERANK_ENABLED=false` and every one scores 0.286–0.466 and passes comfortably.

It ships on because relevancy is what a user experiences, and because the gate is doing the right thing given the scores it is handed. But the reranker is a live defect with two faces, and the second one — silently refusing answerable questions — is the more damaging and was invisible until `refusal_correctness` existed to name it.

**Per-entity retrieval — a comparison is two questions.** A question naming two companies used to share one global top-5, which filled with whichever filing ranked better overall; the model then correctly reported it could not compare, and RAGAS scored that refusal 0.000. Each named company now gets its own `retrieval_top_k` slots, filtered at the ChromaDB level and merged by cross-encoder relevance, so citation [1] is still the best passage overall. Set retrieval confidence on "What regulatory risks do JPMorgan and Goldman Sachs face?" goes 0.30 → 0.971. Entity detection is a literal alias match over the five known companies (`src/retrieval/entities.py`), not an NER model or an LLM call — the set of surface forms is small, closed, and writable by hand, and anything cleverer would put another model in front of retrieval.

**Hybrid search — the previous −0.23 recall claim does not reproduce.** On the fixed corpus every hybrid delta lands inside its noise floor (recall −0.038 against a 0.038 floor; faithfulness −0.091 against 0.138). The earlier regression was measured against an index missing ~90% of two of the five filings. It stays off by default, but the honest statement is now *"not established as helpful"*, not *"measurably harmful"*.

**HyDE — buys back precision, spends relevancy.** Precision +0.148 over reranking is real; relevancy −0.152 is also real (floor 0.044). Consistent with the mechanism: a generated hypothetical passage matches prose style, which suits the cross-encoder, while drifting from what was literally asked. Off by default.

**Semantic chunking — no measured benefit.** Recall and precision are flat-to-worse and relevancy is down 0.126. Splitting on embedding-distance spikes is more principled than splitting every 512 characters, but principle is not evidence.

**The comparative stratum is broken, and stratification is what showed it.** All three cross-company questions score Answer Relevancy exactly 0.000 in every config. They are not noisy — they fail identically. With `top_k=5`, all five slots fill with one company's chunks, the model correctly answers *"I don't have enough information"*, and RAGAS scores a refusal as 0. This is a retrieval-budget bug (comparatives need per-entity retrieval, not a global top-5), and an aggregate over 20 questions buries it as a mild drag on the mean.

The methodological point: with n=20 and an LLM judge, a single run cannot distinguish a 0.04 improvement from noise — and for Faithfulness it cannot distinguish 0.13. Repeating one config was the cheapest way to learn which deltas were worth believing.

### Iteration Story (interview talking point)

The first RAGAS run showed Context Recall of **0.08**. Root cause: the shipped dataset had placeholder ground truths (*"The exact figure will be populated from the actual downloaded filing"*). Regenerating them from real filings took recall to 0.57.

That fix contained a subtler bug that survived much longer. The regenerated answers were produced by asking an LLM to answer from **this project's own top-20 retrieval** — so the reference was downstream of the system being scored, and any blind spot was invisible by construction.

Removing it required reading the filings directly, which surfaced the real problem: **the corpus was missing most of two filings.** Section extraction was silently returning near-empty sections — 11% of JPM, 8% of MSFT — for two unrelated reasons. Microsoft repeats `PART I` / `Item 1` as a running page header, and each repeat was being treated as a section boundary, so a "section" was one page. JPMorgan satisfies Items 7 and 8 by *incorporation by reference*: the section body is a pointer reading *"appears on pages 46–160"*, while 1.03M characters of actual financials sit past every Item header that header-delimited slicing can reach. Nothing errored in either case. Sections were extracted; they were just tiny.

Fixing both took the index from 6,394 to 9,572 chunks and invalidated every retrieval number measured before it. Hence the corpus fingerprint now written into every result file.

```bash
make eval                     # Runs RAGAS on eval_questions_v3.json

# Retrieval stages toggle independently, so each can be measured in isolation
RERANK_ENABLED=false HYBRID_SEARCH_ENABLED=false make eval  # Dense only (baseline)
RERANK_ENABLED=true  HYBRID_SEARCH_ENABLED=false make eval  # + cross-encoder rerank
RERANK_ENABLED=false HYBRID_SEARCH_ENABLED=true  make eval  # + BM25 hybrid
RERANK_ENABLED=true  HYBRID_SEARCH_ENABLED=true  make eval  # both
HYDE_ENABLED=true    RERANK_ENABLED=true         make eval  # + HyDE query rewrite

# Chunking is baked into an index, so comparing strategies needs one index each
python scripts/build_semantic_index.py
CHROMA_COLLECTION=rag_enterprise_semantic make eval

# Near-duplicate report for an existing collection — read-only, no API calls
python scripts/analyze_duplicates.py
```

## Testing

```bash
make test          # Run all tests
make test-cov      # Run with coverage
make lint          # Lint + type check
```

## Deployment

### Live Demo (Google Cloud Run)

The app is deployed on **Google Cloud Run** at **https://rag-enterprise-laa65asupq-uc.a.run.app**. The SEC filing index (~6,400 chunks) is baked into the Docker image at build time, so the service scales to zero and cold-starts instantly. Try it:

```bash
# Health check
curl https://rag-enterprise-laa65asupq-uc.a.run.app/health

# Login as research analyst -> returns a JWT
curl -X POST https://rag-enterprise-laa65asupq-uc.a.run.app/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"research_analyst","password":"research1!"}'

# Query (use the access_token from the login response)
curl -X POST https://rag-enterprise-laa65asupq-uc.a.run.app/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"What was Apple total net revenue in fiscal year 2024?"}'
```

Or use the **interactive demo on the landing page** (pick a role, ask a question), or explore the [API docs](https://rag-enterprise-laa65asupq-uc.a.run.app/docs).

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
