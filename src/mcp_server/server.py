"""SEC EDGAR RAG — MCP server.

Wraps the existing Retrieval-Augmented Generation pipeline (ChromaDB +
LangChain + Gemini/Groq) as Model Context Protocol tools. Any MCP client
can now invoke these tools natively to query real SEC 10-K filings.

The same RBAC department filtering, Chinese Wall information barriers, and
financial compliance guardrails used by the REST API are applied here.

Run locally (stdio — for Claude Desktop / Cursor):
    python -m src.mcp_server.server

Run over HTTP (streamable-http — for remote clients):
    python -m src.mcp_server.server --http --port 8001
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from src.auth.rbac import (
    get_accessible_departments,
    get_information_barriers_for_user,
)
from src.common.logging import get_logger, setup_logging
from src.edgar.client import COMPANY_REGISTRY
from src.generation.chains import query_with_context
from src.guardrails.financial_compliance import (
    apply_financial_disclaimers,
    check_financial_compliance,
)
from src.guardrails.input_validator import validate_input
from src.guardrails.prompt_injection import detect_prompt_injection
from src.retrieval.retriever import RBACRetriever

logger = get_logger(__name__)

# Valid RBAC roles a client may assume (mirrors src/auth/rbac.py).
VALID_ROLES = {
    "admin",
    "trading",
    "risk",
    "compliance",
    "research",
    "wealth_management",
    "operations",
    "auditor",
    "viewer",
}

INSTRUCTIONS = """\
SEC EDGAR RAG — query real 10-K filings from Apple (AAPL), JPMorgan (JPM),
Tesla (TSLA), Microsoft (MSFT), and Goldman Sachs (GS).

Access is role-scoped: pass a `role` to each tool to control which documents
are visible. Chinese Wall information barriers are enforced (e.g. a `research`
role cannot read trading-desk documents). Answers are grounded only in the
retrieved filings and carry financial-compliance disclaimers where relevant.
"""

mcp = FastMCP("sec-edgar-rag", instructions=INSTRUCTIONS)


def _run_rag(question: str, role: str) -> str:
    """Shared RAG path: guardrails -> RBAC retrieval -> grounded generation."""
    role = role.lower().strip()
    if role not in VALID_ROLES:
        return f"Invalid role '{role}'. Valid roles: {', '.join(sorted(VALID_ROLES))}."

    # Input guardrails
    validate_input(question)
    injection = detect_prompt_injection(question)
    if injection.is_blocked:
        return f"Query blocked by safety system: {injection.reason}"

    user_roles = {role}

    # RBAC-filtered retrieval (Chinese Wall enforced inside the retriever)
    retriever = RBACRetriever(user_roles=user_roles)
    documents = retriever.retrieve(question)

    barriers = get_information_barriers_for_user(user_roles)
    barrier_note = ""
    if barriers:
        names = ", ".join(b["name"] for b in barriers)
        barrier_note = f"\n\n[Information barriers active for '{role}': {names}]"

    if not documents:
        return f"No relevant documents found within your access level.{barrier_note}"

    # Grounded generation
    answer = query_with_context(question, documents)

    # Financial compliance guardrails on the output
    compliance = check_financial_compliance(query=question, response=answer, user_roles=user_roles)
    answer = apply_financial_disclaimers(answer, compliance)

    # Append source citations
    sources = []
    for doc in documents:
        ticker = doc.metadata.get("ticker", "?")
        section = doc.metadata.get("section_name", "")
        filing = doc.metadata.get("filing_type", "")
        date = doc.metadata.get("filing_date", "")
        sources.append(f"  - {ticker} {filing} ({date}) — {section}")
    unique_sources = sorted(set(sources))

    flags_note = ""
    if compliance.flags:
        flags_note = f"\n\nGuardrail flags: {', '.join(compliance.flags)}"

    return (
        f"{answer}\n\n"
        f"Sources ({len(documents)} chunks):\n"
        + "\n".join(unique_sources)
        + flags_note
        + barrier_note
    )


@mcp.tool()
def query_sec_filings(question: str, role: str = "research") -> str:
    """Answer a natural-language question using real SEC 10-K filings.

    Retrieves the most relevant filing sections (RBAC-filtered for the given
    role) and generates a grounded answer with source citations.

    Args:
        question: A question about SEC filings, e.g.
            "What was Apple's total net revenue in fiscal year 2024?"
        role: RBAC role to assume (research, trading, risk, compliance,
            wealth_management, operations, auditor, viewer, admin).
            Controls which documents are visible. Defaults to "research".
    """
    logger.info("mcp_query_sec_filings", role=role, question=question[:80])
    return _run_rag(question, role)


@mcp.tool()
def compare_companies(ticker_a: str, ticker_b: str, topic: str, role: str = "research") -> str:
    """Compare two companies on a specific topic using their 10-K filings.

    Args:
        ticker_a: First company ticker (AAPL, JPM, TSLA, MSFT, GS).
        ticker_b: Second company ticker.
        topic: What to compare, e.g. "credit risk disclosures",
            "revenue growth", "key risk factors".
        role: RBAC role to assume. Defaults to "research".
    """
    a = ticker_a.upper().strip()
    b = ticker_b.upper().strip()
    for t in (a, b):
        if t not in COMPANY_REGISTRY:
            return f"Unknown ticker '{t}'. Available: {', '.join(COMPANY_REGISTRY.keys())}."
    question = (
        f"Compare {a} ({COMPANY_REGISTRY[a]['name']}) and "
        f"{b} ({COMPANY_REGISTRY[b]['name']}) regarding {topic}. "
        f"Clearly attribute each data point to the correct company and filing."
    )
    logger.info("mcp_compare_companies", a=a, b=b, topic=topic[:60], role=role)
    return _run_rag(question, role)


@mcp.tool()
def list_indexed_companies() -> str:
    """List the companies whose SEC 10-K filings are indexed and queryable."""
    lines = ["Indexed SEC 10-K filings:"]
    for ticker, info in COMPANY_REGISTRY.items():
        lines.append(f"  - {ticker}: {info['name']} (CIK {info['cik']})")
    lines.append(
        "\nQueryable 10-K sections: Item 1 (Business), Item 1A (Risk Factors), "
        "Item 7 (MD&A), Item 7A (Market Risk), Item 8 (Financial Statements)."
    )
    return "\n".join(lines)


@mcp.tool()
def describe_access(role: str = "research") -> str:
    """Show which document departments a given RBAC role can access.

    Demonstrates the Chinese Wall information barriers — e.g. a research
    analyst is walled off from trading and compliance documents.

    Args:
        role: RBAC role to inspect. Defaults to "research".
    """
    role = role.lower().strip()
    if role not in VALID_ROLES:
        return f"Invalid role '{role}'. Valid roles: {', '.join(sorted(VALID_ROLES))}."

    departments = get_accessible_departments({role})
    barriers = get_information_barriers_for_user({role})

    lines = [f"Role '{role}' can access departments:"]
    lines.extend(f"  - {d}" for d in sorted(departments))
    if barriers:
        lines.append("\nActive information barriers (Chinese Walls):")
        for b in barriers:
            blocked = ", ".join(sorted(b["blocked_departments"]))
            lines.append(f"  - {b['name']}: blocked from [{blocked}]")
    else:
        lines.append("\nNo information barriers active for this role.")
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="SEC EDGAR RAG MCP server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over streamable HTTP instead of stdio",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port")
    args = parser.parse_args()

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info("mcp_server_starting", transport="streamable-http", port=args.port)
        mcp.run(transport="streamable-http")
    else:
        logger.info("mcp_server_starting", transport="stdio")
        mcp.run()


if __name__ == "__main__":
    main()
