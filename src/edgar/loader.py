"""LangChain Document adapter for SEC EDGAR filings.

Bridges the EDGAR client and parser into the existing ingestion pipeline
by producing langchain_core.documents.Document objects with rich metadata.
"""

import json
from pathlib import Path

from langchain_core.documents import Document

from src.common.logging import get_logger
from src.config import settings
from src.edgar.client import COMPANY_REGISTRY, EdgarClient
from src.edgar.parser import FilingSection, parse_10k_sections

logger = get_logger(__name__)


def _sections_to_documents(
    sections: list[FilingSection],
    ticker: str,
    company_name: str,
    cik: str,
    filing_type: str,
    filing_date: str,
) -> list[Document]:
    """Convert parsed filing sections into LangChain Documents."""
    documents = []
    source_name = f"{ticker}_{filing_type}_{filing_date}"

    for section in sections:
        doc = Document(
            page_content=section.content,
            metadata={
                "source_file": source_name,
                "ticker": ticker,
                "cik": cik,
                "company_name": company_name,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "section_id": section.section_id,
                "section_name": section.section_name,
                "department": "sec_filings",
                "access_roles": "trading,risk,compliance,research,wealth_management,auditor,viewer,admin",
            },
        )
        documents.append(doc)

    logger.info(
        "documents_created",
        ticker=ticker,
        filing=f"{filing_type} {filing_date}",
        sections=len(documents),
    )
    return documents


async def load_edgar_filing(
    ticker: str,
    filing_type: str = "10-K",
    filing_index: int = 0,
) -> list[Document]:
    """Fetch, parse, and convert a single EDGAR filing into Documents.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        filing_type: SEC form type (e.g., "10-K", "10-Q")
        filing_index: 0 = most recent, 1 = second most recent, etc.

    Returns:
        List of Documents, one per filing section
    """
    client = EdgarClient()
    try:
        filings = await client.get_company_filings(
            ticker, filing_type, count=filing_index + 1,
        )

        if not filings or filing_index >= len(filings):
            logger.warning("filing_not_found", ticker=ticker, type=filing_type, index=filing_index)
            return []

        filing = filings[filing_index]
        html = await client.download_filing_document(ticker, filing)

        sections = parse_10k_sections(html)
        company = COMPANY_REGISTRY[ticker.upper()]

        return _sections_to_documents(
            sections=sections,
            ticker=ticker.upper(),
            company_name=company["name"],
            cik=company["cik"],
            filing_type=filing.form_type,
            filing_date=filing.filing_date,
        )
    finally:
        await client.close()


async def load_multiple_filings(
    tickers: list[str],
    filing_type: str = "10-K",
    count_per_company: int = 1,
) -> list[Document]:
    """Load filings for multiple companies."""
    client = EdgarClient()
    all_documents: list[Document] = []

    try:
        for ticker in tickers:
            company = COMPANY_REGISTRY.get(ticker.upper())
            if company is None:
                logger.warning("unknown_ticker", ticker=ticker)
                continue

            filings = await client.get_company_filings(
                ticker, filing_type, count=count_per_company,
            )

            for filing in filings:
                html = await client.download_filing_document(ticker, filing)
                sections = parse_10k_sections(html)

                docs = _sections_to_documents(
                    sections=sections,
                    ticker=ticker.upper(),
                    company_name=company["name"],
                    cik=company["cik"],
                    filing_type=filing.form_type,
                    filing_date=filing.filing_date,
                )
                all_documents.extend(docs)

        logger.info(
            "all_filings_loaded",
            companies=len(tickers),
            total_documents=len(all_documents),
        )
    finally:
        await client.close()

    return all_documents


def load_filing_from_disk(
    html_path: str | Path,
    metadata_path: str | Path | None = None,
) -> list[Document]:
    """Load a previously downloaded filing from disk.

    This is useful for offline operation or when filings have already
    been downloaded via scripts/download_filings.py.
    """
    html_path = Path(html_path)
    html = html_path.read_text(encoding="utf-8")

    # Try to load metadata from companion JSON file
    if metadata_path is None:
        metadata_path = html_path.with_suffix(".json")

    if Path(metadata_path).exists():
        meta = json.loads(Path(metadata_path).read_text())
        ticker = meta["ticker"]
        company_name = meta["company_name"]
        cik = meta["cik"]
        filing_type = meta["form_type"]
        filing_date = meta["filing_date"]
    else:
        # Infer from filename: e.g., "10-K_2024-11-01.html"
        ticker = html_path.parent.name
        filing_type = html_path.stem.split("_")[0]
        filing_date = html_path.stem.split("_")[1] if "_" in html_path.stem else "unknown"
        company = COMPANY_REGISTRY.get(ticker.upper(), {})
        company_name = company.get("name", ticker)
        cik = company.get("cik", "")

    sections = parse_10k_sections(html)

    return _sections_to_documents(
        sections=sections,
        ticker=ticker.upper(),
        company_name=company_name,
        cik=cik,
        filing_type=filing_type,
        filing_date=filing_date,
    )
