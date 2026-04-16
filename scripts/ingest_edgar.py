"""Ingest SEC EDGAR filings into the vector store.

Reads previously downloaded filings from data/edgar/ and ingests them
into ChromaDB. Can also fetch directly from EDGAR API if files aren't cached.

Usage:
    python scripts/ingest_edgar.py
    python scripts/ingest_edgar.py --tickers AAPL,JPM,TSLA,MSFT,GS --from-disk
    python scripts/ingest_edgar.py --tickers AAPL --filing-type 10-K --from-api
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import setup_logging
from src.config import settings
from src.edgar.client import COMPANY_REGISTRY

DEFAULT_TICKERS = ["AAPL", "JPM", "TSLA", "MSFT", "GS"]


async def ingest_from_disk(tickers: list[str]) -> int:
    """Ingest filings that were previously downloaded to data/edgar/."""
    from src.edgar.loader import load_filing_from_disk
    from src.ingestion.chunking import split_documents
    from src.retrieval.vector_store import get_vector_store

    edgar_dir = Path(settings.edgar_data_dir)
    vector_store = get_vector_store()
    total_chunks = 0

    for ticker in tickers:
        ticker_dir = edgar_dir / ticker.upper()
        if not ticker_dir.exists():
            print(f"  No downloaded filings for {ticker} in {ticker_dir}")
            continue

        for html_file in sorted(ticker_dir.glob("*.html")):
            print(f"  Parsing {html_file.name}...")
            docs = load_filing_from_disk(html_file)
            print(f"    Sections extracted: {len(docs)}")

            chunks = split_documents(docs)
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"    Chunks ingested: {len(chunks)}")

    return total_chunks


async def ingest_from_api(tickers: list[str], filing_type: str, count: int) -> int:
    """Fetch filings from EDGAR API and ingest directly."""
    from src.ingestion.pipeline import ingest_edgar_filings

    chunks = await ingest_edgar_filings(
        tickers=tickers,
        filing_type=filing_type,
        count_per_company=count,
    )
    return len(chunks)


async def main(tickers: list[str], filing_type: str, count: int, from_disk: bool) -> None:
    setup_logging()
    print("=" * 60)
    print("SEC EDGAR Filing Ingestion")
    print("=" * 60)
    print(f"Companies: {', '.join(tickers)}")
    print(f"Source: {'disk (data/edgar/)' if from_disk else 'EDGAR API'}")
    print()

    if from_disk:
        total = await ingest_from_disk(tickers)
    else:
        total = await ingest_from_api(tickers, filing_type, count)

    print(f"\nTotal chunks ingested: {total}")
    print("Start the server with 'make dev' and try querying!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR filings")
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help=f"Comma-separated tickers (default: {','.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--filing-type",
        type=str,
        default="10-K",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--from-disk", action="store_true", default=True)
    source.add_argument("--from-api", action="store_true")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    from_disk = not args.from_api

    asyncio.run(main(tickers, args.filing_type, args.count, from_disk))
