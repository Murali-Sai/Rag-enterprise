"""Download SEC EDGAR filings for demo companies.

Downloads real 10-K annual reports from the SEC EDGAR API and saves
them to data/edgar/{ticker}/ for subsequent ingestion.

Usage:
    python scripts/download_filings.py
    python scripts/download_filings.py --tickers AAPL,JPM --filing-type 10-K --count 2
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edgar.client import COMPANY_REGISTRY, EdgarClient


DEFAULT_TICKERS = ["AAPL", "JPM", "TSLA", "MSFT", "GS"]
DEFAULT_FILING_TYPE = "10-K"
DEFAULT_COUNT = 1


async def main(tickers: list[str], filing_type: str, count: int) -> None:
    print("=" * 60)
    print("SEC EDGAR Filing Downloader")
    print("=" * 60)
    print(f"Companies: {', '.join(tickers)}")
    print(f"Filing type: {filing_type}")
    print(f"Count per company: {count}")
    print()

    client = EdgarClient()
    total_files = 0

    try:
        for ticker in tickers:
            if ticker.upper() not in COMPANY_REGISTRY:
                print(f"  WARNING: Unknown ticker {ticker}, skipping")
                continue

            company = COMPANY_REGISTRY[ticker.upper()]
            print(f"Downloading {filing_type} filings for {company['name']} ({ticker})...")

            try:
                paths = await client.download_and_save(
                    ticker=ticker,
                    filing_type=filing_type,
                    count=count,
                )
                for p in paths:
                    print(f"  -> Saved: {p} ({p.stat().st_size // 1024} KB)")
                total_files += len(paths)
            except Exception as e:
                print(f"  ERROR downloading {ticker}: {e}")
    finally:
        await client.close()

    print(f"\nTotal files downloaded: {total_files}")
    print("Run 'python scripts/ingest_edgar.py' to ingest into the vector store.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEC EDGAR filings")
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help=f"Comma-separated tickers (default: {','.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--filing-type",
        type=str,
        default=DEFAULT_FILING_TYPE,
        help=f"Filing type (default: {DEFAULT_FILING_TYPE})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Filings per company (default: {DEFAULT_COUNT})",
    )
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    asyncio.run(main(tickers, args.filing_type, args.count))
