"""SEC EDGAR API client.

Fetches company filings from the SEC EDGAR public API.
No authentication required — only a User-Agent header with
company name and email (SEC fair access policy).

Rate limit: 10 requests/second per SEC guidelines.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.common.logging import get_logger
from src.config import settings

logger = get_logger(__name__)

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"

# Target companies — CIK numbers are zero-padded to 10 digits
COMPANY_REGISTRY: dict[str, dict[str, str]] = {
    "AAPL": {"cik": "0000320193", "name": "Apple Inc."},
    "JPM": {"cik": "0000019617", "name": "JPMorgan Chase & Co."},
    "TSLA": {"cik": "0001318605", "name": "Tesla Inc."},
    "MSFT": {"cik": "0000789019", "name": "Microsoft Corporation"},
    "GS": {"cik": "0000886982", "name": "The Goldman Sachs Group Inc."},
}


@dataclass
class FilingMetadata:
    accession_number: str
    filing_date: str
    primary_document: str
    form_type: str
    description: str


class EdgarClient:
    """Async client for the SEC EDGAR API with rate limiting."""

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or settings.edgar_user_agent
        self._semaphore = asyncio.Semaphore(settings.edgar_rate_limit)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    async def _rate_limited_get(self, url: str) -> httpx.Response:
        async with self._semaphore:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            # Minimum 0.1s between requests to stay well under 10/s limit
            await asyncio.sleep(0.1)
            return response

    async def get_company_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        count: int = 1,
    ) -> list[FilingMetadata]:
        """Fetch filing metadata for a company from EDGAR submissions endpoint."""
        company = COMPANY_REGISTRY.get(ticker.upper())
        if company is None:
            raise ValueError(
                f"Unknown ticker: {ticker}. Available: {list(COMPANY_REGISTRY.keys())}"
            )

        cik = company["cik"]
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)

        logger.info("fetching_filings", ticker=ticker, cik=cik, type=filing_type)
        response = await self._rate_limited_get(url)
        data = response.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        filings: list[FilingMetadata] = []
        for i, form in enumerate(forms):
            if form == filing_type and len(filings) < count:
                filings.append(FilingMetadata(
                    accession_number=accessions[i],
                    filing_date=dates[i],
                    primary_document=primary_docs[i],
                    form_type=form,
                    description=descriptions[i] if i < len(descriptions) else "",
                ))

        logger.info("filings_found", ticker=ticker, count=len(filings))
        return filings

    async def download_filing_document(
        self,
        ticker: str,
        filing: FilingMetadata,
    ) -> str:
        """Download the actual filing HTML content."""
        company = COMPANY_REGISTRY[ticker.upper()]
        cik = company["cik"]
        # Accession number in URL has no dashes
        accession_path = filing.accession_number.replace("-", "")

        url = EDGAR_ARCHIVES_URL.format(
            cik=cik.lstrip("0"),  # EDGAR archives use unpadded CIK
            accession=accession_path,
            filename=filing.primary_document,
        )

        logger.info("downloading_filing", ticker=ticker, url=url)
        response = await self._rate_limited_get(url)
        return response.text

    async def download_and_save(
        self,
        ticker: str,
        filing_type: str = "10-K",
        count: int = 1,
        output_dir: str | None = None,
    ) -> list[Path]:
        """Download filings and save to disk. Returns list of saved file paths."""
        out_dir = Path(output_dir or settings.edgar_data_dir) / ticker.upper()
        out_dir.mkdir(parents=True, exist_ok=True)

        filings = await self.get_company_filings(ticker, filing_type, count)
        saved_paths: list[Path] = []

        for filing in filings:
            filename = f"{filing.form_type}_{filing.filing_date}.html"
            filepath = out_dir / filename

            if filepath.exists():
                logger.info("filing_cached", path=str(filepath))
                saved_paths.append(filepath)
                continue

            html = await self.download_filing_document(ticker, filing)
            filepath.write_text(html, encoding="utf-8")

            # Save metadata alongside
            meta_path = filepath.with_suffix(".json")
            meta_path.write_text(json.dumps({
                "ticker": ticker.upper(),
                "cik": COMPANY_REGISTRY[ticker.upper()]["cik"],
                "company_name": COMPANY_REGISTRY[ticker.upper()]["name"],
                "form_type": filing.form_type,
                "filing_date": filing.filing_date,
                "accession_number": filing.accession_number,
                "primary_document": filing.primary_document,
            }, indent=2))

            logger.info("filing_saved", path=str(filepath), size_kb=len(html) // 1024)
            saved_paths.append(filepath)

        return saved_paths

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
