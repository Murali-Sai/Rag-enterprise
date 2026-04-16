from pathlib import Path

from langchain_core.documents import Document

from src.common.logging import get_logger
from src.ingestion.chunking import split_documents
from src.ingestion.loaders import load_document
from src.ingestion.metadata import enrich_metadata
from src.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


def ingest_document(
    file_path: str | Path,
    department: str,
    access_roles: list[str],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    # Load
    raw_docs = load_document(file_path)

    # Chunk
    chunks = split_documents(raw_docs, chunk_size, chunk_overlap)

    # Enrich metadata
    enriched_chunks = enrich_metadata(chunks, str(file_path), department, access_roles)

    # Store in vector DB
    vector_store = get_vector_store()
    vector_store.add_documents(enriched_chunks)

    logger.info(
        "document_ingested",
        file=str(file_path),
        department=department,
        chunks=len(enriched_chunks),
    )
    return enriched_chunks


def ingest_directory(
    directory: str | Path,
    department: str,
    access_roles: list[str],
) -> int:
    dir_path = Path(directory)
    total_chunks = 0

    from src.ingestion.loaders import get_supported_extensions

    for ext in get_supported_extensions():
        for file_path in dir_path.glob(f"*{ext}"):
            chunks = ingest_document(file_path, department, access_roles)
            total_chunks += len(chunks)

    logger.info(
        "directory_ingested",
        directory=str(directory),
        department=department,
        total_chunks=total_chunks,
    )
    return total_chunks


async def ingest_edgar_filings(
    tickers: list[str],
    filing_type: str = "10-K",
    count_per_company: int = 1,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Fetch real SEC EDGAR filings and ingest into the vector store.

    Downloads filings from the SEC EDGAR API, parses them into sections,
    chunks them, and stores them in the vector store with rich metadata.
    """
    from src.edgar.loader import load_multiple_filings

    # Fetch and parse filings from EDGAR
    raw_docs = await load_multiple_filings(tickers, filing_type, count_per_company)

    if not raw_docs:
        logger.warning("no_edgar_documents", tickers=tickers, type=filing_type)
        return []

    # Chunk (EDGAR metadata is already set on each Document by the loader)
    chunks = split_documents(raw_docs, chunk_size, chunk_overlap)

    # Store in vector DB
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    logger.info(
        "edgar_filings_ingested",
        companies=len(tickers),
        filing_type=filing_type,
        raw_sections=len(raw_docs),
        total_chunks=len(chunks),
    )
    return chunks
