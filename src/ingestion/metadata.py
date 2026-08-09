from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

from src.common.logging import get_logger

logger = get_logger(__name__)


def enrich_metadata(
    documents: list[Document],
    source_file: str,
    department: str,
    access_roles: list[str],
) -> list[Document]:
    enriched = []
    for doc in documents:
        # chunk_index is deliberately not set here. split_documents assigns it
        # per source document at split time, for every ingestion path; setting
        # it again here would overwrite that with an index over whatever slice
        # of chunks this call happened to receive.
        doc.metadata.update(
            {
                "source_file": Path(source_file).name,
                "department": department,
                "access_roles": ",".join(access_roles),  # Stored as comma-separated for ChromaDB
                "ingested_at": datetime.now(UTC).isoformat(),
            }
        )
        enriched.append(doc)

    logger.info(
        "metadata_enriched",
        chunks=len(enriched),
        department=department,
        access_roles=access_roles,
    )
    return enriched
