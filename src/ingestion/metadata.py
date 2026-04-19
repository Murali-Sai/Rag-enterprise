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
    for i, doc in enumerate(documents):
        doc.metadata.update(
            {
                "source_file": Path(source_file).name,
                "department": department,
                "access_roles": ",".join(access_roles),  # Stored as comma-separated for ChromaDB
                "ingested_at": datetime.now(UTC).isoformat(),
                "chunk_index": i,
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
