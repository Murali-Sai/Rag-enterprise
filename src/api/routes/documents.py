import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from src.api.deps import get_current_user, require_role
from src.auth.models import User
from src.auth.rbac import get_accessible_departments
from src.common.schemas import (
    DocumentIngestResponse,
    IndexedDocument,
    IndexedDocumentsResponse,
)
from src.config import settings
from src.ingestion.loaders import get_supported_extensions
from src.ingestion.pipeline import ingest_document
from src.retrieval.retriever import RBACRetriever
from src.retrieval.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/ingest", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(
    file: UploadFile,
    department: str,
    access_roles: str,  # Comma-separated roles
    user: User = Depends(require_role("admin")),
) -> DocumentIngestResponse:
    # The admin role is not a secret here: the demo's credentials are published
    # so a visitor can watch the barriers come down. That makes "admin only" the
    # wrong control for a write, and the deployment turns writes off entirely.
    if not settings.allow_runtime_ingest:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "This deployment serves a fixed, verified corpus and does not accept "
                "document uploads. Run it locally to ingest your own documents."
            ),
        )

    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in get_supported_extensions():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {get_supported_extensions()}",
        )

    roles_list = [r.strip() for r in access_roles.split(",")]

    # Save to temp file and ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = ingest_document(
            file_path=tmp_path,
            department=department,
            access_roles=roles_list,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return DocumentIngestResponse(
        filename=file.filename,
        chunks_created=len(chunks),
        department=department,
        access_roles=roles_list,
    )


def _group_by_source(metadatas: list[dict]) -> list[IndexedDocument]:
    """Roll chunk metadata up into the documents it came from.

    Provenance is stamped per chunk at ingestion, so the document view is a
    derived one. Sections are collected across a filing's chunks because that
    is the only place the answer lives — "which Items did the parser actually
    recover for GS" is a question this listing can answer and a manifest
    written at upload time could not.
    """
    grouped: dict[str, dict] = {}
    for meta in metadatas:
        source = str(meta.get("source_file", "unknown"))
        entry = grouped.setdefault(
            source,
            {
                "department": str(meta.get("department", "unknown")),
                "chunks": 0,
                "ticker": meta.get("ticker"),
                "filing_type": meta.get("filing_type"),
                "filing_date": meta.get("filing_date"),
                "sections": {},
            },
        )
        entry["chunks"] += 1
        section = meta.get("section_name")
        if section:
            entry["sections"].setdefault(str(section), None)

    return [
        IndexedDocument(
            source=source,
            department=entry["department"],
            chunks=entry["chunks"],
            ticker=entry["ticker"],
            filing_type=entry["filing_type"],
            filing_date=entry["filing_date"],
            sections=list(entry["sections"]),
        )
        for source, entry in sorted(grouped.items())
    ]


@router.get("", response_model=IndexedDocumentsResponse)
@router.get("/", response_model=IndexedDocumentsResponse, include_in_schema=False)
async def list_documents(
    user: User = Depends(get_current_user),
) -> IndexedDocumentsResponse:
    """What is in the index, filtered to what this role may read.

    Uses the same where-clause dense retrieval builds, rather than listing
    everything and filtering after: a listing that names documents a role
    cannot retrieve would leak the existence of what the information barrier
    is there to hide, which is a weaker but real version of the disclosure
    the barrier prevents.
    """
    retriever = RBACRetriever(user_roles=user.role_names, vector_store=get_vector_store())
    metadatas = retriever.vector_store.get_all_metadata(retriever.build_role_filter())
    documents = _group_by_source(metadatas)

    return IndexedDocumentsResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(d.chunks for d in documents),
        accessible_departments=sorted(get_accessible_departments(user.role_names)),
    )


@router.get("/supported-types")
async def supported_types(user: User = Depends(get_current_user)) -> dict:
    return {"supported_extensions": get_supported_extensions()}
