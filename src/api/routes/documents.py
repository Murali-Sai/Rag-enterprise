import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from src.api.deps import get_current_user, require_role
from src.auth.models import User
from src.common.schemas import DocumentIngestResponse
from src.ingestion.loaders import get_supported_extensions
from src.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/ingest", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(
    file: UploadFile,
    department: str,
    access_roles: str,  # Comma-separated roles
    user: User = Depends(require_role("admin")),
) -> DocumentIngestResponse:
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


@router.get("/supported-types")
async def supported_types(user: User = Depends(get_current_user)) -> dict:
    return {"supported_extensions": get_supported_extensions()}
