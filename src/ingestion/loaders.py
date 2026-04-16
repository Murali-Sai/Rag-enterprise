from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from src.common.exceptions import DocumentIngestionError
from src.common.logging import get_logger

logger = get_logger(__name__)

LOADER_REGISTRY: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_document(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise DocumentIngestionError(f"File not found: {path}")

    suffix = path.suffix.lower()
    loader_class = LOADER_REGISTRY.get(suffix)
    if loader_class is None:
        raise DocumentIngestionError(
            f"Unsupported file type: {suffix}. Supported: {list(LOADER_REGISTRY.keys())}"
        )

    logger.info("loading_document", file=str(path), type=suffix)
    loader = loader_class(str(path))
    return loader.load()


def get_supported_extensions() -> list[str]:
    return list(LOADER_REGISTRY.keys())
