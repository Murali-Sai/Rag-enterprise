from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
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
    # Generic HTML only. SEC filings are also HTML but do not come through
    # here — src/edgar/parser.py splits them on Item boundaries first, which
    # BSHTMLLoader cannot do and which is what makes a chunk attributable to
    # "Item 1A Risk Factors" instead of "somewhere in a 1.2M-character file".
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}

# Loaders that read bytes off disk and need to be told how. TextLoader and
# BSHTMLLoader both default to `locale.getpreferredencoding()`, which is
# cp1252 on Windows and UTF-8 on Linux — so the same file ingested on a dev
# machine and in the Docker build produces different text. It did: the sample
# 10-K's em dashes (e2 80 94 on disk) went into the index as "â€"", visible in
# any chunk snippet the API returns. Pinning UTF-8 makes the ingest
# reproducible across platforms rather than merely correct on one.
#
# The two spell the argument differently, which is the whole reason this is a
# lookup rather than one keyword passed to everything: `encoding=` raises
# TypeError on BSHTMLLoader. PDF and DOCX are absent because they carry their
# own encoding and take no such argument.
_ENCODING_ARGUMENT: dict[type, str] = {
    TextLoader: "encoding",
    BSHTMLLoader: "open_encoding",
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
    encoding_argument = _ENCODING_ARGUMENT.get(loader_class)
    if encoding_argument:
        loader = loader_class(str(path), **{encoding_argument: "utf-8"})
    else:
        loader = loader_class(str(path))
    return loader.load()


def get_supported_extensions() -> list[str]:
    return list(LOADER_REGISTRY.keys())
