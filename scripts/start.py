"""Startup script.

On Cloud Run (and any platform where the SEC filing index ships inside the
Docker image), the vector store already exists, so startup just reports which
corpus it found, seeds demo users and launches uvicorn — keeping cold starts
fast.

If the index is missing (e.g. a platform that didn't get one at build time), it
falls back to downloading + ingesting filings at runtime. That fallback now
embeds with OpenAI by default and produces a corpus that is *not* the one the
published numbers describe, so the log line above it is the one to read before
believing a deployment's scores.
"""

import os
import subprocess
import sys
from pathlib import Path

# Run as `python scripts/start.py`, so sys.path[0] is scripts/, not the repo
# root — same line every other script in here carries.
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_step(name: str, args: list[str]) -> None:
    """Run a script, warn on failure but continue."""
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}", flush=True)
    result = subprocess.run(args, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode} — continuing\n", flush=True)
    else:
        print(f"OK: {name} complete\n", flush=True)


def report_corpus() -> None:
    """State the corpus and embedding model this process will answer from.

    A deployment serving a different index than its documentation claims looks
    exactly like one serving the right index — until the answers are subtly
    off. One line in the boot log makes that checkable without a query.

    Counts only: materialising the chunk bodies to compute a content digest
    takes seconds and would land on every cold start.
    """
    try:
        import chromadb

        from src.config import settings
        from src.ingestion.embeddings import active_embedding_model_name

        collection = chromadb.PersistentClient(path=settings.chroma_persist_dir).get_collection(
            settings.chroma_collection
        )
        print(
            f"Corpus: {settings.chroma_collection} — {collection.count()} chunks, "
            f"embedded with {active_embedding_model_name()}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — never block startup on a log line
        print(f"WARNING: could not read the corpus fingerprint: {exc}", flush=True)


def main() -> None:
    py = sys.executable

    # The index is baked at build time (see Dockerfile). Only ingest at runtime
    # if it's missing — keeps Cloud Run cold starts fast.
    chroma_ready = Path("chroma_data/chroma.sqlite3").exists()
    if chroma_ready:
        print("Vector store already present (shipped in image) — skipping ingest.", flush=True)
        report_corpus()
    else:
        print("Vector store not found — downloading + ingesting filings now.", flush=True)
        run_step("Downloading SEC filings", [py, "scripts/download_filings.py"])
        run_step("Ingesting filings into ChromaDB", [py, "scripts/ingest_edgar.py", "--from-disk"])

    # Seed demo users (fast; SQLite). Ignores errors if users already exist.
    run_step("Seeding demo users", [py, "scripts/seed_users.py"])

    # Cloud Run injects PORT (8080). Default to 8080 for local parity.
    port = os.environ.get("PORT", "8080")
    print(f"\n=== Starting uvicorn on port {port} ===", flush=True)
    os.execvp(
        "uvicorn",
        ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", port, "--log-level", "info"],
    )


if __name__ == "__main__":
    main()
