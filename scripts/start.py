"""Startup script.

On Cloud Run (and any platform where the SEC filing index ships inside the
Docker image), the vector store already exists, so startup just reports which
corpus it found, seeds demo users and launches uvicorn — keeping cold starts
fast.

If the index is missing, it builds one on first boot: downloads the filings,
ingests them, and ingests the sample documents. That is the path a fresh clone
takes, because `chroma_dist/` is generated from a local index a clone does not
have — so `make docker-up` has to work without one, and without an API key.

An index built that way is *not* the corpus the published numbers describe.
The corpus line printed at boot is the one to read before believing a
deployment's scores.
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

        # Read the provider from the environment, not from `settings`.
        # `settings` is a module-level singleton built at first import, so an
        # override this script writes into os.environ after that point reaches
        # the ingest subprocesses and the uvicorn exec below — but not the
        # already-constructed object here. Reading it would print
        # "text-embedding-3-small" over an index just built with MiniLM, which
        # is the exact misreport this line exists to prevent.
        provider = os.environ.get("EMBEDDING_PROVIDER", settings.embedding_provider.value)
        model = (
            settings.openai_embedding_model if provider == "openai" else settings.embedding_model
        )

        collection = chromadb.PersistentClient(path=settings.chroma_persist_dir).get_collection(
            settings.chroma_collection
        )
        print(
            f"Corpus: {settings.chroma_collection} — {collection.count()} chunks, "
            f"embedded with {provider}/{model}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — never block startup on a log line
        print(f"WARNING: could not read the corpus fingerprint: {exc}", flush=True)


def choose_bootstrap_embeddings() -> str:
    """Pick the embedding provider for an index built at boot, and make the
    server agree with it.

    The image defaults to OpenAI, which a reviewer cloning the repo has no key
    for. Local embeddings need no key, so fall back to them — and *keep* them
    for the server process, because the two providers emit different-sized
    vectors (1536 vs 384) and a collection built under one cannot be queried
    under the other. Choosing per-process would produce an index that silently
    mismatches its own query path.

    Written into the environment because both the ingest subprocesses and the
    uvicorn exec below inherit it, which is what keeps the two in step.
    """
    from src.config import settings

    if settings.openai_api_key:
        return os.environ.get("EMBEDDING_PROVIDER", "openai")

    os.environ["EMBEDDING_PROVIDER"] = "huggingface"
    print(
        "No OPENAI_API_KEY — building the index with local embeddings "
        "(all-MiniLM-L6-v2) and querying it the same way.",
        flush=True,
    )
    return "huggingface"


def main() -> None:
    py = sys.executable

    # The index ships in the image when one was available at build time (see
    # Dockerfile). Only ingest at boot if it's missing — that keeps Cloud Run
    # cold starts instant, and is the path a fresh clone takes.
    chroma_ready = Path("chroma_data/chroma.sqlite3").exists()
    if chroma_ready:
        print("Vector store already present (shipped in image) — skipping ingest.", flush=True)
        report_corpus()
    else:
        provider = choose_bootstrap_embeddings()
        print(f"Vector store not found — building one now ({provider} embeddings).", flush=True)
        run_step("Downloading SEC filings", [py, "scripts/download_filings.py"])
        run_step("Ingesting filings into ChromaDB", [py, "scripts/ingest_edgar.py", "--from-disk"])
        # Not optional. These supply the trading, risk, compliance and research
        # documents — everything outside sec_filings. Without them a trader
        # asking about desk procedures gets "no documents found", which looks
        # exactly like a Chinese Wall block but is an empty corpus, and the
        # information-barrier demo degrades silently.
        run_step("Ingesting sample documents", [py, "scripts/ingest_samples.py"])
        report_corpus()

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
