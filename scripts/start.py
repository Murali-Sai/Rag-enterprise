"""Startup script.

On Cloud Run (and any platform where the SEC filing index is baked into the
Docker image at build time), the vector store already exists, so startup just
seeds demo users and launches uvicorn — keeping cold starts fast.

If the index is missing (e.g. a platform that didn't bake it at build time),
it falls back to downloading + ingesting filings at runtime.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_step(name: str, args: list[str]) -> None:
    """Run a script, warn on failure but continue."""
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}", flush=True)
    result = subprocess.run(args, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode} — continuing\n", flush=True)
    else:
        print(f"OK: {name} complete\n", flush=True)


def main() -> None:
    py = sys.executable

    # The index is baked at build time (see Dockerfile). Only ingest at runtime
    # if it's missing — keeps Cloud Run cold starts fast.
    chroma_ready = Path("chroma_data/chroma.sqlite3").exists()
    if chroma_ready:
        print("Vector store already present (baked into image) — skipping ingest.", flush=True)
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
