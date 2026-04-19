"""Startup script: download filings, ingest, seed users, then launch uvicorn."""

import os
import subprocess
import sys


def run_step(name: str, args: list[str]) -> None:
    """Run a script, warn on failure but continue."""
    print(f"=== {name} ===", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode} — continuing", flush=True)
    else:
        print(f"=== {name} complete ===", flush=True)


def main() -> None:
    py = sys.executable

    # 1. Download SEC filings from EDGAR API
    run_step("Downloading SEC filings", [py, "scripts/download_filings.py"])

    # 2. Ingest filings into ChromaDB vector store
    run_step("Ingesting filings", [py, "scripts/ingest_edgar.py", "--from-disk"])

    # 3. Seed demo users
    run_step("Seeding users", [py, "scripts/seed_users.py"])

    # 4. Start uvicorn
    port = os.environ.get("PORT", "8000")
    print(f"=== Starting uvicorn on port {port} ===", flush=True)
    os.execvp(
        "uvicorn",
        ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", port, "--log-level", "info"],
    )


if __name__ == "__main__":
    main()
