"""Startup script: download filings, ingest, seed users, then launch uvicorn."""

import os
import subprocess
import sys


def run_step(name: str, args: list[str]) -> None:
    """Run a script, capture output, warn on failure but continue."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'=' * 60}", flush=True)
    result = subprocess.run(args, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode} — continuing\n", flush=True)
    else:
        print(f"OK: {name} complete\n", flush=True)


def main() -> None:
    py = sys.executable

    # 1. Download SEC filings from EDGAR API
    run_step("Downloading SEC filings", [py, "scripts/download_filings.py"])

    # 2. Verify files exist
    from pathlib import Path

    edgar_dir = Path("data/edgar")
    html_files = list(edgar_dir.rglob("*.html"))
    print(f"Filing HTML files found on disk: {len(html_files)}", flush=True)
    for f in html_files:
        print(f"  {f} ({f.stat().st_size // 1024} KB)", flush=True)

    # 3. Ingest filings into ChromaDB vector store
    if html_files:
        run_step("Ingesting filings into ChromaDB", [py, "scripts/ingest_edgar.py", "--from-disk"])
    else:
        print("WARNING: No filing HTML files found — skipping ingestion", flush=True)

    # 4. Seed demo users
    run_step("Seeding demo users", [py, "scripts/seed_users.py"])

    # 5. Start uvicorn
    port = os.environ.get("PORT", "8000")
    print(f"\n=== Starting uvicorn on port {port} ===", flush=True)
    os.execvp(
        "uvicorn",
        ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", port, "--log-level", "info"],
    )


if __name__ == "__main__":
    main()
