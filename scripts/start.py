"""Startup script: seed users then launch uvicorn."""

import os
import subprocess
import sys


def main() -> None:
    # Seed demo users (ignore errors — users may already exist on restart)
    print("=== Seeding users ===", flush=True)
    result = subprocess.run([sys.executable, "scripts/seed_users.py"])
    if result.returncode != 0:
        print(f"WARNING: seed_users.py exited with code {result.returncode} — continuing", flush=True)

    # Start uvicorn
    port = os.environ.get("PORT", "8000")
    print(f"=== Starting uvicorn on port {port} ===", flush=True)
    os.execvp(
        "uvicorn",
        ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", port, "--log-level", "info"],
    )


if __name__ == "__main__":
    main()
