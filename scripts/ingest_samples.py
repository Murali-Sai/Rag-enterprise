"""Ingest sample financial documents into the vector store."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import setup_logging
from src.ingestion.pipeline import ingest_directory

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"

# Department access mirrors investment bank information barriers
DEPARTMENTS = {
    "sec_filings": {
        "path": SAMPLE_DIR / "sec_filings",
        "access_roles": ["trading", "risk", "compliance", "research", "wealth_management", "operations", "auditor", "viewer", "admin"],
    },
    "risk_management": {
        "path": SAMPLE_DIR / "risk_management",
        "access_roles": ["trading", "risk", "compliance", "admin"],
    },
    "compliance": {
        "path": SAMPLE_DIR / "compliance",
        "access_roles": ["compliance", "risk", "operations", "auditor", "admin"],
    },
    "research": {
        "path": SAMPLE_DIR / "research",
        "access_roles": ["research", "wealth_management", "admin"],
    },
    "trading": {
        "path": SAMPLE_DIR / "trading",
        "access_roles": ["trading", "risk", "operations", "admin"],
    },
}


def main() -> None:
    setup_logging()
    print("=" * 60)
    print("Ingesting financial services sample documents")
    print("=" * 60)

    total = 0
    for dept, config in DEPARTMENTS.items():
        print(f"\nIngesting {dept} documents from {config['path']}...")
        chunks = ingest_directory(
            directory=config["path"],
            department=dept,
            access_roles=config["access_roles"],
        )
        print(f"  -> {chunks} chunks created (access: {config['access_roles']})")
        total += chunks

    print(f"\nTotal chunks ingested: {total}")
    print("\nInformation barrier note:")
    print("  - Research analysts CANNOT access trading or compliance docs")
    print("  - This enforces the Chinese Wall between research and front office")


if __name__ == "__main__":
    main()
