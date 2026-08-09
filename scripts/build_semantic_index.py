"""Build a semantically-chunked index alongside the fixed-size one.

Chunking strategy is baked into an index at ingestion time, so comparing
strategies means building two indexes and pointing the eval at each. This
writes to a separate Chroma collection; the shipped one is untouched.

Usage:
    python scripts/build_semantic_index.py
    python scripts/build_semantic_index.py --collection rag_enterprise_semantic

Then evaluate against it:
    CHROMA_COLLECTION=rag_enterprise_semantic python -m evaluation.run_evaluation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_COLLECTION = "rag_enterprise_semantic"
DEFAULT_TICKERS = ["AAPL", "JPM", "TSLA", "MSFT", "GS"]


def main(collection: str, tickers: list[str]) -> None:
    # Settings are read at import time by the vector store, so configure the
    # target collection and strategy before importing anything that touches
    # them — otherwise this would rebuild the shipped index in place.
    import os

    os.environ["CHROMA_COLLECTION"] = collection
    os.environ["CHUNKING_STRATEGY"] = "semantic"

    from src.common.logging import setup_logging
    from src.config import settings
    from src.edgar.loader import load_filing_from_disk
    from src.ingestion.chunking import split_documents
    from src.retrieval.vector_store import get_vector_store

    setup_logging()

    print("=" * 60)
    print("Semantic chunking — alternate index build")
    print("=" * 60)
    print(f"Collection: {settings.chroma_collection}")
    print(f"Strategy:   {settings.chunking_strategy.value}")
    print(f"Percentile: {settings.semantic_breakpoint_percentile}")
    print(f"Max chars:  {settings.semantic_max_chunk_chars}")
    print()

    if settings.chroma_collection == "rag_enterprise":
        raise SystemExit("Refusing to overwrite the shipped index — pass --collection.")

    edgar_dir = Path(settings.edgar_data_dir)
    vector_store = get_vector_store()

    # add_documents appends, so re-running without clearing silently doubles
    # the index — and a duplicated corpus quietly changes retrieval results.
    existing = vector_store.get_all_documents()
    if existing:
        print(f"Clearing {len(existing)} existing chunks from {settings.chroma_collection}...")
        vector_store.store.reset_collection()

    total_chunks = 0

    for ticker in tickers:
        ticker_dir = edgar_dir / ticker.upper()
        if not ticker_dir.exists() or not any(ticker_dir.glob("*.html")):
            print(f"  No downloaded filings for {ticker} — run download_filings.py first")
            continue

        for html_file in sorted(ticker_dir.glob("*.html")):
            print(f"  {html_file.name}...", flush=True)
            docs = load_filing_from_disk(html_file)
            chunks = split_documents(docs)
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"    sections={len(docs)} chunks={len(chunks)}", flush=True)

    print(f"\nTotal chunks: {total_chunks}")
    print(f"\nEvaluate with:\n  CHROMA_COLLECTION={collection} python -m evaluation.run_evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a semantically-chunked index")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    args = parser.parse_args()

    main(args.collection, [t.strip().upper() for t in args.tickers.split(",")])
