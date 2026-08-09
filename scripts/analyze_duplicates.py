"""Report near-duplicate chunks in an existing collection. Read-only.

Answers "how much of this index is redundant" without re-ingesting anything.
It reads the vectors Chroma already stores, so it costs no embedding calls
and can be run against a live collection.

Why it matters: retrieval returns a fixed top_k, so a duplicate chunk does
not merely waste disk — it competes for one of those slots and wins it,
displacing whatever the LLM would have seen next.

Usage:
    python scripts/analyze_duplicates.py
    python scripts/analyze_duplicates.py --collection rag_enterprise_semantic
    python scripts/analyze_duplicates.py --show 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Similarity is reported at several thresholds because the right cutoff is a
# judgment call, and seeing the distribution is what makes it an informed one:
# 0.999 is "the same text twice", while 0.90 already includes chunks that
# merely discuss the same topic and must not be discarded.
THRESHOLDS = (0.90, 0.95, 0.98, 0.999)
BLOCK = 512


def main(collection: str | None, show: int) -> None:
    import os

    if collection:
        os.environ["CHROMA_COLLECTION"] = collection

    import numpy as np

    from src.common.logging import setup_logging
    from src.config import settings
    from src.retrieval.vector_store import get_vector_store

    setup_logging()

    raw = get_vector_store().store.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.asarray(raw.get("embeddings"), dtype=np.float32)
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    if not len(vectors):
        raise SystemExit(f"Collection {settings.chroma_collection!r} has no vectors.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1.0, norms)
    total = len(unit)

    print(f"Collection: {settings.chroma_collection}")
    print(f"Chunks:     {total:,}  (dim {unit.shape[1]})\n")

    counts = dict.fromkeys(THRESHOLDS, 0)
    examples: list[tuple[float, int, int]] = []

    for start in range(0, total, BLOCK):
        end = min(start + BLOCK, total)
        similarities = unit[start:end] @ unit.T

        for row in range(end - start):
            i = start + row
            if i == 0:
                continue
            # Compare only against earlier chunks, so each duplicated pair is
            # counted once rather than twice, and the survivor is the first
            # occurrence — matching what ingestion-time dedup actually keeps.
            prior = similarities[row, :i]
            best = float(prior.max())
            j = int(prior.argmax())

            for threshold in THRESHOLDS:
                if best >= threshold:
                    counts[threshold] += 1

            if best >= settings.dedup_threshold and len(examples) < show:
                examples.append((best, i, j))

    for threshold in THRESHOLDS:
        n = counts[threshold]
        marker = "  <- DEDUP_THRESHOLD" if threshold == settings.dedup_threshold else ""
        print(f"  cos >= {threshold:<5} {n:6,} chunks  ({n / total * 100:5.2f}%){marker}")

    def label(index: int) -> str:
        meta = metadatas[index] or {}
        return str(meta.get("ticker") or meta.get("source_file") or "?")

    if examples:
        print(f"\nExamples at or above {settings.dedup_threshold}:")
        for score, i, j in examples:
            print(f"\n  cos={score:.4f}   [{label(i)}] vs [{label(j)}]")
            print(f"    A: {documents[i][:160]!r}")
            print(f"    B: {documents[j][:160]!r}")

    suppressed = counts[settings.dedup_threshold]
    print(
        f"\nIngesting this corpus with DEDUP_ENABLED=true would skip roughly "
        f"{suppressed:,} chunks ({suppressed / total * 100:.1f}%)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report near-duplicate chunks (read-only)")
    parser.add_argument("--collection", default=None, help="Defaults to CHROMA_COLLECTION")
    parser.add_argument("--show", type=int, default=5, help="How many example pairs to print")
    args = parser.parse_args()

    main(args.collection, args.show)
