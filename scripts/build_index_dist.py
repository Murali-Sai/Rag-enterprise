"""Build a shippable copy of the measured index.

The deployed demo ships the index rather than baking a fresh one, so that it
answers off the same corpus every number in the README was measured against
(`eval_20260809_015503`: 8,232 chunks, digest `c2f8c13673cf5ca5`). Baking at
build time cannot do that — it would need an OpenAI key inside the build, and
it would produce a *different* corpus each time it ran.

The local `chroma_data/` cannot be shipped as-is. It is a working directory:
alongside the live `rag_enterprise` collection it carries a semantic-chunking
experiment, a dedup smoke test and orphaned segment directories from
collections deleted long ago — 361 MB, of which the demo reads about 150 MB.
Shipping all of it would also ship the ambiguity: an image containing three
indices invites the next person to point the service at the wrong one.

So this writes a copy holding exactly one collection, and refuses to finish
unless that copy has the chunk count and content digest it was told to expect.
A silently-wrong index is the failure mode worth spending a check on — it
looks identical to a right one until the answers are subtly off.

    python scripts/build_index_dist.py

Never writes to the source directory.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path

# The corpus the current baseline was measured on. Defaults rather than
# required arguments: the point is that running this script with no arguments
# either reproduces the measured index or fails.
EXPECTED_CHUNKS = 8232
EXPECTED_DIGEST = "c2f8c13673cf5ca5"


def content_digest(documents: list[str]) -> str:
    """Hash a corpus by content, order-independently.

    Same construction as `_corpus_fingerprint()` in `evaluation/run_evaluation.py`
    — sorted per-document SHA-256, hashed again — so a digest printed here is
    directly comparable to the one recorded in an eval run's `config` block.
    """
    digest = hashlib.sha256()
    for doc_hash in sorted(hashlib.sha256(d.encode("utf-8")).hexdigest() for d in documents):
        digest.update(doc_hash.encode("ascii"))
    return digest.hexdigest()[:16]


def fingerprint(persist_dir: Path, collection: str) -> tuple[int, str]:
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    result = client.get_collection(collection).get(include=["documents"])
    documents = result["documents"] or []
    return len(documents), content_digest(documents)


def copy_tree(source: Path, dest: Path) -> None:
    """Copy the whole store, segment directories included.

    Copying everything and pruning afterwards — rather than copying only the
    directories that look relevant — keeps the decision about what is
    referenced with Chroma, which owns that mapping, instead of with this
    script's reading of it.
    """
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def prune(persist_dir: Path, keep: str) -> list[str]:
    """Drop every collection but `keep`, then reclaim the space."""
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    names = [c.name for c in client.list_collections()]
    if keep not in names:
        raise SystemExit(f"collection {keep!r} not in the store (found: {names or 'none'})")

    dropped = [name for name in names if name != keep]
    for name in dropped:
        client.delete_collection(name)
    del client  # release the sqlite handle before VACUUM opens its own

    # Deleting rows does not shrink the file; the metadata segment and its
    # full-text index are most of chroma.sqlite3.
    connection = sqlite3.connect(persist_dir / "chroma.sqlite3")
    connection.execute("VACUUM")
    connection.close()
    return dropped


def remove_orphan_segments(persist_dir: Path) -> list[str]:
    """Delete HNSW directories no segment row points at any more.

    `delete_collection` removes the rows but leaves the vector directory on
    disk, and the store carries directories from collections deleted in earlier
    sessions besides. Anything not named in `segments` is unreachable.
    """
    connection = sqlite3.connect(persist_dir / "chroma.sqlite3")
    live = {row[0] for row in connection.execute("SELECT id FROM segments")}
    connection.close()

    removed = []
    for child in persist_dir.iterdir():
        if child.is_dir() and child.name not in live:
            shutil.rmtree(child)
            removed.append(child.name)
    return removed


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("chroma_data"))
    parser.add_argument("--dest", type=Path, default=Path("chroma_dist"))
    parser.add_argument("--collection", default="rag_enterprise")
    parser.add_argument("--expect-chunks", type=int, default=EXPECTED_CHUNKS)
    parser.add_argument("--expect-digest", default=EXPECTED_DIGEST)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Build a dist from a corpus other than the measured one. The image "
        "then ships an index the published numbers do not describe — say so "
        "wherever those numbers appear.",
    )
    args = parser.parse_args()

    if not (args.source / "chroma.sqlite3").exists():
        raise SystemExit(f"no Chroma store at {args.source}")

    print(f"copying {args.source} -> {args.dest} ({directory_size_mb(args.source):.0f} MB)")
    copy_tree(args.source, args.dest)

    dropped = prune(args.dest, args.collection)
    print(f"kept {args.collection!r}; dropped {dropped or 'nothing'}")
    orphans = remove_orphan_segments(args.dest)
    noun = "directory" if len(orphans) == 1 else "directories"
    print(f"removed {len(orphans)} unreferenced segment {noun}")

    chunks, digest = fingerprint(args.dest, args.collection)
    print(f"\n{args.dest}: {chunks} chunks, digest {digest}, {directory_size_mb(args.dest):.0f} MB")

    if args.no_verify:
        print("verification skipped (--no-verify)")
        return 0

    if (chunks, digest) != (args.expect_chunks, args.expect_digest):
        print(
            f"\nFAILED: expected {args.expect_chunks} chunks / digest "
            f"{args.expect_digest}.\nThe source index is not the corpus the "
            f"published numbers were measured on. Re-run with --no-verify only "
            f"if you intend to ship a different one.",
            file=sys.stderr,
        )
        return 1

    print("verified against the measured corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
