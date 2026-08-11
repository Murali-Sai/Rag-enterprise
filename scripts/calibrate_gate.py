"""Measure where the retrieval-confidence gate's floor actually belongs.

The gate in `src/generation/answer.py` refuses when
`retrieval_confidence()` falls below `insufficient_context_threshold`. That
threshold was picked, not measured, and the number it is compared against is a
logistic over a cross-encoder logit — which is a ranking score, not a
calibrated probability of relevance over 10-K prose.

Measured on the 54-question eval suite, the failure has a specific shape. The
gate's score separates *nothing in this corpus is about this topic* cleanly,
and does not separate *the answer is not in these five chunks* at all: the four
`out_of_corpus` questions occupy the four lowest scores in the suite, while
answerable questions the gate refuses are interleaved with `rbac_blocked` and
`no_answer` ones just above them. One of those refused questions retrieves
context scoring 1.000 context recall.

So the floor is worth measuring on the job the score can do. This script runs
retrieval — no generation, no judge, so it costs one embedding per question —
over a probe set held out from the eval suite, and reports the separation
between the two labels.

Why a separate file rather than more rows in eval_questions_v4.json: the suite
is the test set. A threshold tuned on it and then reported against it would be
the same self-grading that generating ground truths from this project's own
retrieval produced, and which cost 0.24 of apparent context recall to undo.

    ./.venv/Scripts/python.exe -m scripts.calibrate_gate

Costs about $0.0003. Writes nothing; read the table.
"""

import json
from collections import defaultdict
from pathlib import Path

from src.generation.confidence import retrieval_confidence
from src.retrieval.retriever import get_retriever, retrieve_scored
from src.retrieval.vector_store import get_vector_store

DATASET = Path("evaluation/datasets/gate_calibration_v1.json")

# The probe runs as a single role so the RBAC where-clause is constant across
# the set. Research reaches sec_filings, which is the whole corpus under test.
PROBE_ROLES = {"research"}

OUT_OF_CORPUS = "out_of_corpus"
IN_CORPUS = "in_corpus"


def verify_evidence(items: list[dict]) -> list[str]:
    """Check every in-corpus item's evidence really is in that ticker's chunks.

    Lexical containment over the whole filing, not a retrieval call. An item
    whose evidence cannot be found is dropped rather than silently relabelled:
    calibrating a floor against a question the corpus does not actually answer
    would move the floor in exactly the wrong direction.
    """
    store = get_vector_store()
    by_ticker: dict[str, list[str]] = defaultdict(list)
    for item in items:
        if item["label"] == IN_CORPUS:
            by_ticker[item["ticker"]].append(item["evidence"])

    present: dict[tuple[str, str], bool] = {}
    for ticker, needles in by_ticker.items():
        docs = store.get_all_documents({"ticker": {"$eq": ticker}})
        blob = "\n".join(doc.page_content for doc in docs).lower()
        for needle in needles:
            present[(ticker, needle)] = needle.lower() in blob
        print(f"  {ticker}: {len(docs)} chunks searched")

    return [
        f"{ticker}: {needle!r}" for (ticker, needle), found in sorted(present.items()) if not found
    ]


def main() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    items = dataset["items"]

    print(f"Verifying evidence for in-corpus items in {DATASET}")
    missing = verify_evidence(items)
    if missing:
        print("\n  NOT FOUND in the corpus — these items are dropped:")
        for entry in missing:
            print(f"    {entry}")
        missing_needles = {entry.split(": ", 1)[1].strip("'") for entry in missing}
        items = [item for item in items if item.get("evidence") not in missing_needles]

    print(f"\nProbing {len(items)} questions (retrieval only)\n")
    scored_items: list[tuple[float | None, dict]] = []
    for item in items:
        retriever = get_retriever(user_roles=PROBE_ROLES)
        scored = retrieve_scored(retriever, item["question"])
        confidence = retrieval_confidence(scored)
        best = max((s.score for s in scored if s.score is not None), default=None)
        scored_items.append((confidence, item))
        print(
            f"  {'   None' if confidence is None else f'{confidence:7.4f}'}"
            f"  {'   --' if best is None else f'{best:6.2f}'}"
            f"  {item['label']:<14} {item['question'][:56]}"
        )

    report(scored_items)


def report(scored_items: list[tuple[float | None, dict]]) -> None:
    out = sorted(c for c, item in scored_items if item["label"] == OUT_OF_CORPUS and c is not None)
    inc = sorted(c for c, item in scored_items if item["label"] == IN_CORPUS and c is not None)

    print("\n" + "=" * 72)
    print(f"{'':16}{'n':>4}{'min':>10}{'median':>10}{'max':>10}")
    for name, group in ((OUT_OF_CORPUS, out), (IN_CORPUS, inc)):
        if group:
            print(
                f"{name:16}{len(group):>4}{group[0]:>10.4f}"
                f"{group[len(group) // 2]:>10.4f}{group[-1]:>10.4f}"
            )

    if not out or not inc:
        return

    print("\nSeparation")
    print(f"  highest out-of-corpus score : {out[-1]:.4f}")
    print(f"  lowest in-corpus score      : {inc[0]:.4f}")

    if out[-1] < inc[0]:
        floor = (out[-1] + inc[0]) / 2
        print(f"  the labels do not overlap; midpoint floor = {floor:.4f}")
    else:
        overlap_in = [c for c in inc if c <= out[-1]]
        overlap_out = [c for c in out if c >= inc[0]]
        print(
            f"  the labels OVERLAP: {len(overlap_in)} in-corpus at or below the "
            f"highest out-of-corpus, {len(overlap_out)} out-of-corpus at or above "
            f"the lowest in-corpus"
        )
        print("\n  Best achievable floor, by cost:")
        candidates = sorted({*out, *inc})
        for threshold in candidates:
            wrongly_refused = sum(1 for c in inc if c < threshold)
            wrongly_admitted = sum(1 for c in out if c >= threshold)
            print(
                f"    floor {threshold:.4f}: refuses {wrongly_refused} answerable, "
                f"admits {wrongly_admitted} out-of-corpus"
            )


if __name__ == "__main__":
    main()
