"""Does the model refuse out-of-corpus questions the gate lets through?

The retrieval-confidence gate is the cheap half of the refusal path: it
declines before an LLM call. The model is the expensive half. Whether the gate
can be lowered depends entirely on whether the half behind it holds, and the
54-question suite cannot answer that — all four of its `out_of_corpus`
questions score below the gate, so none of them has ever reached the model.

`scripts/calibrate_gate.py` shows that is an accident of those four. Ten of the
24 held-out out-of-corpus probes already clear the shipped 0.15 threshold, one
of them at 0.975, because the cross-encoder scores topic match and the corpus
covers CET1 ratios, net interest income and remaining performance obligations
for companies other than the one asked about.

This script runs the probe set through full generation with the gate open, and
reports how often the model declines on its own. Costs one generation per
question — about $0.30 for the out-of-corpus half at gpt-4o prices.

    ./.venv/Scripts/python.exe -m scripts.probe_refusal
    ./.venv/Scripts/python.exe -m scripts.probe_refusal in_corpus

Takes an optional label to probe; defaults to out_of_corpus.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from src.generation.answer import generate_grounded_answer
from src.generation.confidence import is_full_refusal, retrieval_confidence
from src.retrieval.retriever import get_retriever, retrieve_scored

DATASET = Path("evaluation/datasets/gate_calibration_v1.json")
PROBE_ROLES = {"research"}


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "out_of_corpus"
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    items = [item for item in dataset["items"] if item["label"] == label]

    print(f"Probing {len(items)} {label} questions with the gate open\n")
    refused = 0
    answered: list[tuple[float, str, str]] = []

    for item in items:
        retriever = get_retriever(user_roles=PROBE_ROLES)
        scored = retrieve_scored(retriever, item["question"])
        confidence = retrieval_confidence(scored) or 0.0

        # Gate fully open: every question reaches the model, so what this
        # measures is the model's own refusal behaviour rather than the gate's.
        with patch("src.generation.answer.settings.insufficient_context_threshold", -1.0):
            result = generate_grounded_answer(item["question"], scored, verify=False)

        declined = is_full_refusal(result.answer, result.parsed)
        refused += declined
        if not declined:
            answered.append((confidence, item["question"], result.answer))
        print(
            f"  conf={confidence:6.4f}  {'REFUSED ' if declined else 'ANSWERED'}"
            f"  {item['question'][:56]}"
        )

    print("\n" + "=" * 72)
    print(f"  {refused}/{len(items)} refused by the model alone ({refused / len(items):.1%})")

    if answered:
        print(f"\n  The {len(answered)} the model answered anyway:\n")
        for confidence, question, answer in sorted(answered, reverse=True):
            print(f"  [{confidence:.4f}] {question}")
            print(f"      {answer[:300].strip()}\n")


if __name__ == "__main__":
    main()
