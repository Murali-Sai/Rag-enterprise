"""Mean and spread across N runs of the same configuration.

`compare_eval_runs.py` diffs exactly two runs, which is the right shape for an
ablation and the wrong shape for a baseline. A baseline is a claim about where
a configuration sits, and the honest form of that claim is a mean with the
run-to-run spread printed next to it — this project has twice published a
single run as though it were a mean, and the second time cost $2.40 and a
retracted 19x claim to undo.

So this takes as many runs as you have and prints the table the README wants,
including the per-stratum breakdown and the spread that any reported delta has
to clear.

    ./.venv/Scripts/python.exe -m scripts.summarize_runs \
        evaluation/results/eval_A.json evaluation/results/eval_B.json ...

Refuses to average runs measured against different corpora or question sets,
for the same reason compare_eval_runs.py refuses to diff them: the fingerprint
is what makes two runs comparable, and averaging across a corpus change
produces a number describing no system that ever existed.
"""

import json
import sys
from pathlib import Path

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "citation_accuracy",
    "citation_coverage",
    "refusal_correctness",
)

STRATA = (
    "interpretive",
    "exact_figure",
    "comparative",
    "ambiguous",
    "no_answer",
    "out_of_corpus",
    "rbac_blocked",
)


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(run: dict) -> tuple:
    corpus = run.get("corpus") or {}
    config = run.get("config") or {}
    return (
        corpus.get("chunk_count"),
        corpus.get("content_digest"),
        config.get("eval_questions"),
        config.get("judge_model"),
    )


def mean_spread(values: list[float]) -> tuple[float, float]:
    return sum(values) / len(values), (max(values) - min(values))


def main(paths: list[str]) -> None:
    runs = [load(p) for p in paths]

    prints = {fingerprint(run) for run in runs}
    if len(prints) > 1:
        print("REFUSING: these runs do not describe the same measurement.\n")
        for path, run in zip(paths, runs, strict=True):
            chunks, digest, questions, judge = fingerprint(run)
            print(f"  {Path(path).name}: {chunks} chunks @ {digest}, {questions}, judge {judge}")
        sys.exit(1)

    chunks, digest, questions, judge = prints.pop()
    print(f"{len(runs)} runs - {chunks} chunks @ {digest}, {questions}, judge {judge}")
    for path in paths:
        print(f"    {Path(path).name}")

    print(f"\n{'metric':22}{'mean':>10}{'spread':>10}   per run")
    print("-" * 78)
    for metric in METRICS:
        values = [
            run["scores"][metric] for run in runs if run.get("scores", {}).get(metric) is not None
        ]
        if not values:
            continue
        mean, spread = mean_spread(values)
        each = "  ".join(f"{v:.4f}" for v in values)
        print(f"{metric:22}{mean:>10.4f}{spread:>10.4f}   {each}")

    per_stratum(runs)


def per_stratum(runs: list[dict]) -> None:
    """Refusal correctness and faithfulness by stratum, averaged across runs.

    Kept to those two because they are the pair that carries the argument: the
    three unanswerable strata have no faithfulness to report and are entirely
    a claim about refusal, while the answerable ones are the reverse.
    """
    print(f"\n{'stratum':16}{'n':>4}{'faithfulness':>14}{'refusal correctness':>22}")
    print("-" * 78)
    for stratum in STRATA:
        rows = [
            row
            for run in runs
            for row in run.get("per_question_scores", [])
            if row.get("question_type") == stratum
        ]
        if not rows:
            continue
        n = len(rows) // len(runs)
        faith = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
        refusal = [
            r["refusal_correctness"] for r in rows if r.get("refusal_correctness") is not None
        ]
        faith_text = f"{sum(faith) / len(faith):.3f}" if faith else "n/a"
        refusal_text = f"{sum(refusal) / len(refusal):.3f}" if refusal else "n/a"
        print(f"{stratum:16}{n:>4}{faith_text:>14}{refusal_text:>22}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
