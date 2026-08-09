"""Diff two evaluation runs, per metric and per question.

Two uses, and they are the same arithmetic:

  * **Noise floor.** Run the same config twice and diff. The spread is what a
    delta has to clear before it means anything, and it is per-metric rather
    than one number — retrieval is deterministic given a fixed index, so
    context precision barely moves, while faithfulness compounds a
    non-deterministic generator with a non-deterministic judge and swings an
    order of magnitude further.
  * **Ablation.** Change one setting and diff. The per-question table is the
    part worth reading: an aggregate cannot distinguish a broad regression
    from two queries collapsing, and those call for opposite responses.

Refuses to compare runs measured against different corpora or scored on
different question sets. That mismatch is what made an earlier generation of
results unreadable — a config tag alone cannot tell two runs apart when the
index changed underneath them, and the difference reads as noise.

Usage:
    python scripts/compare_eval_runs.py \
        evaluation/results/eval_A.json evaluation/results/eval_B.json
"""

import argparse
import json
from pathlib import Path

# Config keys whose difference makes a comparison meaningless rather than
# interesting. Everything else is a knob an ablation is allowed to move.
INCOMPARABLE_KEYS = ("eval_questions", "judge_model", "judge_provider", "judge_embedding_model")


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _compare_preconditions(a: dict, b: dict, force: bool) -> None:
    problems = []

    corpus_a, corpus_b = a.get("corpus"), b.get("corpus")
    if corpus_a != corpus_b:
        problems.append(f"different corpus: {corpus_a} vs {corpus_b}")

    for key in INCOMPARABLE_KEYS:
        left, right = a["config"].get(key), b["config"].get(key)
        if left != right:
            problems.append(f"different {key}: {left!r} vs {right!r}")

    if not problems:
        return
    message = "These runs are not comparable:\n" + "\n".join(f"  - {p}" for p in problems)
    if not force:
        raise SystemExit(f"{message}\n\nPass --force to diff them anyway.")
    print(f"{message}\n(forced)\n")


def _config_delta(a: dict, b: dict) -> dict:
    keys = set(a["config"]) | set(b["config"])
    return {
        k: (a["config"].get(k), b["config"].get(k))
        for k in sorted(keys)
        if a["config"].get(k) != b["config"].get(k)
    }


def _scores(run: dict) -> dict[str, float]:
    return {k: v for k, v in run["scores"].items() if isinstance(v, int | float)}


def _rows(run: dict) -> dict[str, dict]:
    return {row["question"]: row for row in run.get("per_question_scores", [])}


def main(path_a: str, path_b: str, force: bool, top: int) -> None:
    a, b = load(path_a), load(path_b)
    _compare_preconditions(a, b, force)

    print(f"A  {Path(path_a).name}   ({a['timestamp']})")
    print(f"B  {Path(path_b).name}   ({b['timestamp']})")

    config_delta = _config_delta(a, b)
    if config_delta:
        print("\nConfig differences (B relative to A):")
        for key, (left, right) in config_delta.items():
            print(f"  {key}: {left!r} -> {right!r}")
    else:
        print("\nIdentical config — the spread below is the run-to-run noise floor.")

    scores_a, scores_b = _scores(a), _scores(b)
    metrics = [m for m in sorted(set(scores_a) | set(scores_b)) if not m.endswith("_n")]

    print(f"\n{'metric':28}{'A':>10}{'B':>10}{'B - A':>10}")
    print("-" * 58)
    for metric in metrics:
        left, right = scores_a.get(metric), scores_b.get(metric)
        if left is None or right is None:
            print(f"{metric:28}{_fmt(left):>10}{_fmt(right):>10}{'-':>10}")
            continue
        print(f"{metric:28}{left:>10.3f}{right:>10.3f}{right - left:>+10.3f}")

    rows_a, rows_b = _rows(a), _rows(b)
    shared = [q for q in rows_a if q in rows_b]
    if not shared:
        return

    # Per question, on the metric that moves most. Reported because a mean
    # over 50 questions cannot say whether a delta is broad or concentrated,
    # and only one of those is worth acting on.
    for metric in metrics:
        moved = [
            (rows_b[q][metric] - rows_a[q][metric], q)
            for q in shared
            if rows_a[q].get(metric) is not None and rows_b[q].get(metric) is not None
        ]
        moved = [pair for pair in moved if abs(pair[0]) > 1e-9]
        if not moved:
            continue
        moved.sort(key=lambda pair: abs(pair[0]), reverse=True)
        print(f"\n  {metric} — {len(moved)} of {len(shared)} questions moved, largest first:")
        for delta, question in moved[:top]:
            print(f"    {delta:+.3f}  {question[:64]}")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Compare anyway despite a corpus or question-set mismatch.",
    )
    parser.add_argument("--top", type=int, default=5, help="Questions to list per metric.")
    args = parser.parse_args()
    main(args.run_a, args.run_b, args.force, args.top)
