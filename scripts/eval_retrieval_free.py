"""Evaluate retrieval without an LLM, and therefore without a bill.

A full run of `evaluation.run_evaluation` costs about $0.80: generation, one
judge call per citation, and four RAGAS metrics per question. That is the right
price for a published baseline and the wrong price for the question "did this
retrieval change help?", which is asked far more often.

Everything here is computed by string matching and metadata equality. The only
spend is embedding the queries - about $0.00004 for the whole suite.

What it measures
----------------

**answer_hit_rate** - do the retrieved chunks literally contain the figures the
ground truth asserts? Ground truths are hand-written from the complete filing,
so their numbers are the ones a correct answer must cite. 79% of answerable
questions carry at least one extractable figure. This is the free analogue of
context recall: RAGAS asks a judge whether each ground-truth claim is
attributable to the context; this asks whether the number is in there at all.

**entity_purity** - what fraction of the top-k slots come from the company the
question is about. This is the measurement that would have caught the defect
where Goldman Sachs questions were answered off Tesla's filing and a fabricated
sample document, and it needs no judge because `ticker` is metadata.

**gate_fires** - how often the insufficient-context gate would decline, and the
retrieval-confidence distribution behind it.

Where it lies, and it does lie
------------------------------

A figure can appear in a retrieved chunk in an unrelated context - "14.6%"
occurs in plenty of sentences that are not about JPMorgan's CET1 ratio - so
answer_hit_rate over-credits. It also cannot see a correct answer phrased
differently from the ground truth, so it under-credits interpretive questions,
which is why they are reported separately.

So this is a screen, not a replacement. It is trustworthy in one direction: a
retrieval change that *drops* hit rate has almost certainly broken something. A
change that raises it has produced a hypothesis worth $0.80 to confirm.

That asymmetry is deliberate. This project once rejected a good fix because a
proxy metric - mean retrieval confidence - disagreed with it, and the proxy was
the thing that was wrong; the fix went on to produce the largest measured gain
in the repo. Proxies mislead. Use this to decide what to measure properly, not
to decide what is true.

    ./.venv/Scripts/python.exe -m scripts.eval_retrieval_free
    ./.venv/Scripts/python.exe -m scripts.eval_retrieval_free --json out.json
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from src.config import settings
from src.generation.confidence import retrieval_confidence
from src.retrieval.retriever import get_retriever, retrieve_scored

DATASET = Path("evaluation/datasets/eval_questions_v4.json")

# A figure worth matching on. Standalone years are excluded: "2025" appears in
# every chunk of a 2025 filing, so counting it would score every question a hit
# regardless of what was retrieved.
_FIGURE = re.compile(r"\d[\d,]*\.?\d*%?")
_YEAR = re.compile(r"^(19|20)\d{2}$")


def figures(text: str) -> list[str]:
    """Ground-truth numbers distinctive enough to be evidence of retrieval."""
    found = []
    for raw in _FIGURE.findall(text):
        token = raw.rstrip(".,")
        bare = token.replace(",", "").replace(".", "").rstrip("%")
        if len(bare) < 3 or _YEAR.match(token):
            continue
        found.append(token)
    return sorted(set(found))


def expected_tickers(item: dict) -> list[str]:
    """Tickers the question is about, from its source_filing field."""
    source = item.get("source_filing") or ""
    return sorted({match for match in re.findall(r"\b[A-Z]{1,5}\b", source) if match != "K"})


def hit(figure: str, blob: str) -> bool:
    """Is this figure present in the retrieved text?

    Matched with and without thousands separators, because a filing may write
    58,283 where a ground truth wrote 58283.
    """
    return figure in blob or figure.replace(",", "") in blob


def main() -> None:
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    answerable = [i for i in items if i.get("expected_behavior", "answer") == "answer"]

    rows = []
    for item in answerable:
        retriever = get_retriever(user_roles=set(item["access_roles"]))
        scored = retrieve_scored(retriever, item["question"])
        blob = "\n".join(s.document.page_content for s in scored)
        tickers = [s.document.metadata.get("ticker") for s in scored]

        wanted = expected_tickers(item)
        needles = figures(item["ground_truth"])
        found = [f for f in needles if hit(f, blob)]
        confidence = retrieval_confidence(scored)

        rows.append(
            {
                "question": item["question"],
                "type": item.get("question_type"),
                "expected_tickers": wanted,
                "figures_wanted": len(needles),
                "figures_found": len(found),
                "hit_rate": (len(found) / len(needles)) if needles else None,
                "entity_purity": (
                    sum(1 for t in tickers if t in wanted) / len(tickers)
                    if tickers and wanted
                    else None
                ),
                "confidence": confidence,
                "gate_fires": confidence is not None
                and confidence < settings.insufficient_context_threshold,
            }
        )

    report(rows)
    if "--json" in sys.argv:
        target = Path(sys.argv[sys.argv.index("--json") + 1])
        target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {target}")


def _mean(values: list) -> str:
    clean = [v for v in values if v is not None]
    return f"{sum(clean) / len(clean):.3f}" if clean else "n/a"


def report(rows: list[dict]) -> None:
    print(f"\n{len(rows)} answerable questions, retrieval only, no LLM calls")
    print(f"corpus: {settings.chroma_collection}   gate: {settings.insufficient_context_threshold}")

    print("\nOverall")
    print(f"  answer_hit_rate   {_mean([r['hit_rate'] for r in rows])}")
    print(f"  entity_purity     {_mean([r['entity_purity'] for r in rows])}")
    print(f"  gate fires on     {sum(r['gate_fires'] for r in rows)} of {len(rows)}")

    for key, label in (("type", "stratum"), (None, "company")):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if key:
                groups[row[key] or "untyped"].append(row)
            else:
                for ticker in row["expected_tickers"]:
                    groups[ticker].append(row)

        print(f"\n{label:16}{'n':>4}{'hit_rate':>11}{'purity':>10}{'gate':>7}")
        print("-" * 48)
        for name in sorted(groups):
            group = groups[name]
            print(
                f"{name:16}{len(group):>4}"
                f"{_mean([r['hit_rate'] for r in group]):>11}"
                f"{_mean([r['entity_purity'] for r in group]):>10}"
                f"{sum(r['gate_fires'] for r in group):>7}"
            )

    misses = [r for r in rows if r["hit_rate"] == 0 and r["figures_wanted"]]
    if misses:
        print(f"\nRetrieved none of the ground-truth figures ({len(misses)}):")
        for row in misses:
            print(f"  [{','.join(row['expected_tickers']):9}] {row['question'][:62]}")


if __name__ == "__main__":
    main()
