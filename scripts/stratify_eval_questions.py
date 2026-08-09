"""Tag eval questions with a difficulty stratum.

An aggregate over 20 questions hides the mechanism. Retrieval techniques that
sharpen the query — BM25 fusion, HyDE — should help questions that hinge on a
literal term (a ticker, a line item, a figure) and hurt questions that need
broad coverage of a discursive section. Reporting one mean over both cannot
show that; reporting per stratum can, and it is what makes "hybrid costs
recall" an explanation rather than an observation.

Classification is hand-assigned rather than LLM-inferred: 54 questions is
still small enough to read, and a classifier would introduce another model's
judgment into the measurement chain the eval is trying to keep clean.

    exact_figure   wants a specific number stated in the filing
    interpretive   wants a description synthesized across a discussion
    comparative    spans two filings, needing coverage of both
    no_answer      a real 10-K topic these filers do not disclose
    out_of_corpus  a subject no filing in the corpus covers at all
    rbac_blocked   answerable content the asking role may not read
    ambiguous      more than one defensible reading

The last four arrived with v4 and split what "correct" means. For the first
three the system is right when it answers well; for the next three it is
right when it declines. `ambiguous` spans both — a question that names a
company but not a period has a defensible reading to answer and flag, while
one that names no company at all has none, so `expected_behavior` varies
within that stratum and is carried per question rather than per stratum.

The three refusal strata are kept apart rather than pooled into no_answer
because they fail at different stages and point at different fixes:
out_of_corpus should be caught by the retrieval gate before any LLM call,
no_answer only by the model reading on-topic chunks and finding the fact
missing, and rbac_blocked by the where-clause before either.

Usage:
    python scripts/stratify_eval_questions.py --input evaluation/datasets/eval_questions_v4.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Keyed by a distinctive substring of the question.
QUESTION_TYPES: dict[str, str] = {
    "total net revenue for its most recent fiscal year": "exact_figure",
    "risk factors related to supply chain": "interpretive",
    "describe its business segments": "interpretive",
    "total assets and CET1 capital ratio": "exact_figure",
    "primary credit risk factors": "interpretive",
    "describe its lines of business": "interpretive",
    "automotive revenue and total revenue": "exact_figure",
    "related to production and manufacturing": "interpretive",
    "Intelligent Cloud segment": "exact_figure",
    "cybersecurity and AI-related risk factors": "interpretive",
    "principal business segments": "interpretive",
    "market risk disclosures does Goldman Sachs provide including VaR": "exact_figure",
    "Compare the risk factor disclosures": "comparative",
    "How do Apple and Microsoft compare": "comparative",
    "regulatory risks do JPMorgan and Goldman Sachs face": "comparative",
    "competition in the electric vehicle market": "interpretive",
    "provision for credit losses": "exact_figure",
    "AI strategy and investments": "interpretive",
    "quantitative disclosures about market risk": "exact_figure",
    "legal proceedings are disclosed": "interpretive",
    # --- v4 ---
    "total revenue and operating income for fiscal year 2025": "exact_figure",
    "spend on research and development in fiscal 2025": "exact_figure",
    "More Personal Computing segment revenue": "exact_figure",
    "gross margin percentage for fiscal 2025": "exact_figure",
    "total operating expenses and net income attributable": "exact_figure",
    "total net revenues and return on average common": "exact_figure",
    "competition in the productivity and business processes": "interpretive",
    "human capital and employee programs": "interpretive",
    "energy generation and storage business": "interpretive",
    "intellectual property and patent protection": "interpretive",
    "dependence on single-source suppliers": "comparative",
    "disclosed research and development priorities": "comparative",
    "capital adequacy positions": "comparative",
    "exposure to foreign currency risk": "comparative",
    "climate and environmental risk disclosures": "comparative",
    # Plausible 10-K topics these filers do not disclose. Retrieval lands on
    # the right filing; the specific fact is simply not in it, so only the
    # model can catch these.
    "iPhone units did Apple ship": "no_answer",
    "average selling price per vehicle": "no_answer",
    "Azure revenue came from its OpenAI partnership": "no_answer",
    "total compensation paid to JPMorgan's chief executive": "no_answer",
    "employees work in Goldman Sachs' Platform Solutions": "no_answer",
    # Nothing in the corpus bears on these at all — the retrieval gate should
    # fire before any generation call is spent.
    "Netflix's total streaming revenue": "out_of_corpus",
    "did NVIDIA report": "out_of_corpus",
    "federal funds rate target range": "out_of_corpus",
    "Berkshire Hathaway describe its insurance float": "out_of_corpus",
    # The content is indexed and the answer exists; this role may not read it.
    # The whole information-barrier layer was previously covered by unit tests
    # and by nothing in the eval, because every question listed "admin".
    "internal trading desk procedures manual": "rbac_blocked",
    "internal AML/KYC procedures manual": "rbac_blocked",
    "internal credit risk management policy": "rbac_blocked",
    "net sales for the Americas region": "rbac_blocked",
    # More than one defensible reading. The first three name a company, so a
    # reading exists to answer and flag; the last three name none, so there is
    # nothing to choose between five filings.
    "Apple's revenue last year": "ambiguous",
    "Microsoft's cloud revenue": "ambiguous",
    "Tesla's margin in 2025": "ambiguous",
    "total revenue last year": "ambiguous",
    "the bank set aside for credit losses": "ambiguous",
    "the company's main risks": "ambiguous",
}


def classify(question: str) -> str | None:
    for marker, qtype in QUESTION_TYPES.items():
        if marker.lower() in question.lower():
            return qtype
    return None


def main(path: str) -> None:
    with open(path) as f:
        questions = json.load(f)

    unmatched = []
    counts: dict[str, int] = {}
    for item in questions:
        qtype = classify(item["question"])
        if qtype is None:
            unmatched.append(item["question"])
            continue
        item["question_type"] = qtype
        counts[qtype] = counts.get(qtype, 0) + 1

    with open(path, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"Tagged {len(questions) - len(unmatched)}/{len(questions)} questions in {path}")
    for qtype, n in sorted(counts.items()):
        print(f"  {qtype:14} {n}")
    if unmatched:
        print("\nUnmatched (add a marker to QUESTION_TYPES):")
        for q in unmatched:
            print(f"  - {q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluation/datasets/eval_questions_v4.json")
    args = parser.parse_args()
    main(args.input)
