"""Regenerate eval ground truths from full filing text — not from retrieval.

The previous generator (generate_ground_truths.py) asked the LLM to answer
each question using the top-20 chunks returned by *this project's own
retriever*. That makes the eval self-referential: the reference answer is
built from retrieval output, then used to score retrieval. Three concrete
consequences:

  1. Context Recall degenerates toward "does top-5 match my own top-20",
     not "did we find what the filing actually says".
  2. Any section the retriever systematically misses is invisible to the
     ground-truth generator too, so that blind spot can never be detected.
  3. When retrieval failed outright, the failure was frozen as the correct
     answer — four of twenty ground truths read "The provided filing
     excerpts do not contain enough information to answer", for questions
     whose answers are plainly in sections we index (e.g. Goldman's market
     risk disclosures, which are Item 7A).

This version loads the complete parsed sections for the filing(s) a question
targets and answers from those. Retrieval is never invoked, so the reference
answer is independent of the system under test.

Cost: roughly 40-80k input tokens per question on gpt-4o-mini (~$0.15 total
for the set) — the sections are large but the model's context is larger.

Usage:
    python scripts/generate_ground_truths_from_filings.py
    python scripts/generate_ground_truths_from_filings.py --limit 3   # smoke test
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI  # noqa: E402

from src.config import settings  # noqa: E402
from src.edgar.client import COMPANY_REGISTRY  # noqa: E402
from src.edgar.loader import load_filing_from_disk  # noqa: E402

# A window sits comfortably inside gpt-4o-mini's 128k-token context (roughly
# 4 chars/token for filing prose, so 240k chars is ~60k tokens plus prompt).
#
# Fitting the context is not the same as being read carefully. At 240k chars
# the model reliably finds dense material (financial tables, segment names,
# figures) but misses diffuse material: asked "what legal proceedings are
# disclosed by JPMorgan", it returned NOT_IN_WINDOW for a window containing
# five occurrences of "legal proceedings" and the Note 30 — Litigation
# heading. Smaller windows recover it, at proportionally more calls and cost.
#
# 240k stays the default because it is affordable across the whole set;
# --window-chars narrows it for questions where the material is discursive
# rather than tabular. Anything that still comes back NOT_IN_FILING at a
# smaller window is far more likely to be genuinely absent.
WINDOW_CHARS = 240_000
WINDOW_OVERLAP_CHARS = 5_000  # so an answer straddling a boundary isn't lost

NOT_IN_WINDOW = "NOT_IN_WINDOW"

# Two prompts, one per stage, each naming only its own sentinel. Sharing one
# prompt meant an empty window replied NOT_IN_FILING while the caller checked
# for NOT_IN_WINDOW, so the sentinel survived as a "finding" and poisoned the
# synthesis input — every question spanning many mostly-irrelevant windows came
# back unanswerable.
EXTRACTION_SYSTEM_PROMPT = f"""You extract source material for a retrieval evaluation dataset.

You will be given a question and one window of text from an SEC 10-K filing.

Rules:
- Extract what the text says that bears on the question: figures, segment names, \
dates, specific disclosures. Quote numbers exactly as written.
- Partial material is valuable. Other windows, and the other companies' filings, are \
being read separately and the results combined, so do not withhold something because \
it is incomplete on its own.
- If the question compares two companies, this window covers only one of them. Extract \
what it says about the company it does cover; the comparison is assembled later. Never \
answer that a comparison cannot be made.
- Reply with exactly {NOT_IN_WINDOW} — and nothing else — only if this window \
contains nothing bearing on the question.
- Never reply NOT_IN_FILING; you are seeing one window, not the filing."""

SYNTHESIS_SYSTEM_PROMPT = """You are building reference answers for a retrieval evaluation dataset.

You will be given a question and material extracted from across one or more SEC 10-K \
filings. Write the reference answer.

Rules:
- Be factual and specific. Include exact figures, segment names, and dates.
- 2-4 sentences.
- If the question asks for a comparison, cover both companies.
- Do not hedge, do not add disclaimers, do not mention windows or excerpts.

Your answer becomes the reference that a retrieval system is scored against, so it \
must reflect what the filing actually says — not what a retrieval system happened to \
surface."""


def _tickers_for(question: dict) -> list[str]:
    """Which filings a question targets, from its source_filing field."""
    source = question.get("source_filing", "")
    found = [t for t in COMPANY_REGISTRY if re.search(rf"\b{t}\b", source)]
    return found or list(COMPANY_REGISTRY)  # fall back to everything


_section_cache: dict[str, str] = {}


def _full_filing_text(ticker: str) -> str:
    """Every parsed section of a ticker's filing, concatenated."""
    if ticker in _section_cache:
        return _section_cache[ticker]

    ticker_dir = Path(settings.edgar_data_dir) / ticker
    parts: list[str] = []
    for html_file in sorted(ticker_dir.glob("*.html")):
        for doc in load_filing_from_disk(html_file):
            meta = doc.metadata
            header = f"=== {meta['ticker']} {meta['filing_type']} — {meta['section_name']} ==="
            parts.append(f"{header}\n{doc.page_content}")

    text = "\n\n".join(parts)
    _section_cache[ticker] = text
    return text


def _windows(text: str, window_chars: int = WINDOW_CHARS) -> list[str]:
    """Split a filing into overlapping windows that each fit the context.

    Filings run to 1.2M characters once incorporated-by-reference material is
    included — far past any context window. Truncating to the first N chars
    would silently drop the financial statements, which sit at the end, and
    the resulting ground truth would look fine while being wrong. Scanning
    every window is slower but leaves no part of the filing unexamined.
    """
    if len(text) <= window_chars:
        return [text]

    step = window_chars - WINDOW_OVERLAP_CHARS
    return [text[i : i + window_chars] for i in range(0, len(text), step)]


def _ask_window(question: str, window: str, llm: ChatOpenAI) -> str | None:
    # Ask for relevant *material*, not a complete answer. A window covering one
    # company can never answer "compare JPMorgan and Goldman Sachs", so demanding
    # a full answer per window made every comparative question come back
    # unanswerable even though both filings were scanned. Extraction here,
    # composition in the synthesis step.
    prompt = (
        f"Question: {question}\n\n"
        f"Filing text:\n{window}\n\n"
        f"Extract everything in this text that is relevant to the question — figures, "
        f"segment names, dates, specific disclosures. It does not need to fully answer "
        f"the question on its own; partial material is useful and will be combined with "
        f"other parts of the filing. Reply with exactly {NOT_IN_WINDOW} only if there is "
        f"nothing relevant here at all."
    )
    response = llm.invoke(
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    answer = response.content.strip()
    # Belt and braces: drop any sentinel the model emits, whichever it picks.
    # A sentinel that slips through is indistinguishable from real material to
    # the synthesis step, and enough of them make it declare the whole filing
    # silent on the question.
    if not answer or NOT_IN_WINDOW in answer or "NOT_IN_FILING" in answer:
        return None
    return answer


def generate(
    question: str, tickers: list[str], llm: ChatOpenAI, window_chars: int = WINDOW_CHARS
) -> tuple[str, int]:
    """Answer from an exhaustive scan of the filing(s). Returns (answer, windows_scanned)."""
    findings: list[str] = []
    scanned = 0

    for ticker in tickers:
        for window in _windows(_full_filing_text(ticker), window_chars):
            scanned += 1
            found = _ask_window(question, window, llm)
            if found:
                findings.append(f"[{ticker}] {found}")

    if not findings:
        return "NOT_IN_FILING", scanned

    # Always synthesize: windows return extracted material, not finished
    # answers, so even a single finding needs composing into a reference answer.
    # Multiple findings additionally need reconciling rather than concatenating,
    # which would produce repetition and contradictions.
    synthesis = llm.invoke(
        [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Material extracted from different parts of the filing(s):\n"
                    + "\n\n".join(findings)
                    + "\n\nWrite one consolidated answer to the question (2-4 sentences). "
                    "Prefer the most specific figures. If findings conflict, prefer the "
                    "one that reads as the primary financial statement rather than a "
                    "summary. Reply with exactly NOT_IN_FILING only if none of this "
                    "material bears on the question at all."
                ),
            },
        ]
    )
    return synthesis.content.strip(), scanned


PENDING_PREFIX = "PENDING"


def main(
    input_path: str,
    output_path: str,
    limit: int | None,
    only: str | None,
    window_chars: int,
    pending_only: bool = False,
) -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key, temperature=0.0)

    with open(input_path) as f:
        questions = json.load(f)

    if limit:
        questions = questions[:limit]

    # --only regenerates a subset in place, leaving every other answer as it
    # was. A question that failed at the default window size costs a few cents
    # to retry at a smaller one; regenerating all twenty to fix one costs the
    # whole run again, which in practice means it doesn't get retried.
    targets = questions
    if only:
        targets = [q for q in questions if only.lower() in q["question"].lower()]
        if not targets:
            raise SystemExit(f"No question matches --only {only!r}")

    # Questions whose expected behaviour is a refusal are skipped. Scanning a
    # filing for an answer the question was written not to have is a
    # guaranteed NOT_IN_FILING at full cost — and for the out-of-corpus and
    # RBAC ones there is no filing to scan, so _tickers_for falls back to all
    # five and the scan is at its most expensive precisely where it is most
    # pointless. Their ground truths are hand-written and say why the question
    # cannot be answered, which is the useful thing to record; they are never
    # sent to RAGAS anyway (see evaluation/run_evaluation.py NO_ANSWER_SCORING).
    refusals = [q for q in targets if q.get("expected_behavior") == "refuse"]
    if refusals:
        targets = [q for q in targets if q.get("expected_behavior") != "refuse"]
        print(
            f"Skipping {len(refusals)} question(s) expecting a refusal "
            "— ground truth hand-written"
        )

    # Growing the set is the normal case now, and regenerating the questions
    # that already have a reference answer is worse than wasteful: it costs the
    # whole run again *and* it silently replaces the ground truths that
    # published results were scored against, so those results stop being
    # reproducible from the file that claims to hold their references.
    if pending_only:
        targets = [q for q in targets if q.get("ground_truth", "").startswith(PENDING_PREFIX)]
        if not targets:
            raise SystemExit("No questions are pending a ground truth.")

    print(f"Regenerating {len(targets)} ground truth(s) from full filing text")
    print(f"Source: {input_path}\nOutput: {output_path}\nWindow: {window_chars:,} chars\n")

    not_in_filing = 0
    for i, item in enumerate(targets, 1):
        tickers = _tickers_for(item)
        print(f"[{i}/{len(targets)}] {item['question'][:64]}...  ({'+'.join(tickers)})", flush=True)
        try:
            answer, scanned = generate(item["question"], tickers, llm, window_chars)
            print(f"    scanned {scanned} window(s)", flush=True)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        # Preserve the retrieval-derived answer for comparison rather than
        # overwriting it — the delta between the two is itself evidence of
        # how much the old ground truths were shaped by retrieval. Only on the
        # first pass: re-running over an already-regenerated file would
        # otherwise overwrite that provenance with another filing-derived
        # answer, and the comparison would be lost.
        if item.get("ground_truth_source") != "full_filing_text":
            item["ground_truth_from_retrieval"] = item.get("ground_truth")
        item["ground_truth"] = answer
        item["ground_truth_source"] = "full_filing_text"
        item["ground_truth_window_chars"] = window_chars

        if answer == "NOT_IN_FILING":
            not_in_filing += 1
            print("    NOT_IN_FILING — question is not answerable from this filing")
        else:
            print(f"    -> {answer[:110]}{'...' if len(answer) > 110 else ''}")

    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nSaved {len(questions)} to {output_path}")
    if not_in_filing:
        print(
            f"{not_in_filing} question(s) came back unanswerable. Before dropping one, retry it "
            f"at a smaller window (--only '<substring>' --window-chars 60000): at the default "
            f"size the model misses material that is present but diffuse. Only drop or rewrite "
            f"what still fails there — scoring against a question the filing does not answer "
            f"rewards the retriever for failing too."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluation/datasets/eval_questions_v4.json")
    parser.add_argument("--output", default="evaluation/datasets/eval_questions_v4.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only",
        default=None,
        help="Regenerate only questions containing this substring, in place. "
        "Reads and writes the same file when --input and --output match.",
    )
    parser.add_argument(
        "--window-chars",
        type=int,
        default=WINDOW_CHARS,
        help=f"Scan window size (default {WINDOW_CHARS:,}). Smaller finds diffuse "
        "material the default misses, at proportionally more calls.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help=f"Only questions whose ground_truth still starts with {PENDING_PREFIX}. "
        "Use when growing the set: leaves existing reference answers — and the "
        "results already scored against them — untouched.",
    )
    args = parser.parse_args()
    main(args.input, args.output, args.limit, args.only, args.window_chars, args.pending_only)
