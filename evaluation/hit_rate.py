"""Retrieval hit rate: did the retrieved text literally contain the answer?

The cheapest honest retrieval metric in the project. No LLM, no judge, no
embedding — it asks whether the numbers a correct answer must contain appear
in the passages that were retrieved, and nothing else.

It lives here, rather than inside either script that reports it, because both
do: `evaluation.run_evaluation` reports it beside faithfulness and answer
relevancy so one command covers all three, and `scripts.eval_retrieval_free`
reports it as its headline at roughly one twenty-thousandth of the cost. The
free screen is only useful while it tracks the paid run, and two copies of
this arithmetic would drift the first time either was edited.

**What it is good for.** A retrieval change that drops hit rate has broken
something; that direction is trustworthy. **Where it lies:** a figure can
appear in a retrieved chunk in a sentence that has nothing to do with the
question, so it over-credits, and it cannot see a correct answer phrased
differently from the ground truth, so it under-credits interpretive questions.
It is a screen, not a verdict, and its absolute value is not comparable to
RAGAS context recall — 0.242 against 0.386 on the same corpus, because this is
the stricter literal test.

**One under-credit is systematic and worth naming.** A ground truth writing
"15.0%" can never match this corpus, because the EDGAR parser flattens a table
row into pipe-separated cells and the percent sign lands in its own cell:
`Return on average common equity | 15.0 | %`. So every percentage in a ground
truth is a guaranteed miss, and any question whose answer is a rate scores
lower here than it deserves. Left as it is on purpose — normalising it now
would change every historical number this screen has produced, and the screen
is only useful while runs remain comparable to each other. Read percentages as
a known floor rather than a measurement.
"""

import re

# A figure distinctive enough to be evidence of retrieval. Standalone years are
# excluded: "2025" appears in every chunk of a 2025 filing, so counting it would
# score every question a hit regardless of what came back. Anything under three
# digits goes too — "1" and "13" match inside longer numbers and inside prose.
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


def hit(figure: str, blob: str) -> bool:
    """Is this figure present in the retrieved text?

    Matched with and without thousands separators, because a filing may write
    58,283 where a ground truth wrote 58283.
    """
    return figure in blob or figure.replace(",", "") in blob


def hit_rate(ground_truth: str, contexts: list[str]) -> float | None:
    """Fraction of the ground truth's figures present in the retrieved text.

    None when the ground truth carries no extractable figure — about a fifth
    of the answerable questions, mostly interpretive ones whose answer is
    prose. Scoring those 0.0 would report a retrieval failure where there was
    simply nothing numeric to find, which is why they are excluded from the
    mean rather than counted against it.
    """
    wanted = figures(ground_truth or "")
    if not wanted:
        return None
    blob = "\n".join(contexts or [])
    return sum(1 for figure in wanted if hit(figure, blob)) / len(wanted)
