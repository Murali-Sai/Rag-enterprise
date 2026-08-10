"""Which companies is this question about?

A comparative question is really two questions sharing one retrieval budget,
and a global top-k does not divide it. Asked to compare JPMorgan and Goldman
Sachs on credit risk, the ranker fills all five slots with whichever filing
scores better in aggregate, the model correctly reports it cannot compare,
and the eval scores that refusal 0.000 answer relevancy. All three
comparative questions in eval_questions_v3.json failed exactly this way, in
every retrieval configuration tried — it is a budget bug, not a ranking one.

Detection is a literal alias match rather than an LLM call or an NER model.
The corpus is five known companies; the set of surface forms an analyst uses
for them is small, closed, and writable by hand. Anything cleverer would put
another model's judgment in front of retrieval, and would have to be
re-measured every time that model changed.

Aliases are matched on word boundaries. Tickers match case-sensitively —
lowercase "gs" appears inside ordinary prose far too often, and a false
positive here silently halves the retrieval budget for a single-company
question.
"""

import re

from src.edgar.client import COMPANY_REGISTRY

# Surface forms per ticker, beyond the ticker itself. Ordered longest-first
# within each entry only for readability; matching is by regex alternation,
# which is anchored on word boundaries and so does not need the ordering.
COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("Apple",),
    "MSFT": ("Microsoft",),
    "JPM": ("JPMorgan Chase", "JPMorgan", "JP Morgan", "Chase"),
    "GS": ("Goldman Sachs", "Goldman"),
    "TSLA": ("Tesla",),
}

# Case-insensitive name patterns, and case-sensitive ticker patterns.
_NAME_PATTERNS: dict[str, re.Pattern[str]] = {
    ticker: re.compile(
        r"\b(?:" + "|".join(re.escape(alias) for alias in aliases) + r")\b",
        re.IGNORECASE,
    )
    for ticker, aliases in COMPANY_ALIASES.items()
}
_TICKER_PATTERNS: dict[str, re.Pattern[str]] = {
    ticker: re.compile(rf"\b{re.escape(ticker)}\b") for ticker in COMPANY_REGISTRY
}


def detect_entities(query: str) -> tuple[str, ...]:
    """Tickers named in the query, ordered by first mention.

    Order is preserved because it is the only signal available about which
    company the asker considers primary, and it decides which company's
    chunks land at the top of an otherwise tied merge.
    """
    positions: dict[str, int] = {}
    for ticker in COMPANY_REGISTRY:
        found = [
            match.start()
            for pattern in (_NAME_PATTERNS.get(ticker), _TICKER_PATTERNS[ticker])
            if pattern is not None
            for match in [pattern.search(query)]
            if match is not None
        ]
        if found:
            positions[ticker] = min(found)

    return tuple(sorted(positions, key=lambda t: positions[t]))


def entity_filter(ticker: str) -> dict:
    """A ChromaDB where-clause restricting retrieval to one company's filing."""
    return {"ticker": {"$eq": ticker}}


# Openers that only make sense while a question still names two companies.
# Stripped after the other company is removed, so the leg reads as a question
# about one filing rather than as half of a comparison.
_COMPARATIVE_LEAD = re.compile(
    r"^\s*(?:compare|contrast|how\s+do|how\s+does|what\s+are\s+the\s+differences\s+between|"
    r"differences\s+between|which\s+of)\b[\s,]*",
    re.IGNORECASE,
)

_CONJUNCTION = r"(?:and|or|versus|vs\.?)"


def scope_query(query: str, keep: str, entities: tuple[str, ...]) -> str:
    """The comparative question as it applies to one company.

    A cross-encoder scores "does this passage answer this query?", and no
    single chunk answers "compare A and B" — so every chunk of A's filing is
    judged as half an answer and scored accordingly. Measured on the four
    comparatives in the suite, reranking A's own chunks against the full
    question produced logits from -3.5 to -7.0, which the logistic in
    scores.py turns into relevances of 0.03 down to 0.0008. Retrieval had
    found the right passages; the scoring of them is what tripped the
    insufficient-context gate, and the eval counted that as a refusal of an
    answerable question.

    Scoring each leg against its own scoped question instead moved the merged
    confidence on "Apple and Tesla ... single-source suppliers" from 0.021 to
    0.569, and lifted the JPMorgan/Goldman comparative — which already
    passed — from 0.673 to 0.914.

    Mechanical on purpose. This module's opening argument is that entity
    handling stays a literal alias match rather than an LLM call or an NER
    model, because anything cleverer puts another model's judgment in front of
    retrieval and has to be re-measured whenever that model changes. Rewriting
    the query is exactly the place that temptation returns, so it is answered
    the same way: delete the other companies, delete the conjunction they left
    behind, delete the comparative opener.

    Returns the query unchanged when fewer than two companies are named —
    there is nothing to scope, and a single-company question is already the
    thing the cross-encoder wants.
    """
    if len(entities) < 2:
        return query

    scoped = query
    for ticker in entities:
        if ticker == keep:
            continue
        for alias in (*COMPANY_ALIASES.get(ticker, ()), ticker):
            name = re.escape(alias)
            # Take the conjunction with the name, in whichever order it sits,
            # so "climate and environmental" survives while "Tesla and Apple"
            # loses the right half. Removing names first and tidying stray
            # conjunctions afterwards cannot tell those two cases apart.
            ignore = re.IGNORECASE
            scoped = re.sub(rf"\b{name}\b\s*,?\s*{_CONJUNCTION}\s+", "", scoped, flags=ignore)
            scoped = re.sub(rf"\s*,?\s*{_CONJUNCTION}\s+\b{name}\b", "", scoped, flags=ignore)
            scoped = re.sub(rf"\b{name}\b", "", scoped, flags=ignore)

    scoped = _COMPARATIVE_LEAD.sub("", scoped)
    scoped = re.sub(r"\s{2,}", " ", scoped).strip(" ,")
    # If the rewrite ate the question, the original is the safer input: a
    # slightly depressed score beats scoring against an empty string.
    return scoped or query
