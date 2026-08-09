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
