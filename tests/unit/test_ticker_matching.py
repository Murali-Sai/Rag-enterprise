"""Which surface forms may stand in for a company, and which may not.

`detect_entities` decides whether a question gets scoped to one filing at the
database level, so a false positive here does not degrade an answer — it
answers a different company's question with a full set of confident,
well-formed chunks. That is how Goldman Sachs scored 0.000 context recall for
twelve runs, and nothing in the retrieval tests noticed.

The registry is five companies with three- to four-character tickers today, so
none of this is a live bug. It becomes one the moment a sixth is added, and the
obvious candidates are the dangerous ones: Citigroup trades as `C` and Ford as
`F`. These tests simulate that addition rather than waiting for it.
"""

import re

import pytest

from src.retrieval import entities
from src.retrieval.entities import (
    MIN_TICKER_MATCH_LENGTH,
    detect_entities,
    unmatchable_tickers,
)

# Text a one-character ticker pattern would falsely claim. All of these are
# ordinary financial prose; none of them mentions Citigroup or Ford.
SINGLE_LETTER_TRAPS = (
    "What does the C-suite compensation disclosure say?",
    "What is disclosed in Schedule C?",
    "How many Class C shares are outstanding?",
    "What are the terms of the Series F preferred stock?",
)
# "Item 7C" is deliberately not in that list: `\bC\b` does not match it,
# because `7` and `C` are both word characters and there is no boundary
# between them. The in-test guard below caught that when it was.


class TestTheRegistryIsMatchable:
    def test_every_company_can_be_detected_somehow(self):
        """A company with a too-short ticker and no alias is invisible.

        Every question about it would fall through to an unfiltered search of
        the whole corpus, which is the defect this module exists to prevent.
        """
        assert unmatchable_tickers() == ()

    def test_each_shipped_ticker_is_found_by_its_symbol(self):
        for ticker in ("AAPL", "MSFT", "JPM", "GS", "TSLA"):
            assert detect_entities(f"What was {ticker} revenue?") == (ticker,)

    def test_each_shipped_company_is_found_by_name(self):
        named = {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "JPMorgan": "JPM",
            "Goldman Sachs": "GS",
            "Tesla": "TSLA",
        }
        for name, ticker in named.items():
            assert detect_entities(f"What was {name}'s revenue?") == (ticker,)


class TestShortTickersAreNotMatchedAsSymbols:
    def test_the_floor_is_two_characters(self):
        """Pinned because "GS" depends on it. Raising this to 3 would make
        Goldman undetectable by symbol; lowering it to 1 is what these tests
        exist to prevent."""
        assert MIN_TICKER_MATCH_LENGTH == 2

    @pytest.mark.parametrize("ticker", ["C", "F"])
    def test_a_one_character_ticker_gets_no_symbol_pattern(self, ticker, monkeypatch):
        """Simulates adding Citigroup or Ford.

        Rebuilds the module's pattern tables against an extended registry, the
        same way import does, and asserts the short ticker is excluded from the
        symbol patterns while the long ones survive.
        """
        registry = dict(entities.COMPANY_REGISTRY)
        registry[ticker] = {"cik": "0000000000", "name": "Test Co"}

        patterns = {
            symbol: re.compile(rf"\b{re.escape(symbol)}\b")
            for symbol in registry
            if len(symbol) >= MIN_TICKER_MATCH_LENGTH
        }

        assert ticker not in patterns
        assert "GS" in patterns

    @pytest.mark.parametrize("query", SINGLE_LETTER_TRAPS)
    def test_ordinary_prose_is_not_read_as_a_company(self, query):
        """The reason the floor exists.

        `\\bC\\b` matches 18 times in the current corpus and `\\bF\\b` 12,
        against zero for every multi-character candidate. Under a
        one-character pattern each of these questions would be silently scoped
        to a filing it never mentioned.
        """
        for pattern in (re.compile(r"\bC\b"), re.compile(r"\bF\b")):
            if pattern.search(query):
                break
        else:
            pytest.fail(f"trap no longer contains a bare C or F: {query!r}")

        # No company is detected, so retrieval stays unscoped rather than being
        # pinned to the wrong filing.
        assert detect_entities(query) == ()


class TestARealCompanyStillWins:
    def test_a_named_company_is_found_alongside_a_bare_letter(self):
        """The floor must not cost a detection that should happen."""
        assert detect_entities("What does Apple's C-suite disclosure say?") == ("AAPL",)

    def test_order_of_mention_survives(self):
        assert detect_entities("Compare Tesla and Apple on Class C shares") == ("TSLA", "AAPL")
