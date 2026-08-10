"""Scoping a comparative question down to one company.

A cross-encoder scores "does this passage answer this query?", so a chunk from
Apple's filing judged against "compare Apple and Tesla" is a half-answer and is
scored like one. Measured across the suite's comparatives, that put every
chunk at a relevance between 0.03 and 0.0008 and tripped the
insufficient-context gate on questions retrieval had answered correctly.

These pin the rewrite that fixes it. The interesting cases are not the ones
where a company name disappears — they are the ones where something *else*
nearly does.
"""

from src.retrieval.entities import detect_entities, scope_query


def scoped(query: str, keep: str) -> str:
    return scope_query(query, keep, detect_entities(query))


class TestScopingAComparative:
    def test_it_drops_the_other_company_and_its_conjunction(self):
        query = "Compare how Apple and Tesla describe their dependence on single-source suppliers."

        assert (
            scoped(query, "AAPL")
            == "how Apple describe their dependence on single-source suppliers."
        )
        assert (
            scoped(query, "TSLA")
            == "how Tesla describe their dependence on single-source suppliers."
        )

    def test_it_works_when_the_kept_company_is_named_second(self):
        query = "Contrast the climate and environmental risk disclosures made by Tesla and Apple."

        assert (
            scoped(query, "AAPL") == "the climate and environmental risk disclosures made by Apple."
        )

    def test_an_unrelated_and_survives(self):
        """The failure that motivated taking the conjunction *with* the name.

        Stripping company names first and tidying stray conjunctions afterwards
        cannot tell "Tesla and Apple" from "climate and environmental", and an
        earlier version produced 'the climate environmental risk disclosures
        made by and Apple.'
        """
        query = "Contrast the climate and environmental risk disclosures made by Tesla and Apple."

        assert "climate and environmental" in scoped(query, "TSLA")
        assert " and Apple" not in scoped(query, "TSLA")
        assert not scoped(query, "TSLA").endswith("and")

    def test_it_strips_the_comparative_opener(self):
        """`How do X and Y differ` scoped to X still opens with `How do`, which
        reads as a question about a comparison that is no longer there."""
        query = "How do Microsoft and Tesla differ in their disclosed R&D priorities?"

        assert scoped(query, "MSFT").startswith("Microsoft")

    def test_it_handles_multi_word_aliases(self):
        query = "Compare JPMorgan and Goldman Sachs credit risk disclosures."

        assert scoped(query, "JPM") == "JPMorgan credit risk disclosures."
        assert scoped(query, "GS") == "Goldman Sachs credit risk disclosures."

    def test_three_companies_leave_only_the_one_kept(self):
        query = "Compare Apple, Microsoft and Tesla on research spending."
        result = scoped(query, "MSFT")

        assert "Microsoft" in result
        assert "Apple" not in result
        assert "Tesla" not in result


class TestWhenNotToScope:
    def test_a_single_company_question_is_untouched(self):
        """There is nothing to remove, and the cross-encoder already has the
        query it wants. Rewriting anyway would be a silent behaviour change on
        the 30 answerable questions that name exactly one company."""
        query = "What was Apple's total net revenue for its most recent fiscal year?"

        assert scoped(query, "AAPL") == query

    def test_a_question_naming_nobody_is_untouched(self):
        query = "What are the main risks disclosed in these filings?"

        assert scoped(query, "AAPL") == query

    def test_it_falls_back_rather_than_return_nothing(self):
        """A query that is nothing but company names rewrites to the empty
        string, and scoring against "" is worse than scoring against the
        original — an empty query makes the cross-encoder's output meaningless
        rather than merely pessimistic."""
        query = "Apple and Tesla"

        assert scoped(query, "AAPL") == "Apple"
        assert scope_query("Tesla", "AAPL", ("AAPL", "TSLA")) == "Tesla"
