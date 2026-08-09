"""Tests for SEC EDGAR 10-K HTML parser."""

import pytest

from src.edgar.parser import (
    RUNNING_HEADER_MIN_REPEATS,
    FilingSection,
    _clean_text,
    _html_to_text,
    _page_number,
    _strip_page_furniture,
    _table_to_text,
    parse_10k_sections,
)

SAMPLE_10K_HTML = """
<html>
<head><title>10-K Filing</title></head>
<body>
<p>Table of Contents</p>
<p>PART I</p>
<p><b>Item 1. Business</b></p>
<p>ACME Corp is a global technology company that designs, manufactures, and markets
consumer electronics, computer software, and online services. The Company's products
include smartphones, personal computers, tablets, wearables, and accessories.
The Company was founded in 1976 and is headquartered in Cupertino, California.</p>

<p><b>Item 1A. Risk Factors</b></p>
<p>The Company faces significant risks related to global economic conditions,
competition in the technology industry, supply chain disruptions, and regulatory
changes across multiple jurisdictions. The Company's operations are subject to
risks associated with manufacturing in Asia and consumer demand fluctuations.
Additional risks include cybersecurity threats, intellectual property disputes,
and potential changes in tax legislation that could adversely affect operations.</p>

<p>PART II</p>
<p><b>Item 7. Management Discussion and Analysis</b></p>
<p>Total net revenue for fiscal year 2024 was $391.0 billion, an increase of 2%
compared to $383.3 billion in fiscal year 2023. The increase was driven primarily
by growth in Services revenue, partially offset by lower product revenue.</p>
<table>
<tr><th>Segment</th><th>Revenue 2024</th><th>Revenue 2023</th></tr>
<tr><td>Products</td><td>$295.0B</td><td>$293.1B</td></tr>
<tr><td>Services</td><td>$96.0B</td><td>$90.2B</td></tr>
</table>
<p>Operating income was $123.2 billion with an operating margin of 31.5%.</p>

<p><b>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</b></p>
<p>The Company is exposed to market risk from changes in interest rates, foreign
exchange rates, and equity prices. The Company uses derivatives to manage exposure
to foreign currency risk associated with certain transactions.</p>

<p><b>Item 8. Financial Statements and Supplementary Data</b></p>
<p>Consolidated Balance Sheet as of September 28, 2024:</p>
<table>
<tr><th>Item</th><th>Amount</th></tr>
<tr><td>Total Assets</td><td>$364.98B</td></tr>
<tr><td>Total Liabilities</td><td>$308.03B</td></tr>
<tr><td>Shareholders Equity</td><td>$56.95B</td></tr>
</table>
</body>
</html>
"""


class TestParse10KSections:
    def test_extracts_priority_sections(self):
        sections = parse_10k_sections(SAMPLE_10K_HTML, priority_only=True)
        section_ids = {s.section_id for s in sections}
        assert "item_1" in section_ids
        assert "item_1a" in section_ids
        assert "item_7" in section_ids

    def test_section_content_not_empty(self):
        sections = parse_10k_sections(SAMPLE_10K_HTML)
        for section in sections:
            assert section.char_count > 50
            assert len(section.content) > 50

    def test_section_names_populated(self):
        sections = parse_10k_sections(SAMPLE_10K_HTML)
        names = {s.section_name for s in sections}
        assert "Business" in names or any("Business" in n for n in names)

    def test_returns_filing_sections(self):
        sections = parse_10k_sections(SAMPLE_10K_HTML)
        assert all(isinstance(s, FilingSection) for s in sections)
        assert len(sections) >= 3  # At minimum: Item 1, 1A, 7

    def test_md_and_a_contains_revenue(self):
        sections = parse_10k_sections(SAMPLE_10K_HTML)
        mda = next((s for s in sections if s.section_id == "item_7"), None)
        assert mda is not None
        assert "391.0" in mda.content or "revenue" in mda.content.lower()

    def test_tables_preserved_in_text(self):
        sections = parse_10k_sections(SAMPLE_10K_HTML)
        mda = next((s for s in sections if s.section_id == "item_7"), None)
        assert mda is not None
        assert "Products" in mda.content or "Services" in mda.content


class TestSectionBoundaryEdgeCases:
    """Regressions for two real filings that lost most of their content.

    Both failed silently — sections were extracted, just tiny — so nothing
    errored and the gap only showed up when measuring extracted characters
    against the document's total text.
    """

    def test_repeated_page_headers_do_not_fragment_a_section(self):
        """Microsoft repeats "PART I Item 1 ..." as a running header on every
        page. Slicing to the next match made each page its own section and cut
        the filing from 355k characters to 27k."""
        # Each tag becomes its own line in _html_to_text, so the running header
        # lands at line start exactly as ITEM_PATTERN expects.
        body_pages = "".join(
            f"<p>PART I</p><p>Item 1. Business</p>"
            f"<p>Page {i} of substantive business "
            f"discussion covering products, segments, and competition in depth.</p>"
            for i in range(6)
        )
        html = f"<html><body>{body_pages}"
        html += (
            "<p>Item 1A. Risk Factors</p><p>" + ("Risk discussion. " * 30) + "</p></body></html>"
        )

        sections = parse_10k_sections(html)
        business = next(s for s in sections if s.section_id == "item_1")

        # All six pages belong to one section, not six sections of one page.
        assert sum(1 for s in sections if s.section_id == "item_1") == 1
        assert business.content.count("substantive business discussion") == 6

    def test_table_of_contents_entry_loses_to_the_real_body(self):
        html = (
            "<html><body>"
            "<p>Item 1. Business</p><p>Item 1A. Risk Factors</p>"  # adjacent TOC lines
            "<p>Item 1. Business</p><p>" + ("The Company designs and sells devices. " * 40) + "</p>"
            "<p>Item 1A. Risk Factors</p><p>" + ("Risks include competition. " * 40) + "</p>"
            "</body></html>"
        )

        sections = parse_10k_sections(html)
        business = next(s for s in sections if s.section_id == "item_1")

        assert "The Company designs and sells devices" in business.content
        assert business.char_count > 500

    def test_incorporated_material_after_last_header_is_recovered(self):
        """JPMorgan satisfies Items 7 and 8 by reference — the section body is a
        pointer and 1.03M characters of actual financials sit past every Item
        header, which header-delimited slicing can never reach."""
        pointer = "Management's discussion and analysis appears on pages 46-160."
        appended = "Provision for credit losses was $14,212 million. " * 3000  # ~145k chars

        html = (
            "<html><body>"
            "<p>Item 7. Management Discussion and Analysis</p><p>" + pointer + "</p>"
            "<p>Item 8. Financial Statements and Supplementary Data</p>"
            "<p>The Consolidated Financial Statements appear on pages 162-314.</p>"
            "<p>Item 15. Exhibits</p><p>" + appended + "</p>"
            "</body></html>"
        )

        sections = parse_10k_sections(html)
        recovered = [s for s in sections if s.section_id == "incorporated"]

        assert len(recovered) == 1
        assert "Provision for credit losses was $14,212 million" in recovered[0].content

    def test_ordinary_filing_tail_is_not_emitted_as_incorporated(self):
        """Most filings end in a short exhibit list and signature block. That
        must not be mistaken for incorporated material."""
        html = (
            "<html><body>"
            "<p>Item 1. Business</p><p>" + ("Business discussion. " * 40) + "</p>"
            "<p>Item 8. Financial Statements and Supplementary Data</p><p>"
            + ("Financial detail. " * 40)
            + "</p>"
            "<p>Exhibit index and signatures follow here.</p>"
            "</body></html>"
        )

        sections = parse_10k_sections(html)

        assert not any(s.section_id == "incorporated" for s in sections)


class TestPageFurniture:
    """Running headers and footers, using the strings the real filings print.

    These reached the index as chunks of their own — the recursive splitter
    cannot merge a 32-character line into a neighbour already near the 512
    budget — and then competed for top-k slots. Two of the five contexts
    retrieved for a question about Goldman's net revenues were page footers.
    """

    # Every one of these was copied out of a parsed filing in data/edgar/.
    @pytest.mark.parametrize(
        "furniture",
        [
            "Goldman Sachs 2025 Form 10-K | 1",  # odd pages: title then number
            "2 | Goldman Sachs 2025 Form 10-K",  # even pages: the other order
            "Goldman Sachs 2025 Form 10-K | 201",
            "Apple Inc. | 2025 Form 10-K | 2",  # title itself contains a divider
            "JPMorgan Chase & Co./2025 Form 10-K | 43",
            "44 | JPMorgan Chase & Co./2025 Form 10-K",
            "Table of Contents",
            "Table of contents",  # JPMorgan's casing
            "Page 150",
            "Page 12 of 148",
            "123",  # a page number with nothing else on the line
        ],
    )
    def test_furniture_line_is_removed(self, furniture):
        text = f"Net revenues were $53.51 billion.\n{furniture}\nRevenues increased 16%."
        result = _strip_page_furniture(text)

        assert furniture not in result
        assert "Net revenues were $53.51 billion." in result
        assert "Revenues increased 16%." in result

    # The precision guard. Filings cite themselves in ordinary prose, and a
    # rule keyed on "Form 10-K" alone would delete these sentences outright.
    @pytest.mark.parametrize(
        "prose",
        [
            "The Insider Trading Policy is filed as Exhibit 19 to this 2025 Form 10-K.",
            "For a description of the risks we face related to regulatory matters, refer to "
            "Risk Factors (Part I, Item 1A of this Form 10-K).",
            "We file or furnish periodic reports, including our Annual Reports on Form 10-K.",
            "Refer to Note 1 of the Notes to Financial Statements (Part II, Item 8 of this "
            "Form 10-K).",
        ],
    )
    def test_prose_mentioning_the_form_survives(self, prose):
        assert _strip_page_furniture(prose).strip() == prose

    def test_five_digit_number_is_data_not_a_page_number(self):
        """Page numbers stop at four digits; a longer figure came from a table."""
        assert "10250" in _strip_page_furniture("Total assets\n10250\nTotal liabilities")

    def test_nbsp_terminated_footer_still_matches(self):
        """A trailing non-breaking space is not [ \\t], so it would otherwise
        stop the footer matching its own end-of-line anchor. _clean_text
        normalises before stripping; this pins that ordering."""
        assert "Form 10-K" not in _clean_text(
            "Revenues rose.\nGoldman Sachs 2025 Form 10-K | 5\xa0"
        )

    def test_removal_does_not_leave_a_blank_line_behind(self):
        result = _clean_text(
            "First paragraph.\nGoldman Sachs 2025 Form 10-K | 7\nSecond paragraph."
        )
        assert "\n\n\n" not in result
        assert result == "First paragraph.\n\nSecond paragraph."

    def test_footer_is_stripped_from_a_real_section(self):
        html = (
            "<html><body>"
            "<p>Item 7. Management Discussion and Analysis</p>"
            "<p>Total net revenues were $53.51 billion for 2025. " + ("Detail. " * 40) + "</p>"
            "<p>Goldman Sachs 2025 Form 10-K | 123</p>"
            "<p>Return on average common shareholders' equity was 12.7%.</p>"
            "</body></html>"
        )

        sections = parse_10k_sections(html)
        mda = next(s for s in sections if s.section_id == "item_7")

        assert "Form 10-K | 123" not in mda.content
        assert "$53.51 billion" in mda.content
        assert "12.7%" in mda.content


class TestRunningHeaders:
    """Headers that repeat on every page but carry no page number.

    PAGE_FOOTER_PATTERNS cannot see these — there is no number to anchor on —
    so they survived into the corpus inline at the head of otherwise-good
    chunks: 446 chunks carried one, averaging 52.7 of the 512-character budget.

    Frequency alone cannot be the test, because the content worth keeping
    repeats just as hard. The guard is position: a running header sits directly
    below a page break and a table caption sits wherever its table sits.
    """

    @staticmethod
    def _pages(*, header: str, body: str, footer, count: int) -> str:
        """A run of pages, each footer, then header, then a line of body."""
        return "\n".join(
            f"{footer(page)}\n\n{header}\n{body} on page {page}." for page in range(20, 20 + count)
        )

    # Each layout was read off a filing in data/edgar/: Goldman prints a footer
    # with the page number, Microsoft prints the number alone, and Goldman
    # stacks two headers so the second sits two non-blank lines below the break.
    @pytest.mark.parametrize(
        ("header", "footer"),
        [
            (
                "THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES",
                lambda page: f"Goldman Sachs 2025 Form 10-K | {page}",
            ),
            (
                "Notes to consolidated financial statements",
                lambda page: f"{page} | JPMorgan Chase & Co./2025 Form 10-K",
            ),
            ("PART II", str),  # Microsoft: a bare page number is the break
            ("Item 8", str),
        ],
    )
    def test_running_header_is_removed(self, header, footer):
        text = self._pages(header=header, body="Net revenues rose", footer=footer, count=12)

        result = _strip_page_furniture(text)

        assert header not in result
        assert result.count("Net revenues rose") == 12

    def test_header_two_lines_below_the_break_is_removed(self):
        """Goldman stacks the company name above the section name, so the
        second header is two non-blank lines below the footer."""
        text = "\n".join(
            f"Goldman Sachs 2025 Form 10-K | {page}\n"
            "THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES\n"
            "Management\u2019s Discussion and Analysis\n"
            f"Business Environment in {page}."
            for page in range(60, 72)
        )

        result = _strip_page_furniture(text)

        assert "THE GOLDMAN SACHS GROUP" not in result
        assert "Management\u2019s Discussion and Analysis" not in result

    # The false positives this rule exists to avoid, with the counts they reach
    # in Goldman's filing. Every one of them is period context for a financial
    # table: delete "As of December 2025" and the table above it no longer says
    # which year it reports.
    @pytest.mark.parametrize(
        ("caption", "count"),
        [
            ("As of December 2025", 52),
            ("As of December 2024", 41),
            ("As of December", 71),
            ("Year Ended December", 62),
            ("In the table above:", 77),
            ("(In millions)", 32),
        ],
    )
    def test_repeated_table_caption_survives(self, caption, count):
        """A caption repeats as often as a header, but sits mid-page.

        The header is at the top of each page and the caption is down in the
        body with its table, which is the arrangement measured in the filings:
        every one of these captions clears the search window on at least 93% of
        its occurrences, against 0% for a running header.
        """
        text = "\n".join(
            f"Goldman Sachs 2025 Form 10-K | {page}\n"
            "THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES\n"
            "Management’s Discussion and Analysis\n"
            "Total assets grew across every segment we report.\n"
            "The following table presents our balance sheet.\n"
            f"{caption}\n"
            f"Total assets | {page * 1000}"
            for page in range(20, 20 + count)
        )

        result = _strip_page_furniture(text)

        assert result.count(caption) == count
        # ... and the header alongside it still goes.
        assert "THE GOLDMAN SACHS GROUP" not in result

    def test_table_fragment_below_a_figure_survives(self):
        """The reason the page numbers have to ascend.

        A split around an inline number leaves fragments like "million and $"
        directly below a bare number, which looks exactly like Microsoft's
        layout. The numbers above them are figures from a column, so they do
        not count upward, and that is what tells the two apart.
        """
        text = "\n".join(
            f"Total revenues\n{figure}\nmillion and $\n{figure + 7}"
            for figure in (940, 210, 655, 130, 480, 275, 820, 360, 705, 195)
        )

        assert _strip_page_furniture(text).count("million and $") == 10

    def test_occurrence_without_a_page_break_survives(self):
        """JPMorgan's glossary defines MD&A with the same words as its running
        header. Deciding per line would lose the definition, so the rule
        deletes occurrences rather than strings."""
        header = "Management\u2019s discussion and analysis"
        text = (
            self._pages(
                header=header,
                body="Consolidated results",
                footer=lambda page: f"JPMorgan Chase & Co./2025 Form 10-K | {page}",
                count=12,
            )
            + f"\nGlossary of Terms and Acronyms\nMD&A:\n{header}\nMeasurement alternative:"
        )

        result = _strip_page_furniture(text)

        assert result.count(header) == 1
        assert "MD&A:" in result

    def test_a_line_below_the_repeat_threshold_survives(self):
        header = "Notes to Consolidated Financial Statements"
        text = self._pages(
            header=header,
            body="Revenues",
            footer=lambda page: f"Goldman Sachs 2025 Form 10-K | {page}",
            count=RUNNING_HEADER_MIN_REPEATS - 1,
        )

        assert header in _strip_page_furniture(text)

    def test_long_repeated_line_is_prose_not_a_header(self):
        """A running header is short. Past the length bound the line is a
        sentence that happens to recur, and deleting it loses meaning."""
        sentence = (
            "We may not be able to fully realize the expected benefits of our "
            "restructuring plans, which could adversely affect our results."
        )
        text = self._pages(
            header=sentence,
            body="Risk detail",
            footer=lambda page: f"Goldman Sachs 2025 Form 10-K | {page}",
            count=12,
        )

        assert sentence in _strip_page_furniture(text)

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("Goldman Sachs 2025 Form 10-K | 63", 63),  # trailing page number
            ("64 | Goldman Sachs 2025 Form 10-K", 64),  # leading page number
            ("Page 12 of 148", 12),
            ("50", 50),
            ("THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES", None),
            ("As of December 2025", None),
        ],
    )
    def test_page_number_is_read_off_the_right_end(self, line, expected):
        """The 10 in "10-K" is not the page number, and neither is the year —
        which end holds it depends on which footer form matched."""
        assert _page_number(line) == expected


class TestHtmlToText:
    def test_strips_tags(self):
        text = _html_to_text("<p>Hello <b>world</b></p>")
        assert "Hello" in text
        assert "world" in text
        assert "<p>" not in text
        assert "<b>" not in text

    def test_removes_scripts(self):
        text = _html_to_text("<script>alert('xss')</script><p>Content</p>")
        assert "alert" not in text
        assert "Content" in text


class TestCleanText:
    def test_collapses_newlines(self):
        result = _clean_text("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_collapses_spaces(self):
        result = _clean_text("a     b")
        assert "     " not in result

    def test_removes_page_numbers(self):
        result = _clean_text("Content\n42\nMore content")
        assert result.strip() in ("Content\nMore content", "Content\n\nMore content")

    def test_strips_whitespace(self):
        result = _clean_text("  content  ")
        assert result == "content"


class TestTableToText:
    def test_basic_table(self):
        from bs4 import BeautifulSoup

        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        result = _table_to_text(table)
        assert "A" in result
        assert "B" in result
        assert "1" in result
        assert "2" in result
