"""Tests for SEC EDGAR 10-K HTML parser."""

from src.edgar.parser import (
    FilingSection,
    _clean_text,
    _html_to_text,
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
