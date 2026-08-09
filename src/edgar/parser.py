"""SEC 10-K/10-Q filing HTML parser.

Extracts structured sections from SEC filing HTML documents.
Handles the wide variety of HTML formatting used by different filers
(some use <b>Item 7, others use <font>ITEM 7., etc.).
"""

import re
from collections import Counter
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from src.common.logging import get_logger

logger = get_logger(__name__)

# 10-K section definitions
SECTION_10K: dict[str, str] = {
    "1": "Business",
    "1a": "Risk Factors",
    "1b": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "7": "Management Discussion and Analysis",
    "7a": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
}

# Sections most valuable for RAG (skip boilerplate sections)
PRIORITY_SECTIONS = {"1", "1a", "7", "7a", "8"}

# Minimum size for trailing text to be treated as incorporated-by-reference
# material rather than the usual exhibit list and signature block.
INCORPORATED_TAIL_MIN_CHARS = 100_000

# Dividers that separate a page number from the document title in a running
# header. Every instance in this corpus is a "|", which is what _table_to_text
# joins cells with — the furniture is laid out as a one-row table. The bullet
# and en/em dashes cover filers who print a rule instead. The ASCII hyphen is
# deliberately excluded: "Form 10-K - 2025" is a caption, not a page footer.
_PAGE_DIVIDER = r"[|•·–—]"

# The running header, in the two orders filers alternate between: title then
# page number on odd pages, page number then title on even ones. Goldman's
# filing uses both, which is why a search for one form finds only half of them.
#
# The page number is what makes these matchable, and also what makes
# DEDUP_ENABLED powerless against them — "Goldman Sachs 2025 Form 10-K | 1" and
# "... | 3" are distinct strings, so near-duplicate suppression sees 121 unique
# chunks rather than one repeated 121 times.
#
# Requiring the page number is the precision guard, and it is doing real work.
# Filings cite themselves in ordinary prose — "filed as Exhibit 19 to this 2025
# Form 10-K", "Risk Factors (Part I, Item 1A of this Form 10-K)" — and those
# sentences carry meaning. Only a line that is *nothing but* a title and a page
# number is furniture, so the anchors and the page number are all mandatory.
#
# The page number is captured as "page" because the running-header rule below
# reads it: which of the two forms matched decides whether the page number is
# the leading or the trailing number, and "the number that isn't a year" gets it
# wrong — it reads the 10 out of 10-K.
_TITLE = r"[^\n]{0,90}\bForm[ \t]*10-[KQ]\b[^\n]{0,25}"
PAGE_FOOTER_PATTERNS = [
    # "Goldman Sachs 2025 Form 10-K | 1", "Apple Inc. | 2025 Form 10-K | 2"
    re.compile(rf"(?im)^[ \t]*{_TITLE}[ \t]*{_PAGE_DIVIDER}[ \t]*(?P<page>\d{{1,4}})[ \t]*$"),
    # "2 | Goldman Sachs 2025 Form 10-K", "44 | JPMorgan Chase & Co./2025 Form 10-K"
    re.compile(rf"(?im)^[ \t]*(?P<page>\d{{1,4}})[ \t]*{_PAGE_DIVIDER}[ \t]*{_TITLE}[ \t]*$"),
    # "Page 150", "Page 12 of 148"
    re.compile(r"(?im)^[ \t]*page[ \t]+(?P<page>\d{1,4})(?:[ \t]+of[ \t]+\d{1,4})?[ \t]*$"),
]

# A page number alone on a line — the other half of the paging apparatus, and
# the page break signal for filers who print no footer text (Microsoft, Tesla).
# Bounded at four digits: a longer standalone number is a figure from a table.
BARE_PAGE_NUMBER_PATTERN = re.compile(r"(?m)^[ \t]*(?P<page>\d{1,4})[ \t]*$")

# A running header repeats on every page but carries no page number of its own —
# "THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES" 239 times, "PART II" 54 —
# so PAGE_FOOTER_PATTERNS cannot see it and it lands inline at the head of an
# otherwise-good chunk, spending 40-odd of the 512-character budget on nothing.
#
# Frequency alone cannot be the test. Real content repeats too, and in this
# corpus the repeats that matter are exactly the ones worth keeping: Goldman
# prints "As of December 2025" 52 times and "Year Ended December" 62 times as
# table captions, and "In the table above:" 77 times as a lead-in. Deleting
# those strips the reporting period off its own financial table.
#
# A "no digits" guard does not separate them — "As of December" and "Year Ended
# December" have none. Position does, and cleanly: a running header sits
# directly below a page break, and a caption sits wherever its table sits.
#
# The window is two non-blank lines, which is exactly what this corpus needs and
# no more: Goldman stacks the company name above the section name, so the second
# header sits two lines below the break, and nothing sits three. Widening it to
# three removes not one additional line across the five filings, while doubling
# the room in which a caption could be mistaken for a header.
RUNNING_HEADER_MIN_REPEATS = 8
RUNNING_HEADER_MAX_CHARS = 60
RUNNING_HEADER_SEARCH_LINES = 2

# Regex to match section headers — handles many real-world formatting variants:
# "Item 1.", "ITEM 1A.", "Item 1A -", "Item 7.", "ITEM\n7", etc.
ITEM_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:<[^>]*>\s*)*"  # Optional HTML tags before
    r"(?:ITEM|Item|item)\s*"  # "Item" keyword
    r"(\d+[aAbB]?)\s*"  # Section number (1, 1a, 7, etc.)
    r"[\.\:\-—–\s]*"  # Separator (., :, -, —)
    r"([A-Z][A-Za-z\s,&\'-]{5,80}?)"  # Section title
    r"\s*(?:</[^>]*>)*",  # Optional closing HTML tags
    re.MULTILINE,
)


@dataclass
class FilingSection:
    section_id: str
    section_name: str
    content: str
    char_count: int


def parse_10k_sections(
    html_content: str,
    priority_only: bool = True,
) -> list[FilingSection]:
    """Parse a 10-K filing HTML into clean text sections.

    Args:
        html_content: Raw HTML of the 10-K filing
        priority_only: If True, only extract high-value sections (1, 1A, 7, 7A, 8)

    Returns:
        List of FilingSection with cleaned text content
    """
    # First pass: convert HTML to text while preserving structure
    text = _html_to_text(html_content)

    # Find all section boundaries
    matches = list(ITEM_PATTERN.finditer(text))

    if not matches:
        logger.warning("no_sections_found", text_length=len(text))
        # Fallback: return entire document as single section
        return [
            FilingSection(
                section_id="full",
                section_name="Full Filing",
                content=_clean_text(text[:100000]),  # Cap at 100K chars
                char_count=min(len(text), 100000),
            )
        ]

    target_sections = PRIORITY_SECTIONS if priority_only else set(SECTION_10K.keys())

    # A section header appears many times in a filing, and only one of them
    # starts the real body:
    #   - once per page as a running header ("PART I Item 1 ..."), which is why
    #     Microsoft's document yields 101 matches;
    #   - once in the table of contents, where headers sit adjacent;
    #   - in cross-references ("see Item 7A").
    #
    # Slicing each match to the *immediately next* match treats all of these as
    # section boundaries, so a page header yields one page and a pair of
    # table-of-contents lines yields a few dozen characters. That is how
    # Microsoft's MD&A came through as 1,931 characters and JPMorgan's as 381.
    #
    # Two rules fix it. A repeat of the same item number is a page header, not a
    # boundary, so a candidate span runs to the next *distinct* section. And of
    # the candidates for one item, the real body is the longest — that discards
    # the table-of-contents entry, whose neighbours are adjacent.
    best_by_section: dict[str, tuple[int, int, FilingSection]] = {}

    for i, match in enumerate(matches):
        section_num = match.group(1).lower()

        if section_num not in target_sections:
            continue

        start = match.end()
        end = len(text)
        for later in matches[i + 1 :]:
            if later.group(1).lower() != section_num:
                end = later.start()
                break

        content = _clean_text(text[start:end])

        if len(content) < 100:
            continue  # Skip empty or near-empty candidates

        previous = best_by_section.get(section_num)
        if previous is not None and previous[2].char_count >= len(content):
            continue

        best_by_section[section_num] = (
            match.start(),
            end,
            FilingSection(
                section_id=f"item_{section_num}",
                section_name=SECTION_10K.get(section_num, match.group(2).strip()),
                content=content,
                char_count=len(content),
            ),
        )

    # Document order, so downstream chunking sees the filing as it reads.
    ordered = sorted(best_by_section.values(), key=lambda triple: triple[0])
    sections = [section for _, _, section in ordered]

    # Large banks satisfy Items 7 and 8 by incorporation by reference: the
    # section body is a pointer ("Management's discussion and analysis ...
    # appears on pages 46-160") and the material itself is appended to the same
    # document, past every Item header. JPMorgan's filing carries 1.03M
    # characters — 72% of the document, including all of its financials — in
    # that tail, and header-delimited slicing can never reach it.
    #
    # Ordinary filings end in a short exhibit list and signature block (3-6%),
    # so a high threshold keeps this targeted at the incorporation case.
    covered_end = max((end for _, end, _ in ordered), default=0)
    tail = _clean_text(text[covered_end:])
    if len(tail) >= INCORPORATED_TAIL_MIN_CHARS:
        logger.info("incorporated_material_recovered", chars=len(tail))
        sections.append(
            FilingSection(
                section_id="incorporated",
                section_name="Incorporated Annual Report (MD&A and Financial Statements)",
                content=tail,
                char_count=len(tail),
            )
        )

    for section in sections:
        logger.info(
            "section_extracted",
            section=section.section_id,
            name=section.section_name,
            chars=section.char_count,
        )

    logger.info("parsing_complete", sections_found=len(sections))
    return sections


def _html_to_text(html: str) -> str:
    """Convert HTML to clean text, preserving table structure."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script and style elements
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    # Convert tables to readable text format
    for table in soup.find_all("table"):
        table_text = _table_to_text(table)
        table.replace_with(soup.new_string(f"\n{table_text}\n"))

    # Get text with reasonable whitespace
    text = soup.get_text(separator="\n")
    return text


def _table_to_text(table: Tag) -> str:
    """Convert an HTML table to a readable text format.

    Preserves financial table structure — critical for Item 8
    (Financial Statements) where numbers matter.
    """
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cell_text = td.get_text(strip=True)
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))

    return "\n".join(rows)


def _clean_text(text: str) -> str:
    """Clean extracted text — remove excess whitespace, page artifacts."""
    # Non-breaking spaces first. They read as whitespace but are not [ \t], so
    # one trailing nbsp is enough to stop a running header matching its own
    # end-of-line anchor below.
    text = text.replace("\xa0", " ")

    # Then furniture, and only then whitespace: each removal leaves an empty
    # line behind, and collapsing afterwards is what closes the gap. Doing it
    # in the other order left a blank line everywhere a header used to be.
    text = _strip_page_furniture(text)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_page_furniture(text: str) -> str:
    """Remove running headers, footers, and standalone page numbers.

    This is paging apparatus from the printed document — it repeats on every
    page, says nothing about the company, and survives into the index as its
    own chunk because the recursive splitter cannot merge a 32-character line
    into a neighbouring 500-character one without exceeding the chunk budget.

    Whole lines only. A footer sits on its own line, so the anchors keep this
    from reaching into a sentence that merely mentions the form.
    """
    # Running headers first: the rule keys off the page breaks, so it has to run
    # while the footers and page numbers are still there to be found.
    text = _strip_running_headers(text)

    text = re.sub(r"(?im)^[ \t]*table of contents[ \t]*$", "", text)
    text = BARE_PAGE_NUMBER_PATTERN.sub("", text)
    for pattern in PAGE_FOOTER_PATTERNS:
        text = pattern.sub("", text)
    return text


def _page_number(line: str) -> int | None:
    """The page number a line carries, if the line is paging apparatus.

    Returns None for anything else, including a line that merely contains a
    number. The caller uses "is this a page break" and "which page" as one
    question, because a bare number is only a page break if it is part of an
    ascending run.
    """
    if BARE_PAGE_NUMBER_PATTERN.fullmatch(line):
        return int(line.strip())
    for pattern in PAGE_FOOTER_PATTERNS:
        match = pattern.match(line)
        if match:
            return int(match.group("page"))
    return None


def _strip_running_headers(text: str) -> str:
    """Remove repeated headers that carry no page number of their own.

    Three conjuncts, and each one is load-bearing against a class of real
    content that the others admit:

      repetition  the line occurs at least RUNNING_HEADER_MIN_REPEATS times.
      position    every deleted occurrence sits within
                  RUNNING_HEADER_SEARCH_LINES non-blank lines below a page
                  break, and those page numbers strictly increase. This is what
                  spares the table captions: measured over the five filings in
                  data/edgar, "As of December 2025" and "As of December 2024"
                  satisfy the position test on 0% of their occurrences, "As of
                  December" on 5%, "Year Ended December" on 7%, "In the table
                  above:" on 4%, and "(In millions)" on 7% — against 100% for
                  every running header. The ascending requirement is what stops
                  a table figure standing in for a page number: "million and $",
                  a fragment left by a split around an inline number, sits below
                  a bare number every time it occurs, and those numbers ascend
                  57% of the time rather than always.
      shape       the line starts like a title. Every candidate that survives
                  the first two but is not furniture is a lowercase mid-sentence
                  fragment ("million, respectively.", "billion, $"), so this
                  widens the margin at no cost to any real header.

    Deletion is per occurrence, not per distinct line. JPMorgan's glossary
    defines "MD&A:" as "Management's discussion and analysis" — the same string
    as its running header, in ordinary content. Deciding per line either loses
    the definition or keeps 39 headers; deciding per occurrence keeps the
    definition, which is the one occurrence with no page break above it.
    """
    lines = text.split("\n")
    stripped = [line.strip() for line in lines]

    counts = Counter(line for line in stripped if line)
    candidates = {
        line
        for line, count in counts.items()
        if count >= RUNNING_HEADER_MIN_REPEATS
        and len(line) <= RUNNING_HEADER_MAX_CHARS
        # A "|" means the line came from a table row, where a repeated short
        # string is a column header, not page furniture.
        and "|" not in line
        and _page_number(line) is None
        and _is_title_shaped(line)
    }
    if not candidates:
        return text

    occurrences: dict[str, list[int]] = {}
    for index, line in enumerate(stripped):
        if line in candidates:
            occurrences.setdefault(line, []).append(index)

    doomed: set[int] = set()
    for line, indices in occurrences.items():
        marked = [(i, _preceding_page_number(stripped, i)) for i in indices]
        marked = [(i, page) for i, page in marked if page is not None]
        if len(marked) < RUNNING_HEADER_MIN_REPEATS:
            continue
        pages = [page for _, page in marked]
        if not all(later > earlier for earlier, later in zip(pages, pages[1:], strict=False)):
            continue
        doomed.update(i for i, _ in marked)
        logger.info(
            "running_header_stripped", header=line, removed=len(marked), occurrences=len(indices)
        )

    if not doomed:
        return text

    # Blank rather than delete, so the line count is unchanged and the empty
    # line left behind is closed by the \n{3,} collapse in _clean_text — the
    # same contract the pattern-based rules above work to.
    return "\n".join("" if i in doomed else line for i, line in enumerate(lines))


def _is_title_shaped(line: str) -> bool:
    """Does the line read as a heading rather than a piece of a sentence?"""
    return bool(re.match(r"[A-Z]", line)) and not line.endswith(("$", ","))


def _preceding_page_number(stripped: list[str], index: int) -> int | None:
    """The page number of the break above `index`, if there is one close above.

    Blank lines do not count against the budget: the filings put one on either
    side of the footer, so a header three non-blank lines below a page break can
    be six raw lines below it.
    """
    seen = 0
    for candidate in reversed(stripped[:index]):
        if not candidate:
            continue
        seen += 1
        if seen > RUNNING_HEADER_SEARCH_LINES:
            return None
        page = _page_number(candidate)
        if page is not None:
            return page
    return None
