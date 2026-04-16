"""SEC 10-K/10-Q filing HTML parser.

Extracts structured sections from SEC filing HTML documents.
Handles the wide variety of HTML formatting used by different filers
(some use <b>Item 7, others use <font>ITEM 7., etc.).
"""

import re
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

# Regex to match section headers — handles many real-world formatting variants:
# "Item 1.", "ITEM 1A.", "Item 1A -", "Item 7.", "ITEM\n7", etc.
ITEM_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:<[^>]*>\s*)*'       # Optional HTML tags before
    r'(?:ITEM|Item|item)\s*'              # "Item" keyword
    r'(\d+[aAbB]?)\s*'                   # Section number (1, 1a, 7, etc.)
    r'[\.\:\-—–\s]*'                     # Separator (., :, -, —)
    r'([A-Z][A-Za-z\s,&\'-]{5,80}?)'    # Section title
    r'\s*(?:</[^>]*>)*',                  # Optional closing HTML tags
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
        return [FilingSection(
            section_id="full",
            section_name="Full Filing",
            content=_clean_text(text[:100000]),  # Cap at 100K chars
            char_count=min(len(text), 100000),
        )]

    sections: list[FilingSection] = []
    target_sections = PRIORITY_SECTIONS if priority_only else set(SECTION_10K.keys())

    for i, match in enumerate(matches):
        section_num = match.group(1).lower()

        if section_num not in target_sections:
            continue

        # Extract text between this header and the next
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end]

        # Clean the extracted text
        content = _clean_text(content)

        if len(content) < 100:
            continue  # Skip empty or near-empty sections

        section_name = SECTION_10K.get(section_num, match.group(2).strip())

        sections.append(FilingSection(
            section_id=f"item_{section_num}",
            section_name=section_name,
            content=content,
            char_count=len(content),
        ))

        logger.info(
            "section_extracted",
            section=f"Item {section_num}",
            name=section_name,
            chars=len(content),
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
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove common page header/footer artifacts
    text = re.sub(r"(?m)^Table of Contents\s*$", "", text)
    text = re.sub(r"(?m)^\d+\s*$", "", text)  # Page numbers on their own line
    # Remove Unicode replacement characters
    text = text.replace("\xa0", " ")
    return text.strip()
