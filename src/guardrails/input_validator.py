import re

from src.common.exceptions import GuardrailViolation
from src.common.logging import get_logger

logger = get_logger(__name__)

MAX_QUERY_LENGTH = 1000
MIN_QUERY_LENGTH = 2

# Patterns that indicate non-question inputs
BLOCKED_PATTERNS = [
    r"(?i)(drop\s+table|delete\s+from|insert\s+into|update\s+.*\s+set)",  # SQL
    r"<script[^>]*>",  # XSS
    r"(?i)(exec|eval|import\s+os|subprocess)",  # Code execution
]


def validate_input(query: str) -> None:
    if not query or not query.strip():
        raise GuardrailViolation("Query cannot be empty", violation_type="empty_input")

    if len(query) < MIN_QUERY_LENGTH:
        raise GuardrailViolation(
            f"Query too short (minimum {MIN_QUERY_LENGTH} characters)",
            violation_type="too_short",
        )

    if len(query) > MAX_QUERY_LENGTH:
        raise GuardrailViolation(
            f"Query too long (maximum {MAX_QUERY_LENGTH} characters)",
            violation_type="too_long",
        )

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query):
            logger.warning("blocked_pattern_detected", query_preview=query[:50], pattern=pattern)
            raise GuardrailViolation(
                "Query contains blocked patterns",
                violation_type="blocked_pattern",
            )
