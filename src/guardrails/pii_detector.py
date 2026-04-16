import re

from src.common.logging import get_logger

logger = get_logger(__name__)

# Regex-based PII detection as a lightweight alternative to Presidio
# Presidio can be enabled in production but requires spaCy models
PII_PATTERNS: dict[str, str] = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}

REDACTION_PLACEHOLDER = {
    "SSN": "[SSN_REDACTED]",
    "EMAIL": "[EMAIL_REDACTED]",
    "PHONE": "[PHONE_REDACTED]",
    "CREDIT_CARD": "[CC_REDACTED]",
    "IP_ADDRESS": "[IP_REDACTED]",
}


def detect_pii(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = matches
    return found


def redact_pii(text: str) -> str:
    redacted = text
    for pii_type, pattern in PII_PATTERNS.items():
        placeholder = REDACTION_PLACEHOLDER[pii_type]
        new_text = re.sub(pattern, placeholder, redacted)
        if new_text != redacted:
            logger.info("pii_redacted", type=pii_type)
        redacted = new_text
    return redacted


# Optional: Presidio-based detection for production
def detect_pii_presidio(text: str) -> list[dict]:
    try:
        from presidio_analyzer import AnalyzerEngine

        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        return [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
            }
            for r in results
        ]
    except ImportError:
        logger.warning("presidio_not_available", fallback="regex")
        return [
            {"type": pii_type, "start": 0, "end": 0, "score": 1.0}
            for pii_type, matches in detect_pii(text).items()
            if matches
        ]
