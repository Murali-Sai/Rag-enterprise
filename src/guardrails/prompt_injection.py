import re
from dataclasses import dataclass

from src.common.logging import get_logger

logger = get_logger(__name__)

# Patterns indicating prompt injection attempts, with associated risk weights
INJECTION_PATTERNS: list[tuple[str, float]] = [
    (r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", 0.9),
    (r"(?i)disregard\s+(all\s+)?(previous|prior|above)", 0.9),
    (r"(?i)forget\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|context)", 0.9),
    (r"(?i)you\s+are\s+now\s+(a|an)\s+", 0.8),
    (r"(?i)act\s+as\s+(if|a|an)\s+", 0.6),
    (r"(?i)pretend\s+(you\s+are|to\s+be)", 0.7),
    (r"(?i)system\s*prompt", 0.7),
    (r"(?i)override\s+(your|the)\s+(instructions?|rules?|safety)", 0.9),
    (r"(?i)bypass\s+(your|the|all)\s+(filters?|safety|guardrails?|restrictions?)", 0.9),
    (r"(?i)jailbreak", 0.9),
    (r"(?i)do\s+anything\s+now", 0.8),
    (r"(?i)DAN\s+mode", 0.9),
    (r"(?i)reveal\s+(your|the)\s+(system|secret|hidden)\s+(prompt|instructions?)", 0.8),
    (r"(?i)\[INST\]|\[/INST\]|<<SYS>>|<\|im_start\|>", 0.9),  # Common LLM delimiters
]

BLOCK_THRESHOLD = 0.8


@dataclass
class InjectionResult:
    risk_score: float
    is_blocked: bool
    reason: str
    matched_patterns: list[str]


def detect_prompt_injection(query: str) -> InjectionResult:
    matched: list[tuple[str, float]] = []

    for pattern, weight in INJECTION_PATTERNS:
        if re.search(pattern, query):
            matched.append((pattern, weight))

    if not matched:
        return InjectionResult(
            risk_score=0.0,
            is_blocked=False,
            reason="",
            matched_patterns=[],
        )

    # Risk score is the maximum weight among matched patterns
    max_score = max(weight for _, weight in matched)

    is_blocked = max_score >= BLOCK_THRESHOLD
    reason = "prompt_injection_detected" if is_blocked else "suspicious_patterns_detected"

    if is_blocked:
        logger.warning(
            "prompt_injection_blocked",
            query_preview=query[:80],
            risk_score=max_score,
            patterns_matched=len(matched),
        )

    return InjectionResult(
        risk_score=max_score,
        is_blocked=is_blocked,
        reason=reason,
        matched_patterns=[p for p, _ in matched],
    )
