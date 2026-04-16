import re
from dataclasses import dataclass, field

from src.common.logging import get_logger

logger = get_logger(__name__)

# Patterns that suggest the LLM is hallucinating or going off-context
HALLUCINATION_MARKERS = [
    r"(?i)as an AI( language model)?",
    r"(?i)I don'?t have (access to|real-?time)",
    r"(?i)I cannot browse",
    r"(?i)my training data",
    r"(?i)as of my (last |knowledge )?cut-?off",
]

# Patterns for potentially unsafe outputs
UNSAFE_OUTPUT_PATTERNS = [
    r"(?i)(how to (make|build|create) (a )?(bomb|weapon|explosive))",
    r"(?i)(instructions for (hacking|breaking into))",
]


@dataclass
class OutputSafetyResult:
    is_safe: bool
    flags: list[str] = field(default_factory=list)


def check_output_safety(output: str) -> OutputSafetyResult:
    flags: list[str] = []

    # Check for hallucination markers
    for pattern in HALLUCINATION_MARKERS:
        if re.search(pattern, output):
            flags.append("possible_hallucination")
            break

    # Check for unsafe content
    for pattern in UNSAFE_OUTPUT_PATTERNS:
        if re.search(pattern, output):
            flags.append("unsafe_content_detected")
            logger.warning("unsafe_output_detected", output_preview=output[:100])
            break

    # Check if response is suspiciously short
    if len(output.strip()) < 10:
        flags.append("response_too_short")

    # Check if response is just repeating the context verbatim (potential leak)
    if len(output) > 500 and output.count("\n") > 20:
        flags.append("possible_context_leak")

    is_safe = "unsafe_content_detected" not in flags
    return OutputSafetyResult(is_safe=is_safe, flags=flags)
