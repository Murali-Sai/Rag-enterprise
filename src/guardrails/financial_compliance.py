"""Financial services-specific guardrails.

Enforces regulatory constraints on RAG outputs relevant to
investment banks (JPMC, Morgan Stanley, Goldman Sachs, etc.):
- Investment advice disclaimers
- MNPI (Material Non-Public Information) detection
- Regulatory citation requirements
- Forward-looking statement warnings
"""

import re
from dataclasses import dataclass, field

from src.common.logging import get_logger

logger = get_logger(__name__)

# Patterns that suggest the response contains investment advice
INVESTMENT_ADVICE_PATTERNS = [
    r"(?i)you should (buy|sell|invest|hold|short|trade|allocate)",
    r"(?i)I (recommend|suggest|advise) (buying|selling|investing|holding|trading)",
    r"(?i)(guaranteed|risk-?free) (return|profit|income|yield)",
    r"(?i)can'?t (lose|go wrong|fail)",
    r"(?i)(definitely|certainly|absolutely) (will|going to) (rise|fall|increase|decrease)",
]

# Patterns indicating potential MNPI leakage
MNPI_PATTERNS = [
    r"(?i)(upcoming|pending|unannounced) (merger|acquisition|deal|takeover|IPO|offering)",
    r"(?i)(before (it'?s|the) (public|announced|disclosed))",
    r"(?i)(insider|non-?public|confidential) (information|knowledge|tip)",
    r"(?i)hasn'?t been (announced|disclosed|made public|filed) yet",
    r"(?i)(pre-?announcement|pre-?release|embargoed)",
    r"(?i)this information is (not yet|still) (public|available)",
]

# Forward-looking statement indicators
FORWARD_LOOKING_PATTERNS = [
    r"(?i)(will|expect|project|forecast|anticipate|estimate|believe|intend|plan)s? to",
    r"(?i)(next quarter|next year|going forward|in the future|outlook|guidance)",
    r"(?i)(target|projected|estimated) (price|revenue|earnings|growth)",
]

INVESTMENT_DISCLAIMER = (
    "\n\n---\n**Disclaimer**: This information is retrieved from internal documents "
    "and is provided for informational purposes only. It does not constitute investment "
    "advice, a recommendation, or a solicitation to buy or sell any security. Past "
    "performance does not guarantee future results."
)

MNPI_WARNING = (
    "\n\n**WARNING**: This response may reference material non-public information (MNPI). "
    "Do not use this information for trading decisions. Consult Compliance before acting "
    "on any non-public information. Violations may result in civil and criminal penalties "
    "under Section 10(b) of the Securities Exchange Act of 1934."
)

FORWARD_LOOKING_DISCLAIMER = (
    "\n\n*This response contains forward-looking statements that involve risks and "
    "uncertainties. Actual results may differ materially from those projected.*"
)


@dataclass
class FinancialComplianceResult:
    requires_investment_disclaimer: bool = False
    requires_mnpi_warning: bool = False
    requires_forward_looking_disclaimer: bool = False
    contains_prohibited_advice: bool = False
    flags: list[str] = field(default_factory=list)
    disclaimers_to_append: list[str] = field(default_factory=list)


def check_financial_compliance(
    query: str,
    response: str,
    user_roles: set[str],
) -> FinancialComplianceResult:
    result = FinancialComplianceResult()

    # Check for investment advice in the response
    for pattern in INVESTMENT_ADVICE_PATTERNS:
        if re.search(pattern, response):
            result.requires_investment_disclaimer = True
            result.flags.append("investment_advice_detected")
            # "guaranteed return" type phrases are prohibited
            if re.search(r"(?i)(guaranteed|risk-?free) (return|profit)", response):
                result.contains_prohibited_advice = True
                result.flags.append("prohibited_guarantee_language")
            break

    # Check for MNPI indicators
    for pattern in MNPI_PATTERNS:
        if re.search(pattern, response):
            result.requires_mnpi_warning = True
            result.flags.append("potential_mnpi_detected")
            logger.warning(
                "mnpi_flag_raised",
                query_preview=query[:50],
                user_roles=list(user_roles),
            )
            break

    # Check for forward-looking statements
    for pattern in FORWARD_LOOKING_PATTERNS:
        if re.search(pattern, response):
            result.requires_forward_looking_disclaimer = True
            result.flags.append("forward_looking_statement")
            break

    # Build disclaimers to append
    if result.requires_investment_disclaimer:
        result.disclaimers_to_append.append(INVESTMENT_DISCLAIMER)
    if result.requires_mnpi_warning:
        result.disclaimers_to_append.append(MNPI_WARNING)
    if result.requires_forward_looking_disclaimer:
        result.disclaimers_to_append.append(FORWARD_LOOKING_DISCLAIMER)

    return result


def apply_financial_disclaimers(response: str, compliance_result: FinancialComplianceResult) -> str:
    """Append required disclaimers to the response."""
    if compliance_result.contains_prohibited_advice:
        response = (
            "I cannot provide guaranteed investment returns or risk-free profit projections. "
            "Please consult with a registered investment advisor for personalized advice."
        )
        return response + INVESTMENT_DISCLAIMER

    for disclaimer in compliance_result.disclaimers_to_append:
        response += disclaimer

    return response
