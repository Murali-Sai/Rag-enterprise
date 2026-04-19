from langchain_core.documents import Document

from src.common.logging import get_logger

logger = get_logger(__name__)

# Investment bank role-to-department access mapping
# Mirrors real IB information barrier (Chinese Wall) structure
ROLE_ACCESS_MAP: dict[str, set[str]] = {
    # Senior management / admin — full access across all barriers
    "admin": {
        "sec_filings",
        "risk_management",
        "compliance",
        "research",
        "trading",
        "general",
    },
    # Front Office: Trading — access to trading procedures, risk, and public filings
    "trading": {"trading", "risk_management", "sec_filings", "general"},
    # Risk Management — access to risk frameworks, credit policies, trading metrics
    "risk": {"risk_management", "trading", "sec_filings", "compliance", "general"},
    # Compliance / Legal — access to compliance, AML, regulatory, and all public info
    "compliance": {"compliance", "sec_filings", "risk_management", "general"},
    # Research — access to research reports and public filings ONLY
    # (Chinese Wall: NO access to trading, compliance investigations, or risk limits)
    "research": {"research", "sec_filings", "general"},
    # Wealth Management / Client-facing — access to research and public filings
    "wealth_management": {"research", "sec_filings", "general"},
    # Operations / Back Office — access to trading procedures and compliance
    "operations": {"trading", "compliance", "general"},
    # External auditor — read access to compliance and filings
    "auditor": {"compliance", "sec_filings", "general"},
    # Basic viewer — public information only
    "viewer": {"sec_filings", "general"},
}

# Information barrier (Chinese Wall) rules — these are hard restrictions
# that cannot be overridden by role combinations
INFORMATION_BARRIERS: list[dict] = [
    {
        "name": "Research-Trading Wall",
        "description": "Research analysts cannot access non-public trading positions or strategies",
        "blocked_role": "research",
        "blocked_departments": {"trading"},
    },
    {
        "name": "Research-Compliance Wall",
        "description": "Research cannot access active compliance investigations",
        "blocked_role": "research",
        "blocked_departments": {"compliance"},
    },
]


def get_accessible_departments(user_roles: set[str]) -> set[str]:
    departments: set[str] = set()
    for role in user_roles:
        departments.update(ROLE_ACCESS_MAP.get(role, set()))

    # Enforce information barriers — remove blocked departments even if
    # another role grants access (Chinese Wall is absolute)
    for barrier in INFORMATION_BARRIERS:
        if barrier["blocked_role"] in user_roles and "admin" not in user_roles:
            departments -= barrier["blocked_departments"]
            logger.info(
                "information_barrier_enforced",
                barrier=barrier["name"],
                removed_departments=list(barrier["blocked_departments"]),
            )

    return departments


def filter_documents_by_access(
    documents: list[Document],
    user_roles: set[str],
) -> list[Document]:
    if "admin" in user_roles:
        return documents

    accessible = get_accessible_departments(user_roles)

    filtered = [doc for doc in documents if doc.metadata.get("department", "general") in accessible]

    logger.info(
        "rbac_filter_applied",
        total_docs=len(documents),
        filtered_docs=len(filtered),
        user_roles=list(user_roles),
        accessible_departments=list(accessible),
    )

    return filtered


def check_department_access(user_roles: set[str], department: str) -> bool:
    if "admin" in user_roles:
        return True
    accessible = get_accessible_departments(user_roles)
    return department in accessible


def get_information_barriers_for_user(user_roles: set[str]) -> list[dict]:
    """Return active information barriers affecting this user."""
    active = []
    for barrier in INFORMATION_BARRIERS:
        if barrier["blocked_role"] in user_roles and "admin" not in user_roles:
            active.append(barrier)
    return active
