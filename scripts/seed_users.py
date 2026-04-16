"""Seed demo users with investment bank roles for testing RBAC and information barriers."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.repository import create_user, init_db
from src.common.exceptions import AuthenticationError

DEMO_USERS = [
    {"username": "admin_user", "password": "admin1234!", "roles": ["admin"]},
    {"username": "trader_desk", "password": "trade1234!", "roles": ["trading"]},
    {"username": "risk_analyst", "password": "risk12345!", "roles": ["risk"]},
    {"username": "compliance_officer", "password": "compl1234!", "roles": ["compliance"]},
    {"username": "research_analyst", "password": "research1!", "roles": ["research"]},
    {"username": "wealth_advisor", "password": "wealth123!", "roles": ["wealth_management"]},
    {"username": "ops_manager", "password": "ops1234567!", "roles": ["operations"]},
    {"username": "external_auditor", "password": "audit1234!", "roles": ["auditor"]},
    {"username": "senior_md", "password": "senior123!", "roles": ["trading", "risk"]},
    {"username": "viewer_user", "password": "viewer123!", "roles": ["viewer"]},
]


async def seed() -> None:
    await init_db()
    print("Database initialized.\n")

    for user_data in DEMO_USERS:
        try:
            user = await create_user(**user_data)
            print(f"Created user: {user.username} (roles: {user_data['roles']})")
        except AuthenticationError as e:
            print(f"Skipped {user_data['username']}: {e}")

    print("\nDemo users seeded successfully!")
    print("\n" + "=" * 80)
    print(f"{'Username':<20} {'Password':<15} {'Roles':<25} {'Access'}")
    print("=" * 80)
    role_access = {
        "admin": "ALL departments",
        "trading": "Trading, Risk, SEC Filings",
        "risk": "Risk, Trading, SEC, Compliance",
        "compliance": "Compliance, SEC, Risk",
        "research": "Research, SEC (NO trading/compliance - Chinese Wall)",
        "wealth_management": "Research, SEC",
        "operations": "Trading, Compliance",
        "auditor": "Compliance, SEC",
        "viewer": "SEC Filings only",
    }
    for u in DEMO_USERS:
        roles = u["roles"]
        access = role_access.get(roles[0], "Custom")
        print(f"  {u['username']:<20} {u['password']:<15} {str(roles):<25} {access}")


if __name__ == "__main__":
    asyncio.run(seed())
