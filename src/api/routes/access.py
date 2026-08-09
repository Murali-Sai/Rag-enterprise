"""What the caller's roles may reach — the RBAC state, without a query.

Every fact here is already computable from a `POST /query` response, and
that is exactly the problem: seeing the information barrier move requires
paying for a `gpt-4o` call plus retrieval per role. A dashboard whose whole
argument is "the same question answered differently by role" would spend a
query just to render the header.

Read-only, no LLM, no vector store. The route exists so the barrier is
visible before anything is asked.
"""

from fastapi import APIRouter, Depends

from src.api.deps import get_current_user
from src.auth.models import User
from src.auth.rbac import get_accessible_departments, get_information_barriers_for_user
from src.common.schemas import AccessProfile, InformationBarrier

router = APIRouter(tags=["Access"])


def to_information_barriers(barriers: list[dict]) -> list[InformationBarrier]:
    """The barrier rules as data, not as the flattened audit string.

    `blocked_departments` is a set on the rule and a sorted list here: a set
    has no JSON representation, and an arbitrary iteration order would make
    two identical responses differ.
    """
    return [
        InformationBarrier(
            name=barrier["name"],
            description=barrier["description"],
            blocked_departments=sorted(barrier["blocked_departments"]),
        )
        for barrier in barriers
    ]


@router.get("/access", response_model=AccessProfile)
async def read_access(user: User = Depends(get_current_user)) -> AccessProfile:
    roles = user.role_names
    return AccessProfile(
        username=user.username,
        roles=sorted(roles),
        accessible_departments=sorted(get_accessible_departments(roles)),
        information_barriers=to_information_barriers(get_information_barriers_for_user(roles)),
        unrestricted="admin" in roles,
    )
