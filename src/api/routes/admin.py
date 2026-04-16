from fastapi import APIRouter, Depends

from src.api.deps import require_role
from src.auth.models import User
from src.auth.repository import list_users
from src.common.schemas import UserResponse

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/users", response_model=list[UserResponse])
async def get_all_users(
    user: User = Depends(require_role("admin")),
) -> list[UserResponse]:
    users = await list_users()
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            roles=[r.name for r in u.roles],
            created_at=u.created_at,
        )
        for u in users
    ]
