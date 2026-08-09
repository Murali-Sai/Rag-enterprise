from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import get_optional_user
from src.auth.jwt_handler import create_access_token
from src.auth.models import User
from src.auth.repository import authenticate_user, create_user
from src.common.exceptions import AuthenticationError
from src.common.schemas import TokenRequest, TokenResponse, UserCreate, UserResponse
from src.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])

# What an anonymous caller may give themselves. Everything else is a grant,
# and grants come from an admin.
SELF_SERVICE_ROLES = frozenset({"viewer"})


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: UserCreate,
    caller: User | None = Depends(get_optional_user),
) -> UserResponse:
    """Create a user. Anonymous callers get `viewer`; other roles need an admin.

    `roles` arrives in the request body, and this endpoint is public, so
    without this check anyone could POST `{"roles": ["admin"]}` and mint
    themselves an account that reads every department — which is to say, the
    information barriers this system exists to demonstrate were one unauthenticated
    request away from being bypassed, and `POST /documents/ingest` one more.

    The barriers are enforced correctly everywhere they are checked. The hole
    was upstream of them: nothing stopped a caller from choosing which side of
    the wall to stand on.
    """
    requested = set(request.roles)
    if not requested <= SELF_SERVICE_ROLES and (caller is None or "admin" not in caller.role_names):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Self-registration grants {sorted(SELF_SERVICE_ROLES)} only. "
                f"Assigning {sorted(requested - SELF_SERVICE_ROLES)} requires an admin token."
            ),
        )

    try:
        user = await create_user(request.username, request.password, request.roles)
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e

    return UserResponse(
        id=user.id,
        username=user.username,
        roles=[r.name for r in user.roles],
        created_at=user.created_at,
    )


@router.post("/token", response_model=TokenResponse)
async def login(request: TokenRequest) -> TokenResponse:
    try:
        user = await authenticate_user(request.username, request.password)
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        ) from e

    token = create_access_token(
        user_id=user.id,
        username=user.username,
        roles=[r.name for r in user.roles],
    )

    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expiry_minutes * 60,
    )
