from fastapi import APIRouter, HTTPException, status

from src.auth.jwt_handler import create_access_token
from src.auth.repository import authenticate_user, create_user
from src.common.exceptions import AuthenticationError
from src.common.schemas import TokenRequest, TokenResponse, UserCreate, UserResponse
from src.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: UserCreate) -> UserResponse:
    try:
        user = await create_user(request.username, request.password, request.roles)
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

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
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token(
        user_id=user.id,
        username=user.username,
        roles=[r.name for r in user.roles],
    )

    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expiry_minutes * 60,
    )
