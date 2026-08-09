from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.jwt_handler import decode_access_token
from src.auth.models import User
from src.auth.repository import get_user_by_id
from src.common.exceptions import AuthenticationError
from src.retrieval.retriever import Retriever, get_retriever

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    try:
        payload = decode_access_token(credentials.credentials)
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    user = await get_user_by_id(int(payload["sub"]))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


def get_rbac_retriever(
    user: User = Depends(get_current_user),
) -> Retriever:
    """The retriever the app serves with, for this user's roles.

    Typed as the `Retriever` protocol rather than a union of concrete classes.
    `get_retriever()` composes whichever stages are enabled — the union had to
    be edited every time a stage was added, and went stale the moment one was,
    which is how a route ends up annotated for a pipeline it no longer gets.
    """
    return get_retriever(user_roles=user.role_names)


def require_role(required_role: str):  # noqa: ANN201
    async def _check(user: User = Depends(get_current_user)) -> User:
        if required_role not in user.role_names and "admin" not in user.role_names:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return user

    return _check
