from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.jwt_handler import decode_access_token
from src.auth.models import User
from src.auth.repository import get_user_by_id
from src.common.exceptions import AuthenticationError
from src.retrieval.retriever import RBACRetriever

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
        )

    user = await get_user_by_id(int(payload["sub"]))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


def get_rbac_retriever(user: User = Depends(get_current_user)) -> RBACRetriever:
    return RBACRetriever(user_roles=user.role_names)


def require_role(required_role: str):  # noqa: ANN201
    async def _check(user: User = Depends(get_current_user)) -> User:
        if required_role not in user.role_names and "admin" not in user.role_names:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return user
    return _check
