from datetime import datetime, timedelta, timezone

import jwt

from src.config import settings
from src.common.exceptions import AuthenticationError


def create_access_token(user_id: int, username: str, roles: list[str]) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "roles": roles,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
