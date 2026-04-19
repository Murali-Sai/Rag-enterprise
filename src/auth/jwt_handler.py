from datetime import UTC, datetime, timedelta

import jwt

from src.common.exceptions import AuthenticationError
from src.config import settings


def create_access_token(user_id: int, username: str, roles: list[str]) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "roles": roles,
        "exp": datetime.now(UTC) + timedelta(minutes=settings.jwt_expiry_minutes),
        "iat": datetime.now(UTC),
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
    except jwt.ExpiredSignatureError as e:
        raise AuthenticationError("Token has expired") from e
    except jwt.InvalidTokenError as e:
        raise AuthenticationError("Invalid token") from e
