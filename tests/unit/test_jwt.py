import pytest

from src.auth.jwt_handler import create_access_token, decode_access_token
from src.common.exceptions import AuthenticationError


class TestJWT:
    def test_create_and_decode(self):
        token = create_access_token(user_id=1, username="testuser", roles=["admin"])
        payload = decode_access_token(token)
        assert payload["sub"] == "1"
        assert payload["username"] == "testuser"
        assert payload["roles"] == ["admin"]

    def test_multiple_roles(self):
        token = create_access_token(user_id=2, username="multi", roles=["hr", "engineering"])
        payload = decode_access_token(token)
        assert set(payload["roles"]) == {"hr", "engineering"}

    def test_invalid_token(self):
        with pytest.raises(AuthenticationError, match="Invalid"):
            decode_access_token("invalid.token.here")

    def test_tampered_token(self):
        token = create_access_token(user_id=1, username="test", roles=["admin"])
        # Tamper with the token
        parts = token.split(".")
        parts[1] = parts[1] + "x"
        tampered = ".".join(parts)
        with pytest.raises(AuthenticationError):
            decode_access_token(tampered)
