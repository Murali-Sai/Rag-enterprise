"""Who may hand out which roles at /auth/register.

The endpoint is public and takes `roles` from the request body, so for a while
anyone could POST `{"roles": ["admin"]}` and read every department — the
information barriers were intact, but nothing stopped a caller from choosing
which side of the wall to stand on. These pin the rule that closed it:
anonymous callers get `viewer`, anything else needs an admin token.

The database is stubbed. What is under test is the authorisation decision, and
routing it through SQLite would test the seed script instead — the same reason
`GET /access` is tested through a dependency override in the integration suite.
"""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_optional_user
from src.auth.models import Role, User
from src.main import app


def user_with(*roles: str) -> User:
    # created_at is a column default, so an unsaved instance has none and
    # UserResponse will not serialise without it.
    user = User(
        id=1,
        username="caller",
        password_hash="unused",  # noqa: S106
        created_at=datetime.now(UTC),
    )
    user.roles = [Role(id=i, name=name) for i, name in enumerate(roles, start=1)]
    return user


@pytest.fixture
def created(monkeypatch):
    """Record what the route asked the repository to create."""
    calls = []

    async def fake_create_user(username: str, password: str, role_names: list[str]) -> User:
        calls.append({"username": username, "roles": role_names})
        return user_with(*role_names)

    monkeypatch.setattr("src.api.routes.auth.create_user", fake_create_user)
    return calls


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def as_anonymous() -> None:
    app.dependency_overrides[get_optional_user] = lambda: None


def as_caller(*roles: str) -> None:
    app.dependency_overrides[get_optional_user] = lambda: user_with(*roles)


class TestAnonymousRegistration:
    def test_viewer_is_allowed(self, client, created):
        as_anonymous()

        response = client.post(
            "/auth/register",
            json={"username": "newcomer", "password": "secure123!", "roles": ["viewer"]},
        )

        assert response.status_code == 201
        assert created[0]["roles"] == ["viewer"]

    def test_the_default_is_viewer(self, client, created):
        as_anonymous()

        response = client.post(
            "/auth/register", json={"username": "newcomer", "password": "secure123!"}
        )

        assert response.status_code == 201
        assert created[0]["roles"] == ["viewer"]

    @pytest.mark.parametrize(
        "roles",
        [
            ["admin"],
            ["trading"],
            ["research", "trading"],
            # The interesting one: a privileged role smuggled in beside the
            # permitted one. A membership test on the first element, or an
            # `in` against the list, would let this through.
            ["viewer", "admin"],
        ],
    )
    def test_privileged_roles_are_refused(self, client, created, roles):
        as_anonymous()

        response = client.post(
            "/auth/register",
            json={"username": "escalator", "password": "secure123!", "roles": roles},
        )

        assert response.status_code == 403
        assert created == [], "no user should be created when the grant is refused"

    def test_the_refusal_names_the_offending_roles(self, client, created):
        as_anonymous()

        response = client.post(
            "/auth/register",
            json={"username": "escalator", "password": "secure123!", "roles": ["viewer", "admin"]},
        )

        detail = response.json()["detail"]
        assert "admin" in detail
        assert "viewer" not in detail.split("Assigning")[1]


class TestAdminRegistration:
    def test_an_admin_may_grant_any_role(self, client, created):
        as_caller("admin")

        response = client.post(
            "/auth/register",
            json={"username": "new_trader", "password": "secure123!", "roles": ["trading"]},
        )

        assert response.status_code == 201
        assert created[0]["roles"] == ["trading"]

    def test_a_non_admin_token_grants_nothing_extra(self, client, created):
        """Holding *a* token is not the same as holding an admin one."""
        as_caller("trading", "risk")

        response = client.post(
            "/auth/register",
            json={"username": "new_trader", "password": "secure123!", "roles": ["trading"]},
        )

        assert response.status_code == 403
        assert created == []
