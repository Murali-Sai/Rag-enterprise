"""The server-rendered surfaces and the access endpoint behind them.

The landing page used to be a 780-line string literal inside src/main.py.
Moving it into templates is the kind of change that breaks silently — the
route keeps returning 200 while serving a page with no stylesheet — so these
assert on the wiring rather than on the markup: that the template renders,
that the extracted assets are actually reachable at the paths the template
asks for, and that the dashboard's stages are present for its script to fill.

`GET /access` is tested through a dependency override rather than a real
login. The endpoint's job is to turn roles into an access profile; going
through the database would test the seed script instead.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_current_user
from src.auth.models import Role, User
from src.config import settings
from src.main import app


def user_with(*roles: str) -> User:
    user = User(id=1, username="test_user", password_hash="unused")  # noqa: S106
    user.roles = [Role(id=i, name=name) for i, name in enumerate(roles, start=1)]
    return user


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def as_role(*roles: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: user_with(*roles)


class TestLandingPage:
    def test_it_renders(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "an annual report on retrieval" in response.text

    def test_it_links_the_extracted_assets(self, client):
        """The failure this catches: template renders, stylesheet 404s, page
        is unstyled and the route still reports 200."""
        response = client.get("/")

        for asset in ("/static/css/tokens.css", "/static/css/landing.css", "/static/js/landing.js"):
            assert asset in response.text
            assert client.get(asset).status_code == 200

    def test_it_links_out_to_the_dashboard_service(self, client):
        """The dashboard is a separate Streamlit process on another port — and,
        deployed, another host — so a relative link would 404 against the API."""
        page = client.get("/").text

        assert settings.dashboard_url in page
        assert 'href="/dashboard"' not in page

    def test_the_dashboard_url_is_configurable(self, client, monkeypatch):
        monkeypatch.setattr(settings, "dashboard_url", "https://dash.example.test")

        assert "https://dash.example.test" in client.get("/").text

    def test_the_api_no_longer_serves_a_dashboard_route(self, client):
        """Removed when the dashboard moved to Streamlit. A route left behind
        would serve a second, diverging dashboard."""
        assert client.get("/dashboard").status_code == 404


class TestAccessEndpoint:
    def test_it_requires_a_token(self, client):
        assert client.get("/access").status_code == 401

    def test_research_reports_both_walls(self, client):
        as_role("research")

        profile = client.get("/access").json()

        assert profile["accessible_departments"] == ["general", "research", "sec_filings"]
        assert [b["name"] for b in profile["information_barriers"]] == [
            "Research-Trading Wall",
            "Research-Compliance Wall",
        ]
        assert profile["unrestricted"] is False

    def test_barriers_carry_the_departments_they_removed(self, client):
        """The whole reason the field exists: `guardrail_flags` flattens the
        barriers to their names and drops this."""
        as_role("research")

        barriers = client.get("/access").json()["information_barriers"]

        assert barriers[0]["blocked_departments"] == ["trading"]
        assert barriers[1]["blocked_departments"] == ["compliance"]
        assert all(b["description"] for b in barriers)

    def test_trading_is_behind_no_wall(self, client):
        as_role("trading")

        profile = client.get("/access").json()

        assert profile["information_barriers"] == []
        assert "trading" in profile["accessible_departments"]

    def test_admin_is_unrestricted(self, client):
        """The department list still populates. It says what the roles grant,
        not what retrieval filtered on — admin applies no where-clause."""
        as_role("admin")

        profile = client.get("/access").json()

        assert profile["unrestricted"] is True
        assert profile["information_barriers"] == []
        assert "trading" in profile["accessible_departments"]

    def test_a_wall_beats_a_second_role_that_grants_the_department(self, client):
        """The Chinese Wall is absolute: holding `trading` alongside
        `research` does not buy back the trading department."""
        as_role("research", "trading")

        profile = client.get("/access").json()

        assert "trading" not in profile["accessible_departments"]
        assert "compliance" not in profile["accessible_departments"]

    def test_departments_are_ordered(self, client):
        """They come from a set. An arbitrary order would make two identical
        responses differ."""
        as_role("risk")

        departments = client.get("/access").json()["accessible_departments"]

        assert departments == sorted(departments)
