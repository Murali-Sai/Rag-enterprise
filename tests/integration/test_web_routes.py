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

    def test_it_offers_the_dashboard(self, client):
        assert '"/dashboard"' in client.get("/").text


class TestDashboardPage:
    def test_it_renders(self, client):
        response = client.get("/dashboard")

        assert response.status_code == 200
        assert "Query Dashboard" in response.text

    def test_it_links_the_extracted_assets(self, client):
        response = client.get("/dashboard")

        for asset in ("/static/css/tokens.css", "/static/css/dashboard.css", "/static/js/dashboard.js"):
            assert asset in response.text
            assert client.get(asset).status_code == 200

    def test_every_stage_the_script_fills_is_present(self, client):
        """The template is a shell; the script writes into these by id. A
        renamed or dropped id fails as a silently blank section."""
        page = client.get("/dashboard").text

        for element_id in (
            "roleGrid",
            "deptChips",
            "barrierList",
            "questionInput",
            "sourceList",
            "answerText",
            "declinedPanel",
            "claimList",
            "confidencePanel",
            "flagChips",
        ):
            assert f'id="{element_id}"' in page

    def test_it_bakes_in_no_scores(self, client):
        """Nothing on this page is precomputed. A number in the template is a
        number nobody measured."""
        page = client.get("/dashboard").text

        assert "faithfulness" not in page.lower()
        assert "0.6" not in page and "0.7" not in page


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
