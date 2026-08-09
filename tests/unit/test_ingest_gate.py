"""Whether a running deployment will accept documents into its index.

`POST /documents/ingest` requires the admin role, which is the right control
almost everywhere and the wrong one here: the demo publishes its admin
credentials — in the README, and on the landing page as a button — so a visitor
can watch the information barriers come down. That makes admin a public role,
and a public role must not be able to write to a corpus the README fingerprints
by digest.

`allow_runtime_ingest` is the switch, off in the image. These pin both sides of
it, because the failure mode is silent: the endpoint keeps returning 201 and
the corpus stops being the one the numbers describe.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_current_user
from src.auth.models import Role, User
from src.config import settings
from src.main import app


def admin() -> User:
    user = User(id=1, username="admin_user", password_hash="unused")  # noqa: S106
    user.roles = [Role(id=1, name="admin")]
    return user


@pytest.fixture
def client():
    app.dependency_overrides[get_current_user] = admin
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def upload():
    return {"file": ("memo.txt", b"Fabricated filing content.", "text/plain")}


def ingest(client, upload):
    return client.post(
        "/documents/ingest",
        params={"department": "sec_filings", "access_roles": "viewer"},
        files=upload,
    )


class TestIngestGate:
    def test_a_read_only_deployment_refuses_an_admin_upload(
        self, client, upload, monkeypatch
    ):
        monkeypatch.setattr(settings, "allow_runtime_ingest", False)

        response = ingest(client, upload)

        assert response.status_code == 403
        assert "does not accept document uploads" in response.json()["detail"]

    def test_the_refusal_happens_before_the_file_is_read(
        self, client, upload, monkeypatch
    ):
        """The gate is not a validation step — nothing should reach the
        ingestion pipeline, which embeds (and bills) before it writes."""
        monkeypatch.setattr(settings, "allow_runtime_ingest", False)
        called = []
        monkeypatch.setattr(
            "src.api.routes.documents.ingest_document",
            lambda **kwargs: called.append(kwargs),
        )

        ingest(client, upload)

        assert called == []

    def test_local_runs_still_accept_uploads(self, client, upload, monkeypatch):
        """The default is on: turning the deployment read-only must not take
        document ingestion away from someone running this locally."""
        monkeypatch.setattr(settings, "allow_runtime_ingest", True)
        monkeypatch.setattr(
            "src.api.routes.documents.ingest_document",
            lambda **kwargs: ["chunk"],
        )

        response = ingest(client, upload)

        assert response.status_code == 201
        assert response.json()["chunks_created"] == 1
