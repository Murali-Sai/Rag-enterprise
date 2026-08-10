"""The ceiling on `POST /documents/ingest`.

`UploadFile` spools to disk past a threshold, so an unbounded upload is a
disk-fill rather than a memory exhaustion — slower to notice and no less
effective. Ingest is off in the deployment, so this guards the from-source and
compose paths, where the admin credentials are still the published ones.

The size is counted as bytes are written, not read off `file.size`: that is the
client's Content-Length, and a caller who lies about it walks straight past a
check that trusts it.
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.deps import get_current_user
from src.auth.models import Role, User
from src.config import settings
from src.main import app

CAP = 4096


def admin() -> User:
    user = User(id=1, username="admin_user", password_hash="unused")  # noqa: S106
    user.roles = [Role(id=1, name="admin")]
    return user


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "allow_runtime_ingest", True)
    monkeypatch.setattr(settings, "max_upload_bytes", CAP)
    monkeypatch.setattr("src.api.routes.documents.ingest_document", lambda **kwargs: ["chunk"])
    app.dependency_overrides[get_current_user] = admin
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def upload(client, payload: bytes):
    return client.post(
        "/documents/ingest",
        params={"department": "sec_filings", "access_roles": "viewer"},
        files={"file": ("filing.txt", payload, "text/plain")},
    )


class TestUploadCap:
    def test_a_file_over_the_cap_is_refused(self, client):
        response = upload(client, b"x" * (CAP * 3))

        assert response.status_code == 413
        assert "upload limit" in response.json()["detail"]

    def test_a_file_under_the_cap_still_ingests(self, client):
        """The cap must not be the reason local document upload stops working."""
        response = upload(client, b"Fabricated filing content.")

        assert response.status_code == 201
        assert response.json()["chunks_created"] == 1

    def test_an_oversized_upload_never_reaches_the_pipeline(self, client, monkeypatch):
        """Ingestion embeds, and embedding bills. The refusal has to happen
        before that, not as a validation step after it."""
        called = []
        monkeypatch.setattr(
            "src.api.routes.documents.ingest_document",
            lambda **kwargs: called.append(kwargs),
        )

        upload(client, b"x" * (CAP * 3))

        assert called == []

    def test_the_rejected_file_is_not_left_on_disk(self, client, monkeypatch):
        """An upload refused for being too large that still consumed the disk
        the cap protects is the same failure wearing a 413."""
        created: list[str] = []
        real = tempfile.NamedTemporaryFile

        def track(*args, **kwargs):
            handle = real(*args, **kwargs)
            created.append(handle.name)
            return handle

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", track)

        upload(client, b"x" * (CAP * 3))

        assert created, "expected the route to have opened a temp file"
        assert not any(Path(name).exists() for name in created)
