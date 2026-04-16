import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.auth.repository import create_user, init_db
from src.auth.jwt_handler import create_access_token


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def setup_db():
    await init_db()


@pytest.fixture
async def client():
    from src.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def admin_token() -> str:
    return create_access_token(user_id=1, username="admin_user", roles=["admin"])


@pytest.fixture
def hr_token() -> str:
    return create_access_token(user_id=2, username="hr_user", roles=["hr"])


@pytest.fixture
def eng_token() -> str:
    return create_access_token(user_id=3, username="eng_user", roles=["engineering"])


@pytest.fixture
def finance_token() -> str:
    return create_access_token(user_id=4, username="finance_user", roles=["finance"])


@pytest.fixture
def viewer_token() -> str:
    return create_access_token(user_id=5, username="viewer_user", roles=["viewer"])
