import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.auth.jwt_handler import create_access_token
from src.auth.repository import init_db


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
def trading_token() -> str:
    return create_access_token(user_id=2, username="trader_desk", roles=["trading"])


@pytest.fixture
def research_token() -> str:
    return create_access_token(user_id=3, username="research_analyst", roles=["research"])


@pytest.fixture
def compliance_token() -> str:
    return create_access_token(user_id=4, username="compliance_officer", roles=["compliance"])


@pytest.fixture
def viewer_token() -> str:
    return create_access_token(user_id=5, username="viewer_user", roles=["viewer"])
