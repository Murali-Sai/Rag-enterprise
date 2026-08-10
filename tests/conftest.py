import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.auth.jwt_handler import create_access_token
from src.auth.repository import init_db


@pytest.fixture(autouse=True)
def _rate_limit_off():
    """Off for every test unless one asks for it.

    The limiter keys on client IP and every test shares one, so a suite that
    left it on would have tests failing according to how many requests the
    tests before them happened to make — a green run that goes red when a
    test is added elsewhere in the file. The tests that care about the limit
    turn it back on through `rate_limited`.
    """
    from src.api.middleware import limiter

    previous, limiter.enabled = limiter.enabled, False
    yield
    limiter.enabled = previous


@pytest.fixture
def rate_limited(_rate_limit_off):
    """Turn the limiter on, with the counters cleared.

    Depends on `_rate_limit_off` so it is guaranteed to run after it rather
    than being undone by it.
    """
    from src.api.middleware import limiter

    limiter.enabled = True
    limiter.reset()
    yield limiter
    limiter.reset()


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
