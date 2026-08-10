import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.common.logging import get_logger
from src.config import settings

logger = get_logger(__name__)


def client_ip(request: Request) -> str:
    """The caller's address, as seen from behind Cloud Run's front end.

    `get_remote_address` reads `request.client.host`, which on Cloud Run is
    Google's load balancer rather than the visitor — so every visitor in the
    world would share one bucket and the first busy minute would lock the demo
    for everybody. The real address is the first hop of X-Forwarded-For.

    That header is caller-supplied and therefore spoofable: anyone willing to
    rotate it can have a fresh bucket per request. This trades that away
    deliberately. The limit exists to stop casual scripted abuse from running
    up a bill on a public unauthenticated demo, and a determined attacker was
    never going to be stopped by a per-IP counter. Locking out real visitors to
    inconvenience an attacker who can trivially step around it is the worse
    failure of the two.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


# `enabled` is read once, at import. Tests flip `limiter.enabled` directly
# rather than re-reading settings, because this object outlives any monkeypatch
# of the settings singleton.
limiter = Limiter(
    key_func=client_ip,
    enabled=settings.rate_limit_enabled,
    # X-RateLimit-Limit/Remaining/Reset on every limited response, and
    # Retry-After on the 429. Off by default in slowapi, which leaves a caller
    # to discover the limit by hitting it and then guess how long to wait.
    headers_enabled=True,
)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """429 with a body that says what to do about it.

    slowapi's stock handler returns `{"error": "Rate limit exceeded: 20 per 1
    minute"}`, which reads like a rejection. This is a public demo and the
    person hitting the limit is more likely to be someone clicking around than
    someone attacking it, so the body says the limit is per-IP and per-minute
    and that waiting is the fix.
    """
    logger.warning("rate_limit_exceeded", path=request.url.path, limit=str(exc.detail))

    response = JSONResponse(
        status_code=429,
        content={
            "detail": (
                f"Rate limit exceeded ({exc.detail}). This is a public demo and the limit "
                "is per IP address — wait a minute and the request will go through. "
                "Run it locally if you need to query without one."
            )
        },
    )

    # X-RateLimit-* and Retry-After. Private API, and only cosmetic if it
    # moves, so a failure here must not turn a 429 into a 500.
    try:
        return limiter._inject_headers(response, request.state.view_rate_limit)
    except Exception:  # noqa: BLE001
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        # Add request_id to request state
        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        response.headers["X-Request-ID"] = request_id
        return response


def setup_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestLoggingMiddleware)
    app.state.limiter = limiter
    # Without this the RateLimitExceeded raised by @limiter.limit escapes as an
    # unhandled 500. slowapi's own `init_app` registers it; this app wires the
    # limiter by hand, so it has to register it by hand too.
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
