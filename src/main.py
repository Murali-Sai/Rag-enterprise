from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.middleware import setup_middleware
from src.api.router import api_router
from src.auth.repository import init_db
from src.common.exceptions import GuardrailViolation, RAGEnterpriseError
from src.common.logging import get_logger, setup_logging
from src.config import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    # Startup
    setup_logging()
    logger.info("starting_app", environment=settings.environment.value)

    await init_db()
    logger.info("database_ready")

    yield

    # Shutdown
    logger.info("shutting_down")


app = FastAPI(
    title="RAG Enterprise",
    description="Production-grade RAG system with RBAC, guardrails, and evaluation",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

setup_middleware(app)
app.include_router(api_router)


@app.exception_handler(GuardrailViolation)
async def guardrail_handler(request: Request, exc: GuardrailViolation) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": "guardrail_violation",
            "violation_type": exc.violation_type,
            "message": str(exc),
        },
    )


@app.exception_handler(RAGEnterpriseError)
async def app_error_handler(request: Request, exc: RAGEnterpriseError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": str(exc)},
    )
