from fastapi import APIRouter

from src.common.schemas import HealthResponse, ReadinessResponse
from src.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        environment=settings.environment.value,
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    vector_store_status = "unknown"
    db_status = "unknown"
    llm_status = "unknown"

    # Check vector store
    try:
        from src.retrieval.vector_store import get_vector_store
        get_vector_store()
        vector_store_status = "connected"
    except Exception:
        vector_store_status = "disconnected"

    # Check database
    try:
        from src.auth.repository import async_session
        async with async_session() as session:
            await session.execute("SELECT 1")  # type: ignore[arg-type]
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    # Check LLM (just verify config is present)
    if settings.groq_api_key or settings.google_api_key:
        llm_status = "configured"
    else:
        llm_status = "not_configured"

    overall = "ready" if vector_store_status == "connected" else "not_ready"

    return ReadinessResponse(
        status=overall,
        vector_store=vector_store_status,
        database=db_status,
        llm=llm_status,
    )
