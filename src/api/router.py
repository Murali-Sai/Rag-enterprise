from fastapi import APIRouter

from src.api.routes import access, admin, auth, documents, health, query, v1

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(access.router)
api_router.include_router(query.router)
api_router.include_router(documents.router)
api_router.include_router(admin.router)
# Last so the paths it shadows are already registered. See routes/v1.py: these
# are the endpoint names Project 6 §5.1 asks for, delegating to the handlers
# above rather than reimplementing them.
api_router.include_router(v1.router)
