from fastapi import APIRouter

from backend.api.routes.assets import router as assets_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.documents import router as documents_router
from backend.api.routes.health import router as health_router
from backend.api.routes.sessions import router as sessions_router

router = APIRouter()
# First, and unauthenticated: probes are made by load balancers and orchestrators,
# which hold no credentials.
router.include_router(health_router)
# No auth router. Login, registration and token refresh are the identity service's
# (see identity/routes.py); this app only verifies the tokens it is handed.
router.include_router(sessions_router)
router.include_router(chat_router)
router.include_router(documents_router)
router.include_router(assets_router)
