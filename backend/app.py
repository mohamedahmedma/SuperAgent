import os
import sys
from pathlib import Path

# Supports both `python backend/app.py` and `uvicorn backend.app:app` startup methods
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.env import PROJECT_ROOT, load_env

load_env()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.router import router
from backend.infra.database import init_db
from backend.profiles import get_profile

FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"


def create_app() -> FastAPI:
    profile = get_profile()

    # LangSmith reads LANGSMITH_PROJECT from the environment itself, so the profile
    # can only supply it as a default — an explicit env var still wins.
    os.environ.setdefault("LANGSMITH_PROJECT", profile.identity.langsmith_project)

    app = FastAPI(
        title=profile.identity.api_title,
        description=profile.identity.description,
    )

    @app.on_event("startup")
    async def _startup_init_db():
        init_db()
        # create_all() creates missing TABLES but never adds columns to existing ones,
        # so a model change ships silently and surfaces as UndefinedColumn partway
        # through a user's upload. Report it at boot instead.
        from backend.db.migrate import check_and_report

        check_and_report()
        # Vision misconfiguration is otherwise silent: a profile can ask for it, the
        # credentials can be missing, and extraction quietly degrades forever. One
        # line at boot makes the state visible.
        if profile.assets.enabled:
            from backend.assets.vision import log_vision_status

            log_vision_status(profile)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(router)

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
