import os
import sys
from pathlib import Path

# Supports both `python backend/app.py` and `uvicorn backend.app:app` startup methods
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.env import PROJECT_ROOT, env_bool, env_value, load_env

load_env()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.router import router
from backend.infra.database import init_db
from backend.profiles import get_profile

FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"

# Response headers a cross-origin caller is allowed to read. The list is short by
# design: only headers a UI acts on belong here.
#
# ETag earns its place. Asset delivery is conditional on it
# (backend/api/routes/assets.py), and the UI fetches asset bytes from JavaScript
# rather than via <img src> because the request needs a bearer token. A fetch()
# that cannot read ETag cannot send If-None-Match on the next request, so every
# image is re-downloaded in full instead of hitting the 304 path.
_EXPOSED_HEADERS = ["ETag", "Content-Disposition"]


def _cors_origins() -> list[str]:
    """Origins allowed to call this API, from CORS_ALLOW_ORIGINS (comma-separated).

    Unset means "any origin", which is the right default for a token-authenticated
    API that ships without knowing where its UI will live. Set it once the UI has a
    fixed origin.
    """
    raw = env_value("CORS_ALLOW_ORIGINS")
    if raw is None:
        return ["*"]
    # An Origin header carries no trailing slash, so a pasted "https://app.example.com/"
    # would silently never match.
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return origins or ["*"]


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

        # Build the embedder now. It used to be constructed at import of
        # backend.indexing.embedding, which made merely importing most of this backend
        # cost a model load — including in processes that never embed anything, and in
        # one configured to embed remotely. Construction is lazy now, so this is where
        # a serving process pays it: at boot, before traffic, and visibly in the log
        # rather than inside whichever request happened to be first.
        from backend.indexing.embedding import embedding_service

        embedding_service.warm_up()

    # This API authenticates with a bearer token in the Authorization header
    # (backend/infra/auth.py), never with a cookie. That is what settles the
    # credentials flag: browsers refuse to combine `allow_credentials=True` with
    # `allow_origins=["*"]` — the wildcard is discarded rather than honoured, so
    # every cross-origin preflight fails. Nothing here needs credentialed mode,
    # so it stays off and the wildcard keeps working for any UI that sends a token.
    #
    # Turning it back on is only correct alongside an explicit CORS_ALLOW_ORIGINS
    # list, and only if this API ever starts setting cookies.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=_EXPOSED_HEADERS,
    )

    # Serving the bundled UI is a decision, not an accident of which files landed in
    # the image. Keying it solely on FRONTEND_DIR.exists() means a broad `COPY . .`
    # silently turns an API-only deployment into a web server for whatever stale
    # bundle it picked up — and, because the mount answers "/" with index.html and a
    # 200, a load balancer health-checking "/" then passes no matter what state the
    # backend is in. With SERVE_FRONTEND=false, "/" is a 404 like any other unrouted
    # path and the check has to name a real endpoint.
    serve_frontend = env_bool("SERVE_FRONTEND", True) and FRONTEND_DIR.exists()

    if serve_frontend:
        # Scoped to the static mount: every path it tests for belongs to the bundled
        # UI, so an API-only process would run it on each request to no effect.
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

    if serve_frontend:
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
