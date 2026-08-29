"""The identity application.

    uvicorn identity.app:app --port 8200

Owns authentication for the whole system and nothing else. It stores no grades, no
students, and no chat history — a service holding credentials should be the least
interesting target in the estate, not the most.

## This file is the composition root

Everything that reads the environment, opens a socket or holds a key is built here, once,
at startup, and put on `app.state` for `api/deps.py` to hand to a request. Nothing below
`api/` reads configuration; nothing in `application/` or `domain/` imports FastAPI,
SQLAlchemy or `httpx`.

That is also where the misconfigurations are caught. A signing key that cannot be loaded,
a school with no WhatsApp number, a SIS base URL with no key — each stops the deploy here,
rather than surfacing as one parent's login failing at eight in the morning.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from identity.api import errors
from identity.api.routers import admin, auth, health, wellknown, whatsapp
from identity.config import settings
from identity.env import env_value, load_env

# Before anything below reads the environment. This service is deployed as its own
# process, so nothing else has loaded the project's `.env` for it — see identity/env.py
# for what that cost.
load_env()

from identity.infrastructure.crypto.jwt import JwtTokenIssuer  # noqa: E402
from identity.infrastructure.crypto.keys import signing_key_from  # noqa: E402
from identity.infrastructure.crypto.passwords import Pbkdf2PasswordHasher  # noqa: E402
from identity.infrastructure.db.schema import init_db  # noqa: E402
from identity.infrastructure.directory.fake import FakeGuardianDirectory  # noqa: E402
from identity.infrastructure.directory.sis import SisGuardianDirectory  # noqa: E402
from identity.infrastructure.whatsapp.channels import build_channels  # noqa: E402

logger = logging.getLogger(__name__)


def _build_directory(resolved):
    """Point the guardian directory at the school's system of record.

    Left as the in-memory fake when `IDENTITY_SIS_BASE_URL` is unset, which means an
    unconfigured deployment refuses every parent rather than authenticating them against
    nothing. That is the safe direction: a login that cannot succeed is a support call, and
    a login that succeeds against an empty directory is a stranger holding a token.
    """
    if not resolved.sis_base_url:
        logger.warning(
            "IDENTITY_SIS_BASE_URL is not set; guardian lookups use an empty in-memory "
            "directory and every parent will be told their number is not registered."
        )
        return FakeGuardianDirectory()

    if not resolved.sis_api_key:
        # It used to be demanded at startup, because SIS answered an unkeyed lookup with a
        # 401 that read downstream as "the school has no such number" — every family in
        # the school told they are not registered. SIS no longer authenticates anyone
        # (`sis/api/deps.py`), so an unset key costs nothing today. Warned rather than
        # dropped, because it is the line that has to come back when SIS has sign-in.
        logger.warning(
            "IDENTITY_SIS_BASE_URL is set without IDENTITY_SIS_API_KEY. Harmless only "
            "while SIS authenticates nobody; set it again when SIS has sign-in."
        )

    return SisGuardianDirectory(
        base_url=resolved.sis_base_url,
        api_key=resolved.sis_api_key,
        timeout_seconds=resolved.directory_timeout_seconds,
        children_timeout_seconds=resolved.children_timeout_seconds,
    )


def _cors_origins() -> list[str]:
    """Origins allowed to call identity, from CORS_ALLOW_ORIGINS (comma-separated).

    The same variable `backend/` reads, and deliberately so: the frontend talks to both,
    and two variables would let them disagree about which origin is the UI. It is read
    here rather than imported from `backend.env` because these are independent projects
    with no import in either direction -- see SERVICES.md. The duplication is the price of
    that, and it is one function.

    Unset means "any origin", which is the right default for a token-authenticated API
    that ships not knowing where its UI will live. Set it once the UI has a fixed origin.
    """
    raw = env_value("CORS_ALLOW_ORIGINS")
    if not raw:
        return ["*"]
    # An Origin header never carries a trailing slash, so a pasted
    # "https://aurexis.cc/" would silently match nothing.
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return origins or ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build everything process-wide, once, and take it down cleanly.

    The order matters in one place: the signing key is forced to load before the app
    starts serving, so a misconfigured key fails the deploy rather than the first parent's
    login.
    """
    resolved = settings()
    init_db()

    app.state.settings = resolved

    app.state.signing_key = signing_key_from(resolved)
    # Force the load here rather than lazily. A key that cannot be read must stop the
    # deploy, not the first sign-in.
    app.state.signing_key.kid

    app.state.token_issuer = JwtTokenIssuer(
        key=app.state.signing_key,
        issuer=resolved.issuer,
        audience=resolved.audience,
        access_ttl_minutes=resolved.access_ttl_minutes,
        refresh_ttl_days=resolved.refresh_ttl_days,
    )

    # One hasher for the process. It carries the precomputed dummy hash that equalises a
    # failed login's timing, and computing that costs one key derivation — worth paying
    # once at startup and not once per unknown username.
    app.state.hasher = Pbkdf2PasswordHasher(rounds=resolved.pbkdf2_rounds)

    app.state.channels = build_channels(resolved, _build_directory(resolved))

    try:
        yield
    finally:
        # Close the pooled HTTP clients. Without this, `uvicorn --reload` leaks a
        # connection pool per reload until the process runs out of sockets.
        app.state.channels.close()


app = FastAPI(
    title="School Identity Service",
    version="0.1.0",
    description=(
        "Authentication for the school assistant. The only service that decides who "
        "someone is, and the only one that can mint a token.\n\n"
        "**Tokens are RS256.** Other services verify against `/.well-known/jwks.json` "
        "with a public key and hold nothing that could forge one. That is what keeps "
        "identity resolution out of the chat backend and the records facade.\n\n"
        "**The `guardian_id` claim is the whole integration.** It is set only by an "
        "administrator through `PUT /v1/admin/accounts/{username}/guardian-binding`, "
        "never at self-registration and never from a request body on a public route. "
        "A token without the claim can read no student records at all.\n\n"
        "**Parents sign in through WhatsApp, not with a password.** They ask for a "
        "challenge, send a pre-filled message from their own WhatsApp, and type back a "
        "code. WhatsApp proves they control the number; the school's own records decide "
        "whether that number belongs to a parent. Neither the browser nor the parent ever "
        "states which guardian they are."
    ),
    lifespan=lifespan,
)

# One envelope for every refusal, and one place that decides the status. See api/errors.py.
errors.install(app)

# Cross-origin access for the browser, and the reason it is needed at all: until now the
# Vue app reached identity through Vite's dev proxy (`/v1` -> :8200 in
# frontend/vite.config.ts), which made every call same-origin and hid the fact that this
# service sends no CORS headers. Serve the UI from a real origin and that proxy is gone --
# every login, every WhatsApp start/status/verify becomes a cross-origin request, the
# preflight gets no Access-Control-Allow-Origin back, and the browser refuses it. The
# webhook keeps working throughout, because Meta is a server and never sends an Origin,
# so the failure arrives as "OTP delivered but nobody can sign in".
#
# Added AFTER errors.install so it wraps the refusal envelope too. A 401 without CORS
# headers is unreadable to the caller, which turns every expired token into a generic
# network error in the console instead of the 401 the UI knows how to act on.
#
# `allow_credentials=False`: this API authenticates with a bearer token in the
# Authorization header and sets no cookie. Browsers refuse to combine credentialed mode
# with `allow_origins=["*"]` -- the wildcard is discarded rather than honoured, and every
# preflight fails. Turning it on is only correct alongside an explicit
# CORS_ALLOW_ORIGINS list, and only if this service ever starts setting cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wellknown.router)
app.include_router(auth.router)
app.include_router(whatsapp.router)
app.include_router(admin.router)
app.include_router(health.router)
