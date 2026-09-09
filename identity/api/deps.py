"""Request-scoped wiring: which transaction, which use case, and who is calling.

Composition happens here. A router declares `service: SessionServiceDep` and receives an
object already bound to a transaction and already holding its policy; it never constructs
a repository, never reads `identity.config`, and never learns which hasher or which gateway
it is working through. That is what makes every service in `application/` testable with
fakes: the only place that knows an environment and a database exist is this file, and a
test replaces it wholesale through `app.dependency_overrides`.

This is `sis/api/deps.py`'s shape, and it replaces two mechanisms this service used before:

**The old `deps.py`** built one `VerificationService` per request out of module-level
globals in `whatsapp.py` and `guardians.py`. Those globals are now a `WhatsAppChannels`
object on `app.state`, built once at startup — so what a request is wired to is a value
rather than whatever the last `set_gateway()` call left behind.

**Configuration read in three places.** `auth.py`, `tokens.py` and `keys.py` each read
`os.getenv` at import. Everything now comes from `settings()`, resolved lazily and cached,
and nothing below `api/` reads it at all.

## What is built once, and what is built per request

The expensive, process-wide things — the signing key, the password hasher's precomputed
dummy hash, the pooled HTTP clients — live on `app.state` and are built at startup.
Rebuilding any of them per request would mean an RSA key parse, a 100ms key derivation and
a TLS handshake on every parent's sign-in.

The cheap, transaction-bound things — the repositories and the services over them — are
built per request, because they hold a session that belongs to that request and nothing
else. A cached service would pin a closed session and serve one request's transaction to
the next.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from identity.application.dto import SchoolChannel, TokenSubject
from identity.application.ports.directory import GuardianDirectory
from identity.application.services.administration import AdministrationService
from identity.application.services.parent_sessions import ParentSessionService
from identity.application.services.sessions import SessionService
from identity.application.services.whatsapp_login import WhatsAppLoginService
from identity.config import Settings, settings
from identity.domain.accounts import LockoutPolicy
from identity.infrastructure.crypto.jwt import JwtTokenIssuer
from identity.infrastructure.crypto.keys import SigningKey
from identity.infrastructure.crypto.passwords import Pbkdf2PasswordHasher
from identity.infrastructure.db.repositories import (
    SqlAccountRepository,
    SqlAuditSink,
    SqlChallengeRepository,
    SqlRefreshTokenRepository,
)
from identity.infrastructure.db.session import get_db
from identity.infrastructure.whatsapp.channels import WhatsAppChannels

# ---------------------------------------------------------------------------
# Process-wide, from `app.state`. Built once by the lifespan; see `app.py`.
# ---------------------------------------------------------------------------


def get_settings() -> Settings:
    """Resolved configuration. Cached, so this is a dict lookup per request."""
    return settings()


def get_signing_key(request: Request) -> SigningKey:
    return request.app.state.signing_key


def get_hasher(request: Request) -> Pbkdf2PasswordHasher:
    return request.app.state.hasher


def get_channels(request: Request) -> WhatsAppChannels:
    return request.app.state.channels


def get_directory(request: Request) -> GuardianDirectory:
    return request.app.state.channels.directory


def get_token_issuer(request: Request) -> JwtTokenIssuer:
    return request.app.state.token_issuer


SettingsDep = Annotated[Settings, Depends(get_settings)]
SigningKeyDep = Annotated[SigningKey, Depends(get_signing_key)]
HasherDep = Annotated[Pbkdf2PasswordHasher, Depends(get_hasher)]
ChannelsDep = Annotated[WhatsAppChannels, Depends(get_channels)]
DirectoryDep = Annotated[GuardianDirectory, Depends(get_directory)]
TokenIssuerDep = Annotated[JwtTokenIssuer, Depends(get_token_issuer)]
DbDep = Annotated[Session, Depends(get_db)]


# ---------------------------------------------------------------------------
# Request-scoped: one session, and the services over it.
# ---------------------------------------------------------------------------


def get_session_service(
    db: DbDep,
    hasher: HasherDep,
    issuer: TokenIssuerDep,
    resolved: SettingsDep,
) -> SessionService:
    """Password login, registration, refresh and logout, bound to this transaction."""
    return SessionService(
        accounts=SqlAccountRepository(db),
        refresh_tokens=SqlRefreshTokenRepository(db),
        audit=SqlAuditSink(db),
        hasher=hasher,
        issuer=issuer,
        lockout=LockoutPolicy(
            max_failed_attempts=resolved.max_failed_attempts,
            lockout_minutes=resolved.lockout_minutes,
        ),
    )


def get_admin_service(db: DbDep, hasher: HasherDep) -> AdministrationService:
    return AdministrationService(
        accounts=SqlAccountRepository(db),
        refresh_tokens=SqlRefreshTokenRepository(db),
        audit=SqlAuditSink(db),
        hasher=hasher,
    )


def get_whatsapp_service(
    db: DbDep,
    channels: ChannelsDep,
    resolved: SettingsDep,
) -> WhatsAppLoginService:
    """The verification flow, over this request's transaction and the process's channels.

    The channel *resolver* is passed rather than a fixed channel, so the school is chosen
    per call from what the request actually proves — the login page for `start`, the
    WhatsApp number the message arrived on for `claim` — instead of being fixed when this
    object is built.
    """
    return WhatsAppLoginService(
        challenges=SqlChallengeRepository(db),
        channel_for=channels.channel_for,
        ttl_minutes=resolved.verification_ttl_minutes,
    )


def get_parent_session_service(
    db: DbDep,
    directory: DirectoryDep,
    sessions: Annotated[SessionService, Depends(get_session_service)],
) -> ParentSessionService:
    return ParentSessionService(
        accounts=SqlAccountRepository(db),
        sessions=sessions,
        directory=directory,
        audit=SqlAuditSink(db),
    )


SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]
AdminServiceDep = Annotated[AdministrationService, Depends(get_admin_service)]
WhatsAppServiceDep = Annotated[WhatsAppLoginService, Depends(get_whatsapp_service)]
ParentSessionServiceDep = Annotated[
    ParentSessionService, Depends(get_parent_session_service)
]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------




#: Declared so the OpenAPI document says how to authenticate, which is what puts the
#: "Authorize" button on /docs. Until this existed identity published NO security scheme at
#: all — every token was read from a raw `Header` parameter — so the one workflow the admin
#: routes are for, an operator managing accounts through Swagger, could not be performed
#: there: there was nowhere to put the token.
#:
#: `auto_error=False` on purpose. Left to raise on its own, `HTTPBearer` answers with
#: FastAPI's plain-string `{"detail": "Not authenticated"}`, and every refusal this service
#: makes is a structured `{"code": ..., "message": ...}` that the Vue app reads a `code` out
#: of (see `refusal()` in frontend/src/stores/auth.ts). Handling the empty case below keeps
#: the response byte-identical to what callers already get.
_bearer_scheme = HTTPBearer(
    auto_error=False,
    description="An access token from POST /v1/auth/login.",
)


def bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """The token out of an `Authorization: Bearer` header, or a 401.

    The refusal is unchanged from when this read the header itself: same status, same
    `code`, same `message`. Only what the OpenAPI document advertises is different.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Missing bearer token."},
        )
    return credentials.credentials


BearerToken = Annotated[str, Depends(bearer_token)]


def require_admin_token(
    token: BearerToken,
    sessions: SessionServiceDep,
) -> TokenSubject:
    """Guards account management and — critically — guardian binding.

    Replaces `require_admin_key`, which took a shared `X-Admin-Key` secret. What changed,
    and what it costs, stated plainly because both matter:

    **What was lost.** The admin routes used to be separated from the public ones by
    CREDENTIAL TYPE: a parent's token could not reach them at all, because they did not
    accept a bearer token as a credential in the first place. Now they do, and a parent's
    token is turned away on its `role` claim instead. That is a weaker kind of separation
    for the most sensitive write in the system, and it is the price of removing the key.

    **What was gained, and why it is worth more.** A shared secret has no identity. The
    binding route's own docstring says it is audited because "who decided this parent is
    that guardian" is the first question anyone asks after a records leak — and with one
    key held by every script and every operator, that question had no answer. A token names
    an account. The credential can also now be changed through the API rather than by a
    redeploy, and it is subject to the lockout policy every other login obeys.

    The subject is returned rather than discarded so the answer to that question can be
    recorded: callers have the administrator's username without decoding anything.

    Verified through `SessionService.describe_token`, which is the same path `/v1/auth/me`
    uses — signature, issuer, audience and expiry all checked by `decode_own_token`. Not a
    second decoder, because two decoders drift and only one of them gets the audience check
    right.
    """
    try:
        subject = sessions.describe_token(token)
    except ValueError:
        # Bad signature, wrong issuer or audience, or expired. Deliberately one answer:
        # telling a caller WHICH of those failed tells an attacker which half they got right.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid or expired token."},
        ) from None

    if subject.role != "admin":
        # 403, not 401: the credential is genuine and re-presenting it will not help. This
        # matches `backend/infra/auth.py` require_admin, so one role check does not mean two
        # different things in two services.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "not_authorized",
                "message": "Administrator access required.",
            },
        )

    return subject


AdminSubject = Annotated[TokenSubject, Depends(require_admin_token)]


def client_ip(request: Request) -> str:
    """The caller's address, for the audit line. `""` when there is no client."""
    return request.client.host if request.client else ""


ClientIp = Annotated[str, Depends(client_ip)]


__all__ = [
    "AdminSubject",
    "AdminServiceDep",
    "BearerToken",
    "ChannelsDep",
    "ClientIp",
    "DbDep",
    "DirectoryDep",
    "ParentSessionServiceDep",
    "SessionServiceDep",
    "SettingsDep",
    "SigningKeyDep",
    "TokenIssuerDep",
    "WhatsAppServiceDep",
    "bearer_token",
    "client_ip",
    "require_admin_token",
]
