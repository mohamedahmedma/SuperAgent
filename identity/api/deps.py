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

import hmac
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from identity.application.dto import SchoolChannel
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
        admin_invite_code=resolved.admin_invite_code,
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


def require_admin_key(
    resolved: SettingsDep,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> str:
    """Guards account creation and — critically — guardian binding.

    A shared secret rather than a token, because these routes are called by the
    registrar's tooling and by migration scripts, not by a logged-in user. It is the
    credential that can bind an account to a guardian, so it is the most dangerous secret
    in the system: anyone holding it can make themselves any parent.

    Compared with `compare_digest`, not `==`. The comparison is against a value an
    attacker supplies and can vary a byte at a time, which is exactly the shape a timing
    oracle needs.

    **An unset key is a 503, not a 401.** Nothing the caller sends will help, and telling
    them "invalid admin key" for a server that has no admin key sends an operator hunting
    for a wrong value rather than a missing one.
    """
    expected = resolved.admin_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "not_configured", "message": "IDENTITY_ADMIN_KEY is not set."},
        )
    if not x_admin_key or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid admin key."},
        )
    return x_admin_key


AdminKey = Annotated[str, Depends(require_admin_key)]


def bearer_token(authorization: str | None = Header(default=None)) -> str:
    """The token out of an `Authorization: Bearer` header, or a 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Missing bearer token."},
        )
    return authorization.split(" ", 1)[1].strip()


BearerToken = Annotated[str, Depends(bearer_token)]


def client_ip(request: Request) -> str:
    """The caller's address, for the audit line. `""` when there is no client."""
    return request.client.host if request.client else ""


ClientIp = Annotated[str, Depends(client_ip)]


__all__ = [
    "AdminKey",
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
    "require_admin_key",
]
