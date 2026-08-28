"""Request-scoped wiring, and the two credentials every parent-facing read carries.

The **driving** side of the hexagon. A router declares `service: RecordsServiceDep` and
`subject: ParentSubjectDep` and receives objects that are already authenticated and
already bound to this deployment's adapters; it never reads configuration and never learns
which system of record it is talking to.

## The rule this file exists to enforce

> An API key proves **which system** is calling. It never proves **which parent** is
> asking. Both are required before a single grade is returned.

The token's `guardian_id` claim must equal the `guardian_id` in the path. That equality is
what stops the calling system choosing whose records it reads: it can only relay a
parent's own token, because it cannot produce a signature for a different one. A fully
compromised chat backend still reaches one family rather than the school.

Verification is **offline** against a public key, so this service holds nothing that could
mint a token and identity being down does not take records down. It **fails closed**: with
no verification material configured, every parent-facing read is a 503 rather than a
fallback to trusting the path.

## What is built once, and what per request

The adapters — and the pooled HTTP clients inside them — are process-wide, built by the
composition root and read off `app.state`. Only the thin service objects are per request,
because they hold nothing but references. Rebuilding an adapter per request would mean a
new connection pool, and therefore a TCP and TLS handshake, on every parent's question.
"""
from __future__ import annotations

import hmac
import logging
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from records.application import audit
from records.application.access import AccessService
from records.application.reads import RecordsService
from records.config import Settings, api_key, settings
from records.ports.calendar import SchoolCalendar
from records.ports.directory import GuardianDirectory
from records.ports.lms import LmsAdapter

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

#: How much of a presented key may appear in a log line. Enough to tell two callers apart
#: while an operator is reading; useless for authenticating.
_PREFIX_LENGTH = 8


class ServiceCaller:
    """The authenticated *system* behind a request. Never a person."""

    __slots__ = ("prefix", "request_id")

    def __init__(self, prefix: str, request_id: str = ""):
        self.prefix = prefix
        self.request_id = request_id


class ParentSubject:
    """A verified (system, parent) pair. Both halves proved, neither assumed."""

    __slots__ = ("caller", "guardian_id", "school_code")

    def __init__(self, caller: ServiceCaller, guardian_id: str, school_code: str | None = None):
        self.caller = caller
        self.guardian_id = guardian_id
        #: Which school's database answers for this parent, off the token's `school` claim.
        #: `None` in a single-school estate. Carried rather than looked up, because it was
        #: settled at sign-in from the WhatsApp number the parent messaged and nothing
        #: since has been in a position to know better.
        self.school_code = school_code


# ---------------------------------------------------------------------------
# Process-wide, from `app.state`.
# ---------------------------------------------------------------------------


def get_settings() -> Settings:
    return settings()


def get_lms(request: Request) -> LmsAdapter:
    return request.app.state.lms


def get_directory(request: Request) -> GuardianDirectory:
    return request.app.state.directory


def get_calendar(request: Request) -> SchoolCalendar:
    return request.app.state.calendar


def get_records_service(request: Request) -> RecordsService:
    """One service over this deployment's adapters.

    Cheap enough to build per request — it is four attribute assignments and two stateless
    assemblers — and deliberately not cached, so a test that swaps an adapter on
    `app.state` takes effect on the next call rather than on the next process.
    """
    state = request.app.state
    return RecordsService(
        access=AccessService(state.directory),
        calendar=state.calendar,
        lms=state.lms,
        policy=state.policy,
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
RecordsServiceDep = Annotated[RecordsService, Depends(get_records_service)]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def require_agent(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
) -> ServiceCaller:
    """Prove which system is calling. Every route here depends on this.

    Fails closed when `RECORDS_API_KEY` is unset: an unconfigured deployment refuses
    everything rather than admitting everyone. The alternative — treating "no key
    configured" as "no key required" — is how a service ships open.
    """
    presented = (x_api_key or "").strip()
    expected = api_key()
    request_id = (x_request_id or "").strip()

    if not expected:
        logger.error(
            "RECORDS_API_KEY is not set; every request is refused. Set it to the secret "
            "the chat backend presents."
        )
        audit.refused(
            audit.NOT_AUTHORIZED, endpoint=str(request.url.path), request_id=request_id
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "not_configured", "message": "This service is not configured."},
        )

    # Compared as bytes and in constant time. `compare_digest` raises `TypeError` on a
    # `str` holding non-ASCII, and this header is entirely caller-controlled — the str
    # form turns one crafted request into a 500 instead of a refusal.
    if not presented or not hmac.compare_digest(
        presented.encode("utf-8"), expected.encode("utf-8")
    ):
        audit.refused(
            audit.NOT_AUTHORIZED, endpoint=str(request.url.path), request_id=request_id
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Missing or invalid API key."},
        )

    return ServiceCaller(prefix=presented[:_PREFIX_LENGTH], request_id=request_id)


def require_parent_subject(
    guardian_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    caller: ServiceCaller = Depends(require_agent),
) -> ParentSubject:
    """Both credentials, checked together. Every parent-facing route depends on this.

    FastAPI supplies `guardian_id` from the path, so a route that declares this dependency
    without a `{guardian_id}` segment fails at startup rather than silently skipping the
    comparison.
    """
    from records.api import identity

    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()

    endpoint = str(request.url.path)

    try:
        claims = identity.verify_token(token)
        claimed_guardian = identity.guardian_id_from_claims(claims)
    except identity.IdentityNotConfigured as exc:
        # Fail closed. No verification material means no reads, not unverified reads.
        logger.error("Identity verification is not configured: %s", exc)
        audit.refused(
            audit.IDENTITY_NOT_CONFIGURED,
            endpoint=endpoint,
            guardian_id=guardian_id,
            request_id=caller.request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "not_configured", "message": "Identity verification unavailable."},
        )
    except identity.IdentityError:
        audit.refused(
            audit.INVALID_IDENTITY,
            endpoint=endpoint,
            guardian_id=guardian_id,
            request_id=caller.request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Missing or invalid identity token."},
        )

    if claimed_guardian != guardian_id:
        # The signature was valid but named someone else. This is the signal that a caller
        # is relaying one parent's token while asking about another, and it gets its own
        # reason so it can be alerted on by itself.
        audit.refused(
            audit.GUARDIAN_MISMATCH,
            endpoint=endpoint,
            guardian_id=guardian_id,
            request_id=caller.request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "not_authorized", "message": "Token does not authorise this guardian."},
        )

    return ParentSubject(
        caller=caller,
        guardian_id=claimed_guardian,
        school_code=identity.school_from_claims(claims),
    )


AgentCaller = Annotated[ServiceCaller, Depends(require_agent)]
ParentSubjectDep = Annotated[ParentSubject, Depends(require_parent_subject)]


__all__ = [
    "API_KEY_HEADER",
    "AgentCaller",
    "ParentSubject",
    "ParentSubjectDep",
    "RecordsServiceDep",
    "ServiceCaller",
    "SettingsDep",
    "require_agent",
    "require_parent_subject",
]
