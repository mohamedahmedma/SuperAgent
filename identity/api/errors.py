"""The single translation from a domain error to an HTTP response.

Every service in `application/` raises the errors defined in `identity/domain/errors.py`,
and none of them decides what status code that becomes. The mapping lives here, once,
because the alternative is the shape this service had before: an `HTTPException` written
inline at each of nineteen places, agreeing on 401 in most of them and quietly differing
in the rest — so a caller meets the same failure under two status codes depending on which
URL produced it, and writes retry logic against whichever one they met first.

**One shape on the wire, whichever mechanism produced it.** FastAPI renders a raised
`HTTPException(detail={...})` as `{"detail": {"code": ..., "message": ...}}`, so every
handler below produces exactly that envelope — including for a bare 404 from the router
table and for an unhandled crash.

**Status is resolved by walking the exception's MRO**, not by an exhaustive table. A new
`VerificationError` subclass added next year is a 400 the day it is written, without
anyone remembering to edit this file — and a mapping that silently defaulted such an error
to 500 would ship a wrong status with no test failing.

## Why the verification failures all share one status

`VerificationNotFound`, `BadCode` and `VerificationExpired` would each justify a different
status in isolation. They deliberately get the same one: distinguishing them by status
lets somebody holding a stolen poll secret learn which of their guesses was *structurally*
wrong rather than merely incorrect, which is the difference between guessing a code and
mapping the flow. The `code` in the body still tells a legitimate page what to say, because
a legitimate page holds the poll secret it was given.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from identity.domain.errors import (
    AccountLocked,
    BadRequest,
    Conflict,
    DependencyUnavailable,
    Forbidden,
    IdentityError,
    NotAuthorized,
    NotConfigured,
    NotFound,
    SchoolsMisconfigured,
    UnknownSchool,
    VerificationError,
)

logger = logging.getLogger(__name__)


_STATUS_BY_ERROR: Final[Mapping[type[IdentityError], int]] = {
    NotAuthorized: status.HTTP_401_UNAUTHORIZED,
    # 403, not 401: the caller is understood and is asking for something they may not
    # have. Answering 401 sends an operator who mistyped an invite code hunting
    # through credentials that are correct.
    Forbidden: status.HTTP_403_FORBIDDEN,
    # 423 rather than 401, and safe to distinguish where the two causes of a 401 are not:
    # an attacker who triggered the lockout already knows they did, and the parent who is
    # locked out needs telling to wait rather than to keep retyping a correct password.
    AccountLocked: status.HTTP_423_LOCKED,
    NotFound: status.HTTP_404_NOT_FOUND,
    # A school this server does not serve is the caller's problem to fix, not the
    # operator's — which is what separates it from `NotConfigured` below.
    UnknownSchool: status.HTTP_404_NOT_FOUND,
    Conflict: status.HTTP_409_CONFLICT,
    BadRequest: status.HTTP_400_BAD_REQUEST,
    # Every refusal in the WhatsApp flow, under one status. See the module docstring.
    VerificationError: status.HTTP_400_BAD_REQUEST,
    # 503, not 400: nothing the caller sent is wrong, and nothing they can send will help.
    # The operator has to configure the school's number. The `code` is stable so the
    # sign-in screen can say something a parent can act on — "contact the school" — rather
    # than showing them a server message about an environment variable.
    NotConfigured: status.HTTP_503_SERVICE_UNAVAILABLE,
    SchoolsMisconfigured: status.HTTP_503_SERVICE_UNAVAILABLE,
    DependencyUnavailable: status.HTTP_503_SERVICE_UNAVAILABLE,
    IdentityError: status.HTTP_400_BAD_REQUEST,
}


def status_for(error: IdentityError) -> int:
    """The status for this error, or the nearest ancestor's. Never raises."""
    for klass in type(error).__mro__:
        found = _STATUS_BY_ERROR.get(klass)  # type: ignore[arg-type]
        if found is not None:
            return found
    return status.HTTP_400_BAD_REQUEST


def _body(code: str, message: str) -> dict[str, Any]:
    return {"detail": {"code": code, "message": message}}


def install(app: FastAPI) -> None:
    """Register every handler. Called once, by the composition root."""

    @app.exception_handler(IdentityError)
    async def _identity_error(_: Request, error: IdentityError) -> JSONResponse:
        code = status_for(error)
        if code >= 500:
            # An operator has to act on these, so they are logged with a stack. A 4xx is
            # an ordinary answer to an ordinary caller and is not worth a log line each.
            logger.error("Refusing with %s: %s", code, error.message, exc_info=error)
        return JSONResponse(status_code=code, content=_body(error.code, error.message))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, error: RequestValidationError) -> JSONResponse:
        """A malformed body, in the same envelope as everything else.

        422 rather than 400 so a bad value caught by pydantic and a bad value caught by a
        domain rule answer alike. The caller's question is "did I send something usable",
        and it deserves one answer.
        """
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_body("bad_request", _first_validation_message(error)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, error: StarletteHTTPException) -> JSONResponse:
        """Anything raised as an `HTTPException`, plus the router table's own 404/405.

        A handler that raises `HTTPException(detail={"code": ...})` — which the admin-key
        dependency does — passes its detail straight through. A bare 404 from the router
        gets the same envelope built for it, so a client never has to parse two shapes.
        """
        detail = error.detail
        if isinstance(detail, dict) and "code" in detail:
            return JSONResponse(
                status_code=error.status_code, content={"detail": detail}
            )
        return JSONResponse(
            status_code=error.status_code,
            content=_body(_code_for_status(error.status_code), str(detail or "")),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, error: Exception) -> JSONResponse:
        """A crash, in the same envelope, saying nothing about what crashed.

        The stack goes to the log; the caller gets a code and a sentence. An exception
        message rendered to the client is how a stack trace ends up in a screenshot in a
        support ticket, and this service's exceptions can quote a username.
        """
        logger.exception("Unhandled error in the identity service")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_body("internal_error", "Something went wrong."),
        )


def _first_validation_message(error: RequestValidationError) -> str:
    """One readable sentence out of pydantic's list, or a generic one."""
    try:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = str(first.get("msg", "")).strip()
        return f"{location}: {message}" if location else message
    except Exception:  # noqa: BLE001 - never fail while reporting a failure
        return "That request could not be read."


def _code_for_status(code: int) -> str:
    return {
        400: "bad_request",
        401: "not_authorized",
        403: "not_authorized",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        423: "locked",
        503: "not_configured",
    }.get(code, "error")


__all__ = ["install", "status_for"]
