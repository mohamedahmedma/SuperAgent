"""The single translation from a domain error to an HTTP response.

Every use case raises the errors in `records/domain/errors.py`, and none of them decides
what status that becomes. The mapping lives here, once — the alternative is what this
service had: an `HTTPException` written inline at fourteen places, agreeing on 503 in most
of them and quietly differing in the rest, so a caller meets the same failure under two
status codes depending on which URL produced it.

**One shape on the wire, whichever mechanism produced it**, and it is the shape the agent
above already parses: `{"detail": {"code": ..., "message": ...}}`. That includes a bare
404 from the router table and an unhandled crash, so the agent never has to branch on how
the error happened.

**Status is resolved by walking the MRO**, not by an exhaustive table, so a new
`UpstreamUnavailable` subclass added next year is a 503 the day it is written.

## The one code that matters most

`lms_unavailable` is the contract the agent is written against: on it, it must say records
are temporarily unavailable and never a remembered or inferred figure. Every way of
failing to reach a system of record collapses onto it — see `UpstreamUnavailable` — so
there is no path where an outage reaches a parent as a number.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from records.domain.errors import (
    GuardianMismatch,
    NotAuthorized,
    NotConfigured,
    RecordsError,
    StudentNotFound,
    UnknownTerm,
    UpstreamUnavailable,
)

logger = logging.getLogger(__name__)


_STATUS_BY_ERROR: Final[Mapping[type[RecordsError], int]] = {
    NotAuthorized: status.HTTP_401_UNAUTHORIZED,
    # 403, not 401: the signature was valid and named somebody else. The caller's
    # credentials are fine; what they asked for is not theirs to ask.
    GuardianMismatch: status.HTTP_403_FORBIDDEN,
    StudentNotFound: status.HTTP_404_NOT_FOUND,
    UnknownTerm: status.HTTP_404_NOT_FOUND,
    # An unreachable system of record is never a 404. Telling a parent "no such child"
    # because another service was briefly down is a lie about their own family.
    UpstreamUnavailable: status.HTTP_503_SERVICE_UNAVAILABLE,
    NotConfigured: status.HTTP_503_SERVICE_UNAVAILABLE,
    RecordsError: status.HTTP_400_BAD_REQUEST,
}


def status_for(error: RecordsError) -> int:
    """The status for this error, or its nearest ancestor's. Never raises."""
    for klass in type(error).__mro__:
        found = _STATUS_BY_ERROR.get(klass)  # type: ignore[arg-type]
        if found is not None:
            return found
    return status.HTTP_400_BAD_REQUEST


def _body(code: str, message: str) -> dict[str, Any]:
    return {"detail": {"code": code, "message": message}}


def install(app: FastAPI) -> None:
    """Register every handler. Called once, by the composition root."""

    @app.exception_handler(RecordsError)
    async def _records_error(_: Request, error: RecordsError) -> JSONResponse:
        code = status_for(error)
        if code >= 500:
            # An operator has to act on these, and a spike of them against one student is
            # how a sync problem gets noticed before a parent reports it.
            logger.error("Refusing with %s: %s", code, error.message, exc_info=error)
        return JSONResponse(status_code=code, content=_body(error.code, error.message))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_body("bad_request", _first_message(error)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, error: StarletteHTTPException) -> JSONResponse:
        """Anything raised as `HTTPException`, plus the router table's own 404/405.

        A handler that raises `HTTPException(detail={"code": ...})` passes its detail
        straight through, so the credential dependencies keep their exact bodies.
        """
        detail = error.detail
        if isinstance(detail, dict) and "code" in detail:
            return JSONResponse(status_code=error.status_code, content={"detail": detail})
        return JSONResponse(
            status_code=error.status_code,
            content=_body(_code_for_status(error.status_code), str(detail or "")),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, error: Exception) -> JSONResponse:
        """A crash, in the same envelope, saying nothing about what crashed.

        The stack goes to the log; the caller gets a code and a sentence. This service's
        exception messages can quote a guardian handle and a student number.
        """
        logger.exception("Unhandled error in the records facade")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_body("internal_error", "Something went wrong."),
        )


def _first_message(error: RequestValidationError) -> str:
    try:
        first = error.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        msg = str(first.get("msg", "")).strip()
        return f"{where}: {msg}" if where else msg
    except Exception:  # noqa: BLE001 - never fail while reporting a failure
        return "That request could not be read."


def _code_for_status(code: int) -> str:
    return {
        400: "bad_request",
        401: "not_authorized",
        403: "not_authorized",
        404: "not_found",
        405: "method_not_allowed",
        410: "gone",
        503: "not_configured",
    }.get(code, "error")


__all__ = ["install", "status_for"]
