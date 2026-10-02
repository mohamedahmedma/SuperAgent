"""The single translation from a backend error to an HTTP response.

Services raise the errors in `backend.domain.errors` and none of them decides what status
code that becomes. The mapping lives here, once, for the reason `sis/api/errors.py` gives: the
alternative is a `raise HTTPException` in every route, agreeing on 404 in nine of them and on
something else in the tenth, and a client learning its retry logic from whichever it met
first.

**What goes over the wire is exactly what the routes sent before.** A domain error answers
`{"detail": "<message>"}`, the body a route's `HTTPException(detail="...")` produced, because
every page of the web app reads `detail` as a string (`src/frontend/src/stores/documents.ts`,
`sessions.ts`, `chat.ts`). A rejected voice note keeps the object form it already had,
`{"detail": {"code": ..., "message": ...}}`, because the recorder reads the code.

Deliberately NOT installed here: a handler for `Exception`, for `RequestValidationError`, or
for Starlette's `HTTPException`. Each would change what some existing request answers - an
unanticipated crash that is a bare 500 today would start echoing its message, and a malformed
body's 422 would change shape - and this module's job is to move the mapping, not to change it.

**Status is resolved by walking the exception's MRO**, as in `sis`, so a new `NotFound`
subclass is a 404 the day it is written without anyone editing this table.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Final

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from backend.agent.chat.attachments import VoiceNoteRejected
from backend.domain.errors import (
    BackendError,
    Forbidden,
    InvalidInput,
    NotFound,
    OperationFailed,
    TurnRefused,
)

logger = logging.getLogger(__name__)

_STATUS_BY_ERROR: Final[Mapping[type[BackendError], int]] = {
    InvalidInput: status.HTTP_400_BAD_REQUEST,
    Forbidden: status.HTTP_403_FORBIDDEN,
    NotFound: status.HTTP_404_NOT_FOUND,
    TurnRefused: status.HTTP_429_TOO_MANY_REQUESTS,
    OperationFailed: status.HTTP_500_INTERNAL_SERVER_ERROR,
}

#: A rejected voice note's status, by the reason it carries. 413 and 415 are what they say;
#: an empty or over-long recording is a plain 400.
_VOICE_NOTE_STATUS: Final[Mapping[str, int]] = {
    VoiceNoteRejected.TOO_LARGE: status.HTTP_413_CONTENT_TOO_LARGE,
    VoiceNoteRejected.UNSUPPORTED_TYPE: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
}

#: How an upstream model provider's SDK names the status it was refused with.
_PROVIDER_STATUS = re.compile(r"Error code:\s*(\d{3})")


def status_for(exc: BackendError) -> int:
    """The status a backend error renders as, inherited from its nearest mapped ancestor."""
    for klass in type(exc).__mro__:
        mapped = _STATUS_BY_ERROR.get(klass)  # type: ignore[arg-type]
        if mapped is not None:
            return mapped
    return status.HTTP_400_BAD_REQUEST


async def backend_error_handler(request: Request, exc: BackendError) -> JSONResponse:
    """Every deliberate failure, as the `{"detail": message}` body the routes always sent.

    A 500 logs the underlying traceback, which is the only way an operator learns what
    actually broke. Of the routes this replaced, only the media route logged a failure it
    answered with a 500 (as "Failed to read blob for asset <id>"; the path in this line still
    names the asset); the session and document routes logged none of theirs. A 4xx is not
    logged here, as it was not before: the access log already records every request's
    method, path and status.
    """
    status_code = status_for(exc)
    if status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        cause = getattr(exc, "cause", None) or exc
        logger.error(
            "%s %s -> %s (%s)",
            request.method,
            request.url.path,
            status_code,
            exc.code,
            exc_info=(type(cause), cause, cause.__traceback__),
        )

    headers = None
    if isinstance(exc, TurnRefused):
        headers = {"Retry-After": str(exc.retry_after_seconds)}
    return JSONResponse(status_code=status_code, content={"detail": exc.message}, headers=headers)


async def voice_note_rejected_handler(request: Request, exc: VoiceNoteRejected) -> JSONResponse:
    """A recording refused, in the object form the recorder reads its `code` out of."""
    status_code = _VOICE_NOTE_STATUS.get(exc.reason, status.HTTP_400_BAD_REQUEST)
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": exc.reason, "message": str(exc)}},
    )


def provider_failure(exc: Exception) -> HTTPException:
    """A failed chat turn, as the status the model provider refused it with.

    Provider SDKs name the upstream status in the exception text ("Error code: 429 ..."),
    so a turn refused for quota answers 429 and one refused for credentials answers 401,
    rather than all of them collapsing into a 500 that tells the operator nothing.
    Anything without a status in it is a 500. The message is passed through, as it always
    was; see `OperationFailed` for why that is a known item rather than something changed
    here.
    """
    message = str(exc)
    match = _PROVIDER_STATUS.search(message)
    if not match:
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message)
    code = int(match.group(1))
    if code == status.HTTP_429_TOO_MANY_REQUESTS:
        return HTTPException(
            status_code=code,
            detail=(
                "The upstream model service triggered rate limiting/quota limits (429). "
                "Please check your account quota/model status.\n"
                f"Original error: {message}"
            ),
        )
    return HTTPException(status_code=code, detail=message)


def install_error_handlers(app: FastAPI) -> None:
    """Register the translations on an application.

    `create_app()` calls this. So must any test that assembles its own `FastAPI()` around a
    backend router, exactly as it includes the router: without it a service's `NotFound`
    reaches the test as an exception instead of as a 404.
    """
    app.add_exception_handler(BackendError, backend_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(VoiceNoteRejected, voice_note_rejected_handler)  # type: ignore[arg-type]


__all__ = [
    "backend_error_handler",
    "install_error_handlers",
    "provider_failure",
    "status_for",
    "voice_note_rejected_handler",
]
