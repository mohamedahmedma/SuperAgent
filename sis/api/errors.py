"""The single translation from a domain error to an HTTP response.

Every router in this service raises — or simply lets propagate — the errors defined in
`sis.domain.errors`, and none of them decides what status code that becomes. The mapping
lives here, once, because the alternative is a `try/except DuplicateCode` in eleven
handlers that agree on 409 in ten of them and 400 in the eleventh. A caller then sees
the same failure under two status codes depending on which URL produced it, and writes
retry logic against whichever one they met first.

**One shape on the wire, whichever mechanism produced it.** FastAPI renders a raised
`HTTPException(detail={...})` as `{"detail": {"code": ..., "message": ...}}`, so every
handler below produces exactly that envelope — including for a bare 404 from the router
table and for an unhandled crash. A client parses one shape and never branches on how
the error happened to be raised.

**Status is resolved by walking the exception's MRO**, not by an exhaustive table. A new
`DomainRuleViolation` subclass added next year is a 409 the day it is written, without
anyone remembering to edit this file — and a mapping that silently defaulted such an
error to 400 would ship a wrong status code with no test failing.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from sis.domain.errors import (
    DomainRuleViolation,
    DuplicateCode,
    ImmutableFieldChange,
    ImportBatchAlreadyCommitted,
    ImportBatchExpired,
    ImportBatchNotFound,
    ImportContentMismatch,
    ImportFailure,
    NotEnrolled,
    OverlappingEnrolment,
    SisError,
    TermClosed,
    UnknownReference,
    UnreadableImportFile,
    UnsupportedFileType,
    UploadTooLarge,
    ValidationError,
)

logger = logging.getLogger(__name__)


_STATUS_BY_ERROR: Final[Mapping[type[SisError], int]] = {
    # A malformed value is 422 rather than 400 so that a bad class code caught by a
    # value object and a bad body caught by pydantic answer with the same status. The
    # client's question is "did I send something usable", and it deserves one answer.
    ValidationError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    # Well-formed input that the stored state forbids: a conflict, not a bad request.
    DomainRuleViolation: status.HTTP_409_CONFLICT,
    DuplicateCode: status.HTTP_409_CONFLICT,
    ImmutableFieldChange: status.HTTP_409_CONFLICT,
    OverlappingEnrolment: status.HTTP_409_CONFLICT,
    NotEnrolled: status.HTTP_409_CONFLICT,
    TermClosed: status.HTTP_409_CONFLICT,
    # A code that names nothing is a 404 wherever it appeared — path or body. The
    # registrar's next action is the same either way: fix the code or create the thing.
    UnknownReference: status.HTTP_404_NOT_FOUND,
    ImportFailure: status.HTTP_400_BAD_REQUEST,
    UnsupportedFileType: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    UnreadableImportFile: status.HTTP_422_UNPROCESSABLE_CONTENT,
    UploadTooLarge: status.HTTP_413_CONTENT_TOO_LARGE,
    ImportBatchNotFound: status.HTTP_404_NOT_FOUND,
    # 410, not 404: the batch was real and aged out. That distinction tells the UI
    # whether to say "re-upload the file" or "that link is wrong", which are different
    # instructions to a person who has already waited once.
    ImportBatchExpired: status.HTTP_410_GONE,
    ImportBatchAlreadyCommitted: status.HTTP_409_CONFLICT,
    ImportContentMismatch: status.HTTP_409_CONFLICT,
}

# For statuses raised by the framework itself, which carry no domain error to ask.
_CODE_BY_STATUS: Final[Mapping[int, str]] = {
    status.HTTP_401_UNAUTHORIZED: "not_authorized",
    status.HTTP_403_FORBIDDEN: "not_authorized",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    status.HTTP_409_CONFLICT: "conflict",
    status.HTTP_410_GONE: "gone",
    status.HTTP_413_CONTENT_TOO_LARGE: "upload_too_large",
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: "unsupported_file_type",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "invalid_value",
    status.HTTP_503_SERVICE_UNAVAILABLE: "not_available",
}


def status_for(exc: SisError) -> int:
    """The status a domain error renders as, inherited from its nearest mapped ancestor."""
    for klass in type(exc).__mro__:
        mapped = _STATUS_BY_ERROR.get(klass)  # type: ignore[arg-type]
        if mapped is not None:
            return mapped
    return status.HTTP_400_BAD_REQUEST


def error_detail(code: str, message: str, field: str | None = None) -> dict[str, Any]:
    """The envelope body. Pass it as `HTTPException(detail=...)` to match these handlers."""
    body: dict[str, Any] = {"code": code, "message": message}
    if field is not None:
        body["field"] = field
    return body


def _response(status_code: int, detail: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


async def sis_error_handler(request: Request, exc: SisError) -> JSONResponse:
    """Every deliberate failure this service raises, rendered from the error's own `code`.

    Only the code and the path are logged, never `exc.message`. Domain messages quote the
    value that failed — a student number, a child's name from a roster cell — and a log
    that repeats them is a partial copy of the student register, shipped off the box and
    readable by more people than the database ever is.
    """
    status_code = status_for(exc)
    logger.info("%s %s -> %s (%s)", request.method, request.url.path, status_code, exc.code)
    return _response(status_code, error_detail(exc.code, exc.message, exc.field))


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Normalise every `HTTPException` — ours and the framework's — into one envelope.

    Handlers raise `HTTPException(detail={"code": ..., "message": ...})` and that dict is
    passed through untouched. A bare 404 from the router table or a 405 from Starlette
    arrives with a plain string instead, and is wrapped, so a client never has to handle
    `detail` being sometimes an object and sometimes a sentence.
    """
    detail: Any = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        body = dict(detail)
    else:
        code = _CODE_BY_STATUS.get(exc.status_code, "http_error")
        body = error_detail(code, str(detail) if detail else "Request failed.")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": body},
        headers=getattr(exc, "headers", None),
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Pydantic's rejection, in the same envelope as a domain `ValidationError`.

    `errors` is kept alongside the envelope because a registrar uploading a form with
    eleven fields needs to know *which* one, and FastAPI's per-field detail is the only
    place that exists. The envelope stays first-class so a client can read `code` without
    knowing pydantic's error vocabulary.
    """
    first = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    body = error_detail(
        "invalid_value",
        first.get("msg", "Request body failed validation."),
        location or None,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        # `jsonable_encoder` is not needed: pydantic v2 already returns JSON-safe errors,
        # except for `ctx` values, which are dropped here rather than risking a 500 while
        # rendering a 422.
        content={"detail": body, "errors": [
            {"loc": [str(p) for p in e.get("loc", ())], "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]},
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """The last resort: a 500 that says nothing.

    `str(exc)` is deliberately absent from the response. An unexpected exception can have
    formatted anything into its message — a connection string, a row of a roster, in the
    worst case a credential — and a 500 body is the least controlled place in the service
    to discover that. The traceback goes to the log, where the operator is.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_detail("internal_error", "An unexpected error occurred."),
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register the translation. Call once, at application construction."""
    app.add_exception_handler(SisError, sis_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_error_handler)
    _install_multipart_handler(app)


def _install_multipart_handler(app: FastAPI) -> None:
    """Turn Starlette's own oversized-upload refusal into the same 413.

    `UploadTooLarge` covers the ceiling *this* service enforces, but it is not the only
    one: Starlette aborts a multipart body that exceeds its own limits before any code in
    `sis/` runs, and renders that as a 400 carrying a bare string. A registrar whose
    40 MB export is refused is then told "bad request" about a file that was merely too
    big, and goes looking for corruption that is not there.

    Guarded because `MultiPartException` moved into `starlette.formparsers` only in
    recent versions; on an older one the handler is simply not registered rather than the
    whole application failing to import.
    """
    try:
        from starlette.formparsers import MultiPartException
    except ImportError:  # pragma: no cover - depends on the installed starlette
        return

    async def multipart_handler(request: Request, exc: MultiPartException) -> JSONResponse:
        message = str(getattr(exc, "message", "") or exc)
        too_large = "exceed" in message.lower() or "size" in message.lower()
        if too_large:
            return _response(
                status.HTTP_413_CONTENT_TOO_LARGE,
                error_detail("upload_too_large", "The uploaded file is too large."),
            )
        return _response(
            status.HTTP_400_BAD_REQUEST,
            error_detail("unreadable_file", "The upload could not be read."),
        )

    app.add_exception_handler(MultiPartException, multipart_handler)  # type: ignore[arg-type]


__all__ = [
    "error_detail",
    "http_exception_handler",
    "install_error_handlers",
    "sis_error_handler",
    "status_for",
    "unhandled_error_handler",
    "validation_error_handler",
]
