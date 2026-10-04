"""The errors the backend raises on purpose, each with the code a client can branch on.

The same design as `sis.domain.errors`: one root, a machine-readable `code` per class, a
message for a person. Services raise these and never decide what HTTP status one becomes;
`backend/api/errors.py` makes that decision, once, for every route.

The names and statuses are the backend's own, not copied from `sis`. The backend has always
answered a bad upload with 400, where `sis` answers invalid values with 422, and giving the
backend a `ValidationError` that meant 400 would put two errors with one name and different
statuses in a single estate.

What goes over the wire has not changed: a client still receives `{"detail": "<message>"}`,
because every page of the web app reads `detail` as a string. The codes are carried so that
moving to the envelope `sis`, `identity` and `records` answer with is a change to one
handler rather than to every route.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar


class BackendError(Exception):
    """Root of everything this service raises deliberately.

    Catching `BackendError` catches every EXPECTED failure and nothing else. An exception
    of any other type reaching the HTTP layer means the code is broken, not that the
    request was.
    """

    code: ClassVar[str] = "backend_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidInput(BackendError):
    """The request names something this service will not accept: an empty or unsupported
    filename, an upload with no file in it, a pair whose two halves are the same file."""

    code: ClassVar[str] = "invalid_input"


class NotFound(BackendError):
    """What was asked for is not there, or is not the caller's to see.

    The two are deliberately one error. A voice note that belongs to someone else answers
    exactly as one that does not exist, so the answer reveals nothing about other accounts.
    """

    code: ClassVar[str] = "not_found"


class FeatureDisabled(NotFound):
    """The deployment has this feature switched off, so as far as a caller can tell, what
    they asked for does not exist here."""

    code: ClassVar[str] = "feature_disabled"


class Forbidden(BackendError):
    """The caller is known and may not do this."""

    code: ClassVar[str] = "forbidden"


class TurnRefused(BackendError):
    """A chat turn refused at the door, with how long until asking again is worthwhile.

    The message is already the profile's copy in the language of the question, because it
    is shown to the parent as it is.
    """

    code: ClassVar[str] = "too_many_requests"

    def __init__(self, message: str, *, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class OperationFailed(BackendError):
    """An operation failed for a reason nobody anticipated, named by the operation.

    `message` says which operation, in its own words; `cause` is what went wrong. The
    client is told both, as `"<message>: <cause>"`, because that is what these routes have
    always answered and this change is not the place to alter it.

    It is recorded in REFACTOR_FINDINGS.md as a known leak: an unexpected exception can
    format anything into its message, and the sibling services answer an unexpected failure
    with a generic sentence while logging the detail. Moving to that is one line in
    `backend/api/errors.py`.
    """

    code: ClassVar[str] = "operation_failed"

    def __init__(self, message: str | None = None, cause: BaseException | None = None) -> None:
        if message and cause is not None:
            text = f"{message}: {cause}"
        elif message:
            text = message
        else:
            text = str(cause) if cause is not None else "Operation failed"
        super().__init__(text)
        self.cause = cause


@contextmanager
def operation(message: str | None = None) -> Iterator[None]:
    """Run a block whose unanticipated failure is reported as this operation's.

        with operation("Failed to retrieve document list"):
            ...

    A `BackendError` raised inside passes through untouched - a `NotFound` stays a 404, the
    way `except HTTPException: raise` kept it one in the routes this replaced. Anything else
    becomes `OperationFailed(message, cause)`.
    """
    try:
        yield
    except BackendError:
        raise
    except Exception as exc:
        raise OperationFailed(message, exc) from exc


__all__ = [
    "BackendError",
    "FeatureDisabled",
    "Forbidden",
    "InvalidInput",
    "NotFound",
    "OperationFailed",
    "TurnRefused",
    "operation",
]
