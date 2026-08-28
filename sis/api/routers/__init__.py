"""HTTP routers: the only place FastAPI and pydantic appear, and the thinnest layer here.

A router does three things and nothing else — turn a request into a DTO or a domain
value, call one service method, render the answer. It holds no rule and it imports no
SQLAlchemy. A `select()` written in a handler is a query no unit test of the application
layer can reach, and the next caller of the same use case has to write it again.

Two shared pieces live in this module because every router needs them, and duplicating
either is how one service ends up with two error envelopes.

`domain_errors()` turns a `SisError` into an `HTTPException`, and it asks
`sis.api.errors` which status that is. The table is **not** duplicated here. It was, and
the two copies had already drifted: an expired preview batch was a 410 to the app-level
handler and a 409 to every route that wrapped its service call, so the same failure
carried two status codes depending on which mechanism happened to catch it — and a client
that learned one of them wrote retry logic that was wrong half the time. One table, in
`sis/api/errors.py`, keyed on the exception class and walked along the MRO so a subclass
added next year inherits its parent's status instead of falling through to a 500.

`all_routers()` imports the submodules inside the function body. That keeps this package
importable *from* a router module without a circular import, which is what lets the
routers share `ErrorOut` and `domain_errors` from here at all.
"""
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException
from pydantic import BaseModel, Field

from sis.api.errors import status_for
from sis.domain.errors import SisError

if TYPE_CHECKING:
    from fastapi import APIRouter

__all__ = [
    "ErrorDetail",
    "ErrorOut",
    "all_routers",
    "domain_errors",
    "error_responses",
    "status_for",
]


class ErrorDetail(BaseModel):
    """The body of a refusal: a code to branch on, a sentence to show, a cell to blame.

    `code` is the contract and `message` is prose — a client that branches on the wording
    breaks the first time somebody translates it into Arabic, which this school will do.
    """

    code: str = Field(examples=["unknown_reference"])
    message: str = Field(examples=["no academic year 2025-2026"])
    field: str | None = Field(
        default=None,
        description="Which input caused it, when a single one did. Lets a UI highlight "
        "the cell rather than reddening the whole form.",
        examples=["academic_year_code"],
    )


class ErrorOut(BaseModel):
    """Every failure this service returns, nested under FastAPI's own `detail` key."""

    detail: ErrorDetail


_DESCRIPTIONS: Final[Mapping[int, str]] = {
    400: "The request or the uploaded file could not be read.",
    401: "Missing or unknown API key.",
    403: "The presented key does not carry the scope this route requires.",
    404: "Nothing on file under the code named in the path or query.",
    409: "The request conflicts with what is already stored.",
    # 410, not 404: the batch was real and aged out, which tells the UI to say
    # "re-upload the file" rather than "that link is wrong".
    410: "The preview batch expired before it was committed.",
    413: "The upload exceeds SIS_MAX_UPLOAD_BYTES.",
    415: "The file extension is not one this service can read.",
    422: "A supplied value is not valid.",
    # A capability this deployment has not armed, not a fault. Provisioning a school
    # over HTTP needs a credential the service is allowed not to hold.
    503: "This deployment is not configured for this operation.",
}


@contextmanager
def domain_errors() -> Iterator[None]:
    """Run a service call, turning any expected failure into its HTTP shape.

    Wrapped at the call site rather than installed as one app-wide exception handler
    because the boundary is the point: everything inside is application code that may
    fail in a stated way, and anything raised *outside* it — a `KeyError` from a broken
    handler — has no business being rendered as a tidy 4xx. An app-level handler catching
    `SisError` globally is a fine second net and does not conflict with this one.
    """
    try:
        yield
    except SisError as error:
        raise HTTPException(
            status_code=status_for(error), detail=error.as_dict()
        ) from error


def error_responses(*codes: int) -> dict[int | str, dict[str, object]]:
    """OpenAPI entries for the failures a route can actually return.

    Named per route rather than attached to every route wholesale, so the generated
    contract tells an integrator which refusals to handle instead of listing eight
    statuses on a handler that can only produce three.
    """
    return {code: {"model": ErrorOut, "description": _DESCRIPTIONS[code]} for code in codes}


def all_routers() -> tuple["APIRouter", ...]:
    """Every router, in the order the OpenAPI page should read.

    Imported lazily — see the module docstring — and returned rather than mounted, so the
    composition root in `sis/app.py` stays the only thing that knows an app exists.
    """
    from sis.api.routers import (
        admin,
        attendance,
        estate,
        grades,
        guardians,
        health,
        imports,
        structure,
        students,
    )

    # An explicit tuple, not discovery over the package. A new module that nobody adds
    # here is unreachable in production while every test still passes, because the tests
    # build their client from this same function — so `test_api` asserts the mounted paths
    # rather than trusting this list to be complete.
    return (
        health.router,
        structure.router,
        students.router,
        guardians.router,
        grades.router,
        attendance.router,
        imports.router,
        admin.router,
        estate.router,
    )
