"""Enforcement, and nothing else. No tables, no decisions this service is not entitled to.

The rule this module exists to enforce, stated once:

    A service key proves *which system* is calling. It never proves *which parent* is
    asking. Those are two separate facts and both are required before a single grade is
    returned.

A leaked service key must therefore be worth nothing on its own. It buys the ability to
ask on behalf of a guardian; whose children that guardian may be told about is resolved
from the school's own records on every request. There is no code path in which the caller
supplies the answer to "which students may I see".

## What this module stopped doing, and where it went

**It no longer decides.** `sis/` owns the guardian link and re-checks it on the read
itself — the handle travels with the request, see `records/sis_adapter.py`. What
`resolve_permitted_student` does below is fail the request early and fetch the child's
name and year for the response. It is a real check and it is not the only one; a
compromised caller that got past it still meets a second refusal at the system of record.

**It no longer remembers.** There is no `api_keys` table and no `access_audit` table,
because there is no database. The audit lives in `sis/`, written where the decision is
made; what could not be recorded there — a request that never became a SIS call — is
emitted as a structured line by `records.audit`.

**The service key is configuration, not a row.** One secret in the environment, compared
in constant time. That is what a stateless service can verify, and it is honest about what
this credential was already: a single value shared with one caller, the chat backend. It
buys rotation-by-redeploy and no revocation list, which is the trade for holding no state.
Minting short-lived service tokens in `identity/` and verifying them offline through
`schoolauth` is the upgrade path, and it is the only reason this is not simply a header
check — the seam below stays the same shape either way.

## Why the permitted set is applied before the system of record is queried

The caller is a language model. Nothing it says — no clever phrasing, no injected
instruction inside a chat message — reaches this decision, because the decision is made
from the URL's guardian id, the signed claim it must equal, and the school's own link. The
system of record is never asked about a student that check excluded.
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import Depends, Header, HTTPException, Request, status

from records import audit, identity
from records.guardian_directory import (
    GuardianDirectoryUnavailable,
    PermittedStudent,
    get_directory,
)

logger = logging.getLogger(__name__)

#: Names the calling system. Still `X-API-Key`: the value is a shared secret and the header
#: has meant exactly that all along. What changed is where the expected value comes from —
#: the environment rather than a table — which is what lets this service hold no database.
API_KEY_HEADER = "X-API-Key"

#: How much of a presented key may appear in a log line. Enough to tell two callers apart
#: while an operator is reading; useless for authenticating.
_PREFIX_LENGTH = 8


class ServiceCaller:
    """The authenticated *system* behind a request. Never a person."""

    def __init__(self, prefix: str, request_id: str = ""):
        self.prefix = prefix
        self.request_id = request_id


def _expected_key() -> str:
    """The secret this service admits, or `""` when none is configured.

    Read per request rather than captured at import, so rotating it is a restart of this
    process and not a rebuild — and so a test can set it without reloading the module.
    """
    return (os.getenv("RECORDS_API_KEY") or "").strip()


def require_agent(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
) -> ServiceCaller:
    """Prove which system is calling. Every route here depends on this.

    Fails closed when `RECORDS_API_KEY` is unset: an unconfigured deployment refuses
    everything rather than admitting everyone. The alternative — treating "no key
    configured" as "no key required" — is how a service ships open, and it is exactly the
    state `sis/` spent a commit recovering from.
    """
    presented = (x_api_key or "").strip()
    expected = _expected_key()
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


class ParentSubject:
    """A verified (system, parent) pair. Both halves proved, neither assumed."""

    def __init__(self, caller: ServiceCaller, guardian_id: str, school_code: str | None = None):
        self.caller = caller
        self.guardian_id = guardian_id
        #: Which school's database answers for this parent, off the token's `school` claim.
        #: `None` in a single-school estate. Carried rather than looked up, because it was
        #: settled at sign-in from the WhatsApp number the parent messaged and nothing
        #: since has been in a position to know better.
        self.school_code = school_code


def require_parent_subject(
    guardian_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    caller: ServiceCaller = Depends(require_agent),
) -> ParentSubject:
    """Both credentials, checked together. Every parent-facing route depends on this.

    The service key has already proved which system is calling. This adds the second,
    independent proof: a token signed by the identity service naming the guardian. The
    `guardian_id` in the path must equal the `guardian_id` in the signed claim.

    That equality check is the point. It means the calling system cannot choose whose
    records it reads — it can only relay a parent's own identity, because it has no way to
    produce a signature for a different one. A fully compromised chat backend still cannot
    read a family it does not hold a token for.

    FastAPI supplies `guardian_id` from the path, so a route that declares this dependency
    without a `{guardian_id}` segment fails at startup rather than silently skipping the
    comparison.
    """
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


def resolve_permitted_student(
    *,
    guardian_external_id: str,
    student_external_id: str,
    caller: ServiceCaller,
    endpoint: str,
    school_code: str | None = None,
) -> PermittedStudent:
    """The child, when the school says this guardian may be told about her.

    **The first of two checks, and no longer the authority.** `sis/` makes the same
    decision again, from the same data, before it answers — the guardian handle travels
    with the read. This one still earns its place: it fails the request before a second
    service is troubled, and it fetches the child's name and year group, which the response
    needs and which the marks call does not return.

    **The answer is asked for, never remembered.** This service holds no tables at all now.
    A registrar revoking access the minute a court order arrives takes effect on the next
    question rather than whenever something here was next synchronised.

    Note what stays deliberately indistinguishable from the caller's side: an unknown
    student, a student who exists but is not this guardian's, and a student whose records
    are restricted all produce the same 404 and the same message. **`sis/` records which
    one actually happened**; the response does not, because a caller who could tell them
    apart could enumerate the student body and detect custody restrictions by their error
    code alone.

    A directory that cannot be reached is the one case that is *not* a 404. Telling a
    parent "no such child" because another service was briefly down is a lie about their
    own family, so it is a 503 and says so.
    """
    try:
        children = get_directory().children_of(
            guardian_external_id, school_code=school_code
        )
        student = next(
            (child for child in children if child.student_id == str(student_external_id)),
            None,
        )
    except GuardianDirectoryUnavailable as error:
        logger.error("Guardian directory unavailable: %s", error)
        audit.refused(
            audit.DIRECTORY_UNAVAILABLE,
            endpoint=endpoint,
            guardian_id=guardian_external_id,
            student_id=student_external_id,
            request_id=caller.request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "not_configured",
                "message": "The school's records are temporarily unavailable.",
            },
        ) from error

    if student is None:
        # Not logged as a refusal: this is an *answer about a child*, and `sis/` records
        # it with the real reason — `no_link` or `no_children` — in a table a school can
        # query. Emitting it here too would double-count the one event that already has a
        # proper home, and would put the distinction in a log the response deliberately
        # withholds.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "No such student record for this guardian."},
        )

    return student


def permitted_students(
    *, guardian_external_id: str, school_code: str | None = None
) -> list[PermittedStudent]:
    """Every child this guardian may ask about. Restricted links are already excluded.

    The filtering happens in `sis/` rather than here — it returns only links carrying
    `can_view_records` — so a barred parent arrives holding no children at all rather than
    holding children this service would have to remember to hide.

    An unreachable directory returns empty rather than raising, because the only caller is
    the "list my children" route and an empty list there renders as "no children on file",
    which is recoverable. The read routes below it raise properly.
    """
    try:
        return list(
            get_directory().children_of(guardian_external_id, school_code=school_code)
        )
    except GuardianDirectoryUnavailable as error:
        logger.error("Guardian directory unavailable while listing children: %s", error)
        return []


__all__ = [
    "API_KEY_HEADER",
    "ParentSubject",
    "ServiceCaller",
    "permitted_students",
    "require_agent",
    "require_parent_subject",
    "resolve_permitted_student",
]
