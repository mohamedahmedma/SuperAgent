"""Every way this service refuses, as one hierarchy.

The point of naming them here rather than raising `HTTPException` where the rule lives:
a use case in `application/services/` states *what* is wrong, and `api/errors.py` decides
once what status code that becomes. Raise HTTP from a service and the service can only
ever be called over HTTP — the import-legacy-accounts script and every unit test then
need a FastAPI request context to exercise a password rule.

**Status is resolved by walking the MRO**, exactly as `sis/api/errors.py` does it, so a
subclass added next year gets its parent's status the day it is written and no mapping
table silently defaults it to 400.

## Two axes, deliberately kept apart

`NotConfigured` is about the *deployment* — an operator has to act, and the answer is a
503. Everything else is about one caller or one challenge — the caller can act, and the
answer is a 4xx. Collapsing them means a school with no WhatsApp number configured tells
parents they typed something wrong.
"""
from __future__ import annotations


class IdentityError(Exception):
    """Base for every refusal this service raises deliberately.

    Carries a stable machine-readable `code` because the sign-in page branches on it:
    a parent who is locked out, a parent whose number is not registered, and a parent
    whose code has expired each need different words, and the page cannot get those from
    a status code alone.
    """

    #: Overridden per class. Stable — a front end matches on it.
    code: str = "error"
    #: Shown to the caller. Written for a parent, never for an operator.
    message: str = "Something went wrong."

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        if code is not None:
            self.code = code
        if message is not None:
            self.message = message
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# The deployment is wrong. 503; the operator acts, not the parent.
# ---------------------------------------------------------------------------


class NotConfigured(IdentityError):
    """This server cannot run the flow that was asked for at all.

    Distinct from every other refusal here, which are about one challenge or one caller.
    Nothing the caller sends will help and nothing they did is wrong, so the message is
    written to be shown to a parent as "contact the school" rather than as a server
    complaint about an environment variable.
    """

    code = "not_configured"
    message = "This service is not configured for that."


class SchoolsMisconfigured(IdentityError):
    """The multi-school registry is wrong and no parent can be signed in correctly.

    Raised at startup, not per request. A school named with no WhatsApp number behind it
    would otherwise fail on the first parent who tapped the link, at whatever hour that
    happened to be, with an error naming a gateway rather than a missing setting.
    """

    code = "schools_misconfigured"
    message = "The school registry is misconfigured."


# ---------------------------------------------------------------------------
# The caller is wrong.
# ---------------------------------------------------------------------------


class NotAuthorized(IdentityError):
    """Credentials were absent, malformed, or wrong.

    One class for "unknown username" and "wrong password" on purpose, and the routes
    render both with the same words. Distinguishing them turns login into an account
    enumerator, and for a school that means confirming which parents are registered.
    """

    code = "not_authorized"
    message = "Invalid username or password."


class AccountLocked(IdentityError):
    """Too many failed attempts. Distinct from `NotAuthorized` because it is actionable.

    Safe to distinguish where `NotAuthorized`'s two causes are not: an attacker who
    triggered the lockout already knows they did, and the parent who is locked out needs
    to be told to wait rather than to keep retyping a password that is correct.
    """

    code = "locked"
    message = "Account temporarily locked. Try again later."


class Forbidden(IdentityError):
    """The caller is understood, and may not have what they asked for.

    Distinct from `NotAuthorized`, and the distinction is the status code. A wrong
    administrator invite code is the case this exists for: the request is not an
    authentication attempt that failed, it is a well-formed registration asking for a role
    the caller cannot have. Answering 401 would tell an operator who mistyped the code to
    go and check their *credentials*, which are fine.
    """

    code = "not_authorized"
    message = "That is not permitted."


class NotFound(IdentityError):
    """A named account, or school, this service does not hold."""

    code = "not_found"
    message = "No such record."


class UnknownSchool(NotFound):
    """A `phone_number_id`, or a school code, this service does not serve.

    Carries the value so a log line can name it: an unrecognised `phone_number_id` almost
    always means a school was onboarded at Meta and never added to `.env`, and the id is
    the one piece of information that says which.
    """

    code = "unknown_school"
    message = "This server does not serve that school."

    def __init__(self, value: str = "") -> None:
        super().__init__()
        self.value = value


class Conflict(IdentityError):
    """The write cannot be made because of what is already stored."""

    code = "conflict"
    message = "That already exists."


class BadRequest(IdentityError):
    """Well-formed enough to parse and not usable."""

    code = "bad_request"
    message = "That request cannot be processed."


# ---------------------------------------------------------------------------
# The WhatsApp verification flow.
# ---------------------------------------------------------------------------


class VerificationError(IdentityError):
    """A challenge cannot proceed.

    Every subclass answers with the same HTTP status — see `api/errors.py` — and differs
    only in `code`. "No such verification", "wrong code" and "expired" would each justify
    a different status in isolation, but distinguishing them by status lets somebody
    holding a stolen poll secret learn which of their guesses was structurally wrong
    rather than merely incorrect.
    """

    code = "verification_failed"
    message = "That verification cannot be completed."


class VerificationNotFound(VerificationError):
    code = "not_found"
    message = "No such verification is in progress."


class VerificationExpired(VerificationError):
    code = "expired"
    message = "That verification has expired."


class VerificationNotReady(VerificationError):
    code = "not_ready"
    message = "No code has been sent for this verification yet."


class VerificationAlreadyUsed(VerificationError):
    code = "already_used"
    message = "That verification has already been used."


class BadCode(VerificationError):
    code = "bad_code"
    message = "That code is not correct."


class TooManyAttempts(VerificationError):
    code = "too_many_attempts"
    message = "Too many incorrect codes."


# ---------------------------------------------------------------------------
# Outbound dependencies. Named so a service can decide what to do about them.
# ---------------------------------------------------------------------------


class DependencyUnavailable(IdentityError):
    """A seam outside this service could not be reached.

    Separate from every refusal above because the caller did nothing wrong and retrying
    may work. A use case catching this decides whether to fail the request or to carry on
    with less — `children_of` failing costs a convenience claim, `resolve` failing costs
    the sign-in.
    """

    code = "dependency_unavailable"
    message = "A service we depend on could not be reached."


class GuardianDirectoryUnavailable(DependencyUnavailable):
    """The school's system of record could not answer.

    Never conflated with "this number is not a guardian". One means try again in a
    minute; the other means contact the school office — and telling a parent the second
    when the first is true is how an outage becomes a hundred phone calls.
    """

    code = "directory_unavailable"
    message = "We could not reach the school's records just now."


class WhatsAppUnavailable(DependencyUnavailable):
    """A message could not be delivered to WhatsApp."""

    code = "whatsapp_unavailable"
    message = "We could not send a WhatsApp message just now."


__all__ = [
    "AccountLocked",
    "BadCode",
    "BadRequest",
    "Conflict",
    "DependencyUnavailable",
    "Forbidden",
    "GuardianDirectoryUnavailable",
    "IdentityError",
    "NotAuthorized",
    "NotConfigured",
    "NotFound",
    "SchoolsMisconfigured",
    "TooManyAttempts",
    "UnknownSchool",
    "VerificationAlreadyUsed",
    "VerificationError",
    "VerificationExpired",
    "VerificationNotFound",
    "VerificationNotReady",
    "WhatsAppUnavailable",
]
