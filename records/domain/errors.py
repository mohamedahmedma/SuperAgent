"""Every way this facade refuses, as one hierarchy.

The point of naming them here rather than raising `HTTPException` where the rule lives:
a use case in `application/` states *what* is wrong, and `api/errors.py` decides once what
status code that becomes. Raise HTTP from a use case and the use case can only ever be
called over HTTP.

## Three of these existed before and are unchanged in meaning

`LmsUnavailable`, `CalendarUnavailable` and `GuardianDirectoryUnavailable` each already
carried one rule that is worth restating, because it is the same rule three times and it
is the most important one in the service:

> **"Could not ask" is never "the answer is no."**

A directory that is briefly unreachable must not tell a mother she has no children. A SIS
that is down must not report a term of perfect attendance. Each of these is a 503 saying
records are temporarily unavailable, and the agent above is required to say exactly that
rather than a remembered or inferred figure.

The one deliberate collapse runs the other way: an unknown student, a student who is not
this guardian's, and a student whose records are restricted are all `StudentNotFound` with
one message. A caller who could tell them apart could enumerate the student body and
detect custody restrictions by error code alone. `sis/` records which actually happened,
in a table a school can query.
"""
from __future__ import annotations


class RecordsError(Exception):
    """Base for every refusal this service raises deliberately.

    Carries a stable machine-readable `code`, because the agent above branches on it —
    `lms_unavailable` is the one it must translate into "I cannot reach the school's
    records right now" rather than into a plausible-sounding number.
    """

    code: str = "error"
    message: str = "Something went wrong."

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        if code is not None:
            self.code = code
        if message is not None:
            self.message = message
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# Something outside this service could not answer. Never "the answer is no".
# ---------------------------------------------------------------------------


class UpstreamUnavailable(RecordsError):
    """A system of record could not be reached, or refused.

    One code on the wire — `lms_unavailable` — for every cause: transport failure, a
    revoked key, a timeout, an unreadable body. Distinguishing them in the response would
    let a caller probe the SIS's configuration through this facade, and none of the
    distinctions change what the agent does next.
    """

    code = "lms_unavailable"
    message = "The school's records are temporarily unavailable."


class LmsUnavailable(UpstreamUnavailable):
    """The marks could not be read."""


class CalendarUnavailable(UpstreamUnavailable):
    """The academic calendar could not be read. Distinct from "no such term", a `None`."""


class GuardianDirectoryUnavailable(UpstreamUnavailable):
    """The guardian links could not be read.

    Never conflated with "this guardian has no children". Telling a parent "no such child"
    because another service was briefly down is a lie about their own family.

    **The code is `not_configured`, not `lms_unavailable`, and that is inherited rather
    than chosen.** The route this replaced answered a directory outage with
    `not_configured`, `tests/general/test_parent_journey.py` asserts it, and the chat
    backend may branch on it — so it is preserved exactly. It reads oddly: nothing is
    misconfigured when a healthy service is briefly unreachable, and the README documents
    `lms_unavailable` as *the* signal for "do not answer from memory". Worth aligning, but
    that is a contract change for the agent above to agree to, not a refactor's to make.
    """

    code = "not_configured"


# ---------------------------------------------------------------------------
# The caller is wrong, or the deployment is.
# ---------------------------------------------------------------------------


class NotConfigured(RecordsError):
    """This deployment cannot answer at all. The operator acts, not the caller.

    Fails closed: an unset service key, or no verification material for identity tokens,
    refuses every read rather than admitting everyone. The alternative — treating "no key
    configured" as "no key required" — is how a service ships open.
    """

    code = "not_configured"
    message = "This service is not configured."


class NotAuthorized(RecordsError):
    """A credential was missing, malformed, or wrong."""

    code = "not_authorized"
    message = "Missing or invalid credentials."


class GuardianMismatch(NotAuthorized):
    """The token was validly signed and named a different guardian than the path.

    Its own type because it is the signal that a caller is relaying one parent's token
    while asking about another — the thing the two-credential rule exists to stop — and it
    deserves to be alertable on by itself.
    """

    message = "Token does not authorise this guardian."


class StudentNotFound(RecordsError):
    """Unknown, unrelated, or restricted. Deliberately indistinguishable. See the module docstring."""

    code = "not_found"
    message = "No such student record for this guardian."


class UnknownTerm(RecordsError):
    """The school has no term by that code."""

    code = "unknown_term"
    message = "No such term."


__all__ = [
    "CalendarUnavailable",
    "GuardianDirectoryUnavailable",
    "GuardianMismatch",
    "LmsUnavailable",
    "NotAuthorized",
    "NotConfigured",
    "RecordsError",
    "StudentNotFound",
    "UnknownTerm",
    "UpstreamUnavailable",
]
