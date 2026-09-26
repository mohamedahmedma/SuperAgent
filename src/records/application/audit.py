"""What this service still has to say about an access, now that the table has moved.

`sis/` records the access decisions it makes — see `sis/domain/access.py`, and
`GET /v1/admin/access-audit` to read them. That covers every question a school actually
asks of an audit: who was told about this child, when, and on whose behalf.

**What SIS cannot record is everything that fails before it is reached.** A request with a
bad service key, a token signed by the wrong issuer, or a signed token naming one guardian
while the path names another never becomes a SIS call at all — so if this service does not
report it, nothing does. Those are exactly the events worth alerting on: they are not a
parent getting a "no such child", they are somebody trying something.

So this emits **one structured line per refusal**, and nothing else. It is not a second
audit trail: it is the record of requests that died at the front door.

## Why a log line and not a row

This service holds no database any more. Giving it one back for this would undo the whole
point — and a queryable trail already exists in `sis/`, in the service that owns both the
data and the decision. What is left here is operational: the events belong in whatever
collects logs, beside the 401s and the 5xxs they resemble, because the response to a run of
them is an alert rather than a subject-access request.

One JSON object per line, so a collector can index the fields rather than regex the prose.
No parent phone numbers, no child names — a guardian handle and a student number, the same
discipline the rest of the estate keeps.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager

logger = logging.getLogger("records.audit")

#: Refusals this service makes on its own, before `sis/` is asked anything.
#:
#: Each is a *failure to get in the door*, not an answer about a child. `sis/` records the
#: answers; these have nowhere else to be reported.
NOT_AUTHORIZED = "not_authorized"
"""No service key, or one that does not match."""

INVALID_IDENTITY = "invalid_identity"
"""The parent token was missing, malformed, expired, or not signed by identity."""

GUARDIAN_MISMATCH = "guardian_mismatch"
"""The signature was valid and named somebody else.

The loudest signal in the system: a caller relaying one parent's token while asking about
another. It has its own reason so it can be alerted on by itself.
"""

IDENTITY_NOT_CONFIGURED = "identity_not_configured"
"""No verification material. Every parent read is failing closed, and somebody must know."""

DIRECTORY_UNAVAILABLE = "directory_unavailable"
"""The school's records could not be reached — a 503, not a refusal about a child."""

LMS_UNAVAILABLE = "lms_unavailable"
"""Same, one call further in. A spike is how a sync problem is noticed before a parent
reports it."""


def refused(
    reason: str,
    *,
    endpoint: str = "",
    guardian_id: str = "",
    student_id: str = "",
    request_id: str = "",
) -> None:
    """Emit one line for a request this service refused or could not serve.

    Always at WARNING. These are not routine: a parent being told "no such child" is
    recorded in `sis/` and never reaches here, so anything that does is either an attack, a
    misconfiguration, or an outage — and all three want a human eventually.
    """
    logger.warning(
        "records.access.refused %s",
        json.dumps(
            {
                "reason": reason,
                "endpoint": endpoint,
                # Handles and numbers only. A guardian is her opaque id here exactly as she
                # is in SIS's table; her phone number has never been in this process.
                "guardian_id": guardian_id,
                "student_id": student_id,
                # Correlates to the chat turn, and onward to the SIS row when the request
                # got far enough to make one.
                "request_id": request_id,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


__all__ = [
    "DIRECTORY_UNAVAILABLE",
    "GUARDIAN_MISMATCH",
    "IDENTITY_NOT_CONFIGURED",
    "INVALID_IDENTITY",
    "LMS_UNAVAILABLE",
    "NOT_AUTHORIZED",
    "refused",
]


@contextmanager
def reporting_unavailable(request, subject, student_id: str):
    """Record a failed read against a system of record, then let it propagate.

    The report matters as much as the error: a spike of `lms_unavailable` against one
    student is how a sync problem gets noticed before a parent reports it. It is a
    structured log line rather than a row because this service holds no database, and
    `sis/` keeps the trail of answers about children.

    A context manager rather than a helper called from an `except` block, because it was
    the latter at four call sites and one of them is how a read gets added later that
    fails silently. Wrapping the call means the reporting cannot be forgotten.

    Only upstream failures are reported here. A 404 for a child who is not this guardian's
    is an *answer*, and `sis/` records it with the real reason — emitting it here too
    would double-count the one event that already has a proper home.
    """
    from records.domain.errors import UpstreamUnavailable

    try:
        yield
    except UpstreamUnavailable:
        refused(
            LMS_UNAVAILABLE,
            endpoint=str(request.url.path),
            guardian_id=subject.guardian_id,
            student_id=student_id,
            request_id=subject.caller.request_id,
        )
        raise
