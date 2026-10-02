"""Every attempt to read a child's record on a guardian's behalf, allowed or refused.

This service decides who may be told about a child — the guardian link is here, the
custody restrictions are here, and since the parent-facing routes re-check that link on
every read, the decision is made here too. **So the record of the decision belongs here.**

It used to live in `records/`, the facade in front of this service. That was the right
place while the facade held the guardian tables and made the call. It no longer does
either, and an audit kept by a service that is now a stateless relay is an audit that
answers "what did the relay pass on", not "who was told about this child".

## Why the denials matter more than the successes

A run of `allowed=False` against one guardian is somebody probing. It is the single most
useful signal this table carries, and it is invisible if only successful reads are
recorded. The two refusal reasons are deliberately different facts:

    no_children   the handle reaches nobody this school will talk about — an unknown
                  guardian, or one whose every link is restricted
    no_link       the handle is a real parent here, and the child named is not hers

The first is somebody probing with a handle that resolves to nothing; the second is
somebody walking student numbers against a real parent's handle. **The caller cannot tell
those apart** — both produce the same `UnknownReference` — which is exactly why the
distinction has to be preserved somewhere, and this is the somewhere.

## What it is not

Not a log line. A log is rotated, sampled and shipped by whoever configured the platform;
this table is queried by a school answering "who saw my daughter's marks, and when",
usually months later and sometimes to a lawyer. Append-only by contract: nothing in this
package updates or deletes a row, and the only route over it reads.

Not PII beyond what it must hold. A guardian appears as her opaque `public_id`, never as a
phone number — the same discipline that put a handle on the guardians table in the first
place.

**The domain never reads the clock.** `at` is supplied by the caller, so a service that
records an attempt is unit-testable against a fixed timestamp.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from sis.domain.errors import ValidationError

#: Longest a reason will ever be. Reasons are a closed vocabulary, not free text: a column
#: holding prose cannot be aggregated, and "how often was this refused last term" is the
#: only question anybody asks of it.
REASON_LENGTH: Final[int] = 32


class AccessReason(StrEnum):
    """Why a read was allowed or refused. A closed set, aggregated on."""

    OK = "ok"
    #: The handle reaches nobody this school will talk about.
    NO_CHILDREN = "no_children"
    #: A real parent here, but not of the child she named.
    NO_LINK = "no_link"

    @property
    def is_allowed(self) -> bool:
        return self is AccessReason.OK


@dataclass(frozen=True, slots=True)
class AccessAttempt:
    """One decision, as it was made. Frozen: a recorded fact is not editable.

    `actor` is the API key *prefix* that asked — enough to name a caller in an
    investigation, useless for authenticating as one. It is the same handle
    `sis.domain.auth` puts in a log line, and the reason revocation deactivates a key
    rather than deleting it: an audit row naming a deleted key names nothing.
    """

    guardian_public_id: str
    student_number: str
    reason: AccessReason
    at: datetime
    #: The API key prefix behind the request, or `""` for the bootstrap credential.
    actor: str = ""
    #: Correlates back to the chat turn that caused this, when the caller supplies one.
    #: Empty is ordinary — a registrar console read has no chat turn behind it.
    request_id: str = ""

    def __post_init__(self) -> None:
        if not self.guardian_public_id.strip():
            raise ValidationError(
                "an access attempt names a guardian", field="guardian_public_id"
            )
        if self.at.tzinfo is None:
            # A naive timestamp in an audit is worse than none: it reads as local time to
            # whoever queries it next, and "when did this happen" is the whole point.
            raise ValidationError("at must be timezone-aware", field="at")

    @property
    def allowed(self) -> bool:
        """Stored as its own column as well as derivable, so the index that answers
        "show me the refusals" does not have to know the reason vocabulary."""
        return self.reason.is_allowed


__all__ = ["REASON_LENGTH", "AccessAttempt", "AccessReason"]
