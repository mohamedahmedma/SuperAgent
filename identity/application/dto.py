"""What the use cases return.

Plain frozen dataclasses, not pydantic models. The API layer maps these into the response
schemas in `api/schemas/`, and that one extra hop is deliberate: a use case that returned
its HTTP response body would have the wire format as part of its signature, so renaming a
JSON field would mean editing a service, and the import script — which speaks no HTTP —
would be constructing response models to call it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from identity.application.ports.directory import GuardianDirectory
from identity.application.ports.messaging import WhatsAppGateway


@dataclass(frozen=True, slots=True)
class IssuedSession:
    """A successful sign-in, whichever door it came through.

    Password login, self-registration and WhatsApp verification all end here, which is
    what lets `TokenOut` be one response shape rather than three that drift apart.
    """

    access_token: str
    refresh_token: str
    expires_at: datetime
    username: str
    role: str
    guardian_external_id: str | None
    display_name: str


@dataclass(frozen=True, slots=True)
class IssuedAccessToken:
    """A refresh, which mints no new refresh token."""

    access_token: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class TokenSubject:
    """What a token says about its holder, decoded. For `/v1/auth/me`."""

    username: str
    role: str
    guardian_external_id: str | None
    display_name: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class StartedChallenge:
    """What the browser is handed. `poll_secret` is shown exactly once and never stored."""

    nonce: str
    poll_secret: str
    link: str
    message: str
    business_number: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ChallengeStatus:
    """Where a verification has got to, with nothing in it a stranger could use."""

    status: str
    display_name: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class AccountSummary:
    """What the admin routes echo back. Never a password hash, never a token.

    That is a property of the SHAPE, not of a filtering step somebody has to remember:
    there is nowhere here to put a hash, so no admin response can leak one by accident.

    The last two fields carry defaults because they were added for the management screen
    after `create_account`, `bind_guardian` and `unbind_guardian` were already returning
    this. Those three say nothing about activity or display name, and defaulting is what
    let them stay untouched.
    """

    username: str
    role: str
    guardian_external_id: str | None
    is_active: bool = True
    display_name: str = ""


@dataclass(frozen=True, slots=True)
class SchoolChannel:
    """Everything the flow needs in order to talk to one school.

    Schools are separated physically, so none of these three is shared: the number a
    parent messages, the credentials a code goes back out through, and the guardian
    directory their number is looked up in all belong to one school and must agree. A
    channel that mixed them — one school's number with another's directory — would resolve
    a parent against a database their children are not in, which is the whole failure the
    separation exists to prevent.

    `code` is `None` in a single-school deployment, which is what the rest of this service
    means by "no school".
    """

    code: str | None
    business_number: str
    gateway: WhatsAppGateway
    directory: GuardianDirectory


__all__ = [
    "AccountSummary",
    "ChallengeStatus",
    "IssuedAccessToken",
    "IssuedSession",
    "SchoolChannel",
    "StartedChallenge",
    "TokenSubject",
]
