"""Request and response shapes for the identity service."""
from datetime import datetime

from pydantic import BaseModel, Field


class LoginIn(BaseModel):
    username: str
    password: str


class RegisterIn(BaseModel):
    """Self-registration, ported from the old chat backend's `/auth/register`.

    `role` may only be "user" or "admin", and "admin" requires `admin_code`. There is
    deliberately no way to self-register as a parent: that role is paired with a
    guardian binding, and both are an administrator's decision.
    """

    username: str
    password: str
    role: str = Field(default="user", description="user | admin")
    admin_code: str | None = None
    display_name: str = ""
    preferred_language: str = "ar"


class TokenOut(BaseModel):
    """What a successful login returns.

    `guardian_id` is echoed so a front end can tell a parent session from a staff one
    without decoding the token. It is a convenience for rendering, never a source of
    truth — the value other services trust is the signed claim, not this field.
    """

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_at: datetime
    # Echoed so a client can render the signed-in user without decoding the token.
    # The old backend's AuthResponse carried it and its front end relied on it.
    username: str
    role: str
    guardian_id: str | None = None
    display_name: str = ""


class RefreshIn(BaseModel):
    refresh_token: str


class AccessTokenOut(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_at: datetime


class MeOut(BaseModel):
    # `username`, not `sub` — JWT vocabulary should not leak into the API a front end
    # codes against.
    username: str
    role: str
    guardian_id: str | None = None
    display_name: str = ""
    expires_at: datetime


class AccountIn(BaseModel):
    username: str
    password: str
    role: str = Field(default="parent", description="user | admin | parent | staff")
    phone: str = ""
    display_name: str = ""
    preferred_language: str = "ar"


class GuardianBindingIn(BaseModel):
    """Binds a login to a guardian in the records facade.

    The most sensitive write in the system. There is no self-service path to it: an
    account that could name its own guardian id could read any family's records.
    """

    guardian_external_id: str


class ErrorOut(BaseModel):
    code: str = Field(description="not_authorized | locked | not_found | conflict | not_configured")
    message: str = ""


class WhatsAppStartOut(BaseModel):
    """What a browser gets when it begins a WhatsApp verification.

    `poll_secret` is shown exactly once and is never recoverable — it is this browser's
    half of the proof, and the reason somebody who forwards the link cannot finish the
    login in your place.
    """

    poll_secret: str = Field(
        description="Hold this in the page and send it back with the code. Never shown "
        "again, never sent over WhatsApp, and never put in a URL."
    )
    link: str = Field(
        description="Opens WhatsApp with the message already typed. The parent must "
        "still tap send — WhatsApp never sends it for them.",
        examples=["https://wa.me/201288339613?text=SCHOOL%20VERIFY%3A%20K7QP4M2X"],
    )
    message: str = Field(
        description="The exact text the link pre-fills. Show it beside the button so a "
        "parent whose in-app browser swallowed the link can send it by hand.",
        examples=["SCHOOL VERIFY: K7QP4M2X"],
    )
    business_number: str = Field(
        description="The school's WhatsApp number, for the same manual fallback.",
        examples=["+201288339613"],
    )
    expires_at: datetime


class WhatsAppStatusIn(BaseModel):
    """In a body, not a query string: a poll secret in a URL is a credential in a log."""

    poll_secret: str


class WhatsAppStatusOut(BaseModel):
    """Where a verification has got to, with nothing in it a stranger could use."""

    status: str = Field(
        description="pending | code_sent | verified | rejected",
        examples=["code_sent"],
    )
    display_name: str = Field(
        default="",
        description="The parent's name once the school's records have identified her, so "
        "the page can greet her before she has an account. Empty until then.",
    )
    expires_at: datetime


class WhatsAppVerifyIn(BaseModel):
    poll_secret: str
    code: str = Field(description="The six digits sent over WhatsApp.", examples=["482103"])
