"""Request and response shapes for parent sign-in over WhatsApp."""
from datetime import datetime

from pydantic import BaseModel, Field


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


__all__ = [
    "WhatsAppStartOut",
    "WhatsAppStatusIn",
    "WhatsAppStatusOut",
    "WhatsAppVerifyIn",
]
