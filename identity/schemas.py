"""Request and response shapes for the identity service."""
from datetime import datetime

from pydantic import BaseModel, Field


class LoginIn(BaseModel):
    username: str
    password: str


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
    subject: str
    role: str
    guardian_id: str | None = None
    display_name: str = ""
    expires_at: datetime


class AccountIn(BaseModel):
    username: str
    password: str
    role: str = Field(default="parent", description="parent | staff")
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
