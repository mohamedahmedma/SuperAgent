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
