"""Request and response shapes for signing in.

Split out of the old single `schemas.py` by feature, matching the routers. Nothing here
is imported by `application/` or `domain/` — these are the wire format, and a use case
that returned one would have the JSON field names as part of its signature.
"""
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


__all__ = [
    "AccessTokenOut",
    "LoginIn",
    "MeOut",
    "RefreshIn",
    "TokenOut",
]
