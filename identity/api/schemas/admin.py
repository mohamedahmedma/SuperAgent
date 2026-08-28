"""Request shapes for the admin-key routes."""
from pydantic import BaseModel, Field


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


class AccountOut(BaseModel):
    """What the admin routes echo back. Never a password hash, never a token."""

    username: str
    role: str = ""
    guardian_id: str | None = None


__all__ = ["AccountIn", "AccountOut", "GuardianBindingIn"]
