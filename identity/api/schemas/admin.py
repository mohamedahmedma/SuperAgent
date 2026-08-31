"""Request shapes for the admin-key routes."""
from pydantic import BaseModel, ConfigDict, Field


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


class AccountUpdateIn(BaseModel):
    """Fields an administrator may change. Every one optional; absent means unchanged.

    **`guardian_external_id` is deliberately absent**, and its absence is the whole design.
    `AccountIn` already refuses it on creation so that an extra column in an import sheet
    cannot silently become a records grant; an update route that accepted it would reopen
    exactly that door from the other side. Binding keeps its own route and its own audit
    event because "who decided this parent is that guardian" has to stay a question with
    one answer.

    `model_config = {"extra": "forbid"}` so a caller who sends `guardian_external_id`
    anyway is told no, rather than having it quietly dropped and believing it worked.
    """

    model_config = ConfigDict(extra="forbid")

    password: str | None = None
    role: str | None = Field(default=None, description="user | admin | parent | staff")
    display_name: str | None = None
    phone: str | None = None
    preferred_language: str | None = None
    is_active: bool | None = Field(
        default=None,
        description="False refuses password login without deleting the account.",
    )


class AccountOut(BaseModel):
    """What the admin routes echo back. Never a password hash, never a token."""

    username: str
    role: str = ""
    guardian_id: str | None = None
    is_active: bool = True
    display_name: str = ""


class AccountListOut(BaseModel):
    """One page, and the total so a management screen can render a pager."""

    accounts: list[AccountOut]
    total: int
    limit: int
    offset: int


__all__ = [
    "AccountIn",
    "AccountListOut",
    "AccountOut",
    "AccountUpdateIn",
    "GuardianBindingIn",
]
