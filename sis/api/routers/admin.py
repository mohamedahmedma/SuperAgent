"""Minting API keys — the one write that creates the credential for every other write.

The secret is returned by this route and by nothing else, ever. `sis.domain.auth` stores
a hash and a prefix, so there is no path anywhere in this service that can show a key
again; the response body below is the only copy that will exist. That is stated in the
route description because an operator who assumes otherwise loses a credential and, with
it, whatever integration was holding it.

`scope` is taken from the caller rather than defaulted. `Scope.permits` is exact
equality — a registrar key does **not** satisfy a reader check — so a reporting
integration handed a registrar key "to unblock it" would silently gain the ability to
rewrite a term's grades. Making the field required means somebody types the word.
"""
from datetime import datetime
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from sis.api.deps import Caller, get_api_key_minter, require_registrar
from sis.api.routers import domain_errors, error_responses
from sis.domain.auth import ApiKey, Scope

router = APIRouter(prefix="/v1/admin", tags=["admin"])


class ApiKeyMinter(Protocol):
    """What this route needs from the composition root, stated by the route that needs it.

    Generating a secret, hashing it and storing the record is one act: split across two
    calls there is a moment where a key exists in a log and not in the database, or in the
    database with a hash of the wrong string. So the port is one method, and it hands back
    both halves — the stored record for the response, and the secret that is never stored.
    """

    def mint(
        self, *, label: str, scope: Scope, expires_in_days: int | None = None
    ) -> tuple[ApiKey, str]:
        """Store a new key and return it beside its one-time secret, in that order."""


# `Caller`, not `ApiKey`: what the dependency returns is the authenticated *system* —
# a prefix and a scope — and never the stored credential row.
Registrar = Annotated[Caller, Depends(require_registrar)]
Minter = Annotated[ApiKeyMinter, Depends(get_api_key_minter)]


class ApiKeyIn(BaseModel):
    label: str = Field(
        min_length=1,
        max_length=120,
        description="Which system holds this key. A key nobody can attribute is a key "
        "nobody dares revoke.",
        examples=["records adapter"],
    )
    scope: Scope = Field(
        description="`registrar` writes, `reader` reads. Checked by exact equality: a "
        "registrar key is refused by reader-only routes and vice versa."
    )
    expires_in_days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="Omit for a key that must be revoked deliberately rather than "
        "expiring under someone at 3am.",
    )


class ApiKeyOut(BaseModel):
    """The minted key. `api_key` appears here and is never retrievable again."""

    prefix: str = Field(
        description="The public handle. Appears in logs and audit rows so a human can "
        "point at one key and revoke it.",
        examples=["sis_a1b2c3d4"],
    )
    label: str
    scope: Scope
    api_key: str = Field(
        description="The secret, shown exactly once. Store it now; the service keeps only "
        "a hash and cannot show it again."
    )
    created_at: datetime
    expires_at: datetime | None = None


@router.post(
    "/api-keys",
    response_model=ApiKeyOut,
    status_code=status.HTTP_201_CREATED,
    summary="Mint an API key",
    description="Returns the only copy of the secret that will ever exist. Requires a "
    "`registrar` key — including the bootstrap key, which is how the first one is made.",
    responses=error_responses(401, 403, 422),
)
def create_api_key(body: ApiKeyIn, minter: Minter, caller: Registrar) -> ApiKeyOut:
    with domain_errors():
        key, secret = minter.mint(
            label=body.label, scope=body.scope, expires_in_days=body.expires_in_days
        )
    return ApiKeyOut(
        prefix=key.prefix,
        label=key.label,
        scope=key.scope,
        api_key=secret,
        created_at=key.created_at,
        expires_at=key.expires_at,
    )
