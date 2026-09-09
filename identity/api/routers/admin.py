"""Creating accounts, and binding one to a guardian.

Reached with an administrator's own access token — `require_admin_token`. There is no
shared key any more.

These routes used to be separated from the public ones by **credential type**: they took an
`X-Admin-Key` header and no bearer token at all, so a parent's token could not reach them
even to be rejected. That property is gone, and it is worth being honest that it was worth
something. What replaces it is stronger where it counts: a shared secret has no identity, so
"who bound this parent to that guardian" — the first question asked after a records leak,
and the reason the binding route is audited separately — had no answer while one key was
held by every script and every operator. Now the caller is a named account.
"""
from fastapi import APIRouter, Query, Response, status

from identity.api.deps import AdminServiceDep, AdminSubject
from identity.api.schemas.admin import (
    AccountIn,
    AccountListOut,
    AccountOut,
    AccountUpdateIn,
    GuardianBindingIn,
)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.post("/accounts", response_model=AccountOut, status_code=status.HTTP_201_CREATED)
def create_account(
    body: AccountIn, service: AdminServiceDep, _admin: AdminSubject
) -> AccountOut:
    """Create a login. Note what this route cannot do: bind a guardian.

    Creation and binding are deliberately two calls. A bulk parent import that runs only
    this one produces accounts that can log in and read nothing, which is the safe
    half-finished state.
    """
    created = service.create_account(
        username=body.username,
        password=body.password,
        role=body.role,
        phone=body.phone,
        display_name=body.display_name,
        preferred_language=body.preferred_language,
    )
    return _out(created)


@router.put("/accounts/{username}/guardian-binding", response_model=AccountOut)
def bind_guardian(
    username: str,
    body: GuardianBindingIn,
    service: AdminServiceDep,
    _admin: AdminSubject,
) -> AccountOut:
    """Bind a login to a guardian. The single most sensitive write in the system.

    Audited as its own event type, because "who decided this parent is that guardian" is
    the first question anyone asks after a records leak.
    """
    bound = service.bind_guardian(
        username=username, guardian_external_id=body.guardian_external_id
    )
    return _out(bound)


@router.delete("/accounts/{username}/guardian-binding", response_model=AccountOut)
def unbind_guardian(
    username: str, service: AdminServiceDep, _admin: AdminSubject
) -> AccountOut:
    """Remove a binding — the custody-change path.

    Takes effect for new access tokens immediately, for existing ones within their
    remaining lifetime, and for the session itself now: the refresh tokens are revoked
    here, so the parent cannot mint a fresh access token to carry on with.
    """
    unbound = service.unbind_guardian(username=username)
    return _out(unbound)


def _out(summary) -> AccountOut:
    """One mapping, so five routes cannot drift into five response shapes."""
    return AccountOut(
        username=summary.username,
        role=summary.role,
        guardian_id=summary.guardian_external_id,
        is_active=summary.is_active,
        display_name=summary.display_name,
    )


@router.get("/accounts", response_model=AccountListOut)
def list_accounts(
    service: AdminServiceDep,
    _admin: AdminSubject,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> AccountListOut:
    """Every account, a page at a time.

    Paged rather than returned whole because a school is thousands of parents, and a
    management screen that fetches all of them works perfectly on the developer's fixture
    data and times out on the first real deployment.
    """
    accounts, total = service.list_accounts(limit=limit, offset=offset)
    return AccountListOut(
        accounts=[_out(a) for a in accounts], total=total, limit=limit, offset=offset
    )


@router.patch("/accounts/{username}", response_model=AccountOut)
def update_account(
    username: str,
    body: AccountUpdateIn,
    service: AdminServiceDep,
    _admin: AdminSubject,
) -> AccountOut:
    """Change an account. PATCH, not PUT: absent means unchanged, never "clear it".

    With PUT, a management form that forgot to send `display_name` would erase it, and a
    form that forgot `is_active` would silently reactivate somebody who was suspended.

    This route cannot bind a guardian — `AccountUpdateIn` has no such field and forbids
    extras, so an attempt is refused rather than ignored.
    """
    return _out(service.update_account(username=username, **body.model_dump(exclude_unset=True)))


@router.delete("/accounts/{username}", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(
    username: str, service: AdminServiceDep, _admin: AdminSubject
) -> Response:
    """Delete an account and revoke its sessions.

    Refused for the last active administrator: losing every admin means nobody can bind a
    parent to their children until the seeded account is restored by a restart.

    204 with no body, because there is no longer anything to describe.
    """
    service.delete_account(username=username)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
