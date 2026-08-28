"""Creating accounts, and binding one to a guardian.

Separated from the public routes by **credential type**, not just by prefix. A parent's
token cannot reach these at all, because they do not accept a bearer token as a credential
in the first place — `require_admin_key` is the only way in, and it takes a shared secret
the registrar's tooling holds and a browser never sees.
"""
from fastapi import APIRouter, status

from identity.api.deps import AdminKey, AdminServiceDep
from identity.api.schemas.admin import AccountIn, AccountOut, GuardianBindingIn

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.post("/accounts", response_model=AccountOut, status_code=status.HTTP_201_CREATED)
def create_account(
    body: AccountIn, service: AdminServiceDep, _: AdminKey
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
    return AccountOut(username=created.username, role=created.role, guardian_id=None)


@router.put("/accounts/{username}/guardian-binding", response_model=AccountOut)
def bind_guardian(
    username: str,
    body: GuardianBindingIn,
    service: AdminServiceDep,
    _: AdminKey,
) -> AccountOut:
    """Bind a login to a guardian. The single most sensitive write in the system.

    Audited as its own event type, because "who decided this parent is that guardian" is
    the first question anyone asks after a records leak.
    """
    bound = service.bind_guardian(
        username=username, guardian_external_id=body.guardian_external_id
    )
    return AccountOut(
        username=bound.username,
        role=bound.role,
        guardian_id=bound.guardian_external_id,
    )


@router.delete("/accounts/{username}/guardian-binding", response_model=AccountOut)
def unbind_guardian(
    username: str, service: AdminServiceDep, _: AdminKey
) -> AccountOut:
    """Remove a binding — the custody-change path.

    Takes effect for new access tokens immediately, for existing ones within their
    remaining lifetime, and for the session itself now: the refresh tokens are revoked
    here, so the parent cannot mint a fresh access token to carry on with.
    """
    unbound = service.unbind_guardian(username=username)
    return AccountOut(
        username=unbound.username, role=unbound.role, guardian_id=None
    )


__all__ = ["router"]
