"""Password sign-in, refresh, logout, and reading your own token.

Every handler here is an adapter: read the request, call one use case, map the result to a
response model. The rules — that an unknown user and a wrong password are
indistinguishable, that a refresh re-reads the binding — live in
`application/services/sessions.py`, and no `try/except` appears below because
`api/errors.py` turns a domain error into a status in one place.
"""
from fastapi import APIRouter, status

from identity.api.deps import BearerToken, ClientIp, SessionServiceDep
from identity.api.schemas.auth import (
    AccessTokenOut,
    LoginIn,
    MeOut,
    RefreshIn,
    TokenOut,
)
from identity.api.schemas.common import ErrorOut

router = APIRouter(prefix="/v1/auth", tags=["auth"])

AUTH_RESPONSES = {
    401: {"model": ErrorOut, "description": "Invalid credentials."},
    423: {"model": ErrorOut, "description": "Account temporarily locked."},
}


def _token_out(session) -> TokenOut:
    """One mapping, so the three doors cannot drift into three response shapes."""
    return TokenOut(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_at=session.expires_at,
        username=session.username,
        role=session.role,
        guardian_id=session.guardian_external_id,
        display_name=session.display_name,
    )


@router.post("/login", response_model=TokenOut, responses=AUTH_RESPONSES)
def login(body: LoginIn, service: SessionServiceDep, ip: ClientIp) -> TokenOut:
    """Exchange credentials for tokens.

    An unknown username and a wrong password produce the same 401 with the same message,
    in the same time. Distinguishing them turns this endpoint into an account enumerator,
    and for a school that means confirming which parents are registered.
    """
    return _token_out(service.login(username=body.username, password=body.password, client_ip=ip))


@router.post("/refresh", response_model=AccessTokenOut, responses=AUTH_RESPONSES)
def refresh(body: RefreshIn, service: SessionServiceDep, ip: ClientIp) -> AccessTokenOut:
    """Exchange a refresh token for a fresh access token.

    The guardian binding is re-read from the account, not carried over from the old token.
    That is what makes a revoked or corrected binding take effect within one access-token
    lifetime instead of persisting until the parent happens to log out — the case that
    matters is a custody change, and it must not wait a month.
    """
    issued = service.refresh(refresh_token=body.refresh_token, client_ip=ip)
    return AccessTokenOut(access_token=issued.access_token, expires_at=issued.expires_at)


@router.post("/logout")
def logout(body: RefreshIn, service: SessionServiceDep) -> dict:
    """Revoke a refresh token.

    The access token already issued stays valid until it expires — offline verification is
    the trade made for not calling this service on every request. Keeping access tokens
    short is what bounds that window.
    """
    return {"revoked": service.logout(refresh_token=body.refresh_token)}


@router.get("/me", response_model=MeOut, responses=AUTH_RESPONSES)
def me(token: BearerToken, service: SessionServiceDep) -> MeOut:
    """Decode the caller's own token. Useful to a front end; used by nothing critical."""
    from fastapi import HTTPException

    try:
        subject = service.describe_token(token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid token."},
        ) from None
    return MeOut(
        username=subject.username,
        role=subject.role,
        guardian_id=subject.guardian_external_id,
        display_name=subject.display_name,
        expires_at=subject.expires_at,
    )


__all__ = ["AUTH_RESPONSES", "router"]
