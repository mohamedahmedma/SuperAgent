"""HTTP surface for the identity service.

Two routers. The public one authenticates; the admin one creates accounts and binds
them to guardians. They are separated by credential type, not just by prefix — a
parent's token cannot reach the admin routes at all, because those routes do not
accept a bearer token as a credential in the first place.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from identity import auth, keys, tokens
from identity.db import get_db
from identity.models import Account, RefreshToken
from identity.schemas import (
    AccessTokenOut,
    AccountIn,
    ErrorOut,
    GuardianBindingIn,
    LoginIn,
    MeOut,
    RefreshIn,
    RegisterIn,
    TokenOut,
)

public_router = APIRouter(prefix="/v1/auth", tags=["auth"])
admin_router = APIRouter(prefix="/v1/admin", tags=["admin"])
wellknown_router = APIRouter(tags=["keys"])

_AUTH_RESPONSES = {
    401: {"model": ErrorOut, "description": "Invalid credentials."},
    423: {"model": ErrorOut, "description": "Account temporarily locked."},
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


@wellknown_router.get("/.well-known/jwks.json")
def jwks() -> dict:
    """The public signing key.

    Every other service verifies tokens against this and holds nothing that could
    mint one. Public by design — a public key is not a secret.
    """
    return keys.jwks()


@public_router.post("/login", response_model=TokenOut, responses=_AUTH_RESPONSES)
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)) -> TokenOut:
    """Exchange credentials for tokens.

    An unknown username and a wrong password produce the same 401 with the same
    message. Distinguishing them turns this endpoint into an account enumerator, and
    for a school that means confirming which parents are registered.
    """
    ip = _client_ip(request)
    account = db.query(Account).filter(Account.username == body.username).first()

    # Hash unconditionally so a missing account and a wrong password take comparable
    # time. Returning early on a miss leaks account existence through latency.
    if account is None:
        auth.verify_password(body.password, auth.hash_password("timing-equalizer"))
        auth.write_audit(
            db, username=body.username, event="login", reason="unknown_user", succeeded=False, client_ip=ip
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid username or password."},
        )

    if auth.is_locked(account):
        auth.write_audit(db, username=body.username, event="login", reason="locked", succeeded=False, client_ip=ip)
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail={"code": "locked", "message": "Account temporarily locked. Try again later."},
        )

    if not account.is_active:
        auth.write_audit(db, username=body.username, event="login", reason="inactive", succeeded=False, client_ip=ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid username or password."},
        )

    if not auth.verify_password(body.password, account.password_hash):
        auth.register_failure(db, account)
        auth.write_audit(
            db, username=body.username, event="login", reason="bad_password", succeeded=False, client_ip=ip
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid username or password."},
        )

    auth.register_success(db, account)
    # The plaintext is in hand and proven correct — the only moment a legacy bcrypt
    # hash imported from the old backend can be upgraded. See identity/auth.py.
    auth.upgrade_hash_if_needed(db, account, body.password)

    access_token, expires_at = tokens.mint_access_token(
        subject=account.username,
        role=account.role,
        guardian_external_id=account.guardian_external_id,
        display_name=account.display_name,
    )
    raw_refresh, refresh_hash, refresh_expires = tokens.mint_refresh_token()
    db.add(RefreshToken(account_id=account.id, token_hash=refresh_hash, expires_at=refresh_expires))
    db.commit()

    auth.write_audit(db, username=body.username, event="login", reason="ok", succeeded=True, client_ip=ip)

    return TokenOut(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_at=expires_at,
        username=account.username,
        role=account.role,
        guardian_id=account.guardian_external_id,
        display_name=account.display_name,
    )


@public_router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, request: Request, db: Session = Depends(get_db)) -> TokenOut:
    """Self-registration, ported from the old chat backend.

    Produces an account that can sign in and read no student records whatsoever: the
    guardian binding is a separate, admin-only write, and nothing on this path can
    reach it. That is what makes leaving self-registration open acceptable — the worst
    a stranger gets is a chat account.
    """
    username = (body.username or "").strip()
    password = (body.password or "").strip()
    if not username or not password:
        raise HTTPException(
            status_code=400,
            detail={"code": "bad_request", "message": "Username and password cannot be empty."},
        )

    if db.query(Account).filter(Account.username == username).first():
        raise HTTPException(
            status_code=409, detail={"code": "conflict", "message": "Username already exists."}
        )

    # Raises 403 on a wrong invite code rather than silently downgrading the role.
    role = auth.resolve_registration_role(body.role, body.admin_code)

    account = Account(
        username=username,
        password_hash=auth.hash_password(password),
        role=role,
        display_name=body.display_name,
        preferred_language=body.preferred_language,
    )
    db.add(account)
    db.commit()

    access_token, expires_at = tokens.mint_access_token(
        subject=account.username,
        role=account.role,
        guardian_external_id=None,
        display_name=account.display_name,
    )
    raw_refresh, refresh_hash, refresh_expires = tokens.mint_refresh_token()
    db.add(RefreshToken(account_id=account.id, token_hash=refresh_hash, expires_at=refresh_expires))
    db.commit()

    auth.write_audit(
        db, username=username, event="register", reason="ok", succeeded=True, client_ip=_client_ip(request)
    )

    return TokenOut(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_at=expires_at,
        username=account.username,
        role=account.role,
        guardian_id=None,
        display_name=account.display_name,
    )


@public_router.post("/refresh", response_model=AccessTokenOut, responses=_AUTH_RESPONSES)
def refresh(body: RefreshIn, request: Request, db: Session = Depends(get_db)) -> AccessTokenOut:
    """Exchange a refresh token for a fresh access token.

    The guardian binding is re-read from the account here, not carried over from the
    old token. That is what makes a revoked or corrected binding take effect within
    one access-token lifetime instead of persisting until the parent happens to log
    out — the case that matters is a custody change, and it must not wait a month.
    """
    ip = _client_ip(request)
    record = (
        db.query(RefreshToken)
        .filter(RefreshToken.token_hash == tokens.hash_refresh_token(body.refresh_token))
        .first()
    )

    expires_at = record.expires_at if record else None
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if record is None or record.revoked_at is not None or (expires_at and expires_at < _now()):
        auth.write_audit(
            db, username="", event="refresh", reason="expired_refresh", succeeded=False, client_ip=ip
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid or expired refresh token."},
        )

    account = db.query(Account).filter(Account.id == record.account_id).first()
    if account is None or not account.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid or expired refresh token."},
        )

    access_token, access_expires = tokens.mint_access_token(
        subject=account.username,
        role=account.role,
        guardian_external_id=account.guardian_external_id,
        display_name=account.display_name,
    )
    auth.write_audit(db, username=account.username, event="refresh", reason="ok", succeeded=True, client_ip=ip)
    return AccessTokenOut(access_token=access_token, expires_at=access_expires)


@public_router.post("/logout")
def logout(body: RefreshIn, db: Session = Depends(get_db)) -> dict:
    """Revoke a refresh token.

    The access token already issued stays valid until it expires — offline
    verification is the trade made for not calling this service on every request.
    Keeping access tokens short is what bounds that window.
    """
    record = (
        db.query(RefreshToken)
        .filter(RefreshToken.token_hash == tokens.hash_refresh_token(body.refresh_token))
        .first()
    )
    if record is not None and record.revoked_at is None:
        record.revoked_at = _now()
        db.commit()
    return {"revoked": True}


@public_router.get("/me", response_model=MeOut, responses=_AUTH_RESPONSES)
def me(authorization: str | None = Header(default=None)) -> MeOut:
    """Decode the caller's own token. Useful to a front end; used by nothing critical."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Missing bearer token."},
        )
    try:
        claims = tokens.decode_own_token(authorization.split(" ", 1)[1].strip())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid token."},
        )

    return MeOut(
        username=claims.get("sub", ""),
        role=claims.get("role", ""),
        guardian_id=claims.get("guardian_id"),
        display_name=claims.get("name", ""),
        expires_at=datetime.fromtimestamp(claims["exp"], tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Admin. Admin key only — a parent token cannot reach these at all.
# ---------------------------------------------------------------------------


@admin_router.post("/accounts", status_code=status.HTTP_201_CREATED)
def create_account(
    body: AccountIn,
    db: Session = Depends(get_db),
    _: str = Depends(auth.require_admin_key),
) -> dict:
    """Create a login. Note what this route cannot do: bind a guardian.

    Creation and binding are deliberately two calls. A bulk parent import that runs
    only this one produces accounts that can log in and read nothing, which is the
    safe half-finished state.
    """
    if db.query(Account).filter(Account.username == body.username).first():
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": "Username already exists."})

    account = Account(
        username=body.username,
        phone=body.phone,
        password_hash=auth.hash_password(body.password),
        role=body.role if body.role in auth.ASSIGNABLE_ROLES else "parent",
        display_name=body.display_name,
        preferred_language=body.preferred_language,
    )
    db.add(account)
    db.commit()
    return {"username": account.username, "role": account.role, "guardian_id": None}


@admin_router.put("/accounts/{username}/guardian-binding")
def bind_guardian(
    username: str,
    body: GuardianBindingIn,
    db: Session = Depends(get_db),
    _: str = Depends(auth.require_admin_key),
) -> dict:
    """Bind a login to a guardian. The single most sensitive write in the system.

    Audited as its own event type, because "who decided this parent is that guardian"
    is the first question anyone asks after a records leak.
    """
    account = db.query(Account).filter(Account.username == username).first()
    if account is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "No such account."})

    account.guardian_external_id = body.guardian_external_id
    db.commit()
    auth.write_audit(db, username=username, event="guardian_bind", reason="ok", succeeded=True)
    return {"username": username, "guardian_id": account.guardian_external_id}


@admin_router.delete("/accounts/{username}/guardian-binding")
def unbind_guardian(
    username: str,
    db: Session = Depends(get_db),
    _: str = Depends(auth.require_admin_key),
) -> dict:
    """Remove a binding — the custody-change path.

    Takes effect for new access tokens immediately and for existing ones within their
    remaining lifetime. Revoke the refresh token too when the change is urgent.
    """
    account = db.query(Account).filter(Account.username == username).first()
    if account is None:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "No such account."})

    account.guardian_external_id = None
    db.query(RefreshToken).filter(
        RefreshToken.account_id == account.id, RefreshToken.revoked_at.is_(None)
    ).update({"revoked_at": _now()})
    db.commit()
    auth.write_audit(db, username=username, event="guardian_unbind", reason="ok", succeeded=True)
    return {"username": username, "guardian_id": None}
