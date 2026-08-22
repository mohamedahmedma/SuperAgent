"""HTTP surface for the identity service.

Two routers. The public one authenticates; the admin one creates accounts and binds
them to guardians. They are separated by credential type, not just by prefix — a
parent's token cannot reach the admin routes at all, because those routes do not
accept a bearer token as a credential in the first place.
"""
from datetime import datetime, timezone

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy.orm import Session

from identity import auth, keys, tokens
from identity import verification as verify_flow
from identity import whatsapp as wa
from identity.db import get_db
from identity.deps import get_verification_service
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
    WhatsAppStartOut,
    WhatsAppStatusIn,
    WhatsAppStatusOut,
    WhatsAppVerifyIn,
)

public_router = APIRouter(prefix="/v1/auth", tags=["auth"])
admin_router = APIRouter(prefix="/v1/admin", tags=["admin"])
wellknown_router = APIRouter(tags=["keys"])

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Parent login by WhatsApp
# ---------------------------------------------------------------------------
#
# Why this exists beside the password login above: a parent has no password and should
# never be given one. The school already holds their phone number, entered by a registrar
# from paperwork, and WhatsApp can prove somebody controls that number for nothing. See
# identity/verification.py for the flow and for why it takes two secrets rather than one.

whatsapp_router = APIRouter(prefix="/v1/auth/whatsapp", tags=["auth"])


def _verification_error(error: verify_flow.VerificationError) -> HTTPException:
    """Every refusal in this flow is a 400 carrying a code the page can branch on.

    One status for all of them on purpose. "No such verification", "wrong code" and
    "expired" would each justify a different status in isolation, but distinguishing them
    by status lets somebody holding a stolen poll secret learn which of their guesses was
    structurally wrong rather than merely incorrect.
    """
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"code": error.code, "message": error.message},
    )


@whatsapp_router.post("/start", response_model=WhatsAppStartOut, status_code=201)
def start_whatsapp_verification(
    db: Session = Depends(get_db),
    service: verify_flow.VerificationService = Depends(get_verification_service),
) -> WhatsAppStartOut:
    """Begin a verification. Takes no phone number, and that is the point.

    Because the caller states nothing, there is nothing here to probe: this endpoint cannot
    be used to ask whether a given number belongs to a parent. That question is answered
    only to somebody who can actually send a WhatsApp message from the number, and it is
    answered over WhatsApp rather than in this response.
    """
    started = service.start(db)
    return WhatsAppStartOut(
        poll_secret=started.poll_secret,
        link=started.link,
        message=started.message,
        business_number=service.business_number,
        expires_at=started.expires_at,
    )


@whatsapp_router.get("/webhook", include_in_schema=False)
def verify_whatsapp_webhook(request: Request) -> Response:
    """Meta's subscription handshake.

    Answered with the bare `hub.challenge` as plain text — no JSON, no quotes. Meta
    compares the body byte for byte, and a JSON-wrapped answer fails the subscription with
    no explanation beyond "the callback URL could not be validated".
    """
    params = request.query_params
    expected = wa.get_verify_token()
    if (
        params.get("hub.mode") == "subscribe"
        and expected
        and params.get("hub.verify_token") == expected
    ):
        return PlainTextResponse(params.get("hub.challenge") or "")
    return PlainTextResponse("", status_code=status.HTTP_403_FORBIDDEN)


@whatsapp_router.post("/webhook", include_in_schema=False)
async def receive_whatsapp_webhook(
    request: Request,
    db: Session = Depends(get_db),
    service: verify_flow.VerificationService = Depends(get_verification_service),
) -> dict:
    """Inbound WhatsApp messages.

    **Always answers 200 once the signature checks out.** Meta retries any delivery it does
    not see acknowledged, for up to seven days, so returning an error for a message we
    simply cannot use would have that message replayed for a week.

    The signature is computed over the raw bytes. Re-serialising the parsed JSON produces
    different bytes — Meta escapes non-ASCII — so an Arabic name in a parent's WhatsApp
    profile is enough to break a signature checked against re-encoded JSON, and it breaks
    for only some parents, which is the worst way to find out.
    """
    raw = await request.body()
    if not wa.signature_is_valid(
        raw_body=raw,
        header=request.headers.get("X-Hub-Signature-256"),
        app_secret=wa.get_app_secret(),
    ):
        # 403 and nothing more. An unsigned caller is not Meta, and Meta does not retry a
        # 403 the way it retries a 5xx.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "not_authorized", "message": "Bad signature."},
        )

    for sender, text, message_id in wa.inbound_text_messages(raw):
        outcome = service.claim(db, wa_id=sender, body=text, message_id=message_id)
        # The number is deliberately not logged: it is a parent's phone and this line goes
        # wherever logs go. The message id is enough to trace one delivery.
        logger.info(
            "WhatsApp verification: outcome=%s message_id=%s", outcome, message_id
        )
    return {"received": True}


@whatsapp_router.post("/status", response_model=WhatsAppStatusOut)
def whatsapp_verification_status(
    body: WhatsAppStatusIn,
    db: Session = Depends(get_db),
    service: verify_flow.VerificationService = Depends(get_verification_service),
) -> WhatsAppStatusOut:
    """Where has this verification got to? Polled while the parent goes to tap send."""
    try:
        challenge = service.status(db, poll_secret=body.poll_secret)
    except verify_flow.VerificationError as error:
        raise _verification_error(error) from error
    return WhatsAppStatusOut(
        status=challenge.status,
        display_name=challenge.display_name,
        expires_at=challenge.expires_at,
    )


@whatsapp_router.post("/verify", response_model=TokenOut, responses=_AUTH_RESPONSES)
def complete_whatsapp_verification(
    body: WhatsAppVerifyIn,
    request: Request,
    db: Session = Depends(get_db),
    service: verify_flow.VerificationService = Depends(get_verification_service),
) -> TokenOut:
    """The code, and the tokens if it is right.

    The account is created here on first use rather than by an administrator, and the
    binding written onto it does not come from this request: it came from the school's own
    records, keyed on a number WhatsApp proved. That is the invariant identity/models.py
    states — an account never names its own guardian — held through a second authority
    rather than broken by one.
    """
    ip = _client_ip(request)
    try:
        challenge = service.verify(db, poll_secret=body.poll_secret, code=body.code)
    except verify_flow.VerificationError as error:
        auth.write_audit(
            db,
            username="",
            event="whatsapp_verify",
            reason=error.code,
            succeeded=False,
            client_ip=ip,
        )
        raise _verification_error(error) from error

    account = _account_for_guardian(db, challenge)

    access_token, expires_at = tokens.mint_access_token(
        subject=account.username,
        role=account.role,
        guardian_external_id=account.guardian_external_id,
        display_name=account.display_name,
    )
    raw_refresh, refresh_hash, refresh_expires = tokens.mint_refresh_token()
    db.add(
        RefreshToken(
            account_id=account.id, token_hash=refresh_hash, expires_at=refresh_expires
        )
    )
    db.commit()

    auth.write_audit(
        db,
        username=account.username,
        event="whatsapp_verify",
        reason="ok",
        succeeded=True,
        client_ip=ip,
    )
    return TokenOut(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_at=expires_at,
        username=account.username,
        role=account.role,
        guardian_id=account.guardian_external_id,
        display_name=account.display_name,
    )


def _account_for_guardian(db: Session, challenge) -> Account:
    """Find or create the account this guardian signs in through.

    Keyed on the guardian handle rather than on the phone number, so a parent who verifies
    her second number lands in the account she already had instead of acquiring a duplicate
    holding half her history. The username is derived from the handle for the same reason:
    a username built from a phone would have to change when she changes number, and a
    username is a join key elsewhere.

    The account carries no password. `verify_password` cannot succeed against the empty
    hash stored here, so the password route stays shut for parents — this is the only door
    they have, and it is one the school closes by removing a guardian link rather than by
    resetting anything.

    The binding is **re-asserted on every sign-in**, deliberately. A registrar who corrects
    a guardian record in sis/ should see it take effect the next time that parent signs in,
    without an administrator having to touch this service as well.
    """
    username = f"guardian:{challenge.guardian_external_id}"
    account = db.query(Account).filter(Account.username == username).first()
    if account is None:
        account = Account(
            username=username,
            phone="",
            password_hash="",
            role="parent",
            guardian_external_id=challenge.guardian_external_id,
            display_name=challenge.display_name,
            preferred_language=challenge.preferred_language or "ar",
            is_active=True,
        )
        db.add(account)
        db.commit()
        db.refresh(account)
        return account

    # An account an administrator disabled stays disabled: re-verifying a phone must not
    # become a way to walk back that decision.
    if not account.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "That account is disabled."},
        )
    account.guardian_external_id = challenge.guardian_external_id
    if challenge.display_name:
        account.display_name = challenge.display_name
    db.commit()
    db.refresh(account)
    return account
