"""Password verification, lockout, and the admin credential.

PBKDF2-SHA256 in the same format the chat backend already uses
(`pbkdf2_sha256$rounds$salt$digest`), so accounts can be migrated between the two
without a password reset. Unlike an API key, a password *is* a low-entropy secret,
so stretching is exactly right here.
"""
import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from identity.models import Account, AuthAudit

PBKDF2_ROUNDS = int(os.getenv("IDENTITY_PBKDF2_ROUNDS", "310000"))
MAX_FAILED_ATTEMPTS = int(os.getenv("IDENTITY_MAX_FAILED_ATTEMPTS", "8"))
LOCKOUT_MINUTES = int(os.getenv("IDENTITY_LOCKOUT_MINUTES", "15"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("password is required")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return (
        f"pbkdf2_sha256${PBKDF2_ROUNDS}$"
        f"{base64.b64encode(salt).decode('ascii')}${base64.b64encode(digest).decode('ascii')}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash.startswith("pbkdf2_sha256$"):
        return False
    try:
        _, rounds, salt_b64, digest_b64 = password_hash.split("$", 3)
        calculated = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            base64.b64decode(salt_b64.encode("ascii")),
            int(rounds),
        )
        return hmac.compare_digest(calculated, base64.b64decode(digest_b64.encode("ascii")))
    except Exception:
        return False


def write_audit(
    db: Session,
    *,
    username: str,
    event: str,
    reason: str,
    succeeded: bool,
    client_ip: str = "",
) -> None:
    """Append one authentication event and commit it on its own.

    Committed separately for the same reason as the records audit: a failed login
    rolls its transaction back, and an audit that rolls back with it records only
    the successes.
    """
    db.add(
        AuthAudit(
            username=username[:120],
            event=event[:32],
            reason=reason[:40],
            succeeded=succeeded,
            client_ip=client_ip[:64],
        )
    )
    db.commit()


def is_locked(account: Account) -> bool:
    locked_until = account.locked_until
    if locked_until is None:
        return False
    # SQLite returns naive datetimes even from a timezone-aware column.
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return locked_until > _now()


def register_failure(db: Session, account: Account) -> None:
    """Count a bad password, and lock the account once there have been too many.

    Lockout is on the account rather than the IP because the threat here is credential
    stuffing against a known parent, and an attacker with a botnet has more IPs than
    the school has parents. The cost is that an attacker can lock a parent out
    deliberately, which is the better failure: a locked-out parent phones the school,
    whereas a breached one does not know to.
    """
    account.failed_attempts = (account.failed_attempts or 0) + 1
    if account.failed_attempts >= MAX_FAILED_ATTEMPTS:
        account.locked_until = _now() + timedelta(minutes=LOCKOUT_MINUTES)
        account.failed_attempts = 0
    db.commit()


def register_success(db: Session, account: Account) -> None:
    account.failed_attempts = 0
    account.locked_until = None
    db.commit()


def require_admin_key(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")) -> str:
    """Guards account creation and — critically — guardian binding.

    A shared secret rather than a token, because these routes are called by the
    registrar's tooling and by migration scripts, not by a logged-in user. It is the
    credential that can bind an account to a guardian, so it is the most dangerous
    secret in the system: anyone holding it can make themselves any parent.
    """
    expected = os.getenv("IDENTITY_ADMIN_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "not_configured", "message": "IDENTITY_ADMIN_KEY is not set."},
        )
    if not x_admin_key or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "not_authorized", "message": "Invalid admin key."},
        )
    return x_admin_key
