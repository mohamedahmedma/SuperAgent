"""Password verification, lockout, and the admin credential.

PBKDF2-SHA256 in the format the chat backend used
(`pbkdf2_sha256$rounds$salt$digest`), so accounts migrate here without a password
reset. Unlike an API key, a password *is* a low-entropy secret, so stretching is
exactly right.

Legacy bcrypt hashes are still verified, and **upgraded in place on the next
successful login**. That is the whole reason accounts can be imported from the old
system silently: a parent who has not logged in since the migration keeps their bcrypt
hash, and the first time they sign in it becomes PBKDF2 without them noticing. A
migration that forced a password reset on every family would have been abandoned
halfway and left the old auth running forever.
"""
import base64
import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta, timezone

from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from identity.models import Account, AuthAudit

logger = logging.getLogger(__name__)

PBKDF2_ROUNDS = int(os.getenv("IDENTITY_PBKDF2_ROUNDS", "310000"))
MAX_FAILED_ATTEMPTS = int(os.getenv("IDENTITY_MAX_FAILED_ATTEMPTS", "8"))
LOCKOUT_MINUTES = int(os.getenv("IDENTITY_LOCKOUT_MINUTES", "15"))

# Grants the "admin" role at self-registration. Unset means no self-service route can
# ever produce an admin, which is the correct posture once the first admin exists.
ADMIN_INVITE_CODE = os.getenv("IDENTITY_ADMIN_INVITE_CODE", "")

PBKDF2_PREFIX = "pbkdf2_sha256$"
# Standard bcrypt: "$2a$", "$2b$", "$2y$". passlib's own variant: "$bcrypt-sha256$".
# They need different verification paths, so they are kept apart.
_BCRYPT_PREFIXES = ("$2a$", "$2b$", "$2y$", "$2$")
_BCRYPT_SHA256_PREFIX = "$bcrypt-sha256$"
_LEGACY_PREFIXES = _BCRYPT_PREFIXES + (_BCRYPT_SHA256_PREFIX,)

# Self-registration may produce these and nothing else. "parent" is absent on purpose:
# a parent role is paired with a guardian binding, and both are an administrator's
# decision. Registration must never be a path to either.
SELF_REGISTRABLE_ROLES = frozenset({"user", "admin"})
ASSIGNABLE_ROLES = frozenset({"user", "admin", "parent", "staff"})


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
    """True if the password matches, in either the current or the legacy format."""
    if not password or not password_hash:
        return False

    if password_hash.startswith(PBKDF2_PREFIX):
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

    if password_hash.startswith(_BCRYPT_SHA256_PREFIX):
        # passlib's own bcrypt_sha256 variant. Verified through passlib because
        # reimplementing its pre-hashing from the spec is a subtle thing to get wrong
        # and there is no way to test it here — passlib 1.7.4 cannot even generate one
        # against bcrypt 5.x, which is the same reason it may fail to verify.
        #
        # A failure is logged rather than swallowed. These accounts need a password
        # reset, and an operator can only arrange that if they know which ones.
        try:
            from passlib.context import CryptContext

            return CryptContext(schemes=["bcrypt_sha256"], deprecated="auto").verify(
                password, password_hash
            )
        except Exception as exc:
            logger.warning(
                "Cannot verify a passlib bcrypt_sha256 hash (%s). This account needs a "
                "password reset; passlib 1.7.4 is incompatible with bcrypt 5.x.",
                exc,
            )
            return False

    if password_hash.startswith(_BCRYPT_PREFIXES):
        # Standard bcrypt, verified against the bcrypt library directly rather than
        # through passlib. passlib 1.7.4 raises on bcrypt 5.x before it reaches the
        # comparison, so routing through it would reject every legacy account —
        # silently, since the old backend caught the exception and returned False.
        try:
            import bcrypt as _bcrypt

            # bcrypt has always used only the first 72 bytes; version 5 raises instead
            # of truncating. Truncating here reproduces what the library did when the
            # stored hash was created, which is what makes the comparison valid.
            return _bcrypt.checkpw(
                password.encode("utf-8")[:72], password_hash.encode("utf-8")
            )
        except Exception as exc:
            logger.warning("Legacy bcrypt verification failed: %s", exc)
            return False

    return False


def needs_rehash(password_hash: str) -> bool:
    """Whether a verified hash should be replaced with the current format.

    True for legacy bcrypt, and for PBKDF2 written at fewer rounds than the current
    setting — raising `IDENTITY_PBKDF2_ROUNDS` then upgrades everyone as they sign in,
    rather than only new accounts.
    """
    if not password_hash:
        return True
    if password_hash.startswith(_LEGACY_PREFIXES):
        return True
    if password_hash.startswith(PBKDF2_PREFIX):
        try:
            return int(password_hash.split("$", 2)[1]) < PBKDF2_ROUNDS
        except Exception:
            return True
    return True


def upgrade_hash_if_needed(db: Session, account: Account, password: str) -> None:
    """Re-hash a just-verified password into the current format.

    Called only on the success path, where the plaintext is in hand and already
    proven correct. A failure here must not fail the login — the user authenticated,
    and the worst case is that the upgrade happens on their next sign-in instead.
    """
    if not needs_rehash(account.password_hash):
        return
    try:
        account.password_hash = hash_password(password)
        db.commit()
    except Exception:
        logger.exception("Password hash upgrade failed for %s", account.username)
        db.rollback()


def resolve_registration_role(requested_role: str | None, admin_code: str | None) -> str:
    """What role a self-registration may claim.

    Ported from the old backend's `resolve_role`, with one change: an incorrect invite
    code is rejected rather than silently downgraded to "user". Silently ignoring it
    means an operator who mistypes the code gets an ordinary account and no
    explanation, then files a bug against the wrong system.
    """
    role = (requested_role or "user").strip().lower()
    if role not in SELF_REGISTRABLE_ROLES:
        return "user"
    if role != "admin":
        return "user"
    if ADMIN_INVITE_CODE and admin_code and hmac.compare_digest(admin_code, ADMIN_INVITE_CODE):
        return "admin"
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"code": "not_authorized", "message": "Incorrect administrator invite code."},
    )


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
