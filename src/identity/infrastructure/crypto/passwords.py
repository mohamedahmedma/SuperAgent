"""Password hashing, and verifying the formats this service has inherited.

PBKDF2-SHA256 in the format the chat backend used (`pbkdf2_sha256$rounds$salt$digest`), so
accounts migrate here without a password reset. Unlike an API key or a refresh token, a
password *is* a low-entropy secret, so stretching is exactly right.

Legacy bcrypt hashes are still verified, and **upgraded in place on the next successful
login**. That is the whole reason accounts could be imported from the old system silently:
a parent who has not logged in since the migration keeps their bcrypt hash, and the first
time they sign in it becomes PBKDF2 without them noticing. A migration that forced a
password reset on every family would have been abandoned halfway and left the old auth
running forever.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from typing import Final

logger = logging.getLogger(__name__)

PBKDF2_PREFIX: Final[str] = "pbkdf2_sha256$"

# Standard bcrypt: "$2a$", "$2b$", "$2y$". passlib's own variant: "$bcrypt-sha256$".
# They need different verification paths, so they are kept apart.
_BCRYPT_PREFIXES: Final[tuple[str, ...]] = ("$2a$", "$2b$", "$2y$", "$2$")
_BCRYPT_SHA256_PREFIX: Final[str] = "$bcrypt-sha256$"
_LEGACY_PREFIXES: Final[tuple[str, ...]] = _BCRYPT_PREFIXES + (_BCRYPT_SHA256_PREFIX,)


class Pbkdf2PasswordHasher:
    """`PasswordHasher` over `hashlib.pbkdf2_hmac`.

    The round count is passed in rather than read from the environment, which is what
    lets a test construct one at 10 rounds and exercise the lockout counter without
    spending a second per attempt on key stretching.
    """

    def __init__(self, *, rounds: int) -> None:
        self._rounds = rounds
        self._dummy_hash: str | None = None

    # -- hashing ------------------------------------------------------------

    def hash(self, password: str) -> str:
        if not password:
            raise ValueError("password is required")
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, self._rounds
        )
        return (
            f"{PBKDF2_PREFIX}{self._rounds}$"
            f"{base64.b64encode(salt).decode('ascii')}$"
            f"{base64.b64encode(digest).decode('ascii')}"
        )

    @property
    def dummy_hash(self) -> str:
        """A hash of a random string, computed once, for equalising a failed login.

        Verifying a candidate password against this costs exactly one key derivation —
        the same as verifying a real one — which is the entire point. See
        `SessionService._reject_unknown_user` for what the obvious alternative gets wrong.

        Built lazily and cached, so importing this module costs nothing: at 310,000 rounds
        it is roughly 100ms, which is fine once at first login and not fine at import time
        in every test collection.

        Over a random secret rather than a constant, so the digest is not a fixed value an
        attacker could recognise in a memory dump or a leaked heap and use to confirm which
        code path a request took.
        """
        if self._dummy_hash is None:
            self._dummy_hash = self.hash(secrets.token_urlsafe(32))
        return self._dummy_hash

    # -- verifying ----------------------------------------------------------

    def verify(self, password: str, password_hash: str) -> bool:
        """True if the password matches, in either the current or a legacy format."""
        if not password or not password_hash:
            return False

        if password_hash.startswith(PBKDF2_PREFIX):
            return self._verify_pbkdf2(password, password_hash)
        if password_hash.startswith(_BCRYPT_SHA256_PREFIX):
            return _verify_passlib_bcrypt_sha256(password, password_hash)
        if password_hash.startswith(_BCRYPT_PREFIXES):
            return _verify_bcrypt(password, password_hash)
        return False

    @staticmethod
    def _verify_pbkdf2(password: str, password_hash: str) -> bool:
        """Verified at the hash's *own* round count, not the current setting.

        A hash written before the rounds were raised must still verify, or raising them
        would lock out everyone who had not signed in since. `needs_rehash` is what then
        migrates them forward.
        """
        try:
            _, rounds, salt_b64, digest_b64 = password_hash.split("$", 3)
            calculated = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                base64.b64decode(salt_b64.encode("ascii")),
                int(rounds),
            )
            return hmac.compare_digest(
                calculated, base64.b64decode(digest_b64.encode("ascii"))
            )
        except Exception:
            return False

    def needs_rehash(self, password_hash: str) -> bool:
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
                return int(password_hash.split("$", 2)[1]) < self._rounds
            except Exception:
                return True
        return True


def _verify_passlib_bcrypt_sha256(password: str, password_hash: str) -> bool:
    """passlib's own bcrypt_sha256 variant.

    Verified through passlib because reimplementing its pre-hashing from the spec is a
    subtle thing to get wrong and there is no way to test it here — passlib 1.7.4 cannot
    even generate one against bcrypt 5.x, which is the same reason it may fail to verify.

    A failure is logged rather than swallowed. These accounts need a password reset, and an
    operator can only arrange that if they know which ones.
    """
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


def _verify_bcrypt(password: str, password_hash: str) -> bool:
    """Standard bcrypt, verified against the bcrypt library directly.

    Not through passlib: passlib 1.7.4 raises on bcrypt 5.x before it reaches the
    comparison, so routing through it would reject every legacy account — silently, since
    the old backend caught the exception and returned False.
    """
    try:
        import bcrypt as _bcrypt

        # bcrypt has always used only the first 72 bytes; version 5 raises instead of
        # truncating. Truncating here reproduces what the library did when the stored hash
        # was created, which is what makes the comparison valid.
        return _bcrypt.checkpw(
            password.encode("utf-8")[:72], password_hash.encode("utf-8")
        )
    except Exception as exc:
        logger.warning("Legacy bcrypt verification failed: %s", exc)
        return False


__all__ = ["PBKDF2_PREFIX", "Pbkdf2PasswordHasher"]
