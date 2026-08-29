"""Password hashing and session tokens. Standard library only, no new dependency.

PBKDF2-SHA256 in the same `pbkdf2_sha256$rounds$salt$digest` format `identity/` writes,
so a hash is legible across the estate and a future consolidation of the two account
stores is a copy rather than a forced reset for every member of staff. It is deliberately
*not* an import from `identity/`: the services do not import each other (SERVICES.md), and
breaking that for forty lines of hashlib would couple the school's staff logins to the
parent-facing auth service's deployment.

**Verification is constant-time and always does the work.** `verify` against an account
that does not exist still runs the KDF, against a dummy hash, so "no such user" and "wrong
password" take the same time. Skipping it turns username enumeration into a stopwatch
measurement — and a school's usernames are staff names, so knowing which ones are real is
most of a targeted phishing list.

**Rounds are a stored property of each hash, not a global.** Raising the cost next year
must not invalidate every existing password; it makes new hashes more expensive and leaves
old ones verifiable, which is what `needs_rehash` is for.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Final

# OWASP's floor for PBKDF2-SHA256 at the time of writing. High enough to make an offline
# guess expensive, low enough that a login is not a visible pause on a school laptop.
DEFAULT_ROUNDS: Final[int] = 600_000

PBKDF2_PREFIX: Final[str] = "pbkdf2_sha256$"

_SALT_BYTES: Final[int] = 16
_TOKEN_BYTES: Final[int] = 32


def _b64(raw: bytes) -> str:
    """URL-safe, unpadded — the hash goes in a database column and sometimes a log."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, *, rounds: int = DEFAULT_ROUNDS) -> str:
    """Hash a password with a fresh random salt. The only way a verifier is created."""
    if not isinstance(password, str) or not password:
        raise ValueError("a password is required")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"{PBKDF2_PREFIX}{rounds}${_b64(salt)}${_b64(digest)}"


def dummy_hash() -> str:
    """A verifier that matches nothing, for the no-such-user path.

    Cheap to build — one round rather than six hundred thousand — because it is compared
    against, never derived from. What has to be constant is the *verify* work, and that is
    governed by the rounds recorded inside the hash being verified. This is the one place
    where a low round count is correct rather than a mistake, so it is spelled out.
    """
    return f"{PBKDF2_PREFIX}{DEFAULT_ROUNDS}${_b64(b'0' * _SALT_BYTES)}${_b64(b'0' * 32)}"


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time verification. Malformed hashes are a refusal, never an exception.

    A row whose hash has been corrupted or hand-edited must fail the login, not 500 the
    route — the second one takes the whole console down for everybody while somebody
    works out which account it was.
    """
    if not password or not password_hash:
        return False
    if not password_hash.startswith(PBKDF2_PREFIX):
        return False
    try:
        rounds_text, salt_text, digest_text = password_hash[len(PBKDF2_PREFIX) :].split("$")
        rounds = int(rounds_text)
        salt = _unb64(salt_text)
        expected = _unb64(digest_text)
    except (ValueError, TypeError):
        return False
    if rounds < 1:
        return False
    calculated = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(calculated, expected)


def needs_rehash(password_hash: str, *, rounds: int = DEFAULT_ROUNDS) -> bool:
    """True when this hash was written cheaper than the current cost.

    Checked on a successful login so the estate migrates upward as people sign in, rather
    than in one migration that cannot possibly know anybody's password.
    """
    if not password_hash.startswith(PBKDF2_PREFIX):
        return True
    try:
        stored = int(password_hash[len(PBKDF2_PREFIX) :].split("$", 1)[0])
    except (ValueError, IndexError):
        return True
    return stored < rounds


def generate_session_token() -> tuple[str, str]:
    """A new session token: `(what the browser gets, what the database keeps)`.

    SHA-256 rather than PBKDF2 for the stored half, and the difference matters. A password
    is low-entropy and guessable, so it needs a slow KDF. A 256-bit random token is not
    guessable at any speed, so the only thing hashing it buys is that a leaked database
    holds no usable session — and a fast hash buys that just as completely, while being
    cheap enough to run on every single request.
    """
    raw = secrets.token_urlsafe(_TOKEN_BYTES)
    return raw, hash_session_token(raw)


def hash_session_token(raw: str) -> str:
    """The stored form of a session token. Deterministic, so a lookup is one index hit."""
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_ROUNDS",
    "PBKDF2_PREFIX",
    "dummy_hash",
    "generate_session_token",
    "hash_password",
    "hash_session_token",
    "needs_rehash",
    "verify_password",
]
