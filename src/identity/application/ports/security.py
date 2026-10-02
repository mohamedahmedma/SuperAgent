"""The cryptographic seams: hashing a password, and signing a token.

Ports rather than direct calls, for one reason each.

`PasswordHasher` because PBKDF2 at 310,000 rounds is roughly 100ms of CPU, and a use-case
test that exercises the lockout counter eight times should not spend a second doing
key stretching to prove a rule about integers. A test substitutes a hasher that compares
strings.

`TokenIssuer` because signing needs an RSA key, and a service that reached for one
directly could not be exercised without generating or loading one.

Neither port exists to make the algorithm swappable. PBKDF2 and RS256 are decided, and the
reasons are written down in `infrastructure/crypto/`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol


class PasswordHasher(Protocol):
    """Hashing and verifying a password, in the current format and the legacy ones."""

    def hash(self, password: str) -> str:
        """A new hash, in the current format."""

    def verify(self, password: str, password_hash: str) -> bool:
        """True if the password matches, in either the current or a legacy format."""

    def needs_rehash(self, password_hash: str) -> bool:
        """Whether a just-verified hash should be replaced with the current format.

        True for legacy bcrypt, and for PBKDF2 written at fewer rounds than the current
        setting — raising the round count then upgrades everyone as they sign in, rather
        than only new accounts.
        """

    @property
    def dummy_hash(self) -> str:
        """A hash of nothing in particular, for equalising the timing of a failed login.

        Verifying against this costs exactly what verifying a real password costs, which
        is the point: an unknown username and a wrong password must take the same time, or
        this endpoint becomes an account enumerator and, for a school, that means
        confirming which parents are registered.

        Precomputed once rather than generated per miss. The obvious spelling —
        `verify(password, hash("timing-equalizer"))` — runs the key derivation *twice* on
        a miss and once on a hit, so it does not equalise the timing at all: it inverts
        it, makes an unknown user measurably slower than a known one, and doubles the CPU
        cost of the most-attacked endpoint in the estate.
        """


class TokenIssuer(Protocol):
    """Minting the tokens only this service can mint.

    Verification elsewhere is done offline against the published JWKS with a public key,
    so nothing outside this process can produce one. That asymmetry is the architecture:
    with a shared secret, every service that *verifies* a token also holds the key that
    *mints* one.
    """

    def mint_access_token(
        self,
        *,
        subject: str,
        role: str,
        guardian_external_id: str | None,
        display_name: str = "",
        children: Sequence[Mapping[str, str]] = (),
        school_code: str | None = None,
    ) -> tuple[str, datetime]:
        """Sign a short-lived access token. Returns `(token, expires_at)`."""

    def mint_refresh_token(self) -> tuple[str, str, datetime]:
        """Return `(raw, hash, expires_at)`. Opaque random bytes, not a JWT."""

    def hash_refresh_token(self, raw: str) -> str:
        """The stored form. SHA-256 — a 48-byte random value needs no stretching."""

    def decode_own_token(self, token: str) -> dict:
        """Verify a token this service issued. Raises `ValueError` when it did not."""


__all__ = ["PasswordHasher", "TokenIssuer"]
