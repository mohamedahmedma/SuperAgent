"""Minting access and refresh tokens.

Only this service ever calls into this module. Verification elsewhere is done against the
published JWKS with a public key, so nothing outside can produce a token.

What goes *into* a token is `domain/claims.py`'s decision, not this file's. What is here
is the signing, the opaque refresh token, and the one decode this service does of its own
tokens.
"""
from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from identity.domain.claims import AccessClaims
from identity.infrastructure.crypto.keys import SigningKey


class JwtTokenIssuer:
    """`TokenIssuer` over python-jose and one `SigningKey`.

    Issuer, audience and the two lifetimes arrive as constructor arguments. They used to
    be module-level `os.getenv` calls, which meant a test that set `IDENTITY_AUDIENCE` in
    a fixture was reliably too late — the value had been captured when pytest imported the
    module during collection.
    """

    def __init__(
        self,
        *,
        key: SigningKey,
        issuer: str,
        audience: str,
        access_ttl_minutes: int,
        refresh_ttl_days: int,
        clock=lambda: datetime.now(timezone.utc),
    ) -> None:
        self._key = key
        self._issuer = issuer
        # Tokens are minted for a named audience and verifiers must check it. Without
        # this, a token accepted by the chat backend is replayable against the records
        # facade — same signature, same issuer, different blast radius.
        self._audience = audience
        self._access_ttl = timedelta(minutes=access_ttl_minutes)
        self._refresh_ttl = timedelta(days=refresh_ttl_days)
        self._clock = clock

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
        """Sign a short-lived access token. Returns `(token, expires_at)`.

        Which claims survive into the payload — and which are omitted rather than sent
        empty — is `AccessClaims.as_dict`, where the reasoning lives.
        """
        now = self._clock()
        expires_at = now + self._access_ttl
        claims = AccessClaims(
            issuer=self._issuer,
            audience=self._audience,
            subject=subject,
            role=role,
            display_name=display_name,
            issued_at=now,
            expires_at=expires_at,
            guardian_external_id=guardian_external_id,
            children=children,
            school_code=school_code,
        )
        token = jwt.encode(
            claims.as_dict(),
            self._key.private_pem,
            algorithm=self._key.algorithm,
            headers={"kid": self._key.kid},
        )
        return token, expires_at

    def mint_refresh_token(self) -> tuple[str, str, datetime]:
        """Return `(raw, hash, expires_at)`.

        Opaque random bytes, not a JWT. A refresh token's only job is to be presented back
        to this service and looked up, so signing it would add verification cost and a
        second way to get revocation wrong.
        """
        raw = secrets.token_urlsafe(48)
        return raw, self.hash_refresh_token(raw), self._clock() + self._refresh_ttl

    @staticmethod
    def hash_refresh_token(raw: str) -> str:
        """SHA-256, correct for a 48-byte random value.

        Stretching is for low-entropy secrets — see `crypto/passwords.py`. Applying it
        here would add latency to a request a parent is waiting on without adding
        strength, because there is no dictionary behind a value drawn from `secrets`.
        """
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def decode_own_token(self, token: str) -> dict:
        """Verify a token this service issued. Used by `/v1/auth/me` and by tests."""
        try:
            return jwt.decode(
                token,
                self._key.public_pem,
                algorithms=[self._key.algorithm],
                audience=self._audience,
                issuer=self._issuer,
            )
        except JWTError as exc:
            raise ValueError("invalid token") from exc


__all__ = ["JwtTokenIssuer"]
