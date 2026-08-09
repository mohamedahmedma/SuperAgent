"""Verifying identity tokens minted by the identity service.

This closes the last gap in the authorisation model. Before it, the guardian id came
from the URL path and this service trusted the agent backend to have put the right
one there — so a bug or a compromise in the chat process could ask about any family.
Now the guardian id must also appear as a signed claim, and the signature can only
have come from the identity service.

Verification is **offline**. This service holds a public key and checks a signature;
it does not call the identity service per request. That keeps the records facade
serving even when identity is down, and it means identity is not in the latency path
of every parent question.

It **fails closed**. With no public key configured, every parent-facing read returns
503 rather than falling back to trusting the path. A records service that quietly
accepts unsigned identity is worse than one that is down.
"""
import logging
import os
import threading
import time

from jose import JWTError, jwt

logger = logging.getLogger(__name__)

ISSUER = os.getenv("IDENTITY_ISSUER", "school-identity")
AUDIENCE = os.getenv("IDENTITY_AUDIENCE", "school-services")
JWKS_URL = os.getenv("IDENTITY_JWKS_URL", "")
JWKS_TTL_SECONDS = int(os.getenv("IDENTITY_JWKS_TTL_SECONDS") or 600)


class IdentityError(RuntimeError):
    """Token missing, malformed, expired, or not signed by the identity service."""


class IdentityNotConfigured(RuntimeError):
    """No verification material available. Fail closed, never fall back."""


_lock = threading.Lock()
_cached_jwks: dict | None = None
_cached_at: float = 0.0


def _public_key_from_env() -> str | None:
    """A PEM passed directly.

    The deployment shape for a fixed key, and what the tests use so they never open a
    socket. Takes precedence over JWKS: an operator who pinned a key meant it.
    """
    return os.getenv("IDENTITY_PUBLIC_KEY_PEM") or None


def _fetch_jwks() -> dict:
    global _cached_jwks, _cached_at

    with _lock:
        if _cached_jwks is not None and (time.monotonic() - _cached_at) < JWKS_TTL_SECONDS:
            return _cached_jwks

        if not JWKS_URL:
            raise IdentityNotConfigured("Neither IDENTITY_PUBLIC_KEY_PEM nor IDENTITY_JWKS_URL is set.")

        import httpx

        try:
            response = httpx.get(JWKS_URL, timeout=5.0)
            response.raise_for_status()
            _cached_jwks = response.json()
            _cached_at = time.monotonic()
            return _cached_jwks
        except Exception as exc:
            # A stale key is still a valid key — serving from cache through an identity
            # outage is the difference between "grades unavailable" and "nobody can log
            # in to anything". Only a cold cache is fatal.
            if _cached_jwks is not None:
                logger.warning("JWKS refresh failed, using cached keys: %s", exc)
                return _cached_jwks
            raise IdentityNotConfigured(f"Could not fetch JWKS: {exc}") from exc


def verify_token(token: str) -> dict:
    """Return the token's claims, or raise.

    `audience` and `issuer` are checked, not just the signature. Without the audience
    check a token minted for one service is replayable against another — same key,
    same issuer, different data.
    """
    if not token:
        raise IdentityError("Missing token.")

    key: str | dict
    pem = _public_key_from_env()
    key = pem if pem else _fetch_jwks()

    try:
        return jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER,
        )
    except JWTError as exc:
        raise IdentityError(str(exc)) from exc


def guardian_id_from_claims(claims: dict) -> str:
    """Pull the guardian binding out, or raise.

    An absent claim is a staff token, or a parent account the registrar has not bound
    yet. Both must read nothing, and both arrive here as the same absence.
    """
    guardian_id = claims.get("guardian_id")
    if not guardian_id or not isinstance(guardian_id, str):
        raise IdentityError("Token carries no guardian binding.")
    return guardian_id
