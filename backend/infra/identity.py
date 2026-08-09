"""Verifying identity tokens minted by the identity service.

Deliberately a near-copy of `records/identity.py` rather than a shared import. The two
services must be deployable and replaceable independently, and a shared auth library
is the seam through which that independence quietly disappears — one repo, one version
bump, three services that have to ship together. Forty lines duplicated across a
service boundary is the cheaper mistake.

Verification is **offline**: this process holds a public key and checks a signature. It
never calls the identity service per request, so identity being down does not stop
anyone using a chat session they are already signed in to, and identity is not in the
latency path of every message.

It **fails closed**. With no verification material configured, authentication fails
rather than falling back to anything.
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


def _fetch_jwks() -> dict:
    global _cached_jwks, _cached_at

    with _lock:
        if _cached_jwks is not None and (time.monotonic() - _cached_at) < JWKS_TTL_SECONDS:
            return _cached_jwks

        if not JWKS_URL:
            raise IdentityNotConfigured(
                "Neither IDENTITY_PUBLIC_KEY_PEM nor IDENTITY_JWKS_URL is set."
            )

        import requests

        try:
            response = requests.get(JWKS_URL, timeout=5.0)
            response.raise_for_status()
            _cached_jwks = response.json()
            _cached_at = time.monotonic()
            return _cached_jwks
        except Exception as exc:
            # A stale key is still a valid key. Serving from cache through an identity
            # outage is the difference between "grades unavailable" and "nobody can use
            # the assistant at all". Only a cold cache is fatal.
            if _cached_jwks is not None:
                logger.warning("JWKS refresh failed, using cached keys: %s", exc)
                return _cached_jwks
            raise IdentityNotConfigured(f"Could not fetch JWKS: {exc}") from exc


def verify_token(token: str) -> dict:
    """Return the token's claims, or raise.

    `audience` and `issuer` are checked, not just the signature — without the audience
    check, a token minted for one service is replayable against another.
    """
    if not token:
        raise IdentityError("Missing token.")

    pem = os.getenv("IDENTITY_PUBLIC_KEY_PEM")
    key = pem if pem else _fetch_jwks()

    try:
        return jwt.decode(token, key, algorithms=["RS256"], audience=AUDIENCE, issuer=ISSUER)
    except JWTError as exc:
        raise IdentityError(str(exc)) from exc
