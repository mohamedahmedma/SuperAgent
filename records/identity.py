"""This service's identity configuration. The verification itself lives in `schoolauth`.

What used to be here — JWKS fetching, a TTL cache, the RS256 decode, two exception types
— was a near-copy of the same code in `backend/infra/identity.py`. Both are now one
implementation in `schoolauth/verification.py`, and what remains in this file is the only
part that was ever this service's own: which issuer and audience *it* expects.

**The configuration deliberately did not move.** A shared library that also read the
environment would force every service in the estate to verify against the same issuer,
which is wrong in two ways: a deployment may legitimately run more than one identity
service, and this suite and the backend's set different values in one pytest process. So
`schoolauth` is handed a config and never reads one.

Verification is **offline** — this service holds a public key and checks a signature; it
does not call the identity service per request. That keeps the records path serving while
identity is down, and keeps identity out of the latency path of every parent question.

It **fails closed**. With no verification material configured, every parent-facing read
returns 503 rather than falling back to trusting the path. A records service that quietly
accepts unsigned identity is worse than one that is down.
"""
import os

from schoolauth import (
    IdentityConfig,
    IdentityError,
    IdentityNotConfigured,
    guardian_id_from_claims,
    school_from_claims,
)
from schoolauth import verify_token as _verify_token

# Captured at import, exactly as before. `records/tests/conftest.py` sets both variables
# before this module is first imported, and a later reader would see the deployment's
# values rather than the suite's.
ISSUER = os.getenv("IDENTITY_ISSUER", "school-identity")
AUDIENCE = os.getenv("IDENTITY_AUDIENCE", "school-services")
JWKS_URL = os.getenv("IDENTITY_JWKS_URL", "")
JWKS_TTL_SECONDS = int(os.getenv("IDENTITY_JWKS_TTL_SECONDS") or 600)


def _config() -> IdentityConfig:
    """Built per call rather than captured, so a test that reassigns `ISSUER` on this
    module still changes what is verified — which is how `tests/test_e2e_api.py` makes
    the backend and identity agree without depending on collection order."""
    return IdentityConfig(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url=JWKS_URL,
        jwks_ttl_seconds=JWKS_TTL_SECONDS,
    )


def verify_token(token: str) -> dict:
    """Return the token's claims, or raise.

    `audience` and `issuer` are checked, not just the signature. Without the audience
    check a token minted for one service is replayable against another — same key, same
    issuer, different data.
    """
    return _verify_token(token, _config())


__all__ = [
    "AUDIENCE",
    "ISSUER",
    "JWKS_TTL_SECONDS",
    "JWKS_URL",
    "IdentityError",
    "IdentityNotConfigured",
    "guardian_id_from_claims",
    "school_from_claims",
    "verify_token",
]
