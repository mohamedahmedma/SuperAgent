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
from records.config import settings

from schoolauth import (
    IdentityConfig,
    IdentityError,
    IdentityNotConfigured,
    guardian_id_from_claims,
    school_from_claims,
)
from schoolauth import verify_token as _verify_token

#: Module-level overrides, kept for one caller: `tests/general/test_e2e_api.py` forces
#: this service and the chat backend onto the same issuer and audience without depending
#: on which suite pytest collected first. `None` means "read the resolved settings",
#: which is what every deployment does.
ISSUER: str | None = None
AUDIENCE: str | None = None
JWKS_URL: str | None = None
JWKS_TTL_SECONDS: int | None = None


def _config() -> IdentityConfig:
    """Built per call rather than captured.

    It used to be four `os.getenv` calls at import, which meant the values a suite set in
    a fixture arrived too late and which value was verified against depended on collection
    order. They come from `records.config` now — resolved lazily and cached — so the cost
    is a dict lookup and the timing problem is gone.

    The four module-level names above still win when set, because one cross-service test
    legitimately needs to force agreement between two services in one process.
    """
    resolved = settings()
    return IdentityConfig(
        issuer=ISSUER if ISSUER is not None else resolved.identity_issuer,
        audience=AUDIENCE if AUDIENCE is not None else resolved.identity_audience,
        jwks_url=JWKS_URL if JWKS_URL is not None else resolved.identity_jwks_url,
        jwks_ttl_seconds=(
            JWKS_TTL_SECONDS
            if JWKS_TTL_SECONDS is not None
            else resolved.identity_jwks_ttl_seconds
        ),
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
