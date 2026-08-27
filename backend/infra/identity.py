"""This backend's identity configuration. The verification itself lives in `schoolauth`.

This file used to say it was "deliberately a near-copy of `records/identity.py` rather
than a shared import", on the grounds that a shared auth library is the seam through which
independent deployability quietly disappears. The concern was right; the conclusion was
not. What was duplicated was a JWKS cache, an RS256 decode and two exception types — none
of which is a *policy* either service owns, and both copies had already drifted: this one
fetched with `requests`, the other with `httpx`, and only one of them re-read the pinned
PEM on every call.

So the split is now by what actually differs. `schoolauth` holds the mechanism and reads
no environment at all. This module holds the one thing that is genuinely this process's
own — the issuer and audience it expects — and it is still free to differ from every other
service's, which is what independent deployability actually required.

Verification is **offline**: this process holds a public key and checks a signature. It
never calls the identity service per request, so identity being down does not stop anyone
using a chat session they are already signed in to.

It **fails closed**. With no verification material configured, authentication fails rather
than falling back to anything.
"""
import os

from schoolauth import IdentityConfig, IdentityError, IdentityNotConfigured
from schoolauth import verify_token as _verify_token

# Module constants, captured at import. `tests/test_backend_auth.py` reads them back to
# mint tokens the running configuration will actually accept, and `tests/test_e2e_api.py`
# copies them onto the identity service so the two agree regardless of collection order.
ISSUER = os.getenv("IDENTITY_ISSUER", "school-identity")
AUDIENCE = os.getenv("IDENTITY_AUDIENCE", "school-services")
JWKS_URL = os.getenv("IDENTITY_JWKS_URL", "")
JWKS_TTL_SECONDS = int(os.getenv("IDENTITY_JWKS_TTL_SECONDS") or 600)


def _config() -> IdentityConfig:
    """Built per call, so reassigning `ISSUER` on this module still takes effect."""
    return IdentityConfig(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url=JWKS_URL,
        jwks_ttl_seconds=JWKS_TTL_SECONDS,
    )


def verify_token(token: str) -> dict:
    """Return the token's claims, or raise.

    `audience` and `issuer` are checked, not just the signature — without the audience
    check, a token minted for one service is replayable against another.
    """
    return _verify_token(token, _config())


__all__ = [
    "AUDIENCE",
    "ISSUER",
    "JWKS_TTL_SECONDS",
    "JWKS_URL",
    "IdentityError",
    "IdentityNotConfigured",
    "verify_token",
]
