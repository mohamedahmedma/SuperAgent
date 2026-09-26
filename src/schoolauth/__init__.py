"""Verifying identity tokens — one implementation, used by every service that enforces.

## Why this package exists

Authentication was implemented three times in this estate: once in `records/identity.py`,
once in `backend/infra/identity.py` as an acknowledged near-copy, and once more as the
minting side in `identity/tokens.py`. Two of those three did the same job — resolve a
public key, check a signature, map failures onto two exception types — and they drifted:
one fetched JWKS with `httpx`, the other with `requests`, and only one of them read the
pinned PEM on every call.

That is the duplication worth removing, and removing it does **not** mean moving the check
into a service. It means the opposite:

    the DECISION and the MACHINERY are centralised   -> this package, and identity/
    the ENFORCEMENT stays at every service           -> because it must

A service that trusts "something upstream already checked" is a service that can be
reached around. Verification is the one part of authentication that is *inherently*
distributed — that is what public-key signing is for — so the right shape is one library
compiled into every enforcement point, never one service asked per request.

## What is deliberately NOT here

**No policy.** This package answers "is this token genuine, and what does it say". It
never answers "may this person see that record". The second question depends on data that
lives in the system of record and changes by court order, and an answer cached in a
library — or baked into a token — is an answer that outlives its revocation.

**No configuration.** Issuer, audience and JWKS URL are passed in by the caller through
`IdentityConfig`. Each service keeps capturing those exactly as it always did, so this
package cannot change what any deployment already verifies against, and two services in
one test process can legitimately disagree about the issuer they expect.

**No HTTP client dependency.** The JWKS fetch uses `urllib` from the standard library.
The two copies this replaces disagreed about `httpx` versus `requests`, which meant their
timeout behaviour and their error types differed for no reason a deployment ever chose.
"""
from schoolauth.claims import (
    children_from_claims,
    guardian_id_from_claims,
    role_from_claims,
    school_from_claims,
)
from schoolauth.verification import (
    DEFAULT_AUDIENCE,
    DEFAULT_ISSUER,
    DEFAULT_JWKS_TTL_SECONDS,
    IdentityConfig,
    IdentityError,
    IdentityNotConfigured,
    reset_key_cache,
    verify_token,
)

__all__ = [
    "DEFAULT_AUDIENCE",
    "DEFAULT_ISSUER",
    "DEFAULT_JWKS_TTL_SECONDS",
    "IdentityConfig",
    "IdentityError",
    "IdentityNotConfigured",
    "children_from_claims",
    "guardian_id_from_claims",
    "reset_key_cache",
    "role_from_claims",
    "school_from_claims",
    "verify_token",
]
