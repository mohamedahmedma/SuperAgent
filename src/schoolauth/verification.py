"""Resolve a verification key, check a signature, and fail closed when neither is possible.

Verification is **offline**. A service holding this library checks a signature against a
public key; it does not call the identity service per request. Two consequences, and both
are the reason the estate is shaped this way:

  * identity being down does not stop a signed-in parent using a service they are already
    authenticated to, and
  * identity is not in the latency path of every request, so the process holding the
    private signing key stays the quietest and least-deployed thing in the estate.

It **fails closed**. With no verification material configured at all, every call raises
`IdentityNotConfigured` and the caller is expected to refuse the request. A verifier that
quietly accepts unsigned tokens when it cannot find a key is worse than one that is down,
because nothing reports it.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from jose import JWTError, jwt

logger = logging.getLogger(__name__)

#: How long a fetched JWKS is reused before another fetch is attempted.
DEFAULT_JWKS_TTL_SECONDS = 600

#: What a token says about who minted it, and who it is for, when a deployment does not
#: say. Named here because THREE services and the minter must agree on them, and until
#: this constant existed each of the four carried its own copy of the literal.
#:
#: Four copies that agree look harmless. The failure is what happens when one deployment
#: sets IDENTITY_ISSUER on the identity service and not on the backend: identity mints
#: `iss: acme-school`, the backend still expects `school-identity`, and every request in
#: the estate comes back 401. Nothing in that 401 mentions the issuer, so it reads exactly
#: like a broken signing key.
#:
#: This is a DEFAULT, not configuration. Each service still reads its own environment and
#: is still free to expect something different — which is what lets one test process hold
#: two services that legitimately disagree. See `IdentityConfig`.
DEFAULT_ISSUER = "school-identity"
DEFAULT_AUDIENCE = "school-services"

#: The JWKS fetch is one small GET of a public document. Short, because a request is
#: usually waiting on it, and a slow answer is worse than a fast refusal it can act on.
_FETCH_TIMEOUT_SECONDS = 5.0

#: Only RS256. Declared rather than read from the token, because a verifier that trusts
#: the token's own `alg` header accepts `none` and accepts HS256 signed with the public
#: key it was about to verify against. Both are the classic JWT forgeries.
_ALGORITHMS = ("RS256",)


class IdentityError(RuntimeError):
    """Token missing, malformed, expired, or not signed by the identity service."""


class IdentityNotConfigured(RuntimeError):
    """No verification material available. Fail closed, never fall back."""


@dataclass(frozen=True, slots=True)
class IdentityConfig:
    """What one service expects of a token it is handed.

    Passed in rather than read from the environment here, so that this package cannot
    change what a deployment verifies against and so two services sharing a test process
    can legitimately expect different issuers.

    `audience` is not optional and is checked on every call. Without it a token minted for
    one service is replayable against another — same key, same issuer, different data.
    """

    issuer: str
    audience: str
    #: Where to fetch signing keys when no PEM is pinned. Empty means "PEM or nothing".
    jwks_url: str = ""
    jwks_ttl_seconds: int = DEFAULT_JWKS_TTL_SECONDS
    #: A PEM pinned by the operator, which wins over `jwks_url`. Read per call by the
    #: service shims rather than captured, because a test that sets the variable after
    #: import must still take effect.
    public_key_pem: str = ""


_lock = threading.Lock()
#: `{jwks_url: (document, fetched_at_monotonic)}`. Keyed by URL so a process verifying
#: against two identity services cannot serve one's keys for the other's tokens.
_cached_jwks: dict[str, tuple[dict, float]] = {}
#: `{jwks_url: Event}` while a refresh is in flight, so exactly one thread per URL goes to
#: the network and the rest wait for its result instead of duplicating it.
_refreshing: dict[str, threading.Event] = {}


def reset_key_cache() -> None:
    """Drop every cached JWKS. For tests, and for an operator forcing a key rotation."""
    with _lock:
        _cached_jwks.clear()
        _refreshing.clear()


def _fetch_jwks(url: str, ttl_seconds: int) -> dict:
    """The signing keys, from cache when fresh and from the network otherwise.

    A stale key is still a valid key: serving from cache through an identity outage is the
    difference between "records unavailable" and "nobody can use anything". Only a cold
    cache is fatal.

    ## The fast path takes no lock

    This runs on every verified request in every service that uses this library, and it
    used to acquire one process-wide mutex to do so — including on the hit, where all it
    does is read a dict. Worse, it held that mutex **across the network fetch**: when the
    TTL lapsed, one thread went to the network with a 5-second timeout while every other
    request in the process blocked behind it on a lock, having nothing to do with keys.
    A ten-minute TTL turned into a periodic stall of the whole service.

    Now the hit path reads the dict unlocked — safe under CPython, and safe in principle
    because the value is only ever replaced wholesale with a fully-built tuple, never
    mutated in place — and the lock is taken only to *decide who fetches*. The fetch
    itself happens outside it.

    Two threads arriving at an expired entry still do not both fetch: the loser waits on
    `_refreshing[url]` and then re-reads what the winner stored. That is what stops a
    burst of traffic at TTL expiry becoming a burst of identical requests at the identity
    service — the thundering herd this cache exists to prevent in the first place.
    """
    now = time.monotonic()
    cached = _cached_jwks.get(url)
    if cached is not None and (now - cached[1]) < ttl_seconds:
        return cached[0]

    # Stale or absent. Exactly one thread per URL does the work; the rest wait for it and
    # then re-read. `_refreshing` is only ever touched under `_lock`.
    with _lock:
        cached = _cached_jwks.get(url)
        if cached is not None and (time.monotonic() - cached[1]) < ttl_seconds:
            return cached[0]  # somebody refreshed while we waited
        in_flight = _refreshing.get(url)
        if in_flight is None:
            in_flight = threading.Event()
            _refreshing[url] = in_flight
            leader = True
        else:
            leader = False

    if not leader:
        # Wait for the leader, bounded by the same budget its fetch has, so a wedged
        # leader cannot pin every other thread indefinitely.
        in_flight.wait(timeout=_FETCH_TIMEOUT_SECONDS + 1.0)
        refreshed = _cached_jwks.get(url)
        if refreshed is not None:
            return refreshed[0]
        if cached is not None:
            return cached[0]
        raise IdentityNotConfigured(
            "Could not fetch JWKS: the refresh in progress did not produce keys."
        )

    try:
        return _fetch_and_store(url, cached)
    finally:
        with _lock:
            _refreshing.pop(url, None)
        in_flight.set()


def _fetch_and_store(url: str, cached: tuple[dict, float] | None) -> dict:
    """The network half: run by one thread at a time, and outside the lock.

    Every failure falls back to the cached document when there is one. A stale key is
    still a valid key, and an identity service that is briefly unreachable must not become
    an outage in every service that verifies its tokens.
    """
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
            document = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        if cached is not None:
            logger.warning("JWKS refresh failed, using cached keys: %s", exc)
            return cached[0]
        raise IdentityNotConfigured(f"Could not fetch JWKS: {exc}") from exc

    if not isinstance(document, dict):
        if cached is not None:
            logger.warning("JWKS was not a JSON object; using cached keys")
            return cached[0]
        raise IdentityNotConfigured("JWKS was not a JSON object.")

    _cached_jwks[url] = (document, time.monotonic())
    return document


def _verification_key(config: IdentityConfig) -> str | dict:
    """A pinned PEM if the operator set one, otherwise the published key set.

    The PEM wins deliberately: an operator who pinned a key meant it, and a deployment
    that can reach a JWKS URL it was not supposed to trust should not silently prefer it.
    """
    pem = (config.public_key_pem or os.getenv("IDENTITY_PUBLIC_KEY_PEM") or "").strip()
    if pem:
        return pem
    if not config.jwks_url:
        raise IdentityNotConfigured(
            "Neither IDENTITY_PUBLIC_KEY_PEM nor IDENTITY_JWKS_URL is set."
        )
    return _fetch_jwks(config.jwks_url, config.jwks_ttl_seconds)


def verify_token(token: str, config: IdentityConfig) -> dict:
    """Return the token's claims, or raise.

    `IdentityNotConfigured` when there is no key to check against — the caller must turn
    that into a refusal, never into an unverified read. `IdentityError` for every way a
    token can be wrong, collapsed into one type on purpose: a caller that could tell
    "expired" from "wrong signature" from "wrong audience" would report the difference to
    whoever presented it, which is a probing oracle.
    """
    if not token:
        raise IdentityError("Missing token.")

    key = _verification_key(config)

    try:
        return jwt.decode(
            token,
            key,
            algorithms=list(_ALGORITHMS),
            audience=config.audience,
            issuer=config.issuer,
        )
    except JWTError as exc:
        raise IdentityError(str(exc)) from exc


__all__ = [
    "DEFAULT_JWKS_TTL_SECONDS",
    "IdentityConfig",
    "IdentityError",
    "IdentityNotConfigured",
    "reset_key_cache",
    "verify_token",
]
