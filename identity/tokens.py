"""Minting access and refresh tokens.

Only this service ever calls into this module. Verification elsewhere is done against
the published JWKS with a public key, so nothing outside can produce a token.

The claim set is deliberately minimal. A token says who the subject is, what role
they hold, and — for a parent — which guardian they are. It carries no student list,
no permissions, and no profile. Those are authorisation decisions and they belong to
the service that owns the data, resolved fresh on every request. A permission baked
into a token is a permission that survives being revoked.
"""
import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from identity import keys

ISSUER = os.getenv("IDENTITY_ISSUER", "school-identity")
# Tokens are minted for a named audience and verifiers must check it. Without this, a
# token accepted by the chat backend is replayable against the records facade — same
# signature, same issuer, different blast radius.
AUDIENCE = os.getenv("IDENTITY_AUDIENCE", "school-services")

ACCESS_TTL_MINUTES = int(os.getenv("IDENTITY_ACCESS_TTL_MINUTES") or 30)
REFRESH_TTL_DAYS = int(os.getenv("IDENTITY_REFRESH_TTL_DAYS") or 30)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def mint_access_token(
    *, subject: str, role: str, guardian_external_id: str | None, display_name: str = ""
) -> tuple[str, datetime]:
    """Sign a short-lived access token. Returns `(token, expires_at)`.

    `guardian_id` is omitted entirely when the account has no binding, rather than
    included as null or empty string. An absent claim fails a verifier's required-claim
    check loudly; an empty one invites a downstream `if guardian_id:` that someone
    eventually writes as `if guardian_id is not None:`.
    """
    expires_at = _now() + timedelta(minutes=ACCESS_TTL_MINUTES)
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": subject,
        "role": role,
        "name": display_name,
        "iat": int(_now().timestamp()),
        "exp": int(expires_at.timestamp()),
        # Unique per token, so a specific token can be named in an audit or a
        # revocation list without naming the account.
        "jti": uuid.uuid4().hex,
    }
    if guardian_external_id:
        claims["guardian_id"] = guardian_external_id

    token = jwt.encode(
        claims,
        keys.private_pem(),
        algorithm=keys.algorithm(),
        headers={"kid": keys.kid()},
    )
    return token, expires_at


def mint_refresh_token() -> tuple[str, str, datetime]:
    """Return `(raw, hash, expires_at)`.

    Opaque random bytes, not a JWT. A refresh token's only job is to be presented back
    to this service and looked up, so signing it would add verification cost and a
    second way to get revocation wrong.
    """
    raw = secrets.token_urlsafe(48)
    return raw, hash_refresh_token(raw), _now() + timedelta(days=REFRESH_TTL_DAYS)


def hash_refresh_token(raw: str) -> str:
    # SHA-256 is correct for a 48-byte random value; see the note in records.auth on
    # why stretching is for low-entropy secrets only.
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def decode_own_token(token: str) -> dict:
    """Verify a token this service issued. Used by `/v1/auth/me` and by tests."""
    try:
        return jwt.decode(
            token,
            keys.public_pem(),
            algorithms=[keys.algorithm()],
            audience=AUDIENCE,
            issuer=ISSUER,
        )
    except JWTError as exc:
        raise ValueError("invalid token") from exc
