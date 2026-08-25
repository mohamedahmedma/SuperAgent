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
from collections.abc import Mapping, Sequence
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
    *,
    subject: str,
    role: str,
    guardian_external_id: str | None,
    display_name: str = "",
    children: Sequence[Mapping[str, str]] = (),
    school_code: str | None = None,
) -> tuple[str, datetime]:
    """Sign a short-lived access token. Returns `(token, expires_at)`.

    `guardian_id` is omitted entirely when the account has no binding, rather than
    included as null or empty string. An absent claim fails a verifier's required-claim
    check loudly; an empty one invites a downstream `if guardian_id:` that someone
    eventually writes as `if guardian_id is not None:`.

    ## `children`, and the rule that makes it safe

    **It authorises nothing.** It says which children this parent was known to have when
    the token was signed; it is not permission to read any of them. Every records lookup
    is still re-checked against the school's own guardian link by the service that answers
    it, so a claim that is wrong — or stale — produces a refusal, not a disclosure.

    That rule is load-bearing rather than decorative. This claim is the one piece of the
    token that can go out of date within its lifetime: a custody order revoking access
    takes effect at the school immediately, and a token already in a browser keeps
    asserting the old family until it expires. `ACCESS_TTL_MINUTES` is what bounds that,
    and it is why this claim belongs in a short-lived access token and never in a refresh
    token.

    It is also PII about minors in a bearer credential that rides every request into every
    access log, which is why `ChildRef.as_claim` carries a name, a year and a sex and
    nothing else — no marks, no attendance, no birth date, no contact details.

    Omitted entirely when empty, for the same reason `guardian_id` is: a staff token, or a
    parent whose children could not be looked up during an outage, should carry no claim
    rather than an empty list that reads as "this parent has no children".

    ## `school`

    Which school's database answers for this token. Schools are separated physically — one
    database each, no query spanning two — so every request made on this parent's behalf
    has to name a school before it can be answered, and this claim is where the answer
    comes from after sign-in. It is settled once, at sign-in, from the WhatsApp number the
    parent messaged, so no later request has to re-derive it or be trusted to state it.

    Unlike `children` this claim *does* gate access, in the only sense available here: it
    selects the database. That makes it safe in a way `children` is not — a stale or wrong
    school code reaches a database where this parent's guardian link does not exist, so the
    lookup refuses. It cannot widen access to another school; it can only fail.

    Omitted when absent, like the two claims above, so a single-school deployment mints
    exactly the tokens it always did.
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
    if children:
        claims["children"] = [dict(child) for child in children]
    if school_code:
        claims["school"] = school_code

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
