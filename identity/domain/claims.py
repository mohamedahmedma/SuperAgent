"""What a token says, and the rules about what it may not say.

Pure claim assembly: a dict in, a dict out, no signing key and no clock of its own. The
signing itself is `infrastructure/crypto/jwt.py`; what belongs here is the part that is a
policy decision rather than a cryptographic one.

The claim set is deliberately minimal. A token says who the subject is, what role they
hold, and — for a parent — which guardian they are. It carries no student list beyond the
convenience claim described below, no permissions and no profile. Those are authorisation
decisions and they belong to the service that owns the data, resolved fresh on every
request. **A permission baked into a token is a permission that survives being revoked.**
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AccessClaims:
    """Everything that goes into an access token, before it is signed."""

    issuer: str
    audience: str
    subject: str
    role: str
    display_name: str
    issued_at: datetime
    expires_at: datetime
    guardian_external_id: str | None = None
    children: Sequence[Mapping[str, str]] = ()
    school_code: str | None = None

    def as_dict(self) -> dict:
        """The JWT payload.

        ## `guardian_id`

        **Omitted entirely** when the account has no binding, rather than included as null
        or an empty string. An absent claim fails a verifier's required-claim check
        loudly; an empty one invites a downstream `if guardian_id:` that somebody
        eventually rewrites as `if guardian_id is not None:`. A staff token and an unbound
        parent then arrive at the records facade as the same absence, and both read
        nothing.

        ## `children`, and the rule that makes it safe

        **It authorises nothing.** It says which children this parent was known to have
        when the token was signed; it is not permission to read any of them. Every records
        lookup is still re-checked against the school's own guardian link by the service
        that answers it, so a claim that is wrong — or stale — produces a refusal, not a
        disclosure.

        That rule is load-bearing rather than decorative. This claim is the one piece of
        the token that can go out of date within its lifetime: a custody order revoking
        access takes effect at the school immediately, and a token already in a browser
        keeps asserting the old family until it expires. The access TTL is what bounds
        that, and it is why this claim belongs in a short-lived access token and never in
        a refresh token.

        It is also PII about minors in a bearer credential that rides every request into
        every access log, which is why `ChildRef.as_claim` carries a name, a year and a
        sex and nothing else — no marks, no attendance, no birth date, no contact details.

        ## `school`

        Which school's database answers for this token. Schools are separated physically,
        so every request made on this parent's behalf has to name a school before it can
        be answered, and this claim is where the answer comes from after sign-in. It is
        settled once, at sign-in, from the WhatsApp number the parent messaged, so no
        later request has to re-derive it or be trusted to state it.

        Unlike `children` this claim *does* gate access, in the only sense available here:
        it selects the database. That makes it safe in a way `children` is not — a stale
        or wrong school code reaches a database where this parent's guardian link does not
        exist, so the lookup refuses. It cannot widen access to another school; it can
        only fail.
        """
        claims: dict = {
            "iss": self.issuer,
            "aud": self.audience,
            "sub": self.subject,
            "role": self.role,
            "name": self.display_name,
            "iat": int(self.issued_at.timestamp()),
            "exp": int(self.expires_at.timestamp()),
            # Unique per token, so a specific token can be named in an audit or a
            # revocation list without naming the account.
            "jti": uuid.uuid4().hex,
        }
        if self.guardian_external_id:
            claims["guardian_id"] = self.guardian_external_id
        if self.children:
            claims["children"] = [dict(child) for child in self.children]
        if self.school_code:
            claims["school"] = self.school_code
        return claims


__all__ = ["AccessClaims"]
