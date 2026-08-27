"""Reading claims out of a verified token — and the rule about what they are worth.

Every function here assumes the token has already been through `verify_token`. They read
a signed document; they do not check a signature and they must never be called on a token
that has not been verified.

## The rule these helpers exist to keep visible

**A claim is an identity, never a permission.**

`guardian_id` says which parent signed in. It does not say which children she may be told
about — that is a fact held by the system of record, amended by custody decisions, and it
has to be asked for at the moment a record is actually read. `children` is a convenience
hint for greeting a parent and understanding "my son"; it is *stale by design* within the
token's lifetime, because a court order takes effect at the school immediately while a
token already in a browser keeps asserting the old family until it expires.

That is why `children_from_claims` is named as a hint and returns plain data, and why
nothing here returns anything an authorisation check could be written against. The moment
one of these values gates a read, a revocation stops working for `ACCESS_TTL_MINUTES` and
nothing reports it.
"""
from __future__ import annotations

from collections.abc import Mapping

from schoolauth.verification import IdentityError


def guardian_id_from_claims(claims: Mapping) -> str:
    """The guardian binding, or raise.

    An absent claim is a staff token, or a parent account the registrar has not bound to
    a guardian yet. Both must read nothing, and both arrive here as the same absence — so
    this raises rather than returning `""` and inviting a caller to write `if not id:` and
    carry on.
    """
    guardian_id = claims.get("guardian_id")
    if not guardian_id or not isinstance(guardian_id, str):
        raise IdentityError("Token carries no guardian binding.")
    return guardian_id


def school_from_claims(claims: Mapping) -> str | None:
    """Which school's database answers for this token; `None` in a single-school estate.

    Optional rather than required, so a single-school deployment keeps minting and
    accepting exactly the tokens it always did.

    Unlike the guardian binding, a wrong value here cannot widen access — it selects a
    database, and in the wrong one this parent's guardian link does not exist, so the
    lookup refuses. The failure mode is a refusal, never a disclosure, which is why it can
    be read straight off the token without a second check.
    """
    school = claims.get("school")
    return school if school and isinstance(school, str) else None


def role_from_claims(claims: Mapping) -> str:
    """The account's role, or `""`. Coarse, and never the basis of a records decision."""
    role = claims.get("role")
    return role if isinstance(role, str) else ""


def children_from_claims(claims: Mapping) -> tuple[dict, ...]:
    """The family the token asserted when it was signed. **A hint, not permission.**

    Returned as a tuple of plain dicts so a caller cannot accumulate into it and come to
    treat it as a roster it owns. Every read of an actual record must still be re-checked
    against the system of record; a stale entry here must produce a refusal there, never
    a disclosure.
    """
    children = claims.get("children")
    if not isinstance(children, list):
        return ()
    return tuple(child for child in children if isinstance(child, dict))


__all__ = [
    "children_from_claims",
    "guardian_id_from_claims",
    "role_from_claims",
    "school_from_claims",
]
