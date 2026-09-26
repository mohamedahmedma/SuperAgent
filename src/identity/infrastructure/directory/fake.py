"""A guardian directory held in a dict.

Ships in the service rather than in the test suite, and that is deliberate on two counts.

**It is the default when no SIS is configured.** The whole verification flow then runs end
to end on a laptop with no second service — and a production deployment that forgot to set
`IDENTITY_SIS_BASE_URL` refuses every parent rather than authenticating them against
nothing. A login that cannot succeed is a support call; a login that succeeds against an
empty directory is a stranger holding a token.

**It is the reference implementation of the port.** Anyone writing a second directory —
against a different SIS, or a CSV — reads this to see what the two methods are supposed to
return, including the distinction between "nobody" and "could not ask".
"""
from __future__ import annotations

from identity.domain.errors import GuardianDirectoryUnavailable
from identity.domain.guardians import ChildRef, GuardianRef


class FakeGuardianDirectory:
    """`GuardianDirectory` over two dicts.

    `unavailable` exists so a test can assert what happens when the school's records are
    unreachable, which is the branch nobody writes by hand and everybody needs.
    """

    def __init__(
        self,
        guardians: dict[str, GuardianRef] | None = None,
        *,
        children: dict[str, list[ChildRef]] | None = None,
        unavailable: bool = False,
    ) -> None:
        self.guardians = dict(guardians or {})
        self.unavailable = unavailable
        self.asked: list[str] = []
        #: Which school each lookup was scoped to, so a test can assert that the school
        #: reached the directory and not merely that a lookup happened.
        self.asked_schools: list[str | None] = []
        #: `{public_id: [ChildRef, ...]}`. Empty by default, so a test that only cares
        #: about sign-in gets a token with no children rather than having to say so.
        self.children: dict[str, list[ChildRef]] = dict(children or {})

    def resolve(
        self, phone_e164: str, *, school_code: str | None = None
    ) -> GuardianRef | None:
        self.asked.append(phone_e164)
        self.asked_schools.append(school_code)
        if self.unavailable:
            raise GuardianDirectoryUnavailable("The fake directory is switched off.")
        return self.guardians.get(phone_e164)

    def children_of(
        self, public_id: str, *, school_code: str | None = None
    ) -> list[ChildRef]:
        if self.unavailable:
            raise GuardianDirectoryUnavailable("The fake directory is switched off.")
        return list(self.children.get(public_id, ()))


__all__ = ["FakeGuardianDirectory"]
