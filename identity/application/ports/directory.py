"""The school's system of record, as this service needs to see it.

Declared here and not in `infrastructure/` on purpose: the use cases own the shape of the
question they need answered, and `SisGuardianDirectory` is written to fit. Inverted the
other way — a service importing the concrete HTTP client — a unit test of "does an
unregistered number get a polite refusal" needs a running SIS.

`Protocol` rather than an abstract base class, so a fake in a test is a plain class with
the right two methods: it does not import this module, inherits from nothing, and cannot
be broken by a base class gaining a method it does not use. The type checker still catches
an implementation that drifts.
"""
from __future__ import annotations

from typing import Protocol

from identity.domain.guardians import ChildRef, GuardianRef


class GuardianDirectory(Protocol):
    """Resolving a verified phone number to the parent the school has on file."""

    def resolve(
        self, phone_e164: str, *, school_code: str | None = None
    ) -> GuardianRef | None:
        """The guardian reachable on this number, or `None` when it reaches nobody.

        `None` is an ordinary answer, not an error: most numbers in the world are not this
        school's parents, and the flow that calls this has to say so politely rather than
        fail. Raises `GuardianDirectoryUnavailable` when the question could not be put at
        all, which is a different situation and gets a different reply — one means *try
        again later*, the other means *this number is not a parent here*, and a caller
        that confused them would either tell a real parent she is unknown or promise an
        unknown caller that the school is merely busy.

        `school_code` selects the database. A number that reaches a parent at one branch
        legitimately reaches nobody at another, and under physical separation that is the
        only answer this service can give: the row is not in the file it is connected to.
        """

    def children_of(
        self, public_id: str, *, school_code: str | None = None
    ) -> list[ChildRef]:
        """Every child this guardian may be told about, by her opaque handle.

        Empty is an ordinary answer — a parent whose only link carries a custody
        restriction has no children *to be told about* — and is deliberately not
        distinguishable here from having none.

        Raises `GuardianDirectoryUnavailable` when the question could not be put, so a
        caller can tell an outage from an empty family. A token minted during an outage
        simply carries no children; it must never carry an empty list as though that were
        the answer.

        `school_code` selects which school's database answers. Schools are separated
        physically, so a handle only means anything inside the school that issued it:
        asked of another school's database the same handle resolves to nobody, which is
        the isolation working rather than a failure. `None` is a single-school deployment.
        """


__all__ = ["GuardianDirectory"]
