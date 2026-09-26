"""Which school a parent is talking to, and the WhatsApp number that says so.

Schools are separated physically — one SIS database each, no query spanning two — so
every lookup this service makes has to name a school before it can be answered. For a
parent arriving over WhatsApp there is no login form to ask, and no phone-to-school
directory to consult: the answer is already in the message.

**The number the parent messaged is the school.** Meta stamps every webhook delivery with
`value.metadata.phone_number_id`, naming which of our numbers received it. Give each
school its own WhatsApp number and that field selects the school before any database is
opened — no fan-out across schools, no extra question to the parent, and no shared
directory to become the one piece of infrastructure that knows every family in the estate.

It is trustworthy for exactly the reason the sender's `wa_id` is, and no more: it arrives
inside a body whose signature is checked against the app secret, and never comes from
anything a browser said. A `phone_number_id` this service does not recognise is refused
rather than resolved to a default school — resolving it would answer one branch's parent
out of another branch's database, which is the single failure physical separation exists
to prevent.

## What is not here any more

Reading the environment. The registry used to build itself from `IDENTITY_SCHOOLS_*` in
this module, which made "does a two-school estate reject a duplicated phone id" a test
that had to set six variables and clear an `lru_cache`. Building it is now
`infrastructure/whatsapp/registry.py`'s job; what is left here is the data and the three
lookups, which a test constructs directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from identity.domain.errors import UnknownSchool

#: A school code is short and typed by hand into env names, class codes and URLs. Matches
#: `sis.domain.value_objects.SchoolCode.MAX_LENGTH`, so a code legal in one service is
#: legal in the other — a code accepted here and refused by SIS would be a school that can
#: sign a parent in and then fail every lookup made on their behalf.
MAX_CODE_LENGTH: Final[int] = 16

#: Names the schools this service serves. Unset means single-school mode.
SCHOOLS_VAR: Final[str] = "IDENTITY_SCHOOLS"


def normalise_code(code: str) -> str:
    """Upper-cased and trimmed, the one spelling of a school code this service stores.

    Deliberately the same normalisation `sis.domain.value_objects._Code` applies, so
    `ncs`, ` NCS ` and `NCS` name one school here and there alike. Validation is kept
    light — SIS owns the grammar of a code, and duplicating its pattern here would be a
    second rule to drift from the first.
    """
    return code.strip().upper()


@dataclass(frozen=True, slots=True)
class SchoolWhatsApp:
    """One school, and the WhatsApp identity parents reach it on.

    `number` is what a click-to-chat link points at, in E.164. `phone_number_id` is what
    inbound deliveries are stamped with, and what maps a message back to this school.
    `access_token` sends *from* this number — it is not enough to know which number a
    parent used; the code has to go back out through that same number, or it arrives in a
    conversation with a different school and the parent never sees it.
    """

    code: str
    number: str
    phone_number_id: str
    access_token: str

    @property
    def can_send(self) -> bool:
        """Whether this school has enough credentials to reach a real phone."""
        return bool(self.phone_number_id and self.access_token)


@dataclass(frozen=True, slots=True)
class SchoolRegistry:
    """Every school this service serves.

    An empty registry is single-school mode, which is what keeps a laptop, the test suite
    and any unsplit deployment working untouched.
    """

    schools: tuple[SchoolWhatsApp, ...] = ()

    @property
    def is_multi_school(self) -> bool:
        return bool(self.schools)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(school.code for school in self.schools)

    def by_code(self, code: str) -> SchoolWhatsApp:
        wanted = normalise_code(code)
        for school in self.schools:
            if school.code == wanted:
                return school
        raise UnknownSchool(wanted)

    def by_phone_number_id(self, phone_number_id: str) -> SchoolWhatsApp:
        """The school a delivery belongs to, from the number it was addressed to.

        The whole point of the module. Raises rather than returning `None`, so a caller
        cannot accidentally treat "I don't know which school" as "the default school".
        """
        wanted = (phone_number_id or "").strip()
        if not wanted:
            raise UnknownSchool("")
        for school in self.schools:
            if school.phone_number_id == wanted:
                return school
        raise UnknownSchool(wanted)

    def has(self, code: str) -> bool:
        try:
            self.by_code(code)
        except UnknownSchool:
            return False
        return True


__all__ = [
    "MAX_CODE_LENGTH",
    "SCHOOLS_VAR",
    "SchoolRegistry",
    "SchoolWhatsApp",
    "normalise_code",
]
