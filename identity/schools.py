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
inside a body whose signature is checked against the app secret (`whatsapp.signature_is_valid`)
and never comes from anything a browser said. A `phone_number_id` this service does not
recognise is refused rather than resolved to a default school — resolving it would answer
one branch's parent out of another branch's database, which is the single failure physical
separation exists to prevent.

Environment shape, with `IDENTITY_SCHOOLS=MAIN,NCS`::

    IDENTITY_SCHOOLS=MAIN,NCS

    IDENTITY_WHATSAPP_NUMBER_MAIN=+201000000000
    IDENTITY_WHATSAPP_PHONE_NUMBER_ID_MAIN=111111111111111
    IDENTITY_WHATSAPP_TOKEN_MAIN=EAAG...

    IDENTITY_WHATSAPP_NUMBER_NCS=+201111111111
    IDENTITY_WHATSAPP_PHONE_NUMBER_ID_NCS=222222222222222
    IDENTITY_WHATSAPP_TOKEN_NCS=EAAG...

The suffix is the school code with `.` and `-` folded to `_`, matching `sis.tenancy`, and
two codes folding together are refused at startup rather than quietly sharing a number.

**Single-school mode is the default.** With `IDENTITY_SCHOOLS` unset this service behaves
exactly as it always did: one number from `IDENTITY_WHATSAPP_NUMBER`, one gateway, no
school on anything. That is what keeps a laptop, the test suite and any unsplit deployment
working untouched.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from identity.env import env_value

#: Names the schools this service serves. Unset means single-school mode.
SCHOOLS_VAR: Final[str] = "IDENTITY_SCHOOLS"

_NUMBER_PREFIX: Final[str] = "IDENTITY_WHATSAPP_NUMBER"
_PHONE_ID_PREFIX: Final[str] = "IDENTITY_WHATSAPP_PHONE_NUMBER_ID"
_TOKEN_PREFIX: Final[str] = "IDENTITY_WHATSAPP_TOKEN"

#: A school code is short and typed by hand into env names, class codes and URLs. Matches
#: `sis.domain.value_objects.SchoolCode.MAX_LENGTH` so a code legal in one service is legal
#: in the other — a code accepted here and refused by SIS would be a school that can sign a
#: parent in and then fail every lookup made on their behalf.
MAX_CODE_LENGTH: Final[int] = 16


class SchoolsMisconfigured(RuntimeError):
    """The registry is wrong and no parent can be signed in correctly.

    Raised at startup, not per request. A school named with no WhatsApp number behind it
    would otherwise fail on the first parent who tapped the link, at whatever hour that
    happened to be, with an error naming a gateway rather than a missing setting.
    """


class UnknownSchool(LookupError):
    """A `phone_number_id`, or a school code, this service does not serve.

    Carries the value so a log line can name it: an unrecognised `phone_number_id` almost
    always means a school was onboarded at Meta and not added to `.env`, and the id is the
    one piece of information that identifies which.
    """

    def __init__(self, value: str) -> None:
        super().__init__(f"no school configured for {value!r}")
        self.value = value


def _suffix(code: str) -> str:
    """The environment-variable suffix for a school code; see `sis.tenancy._suffix`."""
    return code.replace(".", "_").replace("-", "_")


def _normalise(code: str) -> str:
    """Upper-case and trimmed, the one spelling of a school code this service stores.

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
    """Every school this service serves, resolved once from the environment."""

    schools: tuple[SchoolWhatsApp, ...]

    @property
    def is_multi_school(self) -> bool:
        return bool(self.schools)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(school.code for school in self.schools)

    def by_code(self, code: str) -> SchoolWhatsApp:
        wanted = _normalise(code)
        for school in self.schools:
            if school.code == wanted:
                return school
        raise UnknownSchool(wanted)

    def by_phone_number_id(self, phone_number_id: str) -> SchoolWhatsApp:
        """The school a delivery belongs to, from the number it was addressed to.

        The whole point of the module. Raises rather than returning `None` so a caller
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


def _configured_codes() -> tuple[str, ...]:
    """School codes from `IDENTITY_SCHOOLS`, normalised, in order, without duplicates."""
    raw = env_value(SCHOOLS_VAR)
    if not raw:
        return ()
    seen: list[str] = []
    for part in raw.split(","):
        code = _normalise(part)
        if not code:
            continue
        if len(code) > MAX_CODE_LENGTH:
            raise SchoolsMisconfigured(
                f"{SCHOOLS_VAR} lists {code!r}, which is longer than the {MAX_CODE_LENGTH} "
                "characters a school code may have."
            )
        if code not in seen:
            seen.append(code)
    return tuple(seen)


@lru_cache(maxsize=1)
def get_registry() -> SchoolRegistry:
    """The process's school registry. Cached; call `reset_registry_cache()` in tests.

    Read lazily rather than at import, because `load_env()` and every test fixture set
    variables after this module is first imported.

    Validation is deliberately asymmetric. A missing **number** is fatal: without it the
    click-to-chat link for that school opens WhatsApp's contact picker instead of a chat,
    and the parent is asked to choose who to send the school's verification code to — the
    failure `identity/env.py` exists to describe. Missing **credentials** are not fatal:
    that is the state every test and every laptop runs in, and the recording gateway keeps
    the flow working end to end there.
    """
    codes = _configured_codes()
    if not codes:
        return SchoolRegistry(schools=())

    # Imported here rather than at module scope: `whatsapp` imports nothing from this
    # module, and keeping the edge one-directional is what stops the two from forming a
    # cycle when `whatsapp` later needs the registry.
    from identity.whatsapp import e164_or_raise

    schools: list[SchoolWhatsApp] = []
    suffixes: dict[str, str] = {}
    missing_numbers: list[str] = []
    phone_ids: dict[str, str] = {}

    for code in codes:
        suffix = _suffix(code)
        clash = suffixes.get(suffix)
        if clash is not None:
            raise SchoolsMisconfigured(
                f"school codes {clash!r} and {code!r} both map to the environment suffix "
                f"{suffix!r}, so they would read the same {_NUMBER_PREFIX}_{suffix} and "
                "share one WhatsApp number. Rename one."
            )
        suffixes[suffix] = code

        number = env_value(f"{_NUMBER_PREFIX}_{suffix}")
        if not number:
            missing_numbers.append(f"{_NUMBER_PREFIX}_{suffix} (for school {code})")
            continue
        number = e164_or_raise(number, setting=f"{_NUMBER_PREFIX}_{suffix}")

        phone_number_id = env_value(f"{_PHONE_ID_PREFIX}_{suffix}")
        if phone_number_id:
            owner = phone_ids.get(phone_number_id)
            if owner is not None:
                # Two schools on one Meta number cannot be told apart on the way in, so
                # every parent of one of them would be resolved against the other's
                # database. Refuse rather than pick.
                raise SchoolsMisconfigured(
                    f"schools {owner!r} and {code!r} share the WhatsApp "
                    f"phone_number_id {phone_number_id!r}. Inbound messages cannot be "
                    "attributed to a school, so each school needs its own number."
                )
            phone_ids[phone_number_id] = code

        schools.append(
            SchoolWhatsApp(
                code=code,
                number=number,
                phone_number_id=phone_number_id,
                access_token=env_value(f"{_TOKEN_PREFIX}_{suffix}"),
            )
        )

    if missing_numbers:
        raise SchoolsMisconfigured(
            f"{SCHOOLS_VAR} names schools with no WhatsApp number behind them: "
            + ", ".join(missing_numbers)
            + ". Every school needs its own number; there is no shared default, because "
            "a link without the right number opens WhatsApp's contact picker and the "
            "parent is asked to choose who to send the school's verification code to."
        )
    return SchoolRegistry(schools=tuple(schools))


def reset_registry_cache() -> None:
    """Drop the cached registry so a test can reconfigure the estate."""
    get_registry.cache_clear()


__all__ = [
    "MAX_CODE_LENGTH",
    "SCHOOLS_VAR",
    "SchoolRegistry",
    "SchoolWhatsApp",
    "SchoolsMisconfigured",
    "UnknownSchool",
    "get_registry",
    "reset_registry_cache",
]
