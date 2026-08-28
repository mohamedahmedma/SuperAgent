"""Which database a school's data lives in, read from the environment in one place.

Schools are separated **physically**: one database per school, same schema in each, no
query that spans two. Isolation is therefore not a `WHERE` clause a caller can forget —
it is the connection. A read that reaches the wrong school returns nothing, because the
rows are not in that file at all.

What this module owns is the registry: the list of school codes the process serves and
the database URL behind each. Nothing else may read those variables, for the reason
`sis.config` exists — a service that resolves its own tenant from `os.getenv` cannot be
tested without arranging the environment, and it stops being obvious which layer decided
which school a request belongs to.

**Single-school mode is the default.** With `SIS_SCHOOLS` unset the process behaves
exactly as it always did: one database at `SIS_DATABASE_URL`, no header required, no
registry. That is what keeps a development laptop, the test suite and any deployment
that has not been split working unchanged. Multi-school mode is opt-in and begins the
moment `SIS_SCHOOLS` names anything.

Environment shape, with `SIS_SCHOOLS=MAIN,NCS`::

    SIS_SCHOOLS=MAIN,NCS
    SIS_DATABASE_URL_MAIN=sqlite:///./schools/MAIN.db
    SIS_DATABASE_URL_NCS=sqlite:///./schools/NCS.db

The suffix is the school code with `.` and `-` folded to `_`, because those are legal in
a school code and illegal in an environment variable name. Two codes that fold to the
same suffix are refused at startup rather than silently sharing a database — see
`env_suffix`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from sis.domain.errors import UnknownReference, ValidationError
from sis.domain.value_objects import SchoolCode

#: Names the schools this process serves. Unset means single-school mode.
SCHOOLS_VAR: Final[str] = "SIS_SCHOOLS"

#: Prefix of the per-school database URL variable; the school's suffix is appended.
DATABASE_URL_PREFIX: Final[str] = "SIS_DATABASE_URL"


class UnknownSchool(UnknownReference):
    """The caller named a school this process does not serve.

    A refusal rather than a fallback to some default school. Falling back is how a
    request meant for one branch is answered out of another's database — the exact
    failure physical separation exists to make impossible, reintroduced in the one
    place that chooses the connection.

    An `UnknownReference` so it renders as the 404 that every other code naming nothing
    gets (`sis.api.errors`), rather than inventing a status for this one case.
    """

    def __init__(self, code: str) -> None:
        super().__init__(
            f"no school {code!r} is configured on this service", field="school_code"
        )
        #: Named `school_code`, not `code`. `SisError.code` is the *error* code that
        #: `sis.api.errors` puts on the wire and clients branch on; assigning the school
        #: to it replaced `unknown_reference` with whatever string the caller sent, so
        #: every unknown school produced a different machine-readable code.
        self.school_code = code


class TenancyMisconfigured(RuntimeError):
    """The registry itself is wrong, and no request can be served correctly.

    Raised at startup rather than per request. A school named in `SIS_SCHOOLS` with no
    database URL behind it would otherwise fail on the first parent who tried to sign in,
    at whatever hour that happened to be, with an error naming a connection rather than a
    missing setting.
    """


def env_suffix(code: str) -> str:
    """The environment-variable suffix for a school code.

    Public because provisioning renders both the variable name and the database name
    from it (`sis.application.services.estate`), and those two must fold a code the same
    way or a school ends up pointed at another school's database.

    School codes may contain `.` and `-` (`sis.domain.value_objects._CODE_PATTERN`);
    environment variable names may not. Folding both to `_` is the only mapping that
    keeps a code like `NC-1` addressable, and it is not injective — `NC-1` and `NC.1`
    fold together. `_registry` detects that collision and refuses, because the
    alternative is two schools quietly sharing one database.
    """
    return code.replace(".", "_").replace("-", "_")


@dataclass(frozen=True, slots=True)
class Tenant:
    """One school and the database its rows live in."""

    code: str
    database_url: str

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@dataclass(frozen=True, slots=True)
class Registry:
    """Every school this process serves, resolved once from the environment.

    `tenants` is empty in single-school mode, which is what `is_multi_school` reports.
    Callers should ask that question rather than testing the mapping's truthiness, so the
    intent reads at the call site.
    """

    tenants: tuple[Tenant, ...]

    @property
    def is_multi_school(self) -> bool:
        return bool(self.tenants)

    @property
    def codes(self) -> tuple[str, ...]:
        """School codes in the order `SIS_SCHOOLS` listed them."""
        return tuple(tenant.code for tenant in self.tenants)

    def get(self, code: str) -> Tenant:
        """The tenant for a code, normalised the way every other school code is.

        Normalising through `SchoolCode` means `ncs`, ` NCS ` and `NCS` all resolve to
        the same school, matching what the rest of the service does with the same string.
        An unrecognised code raises rather than resolving to anything.
        """
        try:
            wanted = str(SchoolCode(code))
        except ValidationError as error:
            raise UnknownSchool(str(code)) from error
        for tenant in self.tenants:
            if tenant.code == wanted:
                return tenant
        raise UnknownSchool(wanted)

    def contains(self, code: str) -> bool:
        try:
            self.get(code)
        except UnknownSchool:
            return False
        return True


def _configured_codes() -> tuple[str, ...]:
    """The school codes from `SIS_SCHOOLS`, normalised, in order, without duplicates.

    Comma-separated. Blank entries are skipped rather than refused, because a trailing
    comma in an env file is a typo that should not stop a service from starting.
    """
    raw = (os.getenv(SCHOOLS_VAR) or "").strip()
    if not raw:
        return ()
    seen: list[str] = []
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        try:
            code = str(SchoolCode(candidate))
        except ValidationError as error:
            raise TenancyMisconfigured(
                f"{SCHOOLS_VAR} lists {candidate!r}, which is not a valid school code: "
                f"{error}"
            ) from error
        if code not in seen:
            seen.append(code)
    return tuple(seen)


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    """The process's school registry. Cached; call `reset_registry_cache()` in tests.

    Read lazily rather than at import, for the reason `sis.config.get_settings` is:
    alembic's `env.py`, pytest fixtures and `uvicorn --reload` all set variables after
    this package is first imported, and an import-time snapshot ignores them.
    """
    codes = _configured_codes()
    if not codes:
        return Registry(tenants=())

    tenants: list[Tenant] = []
    suffixes: dict[str, str] = {}
    missing: list[str] = []
    for code in codes:
        suffix = env_suffix(code)
        clash = suffixes.get(suffix)
        if clash is not None:
            raise TenancyMisconfigured(
                f"school codes {clash!r} and {code!r} both map to the environment "
                f"suffix {suffix!r}, so they would read the same "
                f"{DATABASE_URL_PREFIX}_{suffix} and share one database. Rename one."
            )
        suffixes[suffix] = code
        url = (os.getenv(f"{DATABASE_URL_PREFIX}_{suffix}") or "").strip()
        if not url:
            missing.append(f"{DATABASE_URL_PREFIX}_{suffix} (for school {code})")
            continue
        tenants.append(Tenant(code=code, database_url=url))

    if missing:
        raise TenancyMisconfigured(
            f"{SCHOOLS_VAR} names schools with no database behind them: "
            + ", ".join(missing)
            + ". Every school needs its own database URL; there is no shared default, "
            "because falling back to one would put two schools' rows in one file."
        )
    return Registry(tenants=tuple(tenants))


def reset_registry_cache() -> None:
    """Drop the cached registry so a test can point the service at other databases.

    Pairs with `sis.config.reset_settings_cache` and
    `sis.infrastructure.db.session.reset_engine`: clearing this alone changes nothing,
    because the engines already hold connections to the old URLs.
    """
    get_registry.cache_clear()


__all__ = [
    "DATABASE_URL_PREFIX",
    "Registry",
    "SCHOOLS_VAR",
    "TenancyMisconfigured",
    "Tenant",
    "env_suffix",
    "UnknownSchool",
    "get_registry",
    "reset_registry_cache",
]
