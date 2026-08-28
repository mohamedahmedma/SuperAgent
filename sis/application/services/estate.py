"""Adding a school to the estate: what its database is called, and what must be true first.

Provisioning used to live in `scripts/schools.py`, where the decision and the printing
were the same function and `SystemExit` was the way a rule reported itself. That made the
rules unreachable from anything but a terminal, and untested -- `provision` and `split`
create and carve real databases, and nothing asserted either.

What is here is the deciding. `plan_provision` performs **no I/O at all**: it normalises
the code, renders the URL from the deployment's template, works out which environment
variable will carry it, and refuses everything it should refuse. A test hands it three
strings and asserts a plan. `EstateService.provision` is the thin part that takes a plan
and drives the two ports in the one order that fails safely.

## Naming

One template for the whole estate, because every school is the same provider on the same
server and only the connection differs::

    SIS_DATABASE_URL_TEMPLATE=postgresql+psycopg2://sis:pw@db:5432/sis_{slug}

`{slug}` is the school code lowercased with `.` and `-` folded to `_` -- the same folding
`sis.tenancy` applies to build the variable name, so the database and the variable that
points at it cannot disagree about which school they mean. `{code}` is available too, for
a deployment that wants the code verbatim.

Rendering rather than accepting a URL per school is the point of the request. A manager
creating a school supplies a code; nobody should be typing a connection string into a
console, and no two schools should be able to end up pointed at one database because
somebody pasted the same URL twice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Final

from sis.application.ports.estate import ConfigStore, DatabaseProvisioner
from sis.domain.errors import DomainRuleViolation, ValidationError
from sis.domain.value_objects import SchoolCode
from sis.tenancy import (
    DATABASE_URL_PREFIX,
    SCHOOLS_VAR,
    TenancyMisconfigured,
    env_suffix,
)

#: Names the template every school's connection is rendered from.
TEMPLATE_VAR: Final[str] = "SIS_DATABASE_URL_TEMPLATE"

#: PostgreSQL truncates identifiers at 63 bytes, silently, which would land two schools
#: whose codes share a long prefix in one database. Refused here instead.
_MAX_DATABASE_NAME: Final[int] = 63

_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{(code|slug)\}")


class SchoolAlreadyProvisioned(DomainRuleViolation):
    """This code already has a database. Provisioning again would run over live rows."""

    code: ClassVar[str] = "school_already_provisioned"

    def __init__(self, code: str) -> None:
        super().__init__(
            f"school {code!r} is already configured on this service. Use `migrate` to "
            "bring its database up to date; provisioning is for schools that do not "
            "exist yet.",
            field="code",
        )
        self.school_code = code


@dataclass(frozen=True, slots=True)
class ProvisionPlan:
    """Everything provisioning will do, decided before anything is touched.

    A value, so the console can show a manager exactly which database is about to be
    created and the CLI can offer a dry run, both without a server being reachable.
    """

    #: The normalised school code, as every other layer spells it.
    code: str
    #: Where this school's rows will live.
    database_url: str
    #: The variable that will carry `database_url`, e.g. `SIS_DATABASE_URL_NC_1`.
    env_var: str
    #: What `SIS_SCHOOLS` becomes: the existing codes, in order, plus this one.
    schools_value: str

    @property
    def config_changes(self) -> dict[str, str]:
        """The two variables the store must end up holding."""
        return {self.env_var: self.database_url, SCHOOLS_VAR: self.schools_value}


def _database_name(database_url: str) -> str:
    """The database a URL names, without its query string.

    Enough of a URL parser for the one check that matters. A driver-specific parse would
    tie this module to a driver, which is what the template exists to avoid.
    """
    tail = database_url.rsplit("/", 1)[-1]
    return tail.split("?", 1)[0]


def plan_provision(
    code: str, *, template: str, existing_codes: tuple[str, ...]
) -> ProvisionPlan:
    """Decide what provisioning this code would do. Pure: no server, no file, no clock.

    Raises rather than returning a plan whenever the estate would end up ambiguous --
    an invalid code, a code already served, a template that renders the same URL for
    every school, or a database name the server will silently truncate.
    """
    normalised = str(SchoolCode(code))

    for existing in existing_codes:
        if existing == normalised:
            raise SchoolAlreadyProvisioned(normalised)

    if not template.strip():
        raise TenancyMisconfigured(
            f"{TEMPLATE_VAR} is not set, so there is no rule for naming a new school's "
            "database. Set it to the connection every school shares, with {slug} where "
            "the school's part goes -- see .env.example."
        )

    if not _PLACEHOLDER.search(template):
        raise TenancyMisconfigured(
            f"{TEMPLATE_VAR} contains no {{slug}} or {{code}} placeholder, so every "
            f"school would render the same URL and share one database. Got "
            f"{template!r}."
        )

    suffix = env_suffix(normalised)
    database_url = template.format(code=normalised, slug=suffix.lower())

    name = _database_name(database_url)
    if len(name.encode("utf-8")) > _MAX_DATABASE_NAME:
        raise ValidationError(
            f"the database name this renders, {name!r}, is longer than the "
            f"{_MAX_DATABASE_NAME} bytes PostgreSQL keeps. It would be truncated, and "
            "two schools with a long shared prefix would land in one database. Shorten "
            f"the code or the prefix in {TEMPLATE_VAR}.",
            field="code",
        )

    ordered = (*existing_codes, normalised)
    return ProvisionPlan(
        code=normalised,
        database_url=database_url,
        env_var=f"{DATABASE_URL_PREFIX}_{suffix}",
        schools_value=",".join(ordered),
    )


class EstateService:
    """Creates a school's database and records its connection.

    Holds the ports, not the environment: the template and the current school list are
    passed to `provision`, so a test drives this with three strings and two fakes.
    """

    def __init__(self, provisioner: DatabaseProvisioner, config: ConfigStore) -> None:
        self._provisioner = provisioner
        self._config = config

    def provision(
        self, code: str, *, template: str, existing_codes: tuple[str, ...]
    ) -> ProvisionPlan:
        """Create the school's database, migrate it, then record the connection.

        The order is the contract described in `sis.application.ports.estate`: a crash
        after the database exists but before the configuration is written leaves an
        orphan that the next attempt refuses to overwrite, which is recoverable. Writing
        the configuration first would leave `SIS_SCHOOLS` naming a school with no
        database, and the service refuses to start at all in that state.
        """
        plan = plan_provision(code, template=template, existing_codes=existing_codes)

        if self._provisioner.exists(plan.database_url):
            raise SchoolAlreadyProvisioned(plan.code)

        self._provisioner.create(plan.database_url)
        self._provisioner.migrate(plan.database_url)
        self._config.update(plan.config_changes)
        return plan


__all__ = [
    "EstateService",
    "ProvisionPlan",
    "SchoolAlreadyProvisioned",
    "TEMPLATE_VAR",
    "plan_provision",
]
