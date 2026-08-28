"""Ports for bringing a school's database into existence, and for remembering it.

Provisioning a school is three separate side effects on three different systems: a
database is created on the server, a migration is run against it, and the connection is
written where the next process start will read it. Each is a port here so the *decisions*
around them -- what the database is called, whether this school already exists, what the
environment should say afterwards -- can be tested with no server and no file.

That split is the whole point. `sis.application.services.estate` computes a
`ProvisionPlan` and never performs any of this; a test hands it a fake and asserts the
plan. What is left behind these protocols is genuinely irreducible I/O.

**Ordering is a rule of the caller, not of these ports, and it is load-bearing.** The
database is created and migrated *before* the configuration is written. A crash between
the two leaves an orphaned database, which is inert and which `provision` will refuse to
overwrite on the next attempt. The other order leaves `SIS_SCHOOLS` naming a school with
no reachable database -- and `sis.tenancy.get_registry` raises `TenancyMisconfigured` for
exactly that, at startup, so the service stops booting until someone edits a file by
hand. One of these failures is a retry; the other is an outage.
"""
from typing import Protocol


class DatabaseProvisioner(Protocol):
    """Creates and migrates one school's database.

    Implementations are per driver, not per deployment: the estate runs one provider and
    switches connections, so what varies is `CREATE DATABASE` against a server versus a
    file appearing on disk -- not what a school is.
    """

    def exists(self, database_url: str) -> bool:
        """Whether the database behind this URL is already there.

        Asked before creating, so provisioning refuses rather than runs over a database
        that may hold another school's rows. An unreachable *server* must raise, never
        return `False`: "I could not tell" and "it is not there" lead to opposite
        actions, and only one of them is safe.
        """
        ...

    def create(self, database_url: str) -> None:
        """Create the empty database. Raises if it already exists."""
        ...

    def migrate(self, database_url: str) -> None:
        """Bring one database to the head revision.

        Separate from `create` because it is also the whole of what `migrate` does for
        schools that already exist, and because a created-but-unmigrated database is a
        recoverable state that a re-run fixes.
        """
        ...


class ConfigStore(Protocol):
    """Where the estate's connections are recorded for the next process start.

    A port rather than a direct `.env` write because the file is a property of *this*
    deployment. A container that receives its environment from an orchestrator reads no
    file at all, and the day that becomes true the adapter changes and the service above
    does not.
    """

    def read(self) -> dict[str, str]:
        """Every variable currently recorded, as written."""
        ...

    def update(self, values: dict[str, str]) -> None:
        """Set these variables, atomically, preserving everything else in the store.

        Atomic because a half-written store is the outage described above, and because
        two managers creating schools at the same moment must not interleave into a file
        that names one school and points at the other's database.
        """
        ...


__all__ = ["ConfigStore", "DatabaseProvisioner"]
