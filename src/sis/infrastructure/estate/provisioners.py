"""Creating and migrating one school's database, per driver.

Two adapters for `sis.application.ports.estate.DatabaseProvisioner`. Which one a
deployment gets is decided by the URL scheme, not by configuration, because the scheme is
already the truth: a URL beginning `postgresql` names a database on a server, and one
beginning `sqlite` names a file.

Both share `_alembic_upgrade`, and the reason it shells out rather than calling alembic's
Python API is the reason the script it came from did: alembic's `env.py` configures
logging and holds module-level state, so migrating ten schools in one process layers ten
environments on top of each other, and the failure that produces is a migration silently
running against the previous school's URL.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

#: The repository root: this file is sis/infrastructure/estate/provisioners.py, so the
#: fourth parent. Both the alembic config and the deployment's .env are found from here.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_ALEMBIC_INI: Final[Path] = PROJECT_ROOT / "sis" / "alembic.ini"


class ProvisioningFailed(RuntimeError):
    """The database could not be created or migrated. Nothing was recorded."""


def _alembic_upgrade(database_url: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(_ALEMBIC_INI),
            "-x",
            f"db_url={database_url}",
            "upgrade",
            "head",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProvisioningFailed(
            f"alembic upgrade head failed for {_redacted(database_url)}:\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _redacted(database_url: str) -> str:
    """A URL safe to put in an error or a log: the password removed, nothing else."""
    try:
        return make_url(database_url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001 - a malformed URL must still be reportable
        return "<unparseable database url>"


class PostgresProvisioner:
    """One PostgreSQL database per school, on the server the estate already runs.

    `CREATE DATABASE` cannot run inside a transaction and cannot run on a connection to
    the database being created, so this connects to a maintenance database -- the one
    named by `admin_url` -- in autocommit and issues the statement there.

    **That credential is not the application's.** A role that can create a database can
    drop one, and the service answering parent requests has no business holding it. Point
    `admin_url` at a separate role whose only job is this, and the blast radius of the
    running service stays where it was.
    """

    def __init__(self, admin_url: str) -> None:
        if not admin_url.strip():
            raise ProvisioningFailed(
                "SIS_ADMIN_DATABASE_URL is not set. Creating a school's database needs a "
                "connection to the server's maintenance database (usually .../postgres) "
                "under a role permitted to CREATE DATABASE. The service's own connection "
                "cannot do it, and should not be able to."
            )
        self._admin_url = admin_url

    def exists(self, database_url: str) -> bool:
        name = make_url(database_url).database
        engine = create_engine(self._admin_url, future=True, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as connection:
                found = connection.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": name},
                ).first()
            return found is not None
        except Exception as error:  # noqa: BLE001
            # Deliberately not `return False`. An unreachable server means "I could not
            # tell", and answering "it is not there" sends the caller on to CREATE.
            raise ProvisioningFailed(
                f"could not reach the PostgreSQL server to check whether {name!r} "
                f"exists: {error}"
            ) from error
        finally:
            engine.dispose()

    def create(self, database_url: str) -> None:
        name = make_url(database_url).database
        if not name:
            raise ProvisioningFailed(f"{_redacted(database_url)} names no database.")
        engine = create_engine(self._admin_url, future=True, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as connection:
                # The identifier is interpolated because PostgreSQL does not accept a
                # bind parameter in DDL. It is safe here and only here: `name` was
                # rendered by `plan_provision` from a `SchoolCode`, which is
                # `^[A-Z0-9][A-Z0-9._-]*$` folded to `[a-z0-9_]` -- no quote, no space,
                # no semicolon can reach this line. Quoted anyway, so a template with an
                # unusual prefix cannot change the statement's shape.
                connection.execute(text(f'CREATE DATABASE "{name}"'))
        except Exception as error:  # noqa: BLE001
            raise ProvisioningFailed(
                f"could not create database {name!r}: {error}"
            ) from error
        finally:
            engine.dispose()

    def migrate(self, database_url: str) -> None:
        _alembic_upgrade(database_url)


class SqliteProvisioner:
    """One file per school. What a development machine and the test suite use.

    `create` only makes the directory: alembic creates the file itself on first
    connection, so there is nothing else to do and creating an empty file here would
    make `exists` true for a database that has no schema.
    """

    def exists(self, database_url: str) -> bool:
        path = self._path(database_url)
        return path is not None and path.exists()

    def create(self, database_url: str) -> None:
        path = self._path(database_url)
        if path is None:
            raise ProvisioningFailed(
                f"{database_url!r} is an in-memory SQLite database. A school's rows have "
                "to outlive the process that wrote them."
            )
        path.parent.mkdir(parents=True, exist_ok=True)

    def migrate(self, database_url: str) -> None:
        _alembic_upgrade(database_url)

    @staticmethod
    def _path(database_url: str) -> Path | None:
        if not database_url.startswith("sqlite"):
            return None
        _, _, tail = database_url.partition(":///")
        return Path(tail).resolve() if tail and ":memory:" not in database_url else None


def provisioner_for(database_url: str, *, admin_url: str = "") -> object:
    """The adapter that fits this URL's driver.

    Chosen from the scheme rather than a setting, so a deployment cannot be configured to
    provision SQLite files against a PostgreSQL estate.
    """
    if database_url.startswith("sqlite"):
        return SqliteProvisioner()
    if database_url.startswith(("postgresql", "postgres")):
        return PostgresProvisioner(admin_url)
    raise ProvisioningFailed(
        f"no provisioner for {_redacted(database_url)}. Schools can be created on "
        "PostgreSQL or SQLite; anything else has to be created by hand and its "
        "connection added to the environment."
    )


__all__ = [
    "PostgresProvisioner",
    "ProvisioningFailed",
    "SqliteProvisioner",
    "provisioner_for",
]
