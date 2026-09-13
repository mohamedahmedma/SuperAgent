"""Alembic's entry point for the chat backend: the URL, the metadata, and the lock.

Alembic owns this schema. The backend does not create tables at boot; it checks that the
database is at this build's head revision and refuses to start otherwise
(`backend/db/schema_version.py`). Upgrading is an explicit step —
`alembic -c backend/alembic.ini upgrade head` — which the container runs before uvicorn.

Two choices follow `sis/migrations/env.py`, for the reasons given there: the engine is
built from the URL string rather than through the config object, so a password
containing `%` cannot fail as a ConfigParser interpolation error; and `target_metadata`
is imported defensively, falling back to `None` — which alembic refuses to autogenerate
against — rather than an empty MetaData that would autogenerate dropping every table.

Three are specific to this service:

- **Its own version table** (`VERSION_TABLE`), so it can share a Postgres database with
  another service's alembic history without either reading the other's revision.
- **An advisory lock around the upgrade.** Two backend containers starting together both
  run `upgrade head`; the lock makes the second wait and then find nothing left to do,
  rather than race the first through the same DDL.
- **A connection can be handed in** through `config.attributes["connection"]`, which is
  how the tests run every revision inside a throwaway schema.
"""
from __future__ import annotations

import logging
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

# `prepend_sys_path` in alembic.ini covers the CLI; this covers alembic driven
# programmatically, where that option is never read.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.db.schema_version import MIGRATION_LOCK_KEY, VERSION_TABLE  # noqa: E402
from backend.env import load_env  # noqa: E402

# Before anything reads DATABASE_URL: a local run takes it from .env, as the app does.
load_env()

config = context.config
_injected = config.attributes.get("connection")
if config.config_file_name is not None and _injected is None:
    # disable_existing_loggers=False: the default switches off every logger that already
    # exists, which would silence the application for the rest of the process.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

log = logging.getLogger("alembic.env")

try:
    import backend.db.models  # noqa: F401 — registers every table on Base.metadata
    from backend.infra.database import Base

    target_metadata = Base.metadata
except Exception as exc:  # noqa: BLE001 — see the module docstring
    target_metadata = None
    log.warning(
        "could not import backend.db.models (%s: %s); migrations will still run, but "
        "--autogenerate is unavailable until this import succeeds",
        type(exc).__name__,
        exc,
    )


def _database_url() -> str:
    """`-x db_url=...` if given, else the DATABASE_URL the application itself uses."""
    override = context.get_x_argument(as_dictionary=True).get("db_url")
    if override:
        return override
    from backend.infra.database import DATABASE_URL

    return DATABASE_URL


def run_migrations_offline() -> None:
    """Emit SQL to stdout for review. No connection is opened, so no lock is taken.

    Only from 0001 onwards (`upgrade 0001:head --sql`): the baseline inspects a live
    database to decide what to create, and there is none here.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _migrate(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        compare_type=True,
    )
    with context.begin_transaction():
        # Transaction-scoped, so it is released by the same commit that records the
        # revision — never held by a process that died halfway.
        connection.exec_driver_sql(f"SELECT pg_advisory_xact_lock({MIGRATION_LOCK_KEY})")
        context.run_migrations()


def run_migrations_online() -> None:
    if _injected is not None:
        _migrate(_injected)
        return
    # NullPool because the process exits when this returns.
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            _migrate(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
