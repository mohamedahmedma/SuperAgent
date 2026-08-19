"""Alembic's entry point: where the URL comes from, and where the metadata comes from.

Alembic owns this schema outright (invariant 8). There is no `create_all` anywhere in
`sis/`, so this file is the only thing standing between a running service and a table
that does not exist.

Two decisions here are worth the words:

**The engine is built by hand rather than through `engine_from_config`.** The usual
template writes the URL into the config object and lets alembic construct the engine
from it, but `Config.set_main_option` hands the string to a `ConfigParser`, which treats
`%` as the start of an interpolation token. A Postgres password containing `%` -- the
character a password generator produces roughly one time in twenty -- therefore fails
with `InterpolationSyntaxError` rather than with anything resembling "bad password", and
the operator spends the outage reading alembic's source. Constructing the engine
directly keeps the URL an opaque string from the environment to the driver.

**A missing models module degrades to "cannot autogenerate", never to "drop
everything".** `target_metadata` is imported defensively because `upgrade head` does not
need it -- the migrations are self-contained -- and an import error in the ORM layer
must not be the reason a production database cannot be brought up to date. The
fallback is `None` and pointedly not an empty `MetaData()`: alembic refuses to
autogenerate against `None` with a clear `CommandError`, whereas an empty metadata is a
valid comparison target describing a database with no tables, so `--autogenerate` would
cheerfully emit a migration that drops every table in the school's SIS.
"""
from __future__ import annotations

import logging
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

# `prepend_sys_path` in alembic.ini covers the CLI; this covers alembic driven
# programmatically from a test fixture, where that option is never read.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

log = logging.getLogger("alembic.env")

try:
    from sis.infrastructure.db.models import Base

    target_metadata = Base.metadata
except Exception as exc:  # noqa: BLE001 -- see the module docstring
    target_metadata = None
    log.warning(
        "could not import sis.infrastructure.db.models (%s: %s); migrations will still "
        "run, but --autogenerate is unavailable until this import succeeds",
        type(exc).__name__,
        exc,
    )


def _database_url() -> str:
    """The database to migrate: `-x db_url=...` if given, else `SIS_DATABASE_URL`.

    Routed through `sis.config` rather than reading `os.environ` here so the default
    (`sqlite:///./sis.db`) is stated in exactly one place. A second copy of it in this
    file is a default that drifts, and the symptom is a migration that appears to
    succeed against a scratch SQLite file while the service talks to Postgres.
    """
    from sis.config import get_settings

    override = context.get_x_argument(as_dictionary=True).get("db_url")
    return override or get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout for a DBA to review. No connection is opened."""
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=url.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and apply. NullPool because the process exits when this returns."""
    engine = create_engine(_database_url(), poolclass=pool.NullPool, future=True)
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                # SQLite cannot ALTER a column, so alembic has to rebuild the table.
                # Enabled by dialect rather than always, because batch mode on Postgres
                # would turn a cheap ALTER into a full table copy under a lock.
                render_as_batch=connection.dialect.name == "sqlite",
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
