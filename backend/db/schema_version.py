"""The startup gate: this build serves only a database at its own head revision.

Alembic owns the backend's schema (`backend/alembic.ini`). The backend used to call
`create_all()` on every boot, which built a missing table in whatever shape the models
had that day and could never change an existing one — so a model change reached
production as `UndefinedColumn`, partway through somebody's request.

The replacement follows `sis/app.py`. At startup the backend compares the revision
recorded in the database with the head of its own migration directory, and refuses to
serve on any mismatch: never migrated, part-migrated, or migrated by a newer build. It
does not upgrade on boot. Two processes starting together would race on the same DDL, so
the upgrade is its own step — the container runs it before uvicorn starts, under the
advisory lock `migrations/env.py` takes.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"
UPGRADE_COMMAND = "alembic -c backend/alembic.ini upgrade head"

#: Not alembic's default `alembic_version`: identity and sis are expected to move onto
#: Postgres, and a shared version table would make each service read another's revision.
VERSION_TABLE = "backend_alembic_version"

#: Serialises concurrent upgrades of one database. Any constant works, as long as every
#: migrating process uses the same one and nothing else locks on it.
MIGRATION_LOCK_KEY = 0x5ABAC0001


def expected_heads() -> set[str]:
    """The revisions this build's migration directory ends at."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    return set(ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_heads())


def applied_heads(engine: Engine) -> set[str]:
    """The revisions recorded in the database; empty when it was never migrated."""
    from alembic.runtime.migration import MigrationContext

    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"version_table": VERSION_TABLE})
        return set(context.get_current_heads())


def verify_database_is_migrated(engine: Engine | None = None) -> None:
    """Raise unless the database is at exactly this build's head revision."""
    if engine is None:
        from backend.infra.database import engine as default_engine

        engine = default_engine

    applied = applied_heads(engine)
    expected = expected_heads()
    if applied == expected:
        logger.info("Database schema at revision %s", ", ".join(sorted(applied)))
        return

    state = ", ".join(sorted(applied)) if applied else "<none — never migrated>"
    raise RuntimeError(
        "The database schema does not match this build.\n"
        f"  database: {engine.url.render_as_string(hide_password=True)}\n"
        f"  applied:  {state}\n"
        f"  expected: {', '.join(sorted(expected)) or '<no migrations found>'}\n"
        "Alembic owns this schema; the backend will not create or alter it. Run:\n"
        f"  {UPGRADE_COMMAND}\n"
        "then start the backend again."
    )
