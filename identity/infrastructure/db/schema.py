"""Creating the tables, and adding the columns `create_all` will not.

This service has no alembic — a deliberate choice its models document. `create_all`
creates missing *tables* and never alters an existing one, so a column added to a model
reaches a fresh database and silently misses every database that already exists. The
additive case is handled here, and **only** the additive case: nothing below drops,
renames or retypes anything.
"""
from __future__ import annotations

import logging

from identity.infrastructure.db.base import Base
from identity.infrastructure.db.session import get_engine

logger = logging.getLogger(__name__)

#: Columns added to existing tables after this service was first deployed.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # Which school a verification belongs to. Blank on every row written before schools
    # were separated, which is exactly right: those challenges were single-school, and
    # `""` is what a single-school challenge stores today.
    ("verification_challenges", "school_code", "VARCHAR(16) NOT NULL DEFAULT ''"),
)


def init_db() -> None:
    """Create anything missing. Safe to call on every startup."""
    # Imported for its side effect: the models must be registered on `Base.metadata`
    # before `create_all` can see them.
    import identity.infrastructure.db.models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine) -> None:
    """Add any column in `_ADDED_COLUMNS` that an existing database is missing.

    Idempotent, and silent when there is nothing to do. Each column is added in its own
    transaction so one failure does not roll back the others, and a failure is logged
    rather than raised: the service must still start if it cannot alter a table, because
    refusing to boot over a column that only affects multi-school deployments would take
    parent sign-in down for every single-school one.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table, column, definition in _ADDED_COLUMNS:
        if table not in existing_tables:
            continue  # `create_all` just made it, with the column already on it.
        present = {row["name"] for row in inspector.get_columns(table)}
        if column in present:
            continue
        try:
            with engine.begin() as connection:
                connection.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                )
            logger.info("Added missing column %s.%s", table, column)
        except Exception:  # noqa: BLE001 - reported, never fatal; see the docstring
            logger.exception(
                "Could not add the column %s.%s. Multi-school verification will not work "
                "until it exists; single-school sign-in is unaffected.",
                table,
                column,
            )


__all__ = ["init_db"]
