"""Engine and session for the identity service.

Its own `IDENTITY_DATABASE_URL`, its own engine, its own `Base`. It shares no table
and no connection with the records facade or the chat backend — a service that owns
credentials should not be reachable through another service's SQL injection bug.

## Built lazily, never at import

The engine used to be created at module scope from a `DATABASE_URL` read at module
scope. That reads the environment once, at whatever moment something first imports this
file — and pytest imports every collected module before running anything, so a test
suite that sets `IDENTITY_DATABASE_URL` in a fixture was already too late. The engine
stayed bound to whichever database won the import race, and a cross-service test then
served one suite's data over another suite's API, failing with `no such table` a long
way from the cause.

`sis.infrastructure.db.session` states the same reasoning and solves it the same way,
which is the pattern followed here: `get_engine()` builds on first use and caches,
`reset_engine()` drops the cache so a caller can repoint the service.

There is deliberately no module-level `engine` or `SessionLocal` any more. Re-adding one
as a convenience would re-create the bug, because a caller that binds it at import
captures the engine at import — which is exactly what this file no longer does.
"""
import logging
import os
from typing import Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def database_url() -> str:
    """Read at call time, so a caller that sets the variable is never too late."""
    return os.getenv("IDENTITY_DATABASE_URL", "sqlite:///./identity.db")


def _engine_kwargs(url: str) -> dict:
    """Pool configuration, which differs by driver rather than by preference."""
    if url.startswith("sqlite"):
        # FastAPI serves sync endpoints from a thread pool; SQLite's default
        # single-thread check would reject those connections. Sizing arguments are
        # omitted rather than defaulted — SQLite's pools reject them outright.
        return {"connect_args": {"check_same_thread": False}}
    return {
        "pool_size": int(os.getenv("IDENTITY_DB_POOL_SIZE") or 10),
        "max_overflow": int(os.getenv("IDENTITY_DB_MAX_OVERFLOW") or 10),
        "pool_recycle": int(os.getenv("IDENTITY_DB_POOL_RECYCLE_SECONDS") or 1800),
        "pool_timeout": int(os.getenv("IDENTITY_DB_POOL_TIMEOUT_SECONDS") or 30),
    }


def get_engine() -> Engine:
    """The engine, built on first use and cached for the life of the process."""
    global _engine
    if _engine is None:
        url = database_url()
        _engine = create_engine(url, pool_pre_ping=True, **_engine_kwargs(url))
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory


def new_session():
    """One session. The explicit spelling of what `SessionLocal()` used to be."""
    return get_session_factory()()


def reset_engine() -> None:
    """Drop the cached engine and factory so a caller can repoint the service.

    For tests and for anything that changes `IDENTITY_DATABASE_URL` after import.
    Disposes first: leaving the old engine undisposed leaks its pool, and on Windows an
    open SQLite handle makes the file unreplaceable.
    """
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


logger = logging.getLogger(__name__)


def get_db():
    db = new_session()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    # Delayed import to avoid a circular dependency: models import Base from here.
    import identity.models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _add_missing_columns(engine)


#: Columns added to existing tables after this service was first deployed.
#:
#: `create_all` creates missing *tables* and never alters an existing one, so a column
#: added to a model reaches a fresh database and silently misses every database that
#: already exists. This service has no alembic — a deliberate choice its models document —
#: so the additive case is handled here instead, and only the additive case: nothing below
#: drops, renames or retypes anything.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # Which school a verification belongs to. Blank on every row written before schools
    # were separated, which is exactly right: those challenges were single-school, and
    # `""` is what a single-school challenge stores today.
    ("verification_challenges", "school_code", "VARCHAR(16) NOT NULL DEFAULT ''"),
)


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


__all__ = [
    "Base",
    "database_url",
    "get_db",
    "get_engine",
    "get_session_factory",
    "init_db",
    "new_session",
    "reset_engine",
]
