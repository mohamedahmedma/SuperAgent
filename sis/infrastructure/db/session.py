"""Engine and session lifecycle.

The engine is built lazily and cached, never at import time. `sis.config` reads the
environment lazily for the same reason: alembic's `env.py`, pytest fixtures and
`uvicorn --reload` all set `SIS_DATABASE_URL` after the package is first imported,
and an import-time engine would silently bind to the wrong database — in tests, to
the developer's real `sis.db`.

Nothing in this module creates tables. Alembic owns the schema; see `base.py`.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from sis.config import Settings, get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _is_memory_sqlite(url: str) -> bool:
    return url.startswith("sqlite") and (":memory:" in url or url.endswith("sqlite://"))


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    """Turn on SQLite foreign key enforcement for every new connection.

    SQLite ships with `foreign_keys` OFF and the pragma is per-connection, not
    per-database — so without this listener every `ForeignKey` in the schema is
    decorative. A grade could be written against a deleted class, a membership
    against a student number that never existed, and `ON DELETE` clauses would
    quietly do nothing. The bad rows are invisible until the day someone runs the
    same schema on Postgres and the import blows up on data that "worked for years".

    Registered against the `Engine` class rather than one instance so test engines
    and the alembic engine are covered too — the pragma is worthless if it depends
    on remembering to apply it.
    """
    # Matched on the driver module ("sqlite3", "_sqlite3", "pysqlite3") because the
    # connect event fires before any dialect object is in reach.
    if "sqlite" in type(dbapi_connection).__module__:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _engine_kwargs(settings: Settings) -> dict[str, Any]:
    """Pool configuration, which differs by driver rather than by preference."""
    if settings.is_sqlite:
        kwargs: dict[str, Any] = {
            # FastAPI runs sync endpoints in a threadpool, so the connection that
            # opened a session is rarely the thread that uses it. SQLAlchemy's pool
            # already guarantees one connection is used by one thread at a time;
            # sqlite3's own check is redundant here and only produces false errors.
            "connect_args": {"check_same_thread": False},
        }
        if _is_memory_sqlite(settings.database_url):
            # An in-memory database lives inside a single connection. Without a
            # StaticPool the pool hands out a fresh, *empty* database on the second
            # checkout and the migrations a test just ran appear to have vanished.
            kwargs["poolclass"] = StaticPool
        return kwargs

    return {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        # Recycle before the server's idle timeout so a connection that the database
        # closed overnight is not handed to the first registrar of the morning.
        "pool_recycle": settings.db_pool_recycle_seconds,
        "pool_timeout": settings.db_pool_timeout_seconds,
        "pool_pre_ping": True,
    }


def get_engine() -> Engine:
    """The process-wide engine, built on first use."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            future=True,
            **_engine_kwargs(settings),
        )
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    """The session factory bound to the process engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            # Attributes stay readable after commit. An API handler that commits and
            # then serialises the object it just wrote would otherwise trigger a
            # refresh against a closed session and raise mid-response.
            expire_on_commit=False,
            class_=Session,
        )
    return _session_factory


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding one session per request.

    The rollback on exception is what makes a partially applied import safe: a row
    that raised after three inserts leaves nothing behind. Committing is the caller's
    job — a dependency that auto-commits would persist work an endpoint decided to
    abandon after validation failed.
    """
    session = get_sessionmaker()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session for code outside a request — CLI tasks, bootstrap."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine and factory so a test can repoint the service.

    Pairs with `reset_settings_cache()`: clearing the settings alone changes nothing,
    because the engine already holds a connection to the old URL.
    """
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


__all__ = [
    "get_engine",
    "get_session",
    "get_sessionmaker",
    "reset_engine",
    "session_scope",
]
