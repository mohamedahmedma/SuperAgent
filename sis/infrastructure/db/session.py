"""Engine and session lifecycle, one engine per school.

Schools are separated **physically**: one database each, the same schema in every one,
and no query that spans two. The engine is therefore what enforces the boundary.
`get_sessionmaker(school_code)` returns a factory bound to that school's database and to
no other, which is why no repository in this service takes a school argument — a query
cannot reach another school's rows because those rows are not in the file it is connected
to. Isolation stops being a `WHERE` clause somebody can forget.

`school_code=None` means the process-wide database at `SIS_DATABASE_URL`: single-school
mode, the default, and what every existing caller and test already asks for. See
`sis.tenancy` for how the two modes are chosen.

Engines are built lazily and cached per school, never at import time. `sis.config` reads
the environment lazily for the same reason: alembic's `env.py`, pytest fixtures and
`uvicorn --reload` all set variables after this package is first imported, and an
import-time engine would silently bind to the wrong database — in tests, to the
developer's real `sis.db`.

Nothing in this module creates tables. Alembic owns the schema; see `base.py`.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from sis.config import Settings, get_settings
from sis.tenancy import get_registry

#: Engines and session factories keyed by school code; `None` is single-school mode.
#: Guarded by `_lock` because FastAPI serves sync endpoints from a threadpool, so the
#: first request for a school can arrive on several threads at once — and two engines
#: for one SQLite file means two pools, which is how a test sees a table that another
#: connection has not committed yet.
_engines: dict[str | None, Engine] = {}
_session_factories: dict[str | None, sessionmaker[Session]] = {}
_lock = Lock()


def _database_url(school_code: str | None) -> str:
    """Which database this school's rows live in.

    A named school always resolves through the registry, and an unknown one raises
    `UnknownSchool` rather than falling back to the default database. The fallback is
    the failure worth naming: it would answer a request meant for one branch out of
    another's file, which is the single thing physical separation exists to prevent.
    """
    if school_code is None:
        return get_settings().database_url
    return get_registry().get(school_code).database_url


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


def _engine_kwargs(url: str, settings: Settings) -> dict[str, Any]:
    """Pool configuration, which differs by driver rather than by preference.

    The URL is passed rather than read off `settings`, because in multi-school mode the
    driver is a property of the school's own database: one school may still be on a
    SQLite file while another has been moved to Postgres, and sizing a pool for the
    wrong one of those is either a crash or a silently serialised service. The pool
    *numbers* stay global — they describe this process's appetite for connections, not
    any one school's.
    """
    if url.startswith("sqlite"):
        kwargs: dict[str, Any] = {
            # FastAPI runs sync endpoints in a threadpool, so the connection that
            # opened a session is rarely the thread that uses it. SQLAlchemy's pool
            # already guarantees one connection is used by one thread at a time;
            # sqlite3's own check is redundant here and only produces false errors.
            "connect_args": {"check_same_thread": False},
        }
        if _is_memory_sqlite(url):
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


def get_engine(school_code: str | None = None) -> Engine:
    """The engine for one school's database, built on first use and cached.

    `None` is the process-wide database — single-school mode, and every caller that
    predates physical separation.
    """
    # Resolved before the lock: an unknown school must raise rather than wait on a lock
    # to find that out, and `_database_url` reads caches of its own.
    url = _database_url(school_code)
    with _lock:
        engine = _engines.get(school_code)
        if engine is None:
            engine = create_engine(
                url,
                future=True,
                **_engine_kwargs(url, get_settings()),
            )
            _engines[school_code] = engine
        return engine


def get_sessionmaker(school_code: str | None = None) -> sessionmaker[Session]:
    """The session factory bound to one school's engine.

    This is the seam physical separation is built on. Every repository is constructed
    against a session from here, so binding the factory to a school binds every read and
    write in that unit of work to that school's database — without a single repository
    knowing schools exist.
    """
    engine = get_engine(school_code)
    with _lock:
        factory = _session_factories.get(school_code)
        if factory is None:
            factory = sessionmaker(
                bind=engine,
                autoflush=False,
                # Attributes stay readable after commit. An API handler that commits and
                # then serialises the object it just wrote would otherwise trigger a
                # refresh against a closed session and raise mid-response.
                expire_on_commit=False,
                class_=Session,
            )
            _session_factories[school_code] = factory
        return factory


def get_session(school_code: str | None = None) -> Iterator[Session]:
    """FastAPI dependency yielding one session per request, on one school's database.

    The rollback on exception is what makes a partially applied import safe: a row
    that raised after three inserts leaves nothing behind. Committing is the caller's
    job — a dependency that auto-commits would persist work an endpoint decided to
    abandon after validation failed.
    """
    session = get_sessionmaker(school_code)()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def session_scope(school_code: str | None = None) -> Iterator[Session]:
    """Transactional session for code outside a request — CLI tasks, bootstrap."""
    session = get_sessionmaker(school_code)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop every cached engine and factory so a test can repoint the service.

    Pairs with `reset_settings_cache()` and `sis.tenancy.reset_registry_cache()`:
    clearing those alone changes nothing, because the engines already hold connections
    to the old URLs. All schools are dropped rather than one, because a test that
    repoints the service is repointing the whole registry.
    """
    with _lock:
        for engine in _engines.values():
            engine.dispose()
        _engines.clear()
        _session_factories.clear()


__all__ = [
    "get_engine",
    "get_session",
    "get_sessionmaker",
    "reset_engine",
    "session_scope",
]
