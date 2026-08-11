"""Engine and session for the records facade.

Its own `RECORDS_DATABASE_URL`, its own engine, its own `Base` — deliberately not
importing `backend.infra.database`. The isolation is the point: this service must be
deployable, restartable and replaceable without the chat backend noticing, and a
shared engine would quietly re-couple them at the connection pool.

The default is SQLite so the service runs with no infrastructure at all. Point it at
Postgres for anything holding real student records.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("RECORDS_DATABASE_URL", "sqlite:///./records.db")

# SQLite's pools reject sizing arguments outright rather than ignoring them, so the
# options are only applied to the server dialects that actually pool connections.
_POOL_OPTIONS: dict = {}
if not DATABASE_URL.startswith("sqlite"):
    _POOL_OPTIONS = {
        "pool_size": int(os.getenv("RECORDS_DB_POOL_SIZE") or 10),
        "max_overflow": int(os.getenv("RECORDS_DB_MAX_OVERFLOW") or 10),
        "pool_recycle": int(os.getenv("RECORDS_DB_POOL_RECYCLE_SECONDS") or 1800),
        "pool_timeout": int(os.getenv("RECORDS_DB_POOL_TIMEOUT_SECONDS") or 30),
    }

_CONNECT_ARGS: dict = {}
if DATABASE_URL.startswith("sqlite"):
    # FastAPI serves sync endpoints from a thread pool; SQLite's default single-thread
    # check would reject those connections.
    _CONNECT_ARGS = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args=_CONNECT_ARGS,
    **_POOL_OPTIONS,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    # Delayed import to avoid a circular dependency: models import Base from here.
    import records.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
