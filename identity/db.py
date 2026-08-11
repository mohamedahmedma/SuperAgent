"""Engine and session for the identity service.

Its own `IDENTITY_DATABASE_URL`, its own engine, its own `Base`. It shares no table
and no connection with the records facade or the chat backend — a service that owns
credentials should not be reachable through another service's SQL injection bug.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("IDENTITY_DATABASE_URL", "sqlite:///./identity.db")

_POOL_OPTIONS: dict = {}
if not DATABASE_URL.startswith("sqlite"):
    _POOL_OPTIONS = {
        "pool_size": int(os.getenv("IDENTITY_DB_POOL_SIZE") or 10),
        "max_overflow": int(os.getenv("IDENTITY_DB_MAX_OVERFLOW") or 10),
        "pool_recycle": int(os.getenv("IDENTITY_DB_POOL_RECYCLE_SECONDS") or 1800),
        "pool_timeout": int(os.getenv("IDENTITY_DB_POOL_TIMEOUT_SECONDS") or 30),
    }

_CONNECT_ARGS: dict = {}
if DATABASE_URL.startswith("sqlite"):
    _CONNECT_ARGS = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_CONNECT_ARGS, **_POOL_OPTIONS)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    import identity.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
