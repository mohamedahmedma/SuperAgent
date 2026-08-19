"""SQLite engine, session factory, and Base for the school app."""

import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("SCHOOL_DATABASE_URL", "sqlite:///./school.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


@event.listens_for(engine, "connect")
def _enable_foreign_keys(dbapi_connection, connection_record):
    """SQLite ignores foreign keys unless this pragma is set on every connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_db():
    """FastAPI dependency that yields a session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create any tables that do not exist yet."""
    from school import models  # noqa: F401  (import registers the models on Base)

    Base.metadata.create_all(bind=engine)
