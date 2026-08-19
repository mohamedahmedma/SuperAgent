"""Database wiring: the declarative base, the engine, and session handling.

Re-exported here so callers write `from sis.infrastructure.db import Base` and stay
insulated from whether `Base` sits in `base.py` or moves later.
"""
from sis.infrastructure.db.base import Base
from sis.infrastructure.db.session import (
    get_engine,
    get_session,
    get_sessionmaker,
    reset_engine,
    session_scope,
)

__all__ = [
    "Base",
    "get_engine",
    "get_session",
    "get_sessionmaker",
    "reset_engine",
    "session_scope",
]
