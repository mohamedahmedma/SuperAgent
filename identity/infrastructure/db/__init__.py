"""Storage: the engine, the tables, and the repositories that read and write them.

`base.py` holds the declarative base alone, `session.py` builds the engine lazily,
`models.py` declares the four tables this service owns, `schema.py` creates them, and
`repositories/` implements the ports the use cases declared.
"""
from identity.infrastructure.db.base import Base
from identity.infrastructure.db.schema import init_db
from identity.infrastructure.db.session import (
    database_url,
    get_db,
    get_engine,
    get_session_factory,
    new_session,
    reset_engine,
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
