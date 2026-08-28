"""The declarative base, alone in its own module.

Separate from `session.py` and from `models.py` because both need it and it must not drag
either in. `models.py` imports `Base`; `schema.py` imports `Base.metadata` *and* the
models; `session.py` imports neither. Collapsing these into one file is how a service
acquires an import cycle it then breaks with a function-level import.
"""
from sqlalchemy.orm import declarative_base

Base = declarative_base()

__all__ = ["Base"]
