"""Adapters that bind the domain and application layers to real technology.

Everything framework-specific lives below this package: SQLAlchemy models and
sessions, repository implementations, file parsers. The dependency arrow points
one way only — infrastructure imports from `application` and `domain`, never the
reverse. That is what lets a use case be exercised with fake repositories and no
database at all.
"""
