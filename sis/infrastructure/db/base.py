"""The declarative base every SIS table inherits from.

It carries a constraint naming convention, and that is the whole reason this file
exists separately from the models. SQLite cannot `ALTER TABLE ... DROP CONSTRAINT`,
so alembic rewrites the table wholesale in batch mode — and a batch migration can
only reproduce a constraint it can *name*. Letting SQLite auto-name indexes and
unique constraints means a future migration that renames a class label or widens a
code column fails at 2am on a database holding a school's grades, with no way
forward but hand-written SQL.

Naming them up front also makes autogenerate stable: without a convention, alembic
compares a named model constraint against an anonymous database one and proposes a
spurious drop-and-recreate on every run.
"""
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# `ix_` / `uq_` / `fk_` / `ck_` / `pk_` prefixes match the hand-written names used in
# the models, so a constraint declared explicitly and one derived by convention end
# up spelled identically.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for all SIS ORM models.

    Note what is absent: no `create_all` is ever called on this metadata, here or
    anywhere else in the service. Alembic owns the schema outright. A stray
    `create_all` would build tables that no migration has a record of, and the first
    real migration would then fail against a database it did not create.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
