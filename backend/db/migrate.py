"""Additive schema reconciliation.

`Base.metadata.create_all()` creates tables that are missing. It never alters tables
that already exist — so adding a column to a model that ships to a database with
existing data leaves that column absent, and the first query naming it fails at
runtime with `UndefinedColumn`. That is what happened when `parent_chunks` gained
`modality` and `asset_ids`: the new tables appeared, the new columns did not.

This module closes that gap for the only change shape that can be applied safely
without a full migration framework:

    ADD COLUMN   — applied, with a server default so existing rows backfill

Everything else is deliberately out of scope. Dropping a column, renaming one, or
changing a type are destructive or ambiguous, and guessing at them is how automated
migration tools lose data. Those need Alembic and a reviewed migration.

Usage:
    python -m backend.db.migrate            # report drift, exit 1 if any
    python -m backend.db.migrate --apply    # add the missing columns
    python -m backend.db.migrate --sql      # print the statements, change nothing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)


@dataclass
class MissingColumn:
    table: str
    column: Any  # sqlalchemy Column

    @property
    def name(self) -> str:
        return self.column.name


@dataclass
class SchemaDrift:
    missing_tables: List[str]
    missing_columns: List[MissingColumn]

    @property
    def has_drift(self) -> bool:
        return bool(self.missing_tables or self.missing_columns)

    def summary(self) -> str:
        parts = []
        if self.missing_tables:
            parts.append(f"{len(self.missing_tables)} missing table(s): {', '.join(self.missing_tables)}")
        if self.missing_columns:
            listed = ", ".join(f"{item.table}.{item.name}" for item in self.missing_columns)
            parts.append(f"{len(self.missing_columns)} missing column(s): {listed}")
        return "; ".join(parts) or "schema is up to date"


def _metadata():
    # Imported lazily so importing this module does not pull in every model.
    import backend.db.models  # noqa: F401
    from backend.infra.database import Base

    return Base.metadata


def detect_drift(engine=None) -> SchemaDrift:
    if engine is None:
        from backend.infra.database import engine as default_engine

        engine = default_engine

    metadata = _metadata()
    inspector = inspect(engine)
    live_tables = set(inspector.get_table_names())

    missing_tables: List[str] = []
    missing_columns: List[MissingColumn] = []

    for name, table in metadata.tables.items():
        if name not in live_tables:
            missing_tables.append(name)
            continue
        present = {column["name"] for column in inspector.get_columns(name)}
        for column in table.columns:
            if column.name not in present:
                missing_columns.append(MissingColumn(table=name, column=column))

    return SchemaDrift(sorted(missing_tables), missing_columns)


def _server_default(column) -> Optional[str]:
    """A SQL literal to backfill existing rows with.

    Model-level `default=` is applied by Python on INSERT, not by the database, so a
    NOT NULL column added to a populated table would fail without one of these. The
    value mirrors the model default so new and backfilled rows agree.
    """
    if column.server_default is not None:
        return None  # SQLAlchemy will render it

    default = getattr(column, "default", None)
    value: Any = None
    if default is not None and getattr(default, "arg", None) is not None:
        arg = default.arg
        if callable(arg):
            # SQLAlchemy wraps a callable default so it takes an execution context;
            # the bare callable (`default=list`) does not. Try both.
            try:
                value = arg(None)
            except TypeError:
                try:
                    value = arg()
                except Exception:
                    value = None
        else:
            value = arg

    if value is None:
        if column.nullable:
            return None
        # Fall back on a type-appropriate empty value.
        python_type = None
        try:
            python_type = column.type.python_type
        except (NotImplementedError, AttributeError):
            pass
        value = {str: "", int: 0, float: 0.0, bool: False, dict: {}, list: []}.get(python_type)
        if value is None:
            return None

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        return "'" + json.dumps(value).replace("'", "''") + "'"
    return "'" + str(value).replace("'", "''") + "'"


def generate_sql(drift: SchemaDrift, engine=None) -> List[str]:
    """ALTER statements for the missing columns, in dependency-free order."""
    if engine is None:
        from backend.infra.database import engine as default_engine

        engine = default_engine

    # `ADD COLUMN IF NOT EXISTS` is a PostgreSQL extension — SQLite and MySQL reject
    # it outright. It is only belt-and-braces anyway, since detect_drift() has already
    # established the column is absent, so it is simply omitted elsewhere.
    guard = "IF NOT EXISTS " if engine.dialect.name == "postgresql" else ""

    statements: List[str] = []
    for item in drift.missing_columns:
        column = item.column
        column_type = column.type.compile(dialect=engine.dialect)
        clause = f'ALTER TABLE {item.table} ADD COLUMN {guard}"{column.name}" {column_type}'

        default_sql = _server_default(column)
        if default_sql is not None:
            clause += f" DEFAULT {default_sql}"
        if not column.nullable:
            clause += " NOT NULL"
        statements.append(clause)
    return statements


def apply_drift(drift: SchemaDrift, engine=None) -> List[str]:
    """Create missing tables and add missing columns. Returns what was executed."""
    if engine is None:
        from backend.infra.database import engine as default_engine

        engine = default_engine

    executed: List[str] = []

    if drift.missing_tables:
        _metadata().create_all(bind=engine)
        executed.extend(f"CREATE TABLE {name}" for name in drift.missing_tables)

    statements = generate_sql(drift, engine)
    if statements:
        # One transaction: either the whole reconciliation lands or none of it does,
        # so a half-migrated schema is never left behind.
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
                executed.append(statement)
    return executed


def check_and_report(engine=None) -> SchemaDrift:
    """Called at startup. Reports drift loudly rather than letting it surface as an
    UndefinedColumn error partway through a user's upload."""
    try:
        drift = detect_drift(engine)
    except Exception:
        logger.exception("Could not inspect the database schema")
        return SchemaDrift([], [])

    if drift.has_drift:
        logger.error(
            "DATABASE SCHEMA IS OUT OF DATE — %s. "
            "create_all() cannot add columns to existing tables. "
            "Run: python -m backend.db.migrate --apply",
            drift.summary(),
        )
    return drift


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Add the missing tables and columns.")
    parser.add_argument("--sql", action="store_true", help="Print the statements without running them.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from backend.env import load_env

    load_env()

    drift = detect_drift()
    if not drift.has_drift:
        print("Schema is up to date.")
        return 0

    print(f"Schema drift detected: {drift.summary()}\n")
    for statement in generate_sql(drift):
        print(f"  {statement};")
    for name in drift.missing_tables:
        print(f"  CREATE TABLE {name} (via create_all);")

    if args.sql:
        return 1
    if not args.apply:
        print("\nRe-run with --apply to make these changes.")
        return 1

    executed = apply_drift(drift)
    print(f"\nApplied {len(executed)} change(s).")
    remaining = detect_drift()
    if remaining.has_drift:
        print(f"WARNING: drift remains after applying: {remaining.summary()}")
        return 1
    print("Schema is now up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
