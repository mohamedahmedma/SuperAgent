"""Additive schema reconciliation.

Guards the failure this module was written for: `create_all()` creates missing tables
but never adds columns to existing ones, so a model change reaches production silently
and surfaces as `UndefinedColumn` partway through a user's upload.
"""
import unittest
from unittest.mock import patch

from sqlalchemy import JSON, Boolean, Column, Integer, MetaData, String, Table, create_engine, inspect
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import StaticPool

from backend.db.migrate import (
    SchemaDrift,
    apply_drift,
    check_and_report,
    detect_drift,
    generate_sql,
)


def make_engine():
    return create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )


class DriftFixture:
    """An 'old' table created without the columns a newer model declares."""

    def __init__(self):
        self.engine = make_engine()
        self.Base = declarative_base()

        class Widget(self.Base):
            __tablename__ = "widgets"
            id = Column(Integer, primary_key=True)
            name = Column(String(50), nullable=False)
            modality = Column(String(20), default="text", nullable=False)
            asset_ids = Column(JSON, default=list, nullable=False)
            note = Column(String(50), nullable=True)
            active = Column(Boolean, default=False, nullable=False)
            count = Column(Integer, default=0, nullable=False)

        self.Widget = Widget

        # Create only the pre-existing shape.
        old = MetaData()
        Table("widgets", old, Column("id", Integer, primary_key=True), Column("name", String(50)))
        old.create_all(self.engine)

    def drift(self):
        with patch("backend.db.migrate._metadata", return_value=self.Base.metadata):
            return detect_drift(self.engine)

    def apply(self, drift):
        with patch("backend.db.migrate._metadata", return_value=self.Base.metadata):
            return apply_drift(drift, self.engine)


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = DriftFixture()

    def test_missing_columns_on_an_existing_table_are_detected(self):
        """The exact bug: create_all left parent_chunks without its new columns."""
        drift = self.fixture.drift()
        found = {item.name for item in drift.missing_columns}
        self.assertEqual({"modality", "asset_ids", "note", "active", "count"}, found)
        self.assertEqual([], drift.missing_tables)
        self.assertTrue(drift.has_drift)

    def test_a_missing_table_is_detected_separately_from_columns(self):
        class Extra(self.fixture.Base):
            __tablename__ = "gadgets"
            id = Column(Integer, primary_key=True)

        drift = self.fixture.drift()
        self.assertIn("gadgets", drift.missing_tables)
        # A missing table is not also reported as missing every one of its columns.
        self.assertNotIn("gadgets", {item.table for item in drift.missing_columns})

    def test_an_up_to_date_schema_reports_no_drift(self):
        drift = self.fixture.drift()
        self.fixture.apply(drift)
        self.assertFalse(self.fixture.drift().has_drift)

    def test_the_summary_names_the_offending_columns(self):
        summary = self.fixture.drift().summary()
        self.assertIn("widgets.modality", summary)
        self.assertIn("missing column", summary)

    def test_an_empty_drift_summarises_cleanly(self):
        self.assertEqual("schema is up to date", SchemaDrift([], []).summary())


class SqlGenerationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = DriftFixture()
        self.statements = {
            item.name: sql
            for item, sql in zip(
                self.fixture.drift().missing_columns,
                generate_sql(self.fixture.drift(), self.fixture.engine),
            )
        }

    def test_only_add_column_is_ever_generated(self):
        """Dropping or retyping a column is destructive or ambiguous; guessing at it
        is how automated migrations lose data."""
        for sql in self.statements.values():
            with self.subTest(sql=sql):
                self.assertIn("ADD COLUMN", sql)
                for destructive in ("DROP", "ALTER COLUMN", "RENAME", "TRUNCATE"):
                    self.assertNotIn(destructive, sql.upper())

    def test_not_null_columns_get_a_server_default_so_existing_rows_backfill(self):
        """Model-level default= is applied by Python on INSERT, not by the database, so
        without this the ALTER fails on a populated table."""
        self.assertIn("DEFAULT 'text'", self.statements["modality"])
        self.assertIn("NOT NULL", self.statements["modality"])

    def test_callable_defaults_are_resolved(self):
        """`default=list` is wrapped by SQLAlchemy to take a context argument."""
        self.assertIn("DEFAULT '[]'", self.statements["asset_ids"])

    def test_boolean_and_numeric_defaults_render_as_sql_literals(self):
        self.assertIn("DEFAULT false", self.statements["active"])
        self.assertIn("DEFAULT 0", self.statements["count"])

    def test_a_nullable_column_needs_no_default(self):
        self.assertNotIn("NOT NULL", self.statements["note"])
        self.assertNotIn("DEFAULT", self.statements["note"])

    def test_column_names_are_quoted(self):
        self.assertIn('"modality"', self.statements["modality"])


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = DriftFixture()

    def test_applying_adds_the_columns(self):
        executed = self.fixture.apply(self.fixture.drift())
        self.assertEqual(5, len(executed))
        columns = {c["name"] for c in inspect(self.fixture.engine).get_columns("widgets")}
        self.assertIn("modality", columns)
        self.assertIn("asset_ids", columns)

    def test_existing_rows_are_backfilled_with_the_default(self):
        with self.fixture.engine.begin() as connection:
            from sqlalchemy import text

            connection.execute(text("INSERT INTO widgets (id, name) VALUES (1, 'old row')"))

        self.fixture.apply(self.fixture.drift())

        with self.fixture.engine.begin() as connection:
            from sqlalchemy import text

            row = connection.execute(text("SELECT modality, active FROM widgets WHERE id = 1")).one()
        self.assertEqual("text", row[0])

    def test_applying_is_idempotent(self):
        self.fixture.apply(self.fixture.drift())
        self.assertEqual([], self.fixture.apply(self.fixture.drift()))

    def test_missing_tables_are_created(self):
        class Gadget(self.fixture.Base):
            __tablename__ = "gadgets"
            id = Column(Integer, primary_key=True)

        self.fixture.apply(self.fixture.drift())
        self.assertIn("gadgets", inspect(self.fixture.engine).get_table_names())

    def test_the_orm_can_read_the_table_after_reconciliation(self):
        """The end state that matters: the query that used to raise UndefinedColumn."""
        from sqlalchemy.orm import sessionmaker

        self.fixture.apply(self.fixture.drift())
        session = sessionmaker(bind=self.fixture.engine)()
        try:
            self.assertEqual([], session.query(self.fixture.Widget).all())
        finally:
            session.close()


class StartupCheckTests(unittest.TestCase):
    def test_drift_is_reported_as_an_error_with_the_fix_command(self):
        fixture = DriftFixture()
        with patch("backend.db.migrate._metadata", return_value=fixture.Base.metadata):
            with self.assertLogs("backend.db.migrate", level="ERROR") as captured:
                drift = check_and_report(fixture.engine)
        message = "".join(captured.output)
        self.assertIn("SCHEMA IS OUT OF DATE", message)
        self.assertIn("backend.db.migrate --apply", message)
        self.assertTrue(drift.has_drift)

    def test_a_clean_schema_logs_nothing(self):
        fixture = DriftFixture()
        with patch("backend.db.migrate._metadata", return_value=fixture.Base.metadata):
            fixture.apply(fixture.drift())
            with patch("backend.db.migrate.logger") as logger:
                drift = check_and_report(fixture.engine)
        logger.error.assert_not_called()
        self.assertFalse(drift.has_drift)

    def test_an_unreachable_database_does_not_crash_startup(self):
        broken = Mock = create_engine("sqlite://")
        with patch("backend.db.migrate.inspect", side_effect=RuntimeError("no db")):
            with patch("backend.db.migrate.logger"):
                drift = check_and_report(broken)
        self.assertFalse(drift.has_drift)


class RealModelTests(unittest.TestCase):
    """Against the actual models, so a future column addition is covered too."""

    def test_the_live_metadata_reconciles_onto_an_empty_database(self):
        engine = make_engine()
        drift = detect_drift(engine)
        self.assertTrue(drift.has_drift)  # empty database: every table is missing
        apply_drift(drift, engine)
        self.assertFalse(detect_drift(engine).has_drift)

    def test_parent_chunks_declares_the_asset_columns(self):
        """Regression guard for the columns that caused the original failure."""
        from backend.db.models import ParentChunk

        columns = {column.name for column in ParentChunk.__table__.columns}
        self.assertIn("modality", columns)
        self.assertIn("asset_ids", columns)


if __name__ == "__main__":
    unittest.main()
