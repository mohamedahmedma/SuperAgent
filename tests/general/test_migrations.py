"""The backend's Alembic history, run against real Postgres.

Every database the backend has run on has to reach head through these revisions: an
empty one, one `create_all()` built, and one an older `create_all()` built before some
columns existed. Each case runs in a throwaway schema and ends on the question that
matters — does the result match the models the code queries with?
"""
import subprocess
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url

import backend.db.models  # noqa: F401 — registers every table on Base.metadata
from backend.db.schema_version import (
    ALEMBIC_INI,
    UPGRADE_COMMAND,
    VERSION_TABLE,
    applied_heads,
    expected_heads,
    verify_database_is_migrated,
)
from backend.infra.database import Base
from tests.general.postgres_support import database_url, postgres_schema

REPO_ROOT = Path(__file__).resolve().parents[2]


class MigrationTestCase(unittest.TestCase):
    def setUp(self):
        self.schema = postgres_schema(self)
        self.engine = self.schema.engine

    def migrate(self, action, target):
        config = Config(str(ALEMBIC_INI))
        with self.engine.begin() as connection:
            config.attributes["connection"] = connection
            action(config, target)

    def drift(self):
        """What autogenerate would emit to make this database match the models."""
        with self.engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={
                    "compare_type": True,
                    "include_name": lambda name, kind, _parent: (
                        kind != "table" or name in Base.metadata.tables
                    ),
                },
            )
            return compare_metadata(context, Base.metadata)

    def column_type(self, table, column):
        with self.engine.connect() as connection:
            return connection.execute(
                text(
                    "SELECT data_type FROM information_schema.columns WHERE "
                    "table_schema = current_schema() AND table_name = :t AND column_name = :c"
                ),
                {"t": table, "c": column},
            ).scalar_one()

    def legacy_database(self):
        """What create_all() built: revision 0001's tables, and no revision recorded."""
        self.migrate(command.upgrade, "0001")
        with self.engine.begin() as connection:
            connection.execute(text(f'DROP TABLE "{VERSION_TABLE}"'))


class HistoryTests(unittest.TestCase):
    def test_the_history_has_exactly_one_head(self):
        """Two heads is two branches of revisions, and `upgrade head` refuses to choose."""
        heads = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_heads()
        self.assertEqual(1, len(heads), heads)


class FreshDatabaseTests(MigrationTestCase):
    def test_an_empty_database_upgrades_to_exactly_the_models(self):
        self.migrate(command.upgrade, "head")
        self.assertEqual([], self.drift())

    def test_the_revision_is_recorded_in_the_backends_own_version_table(self):
        self.migrate(command.upgrade, "head")
        tables = inspect(self.engine).get_table_names()
        self.assertIn(VERSION_TABLE, tables)
        self.assertNotIn("alembic_version", tables)
        self.assertEqual(expected_heads(), applied_heads(self.engine))

    def test_downgrading_to_base_leaves_no_tables(self):
        self.migrate(command.upgrade, "head")
        self.migrate(command.downgrade, "base")
        left = [name for name in inspect(self.engine).get_table_names() if name != VERSION_TABLE]
        self.assertEqual([], left)


class AdoptionTests(MigrationTestCase):
    """Databases that existed before alembic did."""

    def test_a_create_all_database_is_adopted_and_keeps_its_rows(self):
        self.legacy_database()
        with self.engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO users (username, password_hash, role, created_at) "
                "VALUES ('parent', 'x', 'user', '2026-09-13 09:00:00')"
            ))
            user_id = connection.execute(text("SELECT id FROM users")).scalar_one()
            connection.execute(
                text(
                    "INSERT INTO chat_sessions (user_id, session_id, metadata_json, updated_at, created_at) "
                    "VALUES (:user_id, 's1', :meta, '2026-09-13 10:00:00', '2026-09-13 09:00:00')"
                ),
                {"user_id": user_id, "meta": '{"title": "Fees"}'},
            )

        self.migrate(command.upgrade, "head")

        self.assertEqual([], self.drift())
        with self.engine.connect() as connection:
            meta, updated = connection.execute(
                text("SELECT metadata_json, updated_at FROM chat_sessions")
            ).one()
        self.assertEqual({"title": "Fees"}, meta)
        # The stored wall-clock time was UTC all along; the cast declares it, not moves it.
        self.assertEqual(datetime(2026, 9, 13, 10, 0, tzinfo=UTC), updated)

    def test_an_older_database_gains_exactly_what_it_was_missing(self):
        self.legacy_database()
        with self.engine.begin() as connection:
            connection.execute(text("ALTER TABLE parent_chunks DROP COLUMN modality, DROP COLUMN asset_ids"))
            connection.execute(text("DROP TABLE corpus_digests"))
            connection.execute(text("DROP INDEX ix_document_pairs_ar"))
            connection.execute(text("ALTER TABLE chat_sessions DROP CONSTRAINT uq_user_session"))
            connection.execute(text(
                "INSERT INTO parent_chunks (chunk_id, text, filename, file_type, file_path, page_number, "
                "parent_chunk_id, root_chunk_id, chunk_level, chunk_idx, updated_at) "
                "VALUES ('c1', 'fees', 'fees.pdf', 'pdf', '', 0, '', '', 1, 0, '2026-09-13 10:00:00')"
            ))

        self.migrate(command.upgrade, "head")

        self.assertEqual([], self.drift())
        with self.engine.connect() as connection:
            modality, asset_ids = connection.execute(
                text("SELECT modality, asset_ids FROM parent_chunks")
            ).one()
        self.assertEqual(("text", []), (modality, asset_ids))
        columns = {column["name"]: column for column in inspect(self.engine).get_columns("parent_chunks")}
        # The backfill default existed only for the rows already there; a fresh database has none.
        self.assertIsNone(columns["modality"]["default"])
        self.assertIsNone(columns["asset_ids"]["default"])


class TypeConversionTests(MigrationTestCase):
    def test_json_becomes_jsonb_and_timestamps_carry_their_zone(self):
        self.migrate(command.upgrade, "head")
        self.assertEqual("jsonb", self.column_type("chat_messages", "rag_trace"))
        self.assertEqual("timestamp with time zone", self.column_type("chat_messages", "timestamp"))

    def test_downgrading_to_the_baseline_restores_the_old_types_without_moving_an_instant(self):
        self.migrate(command.upgrade, "head")
        with self.engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO document_pairs (pair_id, title, filename_ar, filename_en, created_at, updated_at) "
                "VALUES ('p1', 'Fees', '', 'fees_en.docx', '2026-09-13 10:00:00+00', '2026-09-13 10:00:00+00')"
            ))

        self.migrate(command.downgrade, "0001")

        self.assertEqual("timestamp without time zone", self.column_type("document_pairs", "created_at"))
        self.assertEqual("json", self.column_type("section_summaries", "answers"))
        with self.engine.connect() as connection:
            created = connection.execute(text("SELECT created_at FROM document_pairs")).scalar_one()
        self.assertEqual(datetime(2026, 9, 13, 10, 0), created)


class ConcurrentUpgradeTests(MigrationTestCase):
    def test_two_processes_upgrading_together_both_succeed(self):
        """Two containers starting at once. Without the advisory lock in env.py the
        second races the first through the same DDL and dies on a duplicate table."""
        url = make_url(database_url()).set(query={"options": f"-csearch_path={self.schema.name}"})
        argv = [
            sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI),
            "-x", f"db_url={url.render_as_string(hide_password=False)}",
            "upgrade", "head",
        ]
        processes = [
            subprocess.Popen(argv, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for _ in range(2)
        ]
        for process in processes:
            _, stderr = process.communicate(timeout=240)
            self.assertEqual(0, process.returncode, stderr[-2000:])
        self.assertEqual(expected_heads(), applied_heads(self.engine))
        self.assertEqual([], self.drift())


class StartupGateTests(MigrationTestCase):
    def test_a_database_at_head_is_accepted(self):
        self.migrate(command.upgrade, "head")
        verify_database_is_migrated(self.engine)

    def test_a_database_never_migrated_is_refused_with_the_command_that_fixes_it(self):
        with self.assertRaises(RuntimeError) as raised:
            verify_database_is_migrated(self.engine)
        message = str(raised.exception)
        self.assertIn("never migrated", message)
        self.assertIn(UPGRADE_COMMAND, message)

    def test_a_database_behind_this_build_is_refused(self):
        self.migrate(command.upgrade, "0001")
        with self.assertRaises(RuntimeError) as raised:
            verify_database_is_migrated(self.engine)
        self.assertIn("applied:  0001", str(raised.exception))

    def test_the_refusal_does_not_print_the_password(self):
        with self.assertRaises(RuntimeError) as raised:
            verify_database_is_migrated(self.engine)
        password = make_url(database_url()).password
        self.assertNotIn(f":{password}@", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
