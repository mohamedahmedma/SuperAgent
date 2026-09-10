"""Database connectivity diagnosis at boot.

The failure mode these pin took production down on 2026-09-10 and cost two hours to
read. `POSTGRES_PASSWORD` is applied by the postgres image only while it initialises an
empty data directory, so on an estate whose volume already exists, rotating `.env`
changes what the backend sends and never what the role accepts. `pg_isready` does not
authenticate, so the container stayed `(healthy)` and `depends_on: service_healthy` went
green while every connection was refused — and the reason arrived as a hundred and fifty
lines of SQLAlchemy pool internals, identical in shape to an unresolvable host.

So: a permanent credential failure must be reported once, by name, with the remedy, and
never retried; a transient one must be waited for; and neither message may print the
password.
"""
import logging
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy.exc import OperationalError

from backend.infra import database

SECRET = "0c9b367728bb8ed21f3bcb6333d9690db9e6852715da8d94"
PROD_URL = f"postgresql+psycopg2://postgres:{SECRET}@postgres:5432/langchain_app"


class FakeOrig(Exception):
    """Stands in for a psycopg2 error, which carries the SQLSTATE on `pgcode`."""

    def __init__(self, message, pgcode=None):
        super().__init__(message)
        self.pgcode = pgcode


def operational_error(message, pgcode=None):
    return OperationalError("SELECT 1", {}, FakeOrig(message, pgcode))


AUTH_FAILED = 'FATAL:  password authentication failed for user "postgres"'
REFUSED = 'could not connect to server: Connection refused'


class DescribeTests(unittest.TestCase):
    def test_it_never_renders_the_password(self):
        """This string goes into logs, and a leaked password in a log is a rotation."""
        with patch.object(database, "DATABASE_URL", PROD_URL):
            described = database.describe_database()
        self.assertNotIn(SECRET, described)
        self.assertIn("password=set", described)

    def test_it_names_every_part_an_operator_has_to_check(self):
        with patch.object(database, "DATABASE_URL", PROD_URL):
            described = database.describe_database()
        for expected in ("host=postgres:5432", "db=langchain_app", "user=postgres"):
            self.assertIn(expected, described)

    def test_an_absent_password_is_called_out_rather_than_shown_as_empty(self):
        with patch.object(database, "DATABASE_URL", "postgresql+psycopg2://postgres@postgres:5432/app"):
            self.assertIn("password=ABSENT", database.describe_database())

    def test_sqlite_describes_its_file(self):
        """Pointing DATABASE_URL at SQLite is a normal thing to do while developing."""
        with patch.object(database, "DATABASE_URL", "sqlite:////app/data/dev.db"):
            self.assertEqual("sqlite file=/app/data/dev.db", database.describe_database())

    def test_the_boot_line_says_where_the_url_came_from(self):
        with patch.object(database, "DATABASE_URL", PROD_URL):
            with patch.dict("os.environ", {"DATABASE_URL": PROD_URL}):
                with self.assertLogs(database.logger, level=logging.INFO) as captured:
                    database.log_database_status()
        line = "\n".join(captured.output)
        self.assertIn("from DATABASE_URL", line)
        self.assertNotIn(SECRET, line)


class ClassificationTests(unittest.TestCase):
    def test_the_sqlstate_for_a_rejected_password_is_permanent(self):
        self.assertTrue(database._is_auth_failure(operational_error(AUTH_FAILED, "28P01")))

    def test_the_wider_authorization_sqlstate_is_permanent_too(self):
        """28000 covers a role that does not exist, which is equally not transient."""
        self.assertTrue(database._is_auth_failure(operational_error("role does not exist", "28000")))

    def test_it_falls_back_to_the_message_when_no_sqlstate_is_set(self):
        """psycopg2 leaves pgcode unset when the handshake itself failed, which is this case."""
        self.assertTrue(database._is_auth_failure(operational_error(AUTH_FAILED)))

    def test_a_refused_connection_is_not_an_auth_failure(self):
        """Postgres still starting must be waited for, not reported as a bad credential."""
        self.assertFalse(database._is_auth_failure(operational_error(REFUSED)))


class VerifyConnectivityTests(unittest.TestCase):
    def test_a_rejected_credential_is_not_retried(self):
        engine = MagicMock()
        engine.connect.side_effect = operational_error(AUTH_FAILED, "28P01")
        with patch.object(database, "engine", engine), patch.object(database, "DATABASE_URL", PROD_URL):
            with self.assertLogs(database.logger, level=logging.ERROR):
                with self.assertRaises(RuntimeError):
                    database.verify_connectivity(attempts=5, delay_seconds=0)
        self.assertEqual(1, engine.connect.call_count)

    def test_the_rejection_names_the_remedy_and_hides_the_password(self):
        engine = MagicMock()
        engine.connect.side_effect = operational_error(AUTH_FAILED, "28P01")
        with patch.object(database, "engine", engine), patch.object(database, "DATABASE_URL", PROD_URL):
            with self.assertLogs(database.logger, level=logging.ERROR) as captured:
                with self.assertRaises(RuntimeError):
                    database.verify_connectivity(attempts=1, delay_seconds=0)
        line = "\n".join(captured.output)
        self.assertIn("DATABASE CREDENTIALS REJECTED", line)
        self.assertIn("deploy/scripts/apply-env.sh", line)
        self.assertIn("host=postgres:5432", line)
        self.assertNotIn(SECRET, line)

    def test_the_psycopg2_chain_is_dropped_so_it_cannot_bury_the_diagnosis(self):
        engine = MagicMock()
        engine.connect.side_effect = operational_error(AUTH_FAILED, "28P01")
        with patch.object(database, "engine", engine), patch.object(database, "DATABASE_URL", PROD_URL):
            with self.assertLogs(database.logger, level=logging.ERROR):
                with self.assertRaises(RuntimeError) as raised:
                    database.verify_connectivity(attempts=1, delay_seconds=0)
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)

    def test_a_transient_failure_is_retried_and_then_succeeds(self):
        engine = MagicMock()
        engine.connect.side_effect = [operational_error(REFUSED), MagicMock()]
        with patch.object(database, "engine", engine), patch.object(database, "DATABASE_URL", PROD_URL):
            with self.assertLogs(database.logger, level=logging.WARNING):
                database.verify_connectivity(attempts=3, delay_seconds=0)
        self.assertEqual(2, engine.connect.call_count)

    def test_a_transient_failure_that_never_clears_gives_up_and_says_so(self):
        engine = MagicMock()
        engine.connect.side_effect = operational_error(REFUSED)
        with patch.object(database, "engine", engine), patch.object(database, "DATABASE_URL", PROD_URL):
            with self.assertLogs(database.logger, level=logging.ERROR) as captured:
                with self.assertRaises(RuntimeError):
                    database.verify_connectivity(attempts=3, delay_seconds=0)
        self.assertEqual(3, engine.connect.call_count)
        self.assertIn("DATABASE UNREACHABLE", "\n".join(captured.output))

    def test_a_reachable_database_logs_nothing_and_raises_nothing(self):
        engine = MagicMock()
        with patch.object(database, "engine", engine), patch.object(database, "DATABASE_URL", PROD_URL):
            database.verify_connectivity(attempts=3, delay_seconds=0)
        engine.connect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
