"""Throwaway Postgres schemas, for tests that exercise the backend's real SQL.

The backend runs on Postgres and nothing else, so its persistence tests do too. They used
to run on in-memory SQLite, which accepts what Postgres refuses — a string longer than
its VARCHAR, a NOT NULL column added to a populated table — so a green run said little
about production.

Each test gets its own empty schema, created during setUp and dropped on cleanup, and an
engine whose connections resolve unqualified names inside it. Tests never share rows,
run in any order, and leave nothing behind in the database they borrow.

## Where the database comes from

`TEST_DATABASE_URL`, or a local container on port 55432 when that is unset:

    docker run -d --name superagent-test-postgres -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=superagent_test postgres:15.8-alpine

CI starts the same image as a service; it is the image production's compose file runs.

Unreachable is an error, not a skip. `integration_support` skips because it borrows a
live estate a developer may reasonably not be running; these tests have no other way to
run, so skipping them would turn "no database" into a green build that tested nothing.
"""
from __future__ import annotations

import os
import unittest
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import sort_tables

DATABASE_URL_DEFAULT = "postgresql+psycopg2://postgres:postgres@localhost:55432/superagent_test"


def database_url() -> str:
    return (os.getenv("TEST_DATABASE_URL") or "").strip() or DATABASE_URL_DEFAULT


class PostgresSchema:
    """One empty schema, and an engine confined to it."""

    def __init__(self, *tables):
        self.name = f"test_{uuid.uuid4().hex[:12]}"
        self._url = database_url()
        self._sessions = None
        self._admin(f'CREATE SCHEMA "{self.name}"')
        from backend.infra.database import serialize_json

        self.engine: Engine = create_engine(
            self._url,
            connect_args={
                "options": f"-csearch_path={self.name}",
                # How `drop` finds the connections a test left open.
                "application_name": self.name,
            },
            # The application's serialiser, so JSON columns behave as they do in production.
            json_serializer=serialize_json,
        )
        try:
            # Models or tables; created parents first so foreign keys resolve.
            for table in sort_tables([getattr(item, "__table__", item) for item in tables]):
                table.create(self.engine)
        except Exception:
            self.drop()
            raise

    def sessionmaker(self, **options) -> sessionmaker:
        options.setdefault("expire_on_commit", False)
        return sessionmaker(bind=self.engine, **options)

    def unit_of_work(self):
        """A fresh unit of work over this schema — what a service takes as its factory."""
        from backend.infra.unit_of_work import SqlAlchemyUnitOfWork

        if self._sessions is None:
            self._sessions = self.sessionmaker()
        return SqlAlchemyUnitOfWork(self._sessions)

    def drop(self) -> None:
        self.engine.dispose()
        # A session the test never closed still holds a connection, and DROP SCHEMA
        # would wait on its locks indefinitely. End those backends first.
        self._admin(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE application_name = :name AND pid <> pg_backend_pid()",
            f'DROP SCHEMA IF EXISTS "{self.name}" CASCADE',
            name=self.name,
        )

    def _admin(self, *statements: str, **params) -> None:
        engine = create_engine(self._url, poolclass=NullPool)
        try:
            with engine.begin() as connection:
                for statement in statements:
                    connection.execute(text(statement), params)
        except OperationalError as exc:
            raise RuntimeError(_unreachable(self._url, exc)) from None
        finally:
            engine.dispose()


def postgres_schema(test: unittest.TestCase, *tables) -> PostgresSchema:
    """A fresh schema holding `tables`, dropped when `test` finishes."""
    schema = PostgresSchema(*tables)
    test.addCleanup(schema.drop)
    return schema


def _unreachable(url: str, exc: OperationalError) -> str:
    target = make_url(url)
    return (
        f"Postgres for the tests is not reachable at {target.host}:{target.port} "
        f"db={target.database}: {getattr(exc, 'orig', None) or exc}. Start one with "
        "`docker run -d --name superagent-test-postgres -p 55432:5432 "
        "-e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=superagent_test postgres:15.8-alpine`, "
        "or point TEST_DATABASE_URL at another Postgres."
    )
