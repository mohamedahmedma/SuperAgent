"""Test fixtures for the identity service.

The dev signing key is written into the temp directory so a test run never touches
the developer's real `identity-dev-key.pem`, and never leaves one behind.
"""
import os
import tempfile

_TMPDIR = tempfile.mkdtemp(prefix="identity-tests-")
os.environ["IDENTITY_DATABASE_URL"] = f"sqlite:///{_TMPDIR}/test.db"
os.environ["IDENTITY_DEV_KEY_FILE"] = f"{_TMPDIR}/dev-key.pem"
os.environ["IDENTITY_ADMIN_KEY"] = "test-admin-key"
os.environ["IDENTITY_ISSUER"] = "test-identity"
os.environ["IDENTITY_AUDIENCE"] = "test-services"
# Low enough to trigger in a test without a loop of thirty requests.
os.environ["IDENTITY_MAX_FAILED_ATTEMPTS"] = "3"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from identity.app import app  # noqa: E402
from identity.db import Base, get_engine, new_session, reset_engine  # noqa: E402

ADMIN_HEADERS = {"X-Admin-Key": "test-admin-key"}

def _claim_database() -> None:
    """Point IDENTITY_DATABASE_URL back at this suite's database, and drop any engine built from another.

    Set at import above, and re-asserted here because the variable is process-global and
    this is not the only suite that wants one. pytest imports every collected module
    before running anything, so in a session covering several suites the last import
    silently owns it — and the loser fails a long way from the cause, with `no such
    table` from a server pointed at somebody else's file.

    Now that the engine is built lazily, re-asserting actually works: before, the engine
    was captured at import and no later environment change could move it.
    """
    os.environ["IDENTITY_DATABASE_URL"] = f"sqlite:///{_TMPDIR}/test.db"
    reset_engine()


@pytest.fixture()
def db():
    _claim_database()
    Base.metadata.drop_all(bind=get_engine())
    Base.metadata.create_all(bind=get_engine())
    session = new_session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def parent(client):
    """A parent account, created and bound — the normal end state of onboarding."""
    client.post(
        "/v1/admin/accounts",
        headers=ADMIN_HEADERS,
        json={"username": "0501234567", "password": "correct-horse-battery", "display_name": "Umm Layla"},
    )
    client.put(
        "/v1/admin/accounts/0501234567/guardian-binding",
        headers=ADMIN_HEADERS,
        json={"guardian_external_id": "G-1"},
    )
    return {"username": "0501234567", "password": "correct-horse-battery"}


@pytest.fixture()
def unbound_parent(client):
    """Created but not yet bound — the safe half-finished state of a bulk import."""
    client.post(
        "/v1/admin/accounts",
        headers=ADMIN_HEADERS,
        json={"username": "0509999999", "password": "correct-horse-battery"},
    )
    return {"username": "0509999999", "password": "correct-horse-battery"}
