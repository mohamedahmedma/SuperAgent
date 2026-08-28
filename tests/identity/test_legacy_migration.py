"""Migrating off the old chat backend's auth.

The property these protect: **nobody has to choose a new password.** A migration that
forced a reset on every family would have been abandoned halfway, leaving the old auth
running forever — which is exactly the outcome this whole task exists to avoid.
"""
import os
import tempfile

import pytest
from sqlalchemy import create_engine, text

from identity.import_legacy_accounts import import_accounts
from identity.config import settings
from identity.infrastructure.crypto.passwords import PBKDF2_PREFIX, Pbkdf2PasswordHasher
from identity.infrastructure.db.models import Account
from identity.infrastructure.db.session import new_session
from tests.identity.conftest import ADMIN_HEADERS, use_setting

PASSWORD = "correct-horse-battery"

#: The hasher the running service would build, at the suite's round count.
PBKDF2_ROUNDS = settings().pbkdf2_rounds
hasher = Pbkdf2PasswordHasher(rounds=PBKDF2_ROUNDS)


def _legacy_bcrypt_hash(password: str) -> str:
    """A standard bcrypt hash, as the old backend's earliest accounts would hold.

    Generated with the bcrypt library directly. passlib 1.7.4 cannot produce one
    against bcrypt 5.x — which is precisely why identity verifies these itself rather
    than routing through passlib.
    """
    import bcrypt

    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt(rounds=4)).decode("utf-8")


def _legacy_source_db(rows: list[tuple[str, str, str]]) -> str:
    """A throwaway stand-in for the old backend's database."""
    path = tempfile.mkdtemp(prefix="legacy-src-")
    url = f"sqlite:///{path}/legacy.db"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, "
                "password_hash TEXT, role TEXT)"
            )
        )
        for index, (username, password_hash, role) in enumerate(rows, start=1):
            connection.execute(
                text("INSERT INTO users VALUES (:i, :u, :p, :r)"),
                {"i": index, "u": username, "p": password_hash, "r": role},
            )
    return url


# ---------------------------------------------------------------------------
# Legacy hash verification and in-place upgrade
# ---------------------------------------------------------------------------


def test_a_legacy_bcrypt_password_still_verifies():
    legacy = _legacy_bcrypt_hash(PASSWORD)
    assert hasher.verify(PASSWORD, legacy) is True
    assert hasher.verify("wrong", legacy) is False


def test_a_legacy_hash_is_marked_for_upgrade():
    assert hasher.needs_rehash(_legacy_bcrypt_hash(PASSWORD)) is True
    assert hasher.needs_rehash(hasher.hash(PASSWORD)) is False


def test_a_weaker_pbkdf2_hash_is_marked_for_upgrade(monkeypatch):
    """Raising the round count upgrades everyone as they sign in, not just new users."""
    weak = hasher.hash(PASSWORD).replace(
        f"pbkdf2_sha256${PBKDF2_ROUNDS}$", "pbkdf2_sha256$1000$"
    )
    assert hasher.needs_rehash(weak) is True


def test_logging_in_upgrades_a_legacy_hash_in_place(client, db):
    """The migration is invisible: the user signs in normally and the hash moves."""
    db.add(
        Account(
            username="legacy-user",
            password_hash=_legacy_bcrypt_hash(PASSWORD),
            role="user",
        )
    )
    db.commit()

    response = client.post("/v1/auth/login", json={"username": "legacy-user", "password": PASSWORD})
    assert response.status_code == 200

    fresh = new_session()
    try:
        stored = fresh.query(Account).filter(Account.username == "legacy-user").first().password_hash
    finally:
        fresh.close()

    assert stored.startswith(PBKDF2_PREFIX)
    # And the upgraded hash still accepts the same password.
    assert hasher.verify(PASSWORD, stored) is True


def test_a_wrong_password_does_not_upgrade_anything(client, db):
    legacy = _legacy_bcrypt_hash(PASSWORD)
    db.add(Account(username="legacy-two", password_hash=legacy, role="user"))
    db.commit()

    client.post("/v1/auth/login", json={"username": "legacy-two", "password": "wrong"})

    fresh = new_session()
    try:
        assert fresh.query(Account).filter(Account.username == "legacy-two").first().password_hash == legacy
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# The import script
# ---------------------------------------------------------------------------


def test_import_copies_accounts_without_resetting_passwords(client, db):
    source = _legacy_source_db(
        [
            ("old-admin", hasher.hash(PASSWORD), "admin"),
            ("old-user", _legacy_bcrypt_hash(PASSWORD), "user"),
        ]
    )
    result = import_accounts(source)
    assert result["created"] == 2

    # Both sign in with their original password, whichever format it was stored in.
    for username in ("old-admin", "old-user"):
        response = client.post("/v1/auth/login", json={"username": username, "password": PASSWORD})
        assert response.status_code == 200, username


def test_import_preserves_the_admin_role(client, db):
    source = _legacy_source_db([("kept-admin", hasher.hash(PASSWORD), "admin")])
    import_accounts(source)

    body = client.post("/v1/auth/login", json={"username": "kept-admin", "password": PASSWORD}).json()
    assert body["role"] == "admin"


def test_import_never_creates_a_guardian_binding(client, db):
    """The old system had no guardians. Inventing one during a bulk import is how a
    family's records reach the wrong household."""
    source = _legacy_source_db([("no-binding", hasher.hash(PASSWORD), "user")])
    import_accounts(source)

    body = client.post("/v1/auth/login", json={"username": "no-binding", "password": PASSWORD}).json()
    assert body["guardian_id"] is None


def test_import_is_idempotent_and_never_overwrites(client, db):
    source = _legacy_source_db([("twice", hasher.hash(PASSWORD), "user")])

    first = import_accounts(source)
    second = import_accounts(source)

    assert first["created"] == 1
    assert second["created"] == 0
    assert second["skipped"] == 1


def test_import_dry_run_writes_nothing(client, db):
    source = _legacy_source_db([("ghost", hasher.hash(PASSWORD), "user")])
    result = import_accounts(source, dry_run=True)

    assert result["created"] == 1
    assert db.query(Account).filter(Account.username == "ghost").first() is None


def test_import_does_not_promote_an_unknown_role(client, db):
    source = _legacy_source_db([("weird", hasher.hash(PASSWORD), "superuser")])
    import_accounts(source)

    body = client.post("/v1/auth/login", json={"username": "weird", "password": PASSWORD}).json()
    assert body["role"] == "user"


# ---------------------------------------------------------------------------
# Self-registration, ported from the old backend
# ---------------------------------------------------------------------------


def test_registration_creates_an_ordinary_user(client):
    response = client.post(
        "/v1/auth/register", json={"username": "newcomer", "password": PASSWORD}
    )
    assert response.status_code == 201
    assert response.json()["role"] == "user"
    assert response.json()["username"] == "newcomer"


def test_registration_cannot_produce_a_parent(client):
    """A parent role is paired with a guardian binding. Both are an admin's decision."""
    response = client.post(
        "/v1/auth/register",
        json={"username": "pretend-parent", "password": PASSWORD, "role": "parent"},
    )
    assert response.status_code == 201
    assert response.json()["role"] == "user"
    assert response.json()["guardian_id"] is None


def test_registration_as_admin_requires_the_invite_code(client, monkeypatch):
    use_setting(monkeypatch, "IDENTITY_ADMIN_INVITE_CODE", "the-secret-code")

    ok = client.post(
        "/v1/auth/register",
        json={"username": "real-admin", "password": PASSWORD, "role": "admin", "admin_code": "the-secret-code"},
    )
    assert ok.status_code == 201
    assert ok.json()["role"] == "admin"


def test_a_wrong_invite_code_is_refused_not_downgraded(client, monkeypatch):
    """The old backend silently handed out a plain account on a mistyped code.

    An operator then gets an ordinary login, no explanation, and files a bug against
    the wrong system.
    """
    use_setting(monkeypatch, "IDENTITY_ADMIN_INVITE_CODE", "the-secret-code")

    response = client.post(
        "/v1/auth/register",
        json={"username": "nope", "password": PASSWORD, "role": "admin", "admin_code": "guess"},
    )
    assert response.status_code == 403


def test_admin_registration_is_impossible_when_no_code_is_configured(client, monkeypatch):
    """Unset invite code must mean "no self-service admin", never "no check"."""
    use_setting(monkeypatch, "IDENTITY_ADMIN_INVITE_CODE", "")

    response = client.post(
        "/v1/auth/register",
        json={"username": "sneaky", "password": PASSWORD, "role": "admin", "admin_code": "anything"},
    )
    assert response.status_code == 403


def test_registration_rejects_a_duplicate_username(client):
    client.post("/v1/auth/register", json={"username": "dup", "password": PASSWORD})
    again = client.post("/v1/auth/register", json={"username": "dup", "password": PASSWORD})
    assert again.status_code == 409


def test_registration_rejects_empty_credentials(client):
    response = client.post("/v1/auth/register", json={"username": "  ", "password": "  "})
    assert response.status_code == 400
