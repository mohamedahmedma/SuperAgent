"""The administrator this service guarantees exists.

Before this, a deployment with neither IDENTITY_ADMIN_KEY nor IDENTITY_ADMIN_INVITE_CODE set
had no way to create an administrator at all: `/v1/auth/register` with role=admin raised
Forbidden, and every `/v1/admin/*` route answered 503. Both variables were absent from `.env`
AND from `.env.example`, so an operator could not discover them either. The estate shipped
with no door.

The account is seeded on EVERY boot rather than by a migration, and the difference is the
point: a migration runs once per version, while the requirement is that the administrator is
there however the database arrived — fresh, upgraded, restored from a backup, or with the row
deleted by somebody clearing out test data.

The rule that makes that safe to do repeatedly is that seeding never overwrites. These tests
exist mostly to pin that: a password changed through the admin routes must survive the next
deploy, or the feature is a scheduled credential reset.
"""
import os
import unittest
from unittest.mock import patch

import identity.infrastructure.db.bootstrap as bootstrap_module
from identity.config import reset_settings, settings
from identity.infrastructure.crypto.passwords import Pbkdf2PasswordHasher
from identity.infrastructure.db.base import Base
from identity.infrastructure.db.bootstrap import has_any_admin, seed_bootstrap_admin
from identity.infrastructure.db.models import Account
from identity.infrastructure.db.session import get_engine, new_session

from tests.identity.conftest import _claim_database

#: Cheap on purpose. These tests hash on nearly every case, and the round count is a property
#: of the deployment rather than of the rules being asserted.
ROUNDS = 2000


class _DatabaseCase(unittest.TestCase):
    """A test that needs the accounts table to exist.

    The suite's `db` fixture would do this, but pytest cannot hand a fixture to a
    `unittest.TestCase` — only autouse fixtures reach one — so the same two steps are taken
    here explicitly.

    `_claim_database` is shared with the fixture rather than copied: it re-points the engine
    at this suite's temp file, and the comment where it is defined explains why re-asserting
    matters when several suites run in one session. `create_all` and not `drop_all`, because
    this runs mid-session alongside other identity tests and wiping their rows would make
    this file's ordering everybody else's problem.
    """

    def setUp(self):
        _claim_database()
        Base.metadata.create_all(bind=get_engine())


def accounts_named(username):
    db = new_session()
    try:
        return db.query(Account).filter(Account.username == username).all()
    finally:
        db.close()


def wipe(username):
    db = new_session()
    try:
        db.query(Account).filter(Account.username == username).delete()
        db.commit()
    finally:
        db.close()


class SeedingTests(_DatabaseCase):
    """`seed_bootstrap_admin` itself."""

    USERNAME = "bootstrap-admin-under-test"

    def setUp(self):
        super().setUp()
        wipe(self.USERNAME)
        self.addCleanup(wipe, self.USERNAME)

    def seed(self, username=None, password="s3cret-password"):
        seed_bootstrap_admin(
            username=self.USERNAME if username is None else username,
            password=password,
            pbkdf2_rounds=ROUNDS,
        )

    def test_it_creates_an_administrator(self):
        """The happy path: a fresh database gets an account that can reach the admin routes."""
        self.seed()

        rows = accounts_named(self.USERNAME)
        self.assertEqual(1, len(rows))
        self.assertEqual("admin", rows[0].role)
        self.assertTrue(rows[0].is_active)

    def test_the_seeded_password_actually_verifies(self):
        """A row whose hash does not match the configured password is worse than no row.

        It looks correct in the database and refuses the one person who is supposed to be
        able to get in.
        """
        self.seed(password="a-known-password")

        hasher = Pbkdf2PasswordHasher(rounds=ROUNDS)
        stored = accounts_named(self.USERNAME)[0].password_hash
        self.assertTrue(hasher.verify("a-known-password", stored))
        self.assertFalse(hasher.verify("the-wrong-password", stored))

    def test_seeding_twice_does_not_create_a_second_account(self):
        """Every boot calls this. It must be a no-op on all of them but the first."""
        self.seed()
        self.seed()
        self.seed()

        self.assertEqual(1, len(accounts_named(self.USERNAME)))

    def test_a_changed_password_is_not_reset_by_the_next_boot(self):
        """THE rule this feature stands or falls on.

        An administrator changes their password through the admin routes. The service is
        redeployed. If seeding overwrote, the deploy would silently restore the value from
        `.env` — turning the feature into a scheduled credential reset, and one that hands
        the account back to anyone who ever read that file.
        """
        self.seed(password="the-original-password")

        hasher = Pbkdf2PasswordHasher(rounds=ROUNDS)
        db = new_session()
        try:
            account = db.query(Account).filter(Account.username == self.USERNAME).first()
            account.password_hash = hasher.hash("changed-by-the-admin")
            db.commit()
        finally:
            db.close()

        self.seed(password="the-original-password")  # the next boot

        stored = accounts_named(self.USERNAME)[0].password_hash
        self.assertTrue(hasher.verify("changed-by-the-admin", stored))
        self.assertFalse(
            hasher.verify("the-original-password", stored),
            "the deploy reset an administrator's password back to the .env value",
        )

    def test_a_deleted_account_comes_back_on_the_next_boot(self):
        """The reason this is not a migration.

        A migration runs once per version, so a row deleted afterwards stays deleted and the
        estate is locked out until somebody edits the database by hand.
        """
        self.seed()
        wipe(self.USERNAME)
        self.assertEqual([], accounts_named(self.USERNAME))

        self.seed()

        self.assertEqual(1, len(accounts_named(self.USERNAME)))

    def test_an_existing_non_admin_is_not_promoted(self):
        """Seeding must not silently escalate an ordinary account that shares the name.

        "Never overwrite" has to cover the role too, or the rule has a hole in exactly the
        direction that matters.
        """
        db = new_session()
        try:
            db.add(
                Account(
                    username=self.USERNAME,
                    password_hash=Pbkdf2PasswordHasher(rounds=ROUNDS).hash("theirs"),
                    role="user",
                    is_active=True,
                )
            )
            db.commit()
        finally:
            db.close()

        self.seed()

        self.assertEqual("user", accounts_named(self.USERNAME)[0].role)


class HalfConfiguredTests(_DatabaseCase):
    """Partial configuration is refused, never guessed at."""

    USERNAME = "half-configured-admin"

    def setUp(self):
        super().setUp()
        wipe(self.USERNAME)
        self.addCleanup(wipe, self.USERNAME)

    def test_a_username_with_no_password_creates_nothing(self):
        """An account with an empty password is not a degraded state, it is an open door."""
        with self.assertLogs(bootstrap_module.logger, level="WARNING") as logs:
            seed_bootstrap_admin(
                username=self.USERNAME, password="", pbkdf2_rounds=ROUNDS
            )

        self.assertEqual([], accounts_named(self.USERNAME))
        self.assertIn("IDENTITY_BOOTSTRAP_ADMIN_PASSWORD", "".join(logs.output))

    def test_a_password_with_no_username_creates_nothing(self):
        with self.assertLogs(bootstrap_module.logger, level="WARNING") as logs:
            seed_bootstrap_admin(username="", password="orphaned", pbkdf2_rounds=ROUNDS)

        self.assertIn("IDENTITY_BOOTSTRAP_ADMIN_USER", "".join(logs.output))

    def test_whitespace_only_is_treated_as_unset(self):
        """`.env` files collect trailing spaces. A username of spaces is not a username."""
        with self.assertLogs(bootstrap_module.logger, level="WARNING"):
            seed_bootstrap_admin(username="   ", password="orphaned", pbkdf2_rounds=ROUNDS)

    def test_neither_configured_is_silent(self):
        """An estate that manages administrators another way is legitimate.

        A warning on every boot of every such deployment trains people to ignore warnings,
        which costs more than it saves. `identity/app.py` speaks up instead, and only when
        there is genuinely no administrator at all.
        """
        with patch.object(bootstrap_module, "logger") as fake_logger:
            seed_bootstrap_admin(username="", password="", pbkdf2_rounds=ROUNDS)

            fake_logger.warning.assert_not_called()

    def test_it_never_raises(self):
        """A service that cannot seed its admin must still serve parents already signed in.

        Refusing to boot would turn a management inconvenience into a school-wide outage.
        """
        with patch.object(
            bootstrap_module, "new_session", side_effect=RuntimeError("no database")
        ):
            seed_bootstrap_admin(username="anyone", password="p", pbkdf2_rounds=ROUNDS)


class ConfigWiringTests(unittest.TestCase):
    """The settings a deployment actually sets."""

    def setUp(self):
        reset_settings()
        self.addCleanup(reset_settings)

    def resolved(self, **env):
        overrides = {
            "IDENTITY_BOOTSTRAP_ADMIN_USER": "",
            "IDENTITY_BOOTSTRAP_ADMIN_PASSWORD": "",
        }
        overrides.update(env)
        with patch.dict(os.environ, overrides, clear=False):
            reset_settings()
            return settings()

    def test_both_names_are_read(self):
        resolved = self.resolved(
            IDENTITY_BOOTSTRAP_ADMIN_USER="registrar",
            IDENTITY_BOOTSTRAP_ADMIN_PASSWORD="hunter2",
        )

        self.assertEqual("registrar", resolved.bootstrap_admin_user)
        self.assertEqual("hunter2", resolved.bootstrap_admin_password)

    def test_unset_resolves_to_empty_not_a_default(self):
        """There must be no default username or password. A default IS a known credential.

        This is the whole objection to the shared key restated: a value that ships in the
        repository is known to everyone who can read the repository.
        """
        resolved = self.resolved()

        self.assertEqual("", resolved.bootstrap_admin_user)
        self.assertEqual("", resolved.bootstrap_admin_password)


class AdminPresenceTests(_DatabaseCase):
    """`has_any_admin`, which decides whether startup warns."""

    USERNAME = "presence-check-admin"

    def setUp(self):
        super().setUp()
        wipe(self.USERNAME)
        self.addCleanup(wipe, self.USERNAME)

    def test_a_seeded_admin_is_seen(self):
        seed_bootstrap_admin(
            username=self.USERNAME, password="p4ssword", pbkdf2_rounds=ROUNDS
        )

        self.assertTrue(has_any_admin())

    def test_it_does_not_raise_on_a_broken_database(self):
        """It exists only to decide whether to print a warning.

        It must never be the thing that breaks the boot, so an unreachable database has to
        return rather than propagate.
        """
        with patch.object(
            bootstrap_module, "new_session", side_effect=RuntimeError("no database")
        ):
            self.assertTrue(has_any_admin())


if __name__ == "__main__":
    unittest.main()
