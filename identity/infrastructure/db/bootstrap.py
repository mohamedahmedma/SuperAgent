"""The administrator this service guarantees exists.

## Why this is not a migration

The requirement is stronger than "run once when the schema changes". The account has to be
there however the database arrived: brand new, upgraded, restored from last night's backup,
or with the row deleted by somebody clearing out test data. A migration answers the first
case and none of the others.

`init_db()` already runs on every startup and its docstring says it is safe to. Seeding
rides along with it, so the guarantee is "every boot", which is what was actually asked for.
Delete the row and the next restart puts it back.

## What it will not do

**It never overwrites an existing username.** An administrator who changes their password
through the admin routes must not have it silently reset by the next deploy. The rule is
copied from `import_legacy_accounts.py`, which had to answer this same question and answered
it the same way.

**It never creates an account with a blank password.** Half-configuration — a username with
no password, or the reverse — is refused and logged rather than guessed at. A login with an
empty password is not a degraded state, it is an open door.

## The honest caveat

The password comes from the environment, so anyone who can read `.env` can sign in as this
administrator. That is a smaller surface than the shared `X-Admin-Key` it replaces — the
credential belongs to a named account whose every action is attributable, it can be changed
through the API without a redeploy, and it is subject to the same lockout policy as any
other login — but it is not zero. Treat `IDENTITY_BOOTSTRAP_ADMIN_PASSWORD` as a secret, and
change it through the admin routes once the estate is running: seeding skips the account
from then on, so the value in `.env` stops being the live credential.
"""
from __future__ import annotations

import logging

from identity.infrastructure.crypto.passwords import Pbkdf2PasswordHasher
from identity.infrastructure.db.models import Account
from identity.infrastructure.db.session import new_session

logger = logging.getLogger(__name__)

#: The role the seeded account is given. Not configurable: the entire point of this account
#: is that somebody can reach the admin routes, and any other role cannot.
_ROLE = "admin"


def seed_bootstrap_admin(*, username: str, password: str, pbkdf2_rounds: int) -> None:
    """Ensure the administrator named by `username` exists. Idempotent.

    Values are passed in rather than read from the environment here, matching the rule the
    rest of this service follows: configuration is resolved in `identity/config.py` and
    handed down. It also lets a test call this three times with three values without
    arranging an environment.

    Never raises. A service that cannot seed its administrator must still start and serve
    the parents who are already signed in — refusing to boot would turn a management
    inconvenience into an outage for the whole school.
    """
    username = (username or "").strip()
    password = password or ""

    if not username and not password:
        # Nothing configured. Silent by design: an estate that manages its administrators
        # some other way is legitimate, and a warning on every boot trains people to ignore
        # warnings. `identity/app.py` says something only when there is genuinely no way in.
        return

    if not username or not password:
        logger.warning(
            "Only half of the bootstrap administrator is configured: "
            "IDENTITY_BOOTSTRAP_ADMIN_USER is %s and IDENTITY_BOOTSTRAP_ADMIN_PASSWORD is %s. "
            "No account was created. Set both, or neither.",
            "set" if username else "not set",
            "set" if password else "not set",
        )
        return

    # `new_session()` is INSIDE the try, not before it. Opening the session is itself a
    # thing that can fail — an unreachable database, a bad URL — and this function promises
    # never to raise. With the call outside, that promise held for every failure except the
    # most likely one, and the cost was the whole service failing to boot.
    db = None
    try:
        db = new_session()
        existing = db.query(Account).filter(Account.username == username).first()
        if existing is not None:
            # The ordinary steady state, on every boot after the first. Logged at debug so
            # it does not become noise, but logged, because "why did my password change"
            # and "why did it not" are both questions someone eventually asks.
            logger.debug(
                "Bootstrap administrator %r already exists; left untouched.", username
            )
            return

        hasher = Pbkdf2PasswordHasher(rounds=pbkdf2_rounds)
        db.add(
            Account(
                username=username,
                password_hash=hasher.hash(password),
                role=_ROLE,
                display_name=username,
                is_active=True,
            )
        )
        db.commit()
        logger.info(
            "Bootstrap administrator %r created. Change its password through "
            "PATCH /v1/admin/accounts/%s; seeding will not overwrite it afterwards.",
            username,
            username,
        )
    except Exception:
        # Two replicas booting together both find no row and both insert; the unique
        # constraint on username lets exactly one win. Losing that race means the account
        # exists, which is the outcome wanted — so it is rolled back and ignored rather
        # than crashing a perfectly healthy second replica.
        if db is not None:
            db.rollback()
        logger.exception(
            "Could not seed the bootstrap administrator %r. If another replica created it "
            "at the same moment this is harmless; otherwise no administrator exists yet.",
            username,
        )
    finally:
        if db is not None:
            db.close()


def has_any_admin() -> bool:
    """Whether any administrator account exists at all.

    Used only to decide whether startup should warn. Kept here rather than in a repository
    because it answers a question about the deployment, not about a use case.
    """
    # Session opened inside the try, for the same reason as above: this is called during
    # startup, and a question about whether to log a warning must never be what stops a
    # deploy. It answers True on failure — "assume an admin exists" — so an unreadable
    # database produces silence rather than a scary and possibly false warning.
    db = None
    try:
        db = new_session()
        return (
            db.query(Account.id)
            .filter(Account.role == _ROLE, Account.is_active.is_(True))
            .first()
            is not None
        )
    except Exception:
        # A table that does not exist yet, or a database that is unreachable. The caller
        # only wants to know whether to print a warning; it must not be the thing that
        # breaks the boot.
        logger.debug("Could not check for existing administrators.", exc_info=True)
        return True
    finally:
        if db is not None:
            db.close()


__all__ = ["has_any_admin", "seed_bootstrap_admin"]
