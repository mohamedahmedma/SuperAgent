"""Persistence interfaces, declared by the layer that *uses* them.

These live in `application/` and not in `infrastructure/` for the same reason `sis/`'s do:
the use cases own the shape of the storage they need, and the SQLAlchemy repositories are
written to fit. Inverted the other way — services importing concrete repositories — a unit
test of "does unbinding a guardian revoke her sessions" needs a database, an engine and a
schema, so the test that should take a millisecond takes a fixture, and the one that
should assert a rule ends up asserting SQL.

**Nothing here mentions a session, a transaction or a query.** The transaction boundary
belongs to whoever composes the request. A repository that committed on its own would turn
"mint a token and record the refresh" into "the refresh landed, then the audit failed".

Two conventions run through every method below, so implementers do not have to guess:

**Lookups return `None`, never raise, for an absent row.** "No such account" is an
ordinary answer at this layer; whether it is a 401 or a 404 is a decision the use case
makes with context the repository does not have — the same missing account is a 404 to an
administrator and a deliberately indistinguishable 401 to someone guessing usernames.

**Datetimes cross this boundary timezone-aware.** SQLite hands back naive values even from
an aware column, and a comparison against `now` then raises `TypeError` in development and
in tests only. Implementations re-attach UTC on the way out so no use case has to remember
to.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from identity.domain.accounts import LockoutPolicy


class Account(Protocol):
    """The account fields a use case reads and writes.

    Structural, so the SQLAlchemy model satisfies it without importing anything from this
    layer, and a test's stand-in is a `SimpleNamespace`. It is not an attempt to hide that
    an ORM exists — it is what lets `password_login.py` be read without one.
    """

    id: int
    username: str
    phone: str
    password_hash: str
    role: str
    guardian_external_id: str | None
    display_name: str
    preferred_language: str
    is_active: bool
    failed_attempts: int
    locked_until: datetime | None


class AccountRepository(Protocol):
    """Accounts, and the counters that govern signing in to one."""

    def by_username(self, username: str) -> Account | None:
        """The account, or `None`. Never raises for an absent row."""

    def by_id(self, account_id: int) -> Account | None: ...

    def create(
        self,
        *,
        username: str,
        password_hash: str,
        role: str,
        phone: str = "",
        display_name: str = "",
        preferred_language: str = "ar",
        guardian_external_id: str | None = None,
    ) -> Account:
        """Insert one account and return it, populated.

        `guardian_external_id` is accepted here for exactly one caller — the WhatsApp flow
        creating a parent from a handle the school's own records supplied. Every other
        caller passes `None`, and the admin route's separate `bind_guardian` is the only
        other way that column is ever written. See `domain/accounts.py` for why the
        account may never name its own guardian.
        """

    def set_guardian_binding(self, account: Account, guardian_external_id: str | None) -> None:
        """Write the binding. The single most sensitive write in the system."""

    def set_display_name(self, account: Account, display_name: str) -> None: ...

    def set_password_hash(self, account: Account, password_hash: str) -> None:
        """Replace a verified password's hash with one in the current format."""

    def register_success(self, account: Account) -> None:
        """Clear the failure counter and any lock."""

    def register_failure(self, account: Account, policy: LockoutPolicy, *, now: datetime) -> None:
        """Count a bad password and apply `policy`. See `LockoutPolicy.next_failure`."""


class RefreshTokenRepository(Protocol):
    """Opaque refresh tokens, stored only as hashes.

    A refresh token's only job is to be presented back to this service and looked up, so
    signing it would add verification cost and a second way to get revocation wrong.
    """

    def issue(self, *, account_id: int, token_hash: str, expires_at: datetime) -> None: ...

    def find_active(self, token_hash: str) -> tuple[int, datetime] | None:
        """`(account_id, expires_at)` for a token that is neither revoked nor expired.

        One query rather than "fetch, then check two fields in the caller". Expiry and
        revocation are both storage's business, and three call sites re-deriving that
        check is three chances to write `>=` where the others wrote `>`.
        """

    def revoke(self, token_hash: str) -> bool:
        """Revoke one token. `True` if it existed and was live; idempotent otherwise."""

    def revoke_all_for_account(self, account_id: int) -> int:
        """Revoke every live token for an account. The urgent custody path.

        Returns how many were revoked, so the admin route can say what it actually did.
        """


class ChallengeRepository(Protocol):
    """WhatsApp verification challenges.

    Every lookup below is by a hash or an indexed identifier, never by a scan: this table
    grows by one row per sign-in attempt and is read on the webhook path, where Meta is
    counting the milliseconds before it decides to retry.
    """

    def create(
        self,
        *,
        nonce: str,
        poll_secret_hash: str,
        school_code: str,
        expires_at: datetime,
    ): ...

    def by_nonce(self, nonce: str): ...

    def by_poll_secret_hash(self, poll_secret_hash: str): ...

    def message_already_handled(self, message_id: str) -> bool:
        """Has this inbound WhatsApp message already claimed a challenge?

        Meta's retries are guaranteed, not hypothetical — an unacknowledged delivery is
        replayed for up to seven days. Without this, one parent tap sends several
        conflicting codes and burns several challenges.
        """

    def count_recent_for_phone(self, phone_e164: str, *, since: datetime) -> int:
        """Challenges this number has claimed since `since`, for rate limiting.

        Keyed on the number rather than on an account, because at this point in the flow
        there may be no account — which is exactly why the login lockout cannot serve.
        """

    def mark_code_sent(
        self,
        challenge,
        *,
        guardian_phone: str,
        guardian_external_id: str,
        display_name: str,
        preferred_language: str,
        code_hash: str,
        message_id: str,
    ) -> None: ...

    def mark_rejected(self, challenge, *, reason: str, message_id: str = "") -> None: ...

    def mark_verified(self, challenge, *, at: datetime) -> None:
        """Consume the challenge.

        Consuming here rather than in the caller means a challenge cannot mint two tokens
        even if the route is called twice concurrently — the second call finds it consumed.
        """

    def count_attempt(self, challenge) -> int:
        """Increment and persist the guess counter, returning the new value.

        Counted *before* the comparison, so a crash between the two cannot hand an
        attacker a free guess.
        """


class AuditSink(Protocol):
    """Where authentication events go."""

    def write(
        self,
        *,
        username: str,
        event: str,
        reason: str,
        succeeded: bool,
        client_ip: str = "",
    ) -> None:
        """Append one event, committed on its own.

        Committed separately from whatever else the request is doing, for the same reason
        the records audit is: a failed login rolls its transaction back, and an audit that
        rolled back with it would record only the successes — which is precisely the half
        nobody needs.
        """


__all__ = [
    "Account",
    "AccountRepository",
    "AuditSink",
    "ChallengeRepository",
    "RefreshTokenRepository",
]
