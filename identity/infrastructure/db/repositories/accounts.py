"""Accounts and refresh tokens, over one request's SQLAlchemy session.

Each repository is bound to a session the API layer opened and will close. None of them
opens a connection, and none of them decides what a missing row means.

## Where the commits are

These repositories **do** commit, and that is a departure from `sis/`, which routes every
write through a unit of work. It is deliberate and narrow: this service's writes are
single-row and independent — record a failed attempt, issue a refresh token, clear a lock
— and there is no operation here that has to land atomically across two tables. What there
*is* is a rule that a failed login must still record its audit line, which a shared
transaction would roll back along with the failure.

If a multi-row write ever arrives — a bulk parent import that must be all-or-nothing — the
right move is `sis`'s unit of work, not a `flush()` here and a `commit()` three frames up.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.orm import Session

from identity.domain.accounts import LockoutPolicy, as_aware
from identity.infrastructure.db.models import Account, AuthAudit, RefreshToken


class SqlAccountRepository:
    """`AccountRepository` over SQLAlchemy."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def by_username(self, username: str) -> Account | None:
        return self._db.query(Account).filter(Account.username == username).first()

    def by_id(self, account_id: int) -> Account | None:
        return self._db.query(Account).filter(Account.id == account_id).first()

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
        account = Account(
            username=username,
            phone=phone,
            password_hash=password_hash,
            role=role,
            display_name=display_name,
            preferred_language=preferred_language,
            guardian_external_id=guardian_external_id,
            is_active=True,
        )
        self._db.add(account)
        self._db.commit()
        self._db.refresh(account)
        return account

    def set_guardian_binding(self, account: Account, guardian_external_id: str | None) -> None:
        account.guardian_external_id = guardian_external_id
        self._db.commit()

    def set_display_name(self, account: Account, display_name: str) -> None:
        account.display_name = display_name
        self._db.commit()

    def set_password_hash(self, account: Account, password_hash: str) -> None:
        try:
            account.password_hash = password_hash
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def register_success(self, account: Account) -> None:
        account.failed_attempts = 0
        account.locked_until = None
        self._db.commit()

    def register_failure(
        self, account: Account, policy: LockoutPolicy, *, now: datetime
    ) -> None:
        """Count a bad password and apply the policy. The rule itself is in the domain."""
        attempts, locked_until = policy.next_failure(account.failed_attempts, now=now)
        account.failed_attempts = attempts
        if locked_until is not None:
            account.locked_until = locked_until
        self._db.commit()

    # -- administration -----------------------------------------------------

    def list_page(self, *, limit: int, offset: int) -> list[Account]:
        # Ordered by id, not by username: id is immutable and unique, so a rename between
        # two pages cannot move a row across the boundary and make the pager skip it.
        return (
            self._db.query(Account)
            .order_by(Account.id)
            .limit(limit)
            .offset(offset)
            .all()
        )

    def count(self) -> int:
        return self._db.query(Account.id).count()

    def count_active_admins(self, *, excluding_id: int | None = None) -> int:
        query = self._db.query(Account.id).filter(
            Account.role == "admin", Account.is_active.is_(True)
        )
        if excluding_id is not None:
            query = query.filter(Account.id != excluding_id)
        return query.count()

    def set_role(self, account: Account, role: str) -> None:
        account.role = role
        self._db.commit()

    def set_active(self, account: Account, is_active: bool) -> None:
        account.is_active = is_active
        self._db.commit()

    def set_phone(self, account: Account, phone: str) -> None:
        account.phone = phone
        self._db.commit()

    def set_preferred_language(self, account: Account, preferred_language: str) -> None:
        account.preferred_language = preferred_language
        self._db.commit()

    def delete(self, account: Account) -> None:
        self._db.delete(account)
        self._db.commit()


class SqlRefreshTokenRepository:
    """`RefreshTokenRepository` over SQLAlchemy."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def issue(self, *, account_id: int, token_hash: str, expires_at: datetime) -> None:
        self._db.add(
            RefreshToken(
                account_id=account_id, token_hash=token_hash, expires_at=expires_at
            )
        )
        self._db.commit()

    def find_active(self, token_hash: str) -> tuple[int, datetime] | None:
        """`(account_id, expires_at)` for a token that is neither revoked nor expired.

        Expiry is compared in Python rather than in SQL, because SQLite stores these
        naive and a `WHERE expires_at > :now` against an aware parameter raises there and
        nowhere else. `as_aware` is the same fix `LockoutPolicy` applies, in the one place
        the value crosses out of storage.
        """
        record = (
            self._db.query(RefreshToken)
            .filter(RefreshToken.token_hash == token_hash)
            .first()
        )
        if record is None or record.revoked_at is not None:
            return None
        expires_at = as_aware(record.expires_at)
        if expires_at <= datetime.now(timezone.utc):
            return None
        return record.account_id, expires_at

    def revoke(self, token_hash: str) -> bool:
        record = (
            self._db.query(RefreshToken)
            .filter(RefreshToken.token_hash == token_hash)
            .first()
        )
        if record is None or record.revoked_at is not None:
            return False
        record.revoked_at = datetime.now(timezone.utc)
        self._db.commit()
        return True

    def revoke_all_for_account(self, account_id: int) -> int:
        """One UPDATE, not a loop. A parent may hold a session on every device she owns."""
        revoked = (
            self._db.query(RefreshToken)
            .filter(
                RefreshToken.account_id == account_id,
                RefreshToken.revoked_at.is_(None),
            )
            .update({"revoked_at": datetime.now(timezone.utc)}, synchronize_session=False)
        )
        self._db.commit()
        return int(revoked or 0)


class SqlAuditSink:
    """`AuditSink` over SQLAlchemy.

    Committed on its own, separately from whatever else the request is doing. A failed
    login rolls its transaction back, and an audit that rolled back with it would record
    only the successes — precisely the half nobody needs after an incident.

    Never raises. An audit line that cannot be written must not turn a successful login
    into a 500, and must not turn a refusal into a different refusal.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    def write(
        self,
        *,
        username: str,
        event: str,
        reason: str,
        succeeded: bool,
        client_ip: str = "",
    ) -> None:
        import logging

        try:
            self._db.add(
                AuthAudit(
                    username=username[:120],
                    event=event[:32],
                    reason=reason[:40],
                    succeeded=succeeded,
                    client_ip=client_ip[:64],
                )
            )
            self._db.commit()
        except Exception:  # noqa: BLE001 - see the class docstring
            self._db.rollback()
            logging.getLogger(__name__).exception(
                "Could not write an auth audit line (%s/%s)", event, reason
            )


__all__ = ["SqlAccountRepository", "SqlAuditSink", "SqlRefreshTokenRepository"]
