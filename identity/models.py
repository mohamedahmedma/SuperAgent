"""Tables the identity service owns.

Small on purpose. Identity is credentials plus one binding — it is not a profile
store, and every field added here is a field two other services start wanting to
read directly instead of through a token.

The important row is `Account.guardian_external_id`. That single column is the
mapping from "someone who logged in" to "a guardian the records facade will answer
about", and it exists in exactly one place in the whole system. Nothing else may
derive it, infer it, or accept it from a request body.
"""
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from identity.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Account(Base):
    """A login. Possibly a parent, possibly staff.

    `guardian_external_id` is set only for parents, and only by an administrator
    through the admin route — never at self-registration. A parent who could choose
    their own guardian id at signup could read any family's records, which is why
    there is no self-service path that writes this column.

    It is nullable because staff accounts legitimately have none, and because a
    parent account may exist before the registrar has bound it. An account with no
    binding simply cannot read records: the claim is absent, and the records facade
    rejects a token without it.
    """

    __tablename__ = "accounts"
    __table_args__ = (
        UniqueConstraint("username", name="uq_account_username"),
        Index("ix_accounts_guardian", "guardian_external_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Phone is the realistic login for a parent in a school; username is kept generic
    # so staff and integration accounts fit the same table.
    username: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    phone: Mapped[str] = mapped_column(String(32), default="", nullable=False, index=True)

    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # "parent" or "staff". Only "parent" carries a guardian binding.
    role: Mapped[str] = mapped_column(String(20), default="parent", nullable=False)

    # The whole point of this service. See the class docstring.
    guardian_external_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    display_name: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    preferred_language: Mapped[str] = mapped_column(String(8), default="ar", nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Set by the lockout policy in auth.py. Stored rather than held in memory so a
    # restart does not clear an attacker's lockout.
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class RefreshToken(Base):
    """A long-lived credential exchangeable for short-lived access tokens.

    Access tokens are deliberately short — they are bearer credentials that several
    services accept, and they cannot be revoked once issued because verification is
    offline by design. The revocation story lives here instead: delete or revoke the
    refresh token and the session dies at the next exchange, within the access
    token's lifetime rather than immediately.

    Stored hashed. A leaked database must not yield usable sessions.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)


class AuthAudit(Base):
    """Append-only log of authentication attempts.

    Separate from the records facade's access audit and answering a different
    question: that one records who read a child's grades, this one records who tried
    to become someone. Both are needed, and keeping them in separate services means
    neither can be quietly edited from the other.

    Failures are the interesting rows. A burst against one username is a credential
    stuffing attempt, and it is invisible if only successes are written.
    """

    __tablename__ = "auth_audit"
    __table_args__ = (Index("ix_auth_audit_username_time", "username", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    event: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    # "ok", "bad_password", "unknown_user", "locked", "inactive", "expired_refresh".
    reason: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    succeeded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    client_ip: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
