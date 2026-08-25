"""Tables the identity service owns.

Small on purpose. Identity is credentials plus one binding — it is not a profile
store, and every field added here is a field two other services start wanting to
read directly instead of through a token.

The important row is `Account.guardian_external_id`. That single column is the
mapping from "someone who logged in" to "a guardian the records facade will answer
about", and it exists in exactly one place in the whole system. Nothing else may
derive it, infer it, or accept it from a request body.

**Two authorities may write that column, and only two.** An administrator, through the
admin route; and the WhatsApp verification flow, which does not take the value from
anybody's request either — it proves control of a phone number through WhatsApp, asks the
school's own system of record which guardian that number belongs to, and writes the answer
it is given. Both paths share the property the rule is actually about: the account never
names its own guardian. A parent who could choose their own guardian id could read any
family's records, and neither path lets them near it.
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


class VerificationChallenge(Base):
    """One attempt to prove that a browser and a WhatsApp number belong to the same person.

    The flow this row exists for, in the order it happens: a browser asks for a challenge
    and gets back a `wa.me` link carrying `nonce`; the parent sends that link's pre-filled
    message from their own WhatsApp; our webhook receives it, learns the sender's number
    from WhatsApp itself, and — if the school has that number on file — replies with a
    short code; the parent types the code into the browser.

    **Two secrets, and the split is the whole security model.** `nonce` travels out through
    a URL the parent can forward, screenshot or paste, and comes back over WhatsApp;
    `poll_secret_hash` never leaves the browser that asked. Neither half alone is enough,
    which is what makes both of the obvious attacks fail:

      * Somebody sends a nonce they stole from a screenshot. The code is then delivered to
        *their* WhatsApp, but they hold no poll secret, so they cannot finish — and the
        person whose nonce it was is merely blocked, not impersonated.
      * Somebody tricks a parent into clicking a link they generated. The code goes to the
        *parent's* WhatsApp, which the attacker cannot read.

    `code_hash` rather than the code, for the reason `RefreshToken` stores a hash: a leaked
    database must not yield a usable session. `attempts` is on the row rather than on an
    account because at code-entry time there may be no account yet, and a six-digit code
    with no counter is guessed in an afternoon.

    `wa_message_id` is the WhatsApp message id of the inbound message that claimed this
    challenge. It is stored because Meta retries a webhook for up to seven days when a
    delivery is not acknowledged, so the same parent message arrives repeatedly; without
    remembering which message was already handled, one tap sends a parent several
    conflicting codes.

    Rows are consumed, never reused: `consumed_at` is set the moment a challenge mints a
    token, and a consumed challenge is dead whatever else it holds.
    """

    __tablename__ = "verification_challenges"
    __table_args__ = (
        # The webhook's only lookup: it holds a nonce parsed out of a message and nothing
        # else. Unique because two live challenges sharing one nonce would make "which
        # browser is this parent talking to" unanswerable.
        UniqueConstraint("nonce", name="uq_verification_nonce"),
        # Dedupe of Meta's retries, and of a parent who taps send twice.
        Index("ix_verification_wa_message", "wa_message_id"),
        # Sweeping expired rows, and rate-limiting by number.
        Index("ix_verification_phone_time", "guardian_phone", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: Travels out in the wa.me link and back in the parent's message. Short and typable:
    #: a parent may retype it by hand when an in-app browser swallows the link.
    nonce: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Held only by the browser that asked for this challenge. Hashed, like every other
    #: bearer credential in this service.
    poll_secret_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    #: "pending" -> "code_sent" -> "verified", or the terminal "rejected" / "expired".
    #: A string rather than an enum column for the reason the audit `reason` is: it is read
    #: far more often than it is branched on, and a new state must not need a migration in
    #: a service whose schema is built by `create_all`.
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)

    #: Filled by the webhook, from WhatsApp's own view of who sent the message — never
    #: from anything the browser said. Kept in E.164 so it matches what the school stores.
    guardian_phone: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    #: The handle the system of record gave back for that number. This is what the minted
    #: token will carry, and it is the reason the phone itself need not be kept anywhere
    #: else in this service.
    guardian_external_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    #: Her name, so the browser can greet her before she has an account. Display only.
    display_name: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    preferred_language: Mapped[str] = mapped_column(String(8), default="ar", nullable=False)

    code_hash: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    #: Which school this challenge belongs to; `""` in a single-school deployment.
    #:
    #: Set when the browser starts the challenge, because the login page already knows
    #: which school it is rendering for. It is checked again when the parent's message
    #: arrives, against the school that owns the WhatsApp number they sent it to — so two
    #: independent facts have to agree before a code is issued. That closes the
    #: multi-school shape of the attack this table's docstring already worries about: a
    #: parent talked into sending somebody else's nonce to a *different* school's number
    #: would otherwise have their identity resolved against a database they have no
    #: children in.
    school_code: Mapped[str] = mapped_column(String(16), default="", nullable=False)

    #: The inbound WhatsApp message that claimed this challenge, for retry deduplication.
    wa_message_id: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    #: Why a rejected challenge was rejected. Shown to nobody; read when a parent phones
    #: the school to say it did not work.
    reason: Mapped[str] = mapped_column(String(40), default="", nullable=False)

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
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
