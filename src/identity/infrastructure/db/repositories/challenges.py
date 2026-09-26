"""Verification challenges, over one request's SQLAlchemy session.

Every lookup here is by a hash or an indexed identifier. That is not incidental: this is
the webhook path, and Meta is counting the milliseconds before it decides a delivery went
unacknowledged and schedules a retry. A table scan on a table that grows by one row per
sign-in attempt is how a school's parents start receiving duplicate codes.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from identity.domain import challenges as rules
from identity.domain.accounts import as_aware
from identity.infrastructure.db.models import VerificationChallenge


class SqlChallengeRepository:
    """`ChallengeRepository` over SQLAlchemy."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        nonce: str,
        poll_secret_hash: str,
        school_code: str,
        expires_at: datetime,
    ) -> VerificationChallenge:
        challenge = VerificationChallenge(
            nonce=nonce,
            poll_secret_hash=poll_secret_hash,
            status=rules.STATUS_PENDING,
            school_code=school_code,
            expires_at=expires_at,
        )
        self._db.add(challenge)
        self._db.commit()
        return challenge

    def by_nonce(self, nonce: str) -> VerificationChallenge | None:
        return (
            self._db.query(VerificationChallenge)
            .filter(VerificationChallenge.nonce == nonce)
            .first()
        )

    def by_poll_secret_hash(self, poll_secret_hash: str) -> VerificationChallenge | None:
        return (
            self._db.query(VerificationChallenge)
            .filter(VerificationChallenge.poll_secret_hash == poll_secret_hash)
            .first()
        )

    def message_already_handled(self, message_id: str) -> bool:
        """`EXISTS`, not a fetch. The row's contents are never looked at."""
        return (
            self._db.query(VerificationChallenge.id)
            .filter(VerificationChallenge.wa_message_id == message_id)
            .first()
            is not None
        )

    def count_recent_for_phone(self, phone_e164: str, *, since: datetime) -> int:
        """Served by `ix_verification_phone_time`, which exists for this query."""
        return (
            self._db.query(VerificationChallenge.id)
            .filter(
                VerificationChallenge.guardian_phone == phone_e164,
                VerificationChallenge.created_at >= since,
            )
            .count()
        )

    def mark_code_sent(
        self,
        challenge: VerificationChallenge,
        *,
        guardian_phone: str,
        guardian_external_id: str,
        display_name: str,
        preferred_language: str,
        code_hash: str,
        message_id: str,
    ) -> None:
        challenge.guardian_phone = guardian_phone
        challenge.guardian_external_id = guardian_external_id
        challenge.display_name = display_name
        challenge.preferred_language = preferred_language
        challenge.code_hash = code_hash
        challenge.wa_message_id = message_id
        challenge.status = rules.STATUS_CODE_SENT
        self._db.commit()

    def mark_rejected(
        self, challenge: VerificationChallenge, *, reason: str, message_id: str = ""
    ) -> None:
        challenge.status = rules.STATUS_REJECTED
        challenge.reason = reason
        if message_id:
            challenge.wa_message_id = message_id
        self._db.commit()

    def mark_verified(self, challenge: VerificationChallenge, *, at: datetime) -> None:
        challenge.status = rules.STATUS_VERIFIED
        challenge.consumed_at = at
        self._db.commit()

    def count_attempt(self, challenge: VerificationChallenge) -> int:
        challenge.attempts = (challenge.attempts or 0) + 1
        self._db.commit()
        return challenge.attempts

    def purge_expired(self, *, before: datetime) -> int:
        """Delete challenges that expired before `before`. Returns how many.

        Not called on any request path — it is here for an operator or a scheduled job.
        Nothing in this table is needed once a challenge has expired: the guardian handle
        it holds is re-derivable, and the phone number on it is PII with no reason to
        outlive the sign-in attempt that produced it.
        """
        deleted = (
            self._db.query(VerificationChallenge)
            .filter(VerificationChallenge.expires_at < before)
            .delete(synchronize_session=False)
        )
        self._db.commit()
        return int(deleted or 0)


__all__ = ["SqlChallengeRepository"]
