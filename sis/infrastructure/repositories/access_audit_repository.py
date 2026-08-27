"""`AccessAuditRepository` over SQLAlchemy: append rows, and read them back in order.

**There is no update and no delete.** Not "there is none yet" — the port does not declare
them and this class does not implement them, so the append-only rule is enforced by the
absence of a method rather than by everyone remembering. A retention policy that genuinely
needs to expire old rows should do it with a scheduled job against the table, visibly,
rather than through a method sitting here waiting to be called from a request.

Timestamps are reattached to UTC on read, like every other repository here:
`DateTime(timezone=True)` keeps no offset under SQLite, and `AccessAttempt` refuses a naive
datetime because an audit that reads as local time to whoever queries it next is an audit
that answers "when" wrongly. Everything written is UTC, so reattaching is a restoration
and not a guess.

Nothing commits; the caller's transaction boundary decides.
"""
import logging
from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from sis.domain.access import AccessAttempt, AccessReason
from sis.infrastructure.db import models

logger = logging.getLogger(__name__)


class SqlAlchemyAccessAuditRepository:
    """Access decisions, appended and read."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def record(self, attempt: AccessAttempt) -> None:
        """Append one decision. No return value: there is nothing to do with the row."""
        self._session.add(
            models.AccessAudit(
                guardian_public_id=attempt.guardian_public_id,
                student_number=attempt.student_number,
                allowed=attempt.allowed,
                reason=attempt.reason.value,
                actor=attempt.actor,
                request_id=attempt.request_id,
                created_at=attempt.at,
            )
        )
        self._session.flush()

    def recent(
        self,
        *,
        guardian_public_id: str | None = None,
        student_number: str | None = None,
        allowed: bool | None = None,
        limit: int = 100,
    ) -> Sequence[AccessAttempt]:
        """Newest first, because the question is always "what happened lately".

        Every filter is optional and they compose. `allowed=False` on its own is the
        alerting query — a run of refusals against one handle is somebody probing, and it
        is the reason denials are recorded as loudly as successes.
        """
        stmt = select(models.AccessAudit)
        if guardian_public_id:
            stmt = stmt.where(models.AccessAudit.guardian_public_id == guardian_public_id)
        if student_number:
            stmt = stmt.where(models.AccessAudit.student_number == student_number)
        if allowed is not None:
            stmt = stmt.where(models.AccessAudit.allowed.is_(allowed))
        stmt = stmt.order_by(
            models.AccessAudit.created_at.desc(), models.AccessAudit.id.desc()
        ).limit(limit)
        return [_to_domain(row) for row in self._session.scalars(stmt)]


def _as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _to_domain(record: models.AccessAudit) -> AccessAttempt:
    return AccessAttempt(
        guardian_public_id=record.guardian_public_id,
        student_number=record.student_number,
        reason=_reason(record.reason),
        at=_as_utc(record.created_at),
        actor=record.actor,
        request_id=record.request_id,
    )


def _reason(stored: str) -> AccessReason:
    """A reason this build does not know about must not make the row unreadable.

    The vocabulary is closed, but a database outlives a deployment: a row written by a
    later version, or by a migration somebody wrote by hand, would otherwise raise while
    somebody was reading an audit — the one moment when a partial answer beats an
    exception. Unknown values read as a refusal, which is the safe direction to guess.
    """
    try:
        return AccessReason(stored)
    except ValueError:
        logger.warning("access audit row carries an unknown reason %r", stored)
        return AccessReason.NO_LINK


__all__ = ["SqlAlchemyAccessAuditRepository"]
