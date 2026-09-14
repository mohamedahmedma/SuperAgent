"""`ChatAttachmentRepository` over SQLAlchemy.

Every lookup joins the owner in, the way `SqlAlchemyConversationRepository` does: an
attachment is addressed by (username, id), never by id alone, so one parent's recording
cannot be read through a URL that happens to carry its id.
"""
from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.application.ports.repositories import AttachmentRecord, NewAttachment
from backend.db.models import ChatAttachment, User


class SqlAlchemyChatAttachmentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, username: str, attachment: NewAttachment) -> AttachmentRecord | None:
        user_id = self._session.scalar(select(User.id).where(User.username == username))
        if user_id is None:
            return None
        row = ChatAttachment(
            id=attachment.id,
            user_id=user_id,
            kind=attachment.kind,
            sha256=attachment.sha256,
            storage_uri=attachment.storage_uri,
            content_type=attachment.content_type,
            byte_size=attachment.byte_size,
            duration_ms=attachment.duration_ms,
            transcript=attachment.transcript,
            transcript_status=attachment.transcript_status,
        )
        self._session.add(row)
        self._session.flush()
        return _record(row)

    def get(self, username: str, attachment_id: str) -> AttachmentRecord | None:
        row = self._session.scalars(
            self._owned_by(username).where(ChatAttachment.id == attachment_id)
        ).first()
        return None if row is None else _record(row)

    def get_many(self, username: str, attachment_ids: Sequence[str]) -> Sequence[AttachmentRecord]:
        wanted = [item for item in attachment_ids if item]
        if not wanted:
            return []
        rows = self._session.scalars(self._owned_by(username).where(ChatAttachment.id.in_(wanted)))
        return [_record(row) for row in rows]

    @staticmethod
    def _owned_by(username: str):
        return (
            select(ChatAttachment)
            .join(User, User.id == ChatAttachment.user_id)
            .where(User.username == username)
        )


def _record(row: ChatAttachment) -> AttachmentRecord:
    return AttachmentRecord(
        id=row.id,
        kind=row.kind,
        sha256=row.sha256,
        storage_uri=row.storage_uri,
        content_type=row.content_type,
        byte_size=row.byte_size,
        duration_ms=row.duration_ms,
        transcript=row.transcript,
        transcript_status=row.transcript_status,
        created_at=row.created_at,
    )


__all__ = ["SqlAlchemyChatAttachmentRepository"]
