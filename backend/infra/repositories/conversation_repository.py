"""`ConversationRepository` over SQLAlchemy: conversations addressed by username.

Nothing here commits; the unit of work that built the repository decides. Every lookup
by (username, session_id) is one join rather than a user query followed by a session
query, and the session list counts messages in the same statement that reads the
sessions — it used to issue one COUNT per conversation. Reads select the columns a
record needs rather than loading ORM entities only to copy fields out of them.

Messages are only ever added. There is no method that deletes or rewrites a stored
message, because the one caller that did — a save that replaced the conversation whenever
its copy of it disagreed with the database — is how answers, and the images on them, went
missing from conversations that were merely being continued.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import bindparam, delete, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Row
from sqlalchemy.orm import Session

from backend.application.ports.repositories import (
    NewMessage,
    SessionSummary,
    StoredMessage,
    StoredSession,
)
from backend.db.models import ChatMessage, ChatSession, User

_MESSAGE_COLUMNS = (
    ChatMessage.id,
    ChatMessage.message_type,
    ChatMessage.content,
    ChatMessage.timestamp,
    ChatMessage.rag_trace,
    ChatMessage.attachment_id,
)


class SqlAlchemyConversationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def find_session(self, username: str, session_id: str) -> StoredSession | None:
        row = self._find_row(username, session_id)
        return None if row is None else _session(row)

    def open_session(self, username: str, session_id: str, metadata: dict) -> StoredSession | None:
        row = self._find_row(username, session_id)
        if row is not None:
            return _session(row)
        user_id = self._session.scalar(select(User.id).where(User.username == username))
        if user_id is None:
            return None
        row = ChatSession(user_id=user_id, session_id=session_id, metadata_json=dict(metadata))
        self._session.add(row)
        self._session.flush()
        return _session(row)

    def patch_session(
        self, session: StoredSession, *, metadata: dict | None, updated_at: datetime
    ) -> None:
        values: dict = {"updated_at": updated_at}
        if metadata:
            # Postgres merges the two objects itself, right-hand keys winning, so the row
            # never round-trips through this process and a concurrent patch to another
            # key cannot be overwritten with a stale copy.
            values["metadata_json"] = ChatSession.metadata_json.op("||", return_type=JSONB)(
                bindparam("metadata_patch", value=dict(metadata), type_=JSONB)
            )
        self._session.execute(update(ChatSession).where(ChatSession.id == session.id).values(**values))

    def add_messages(self, session: StoredSession, messages: Sequence[NewMessage]) -> Sequence[int]:
        if not messages:
            return []
        rows = [
            ChatMessage(
                session_ref_id=session.id,
                message_type=message.message_type,
                content=message.content,
                timestamp=message.timestamp,
                rag_trace=message.rag_trace,
                attachment_id=message.attachment_id,
            )
            for message in messages
        ]
        self._session.add_all(rows)
        # One flush for the batch: ids are assigned in the order the rows were added,
        # which is the order the conversation is read back in.
        self._session.flush()
        return [row.id for row in rows]

    def messages(self, session: StoredSession) -> Sequence[StoredMessage]:
        rows = self._session.execute(
            select(*_MESSAGE_COLUMNS)
            .where(ChatMessage.session_ref_id == session.id)
            .order_by(ChatMessage.id.asc())
        )
        return [_message(row) for row in rows]

    def latest_messages(
        self, session: StoredSession, *, limit: int, before_id: int | None
    ) -> Sequence[StoredMessage]:
        statement = select(*_MESSAGE_COLUMNS).where(ChatMessage.session_ref_id == session.id)
        if before_id is not None:
            statement = statement.where(ChatMessage.id < before_id)
        rows = self._session.execute(statement.order_by(ChatMessage.id.desc()).limit(limit))
        return [_message(row) for row in rows]

    def summaries(self, username: str) -> Sequence[SessionSummary]:
        rows = self._session.execute(
            select(
                ChatSession.session_id,
                ChatSession.metadata_json,
                ChatSession.updated_at,
                func.count(ChatMessage.id).label("message_count"),
            )
            .join(User, User.id == ChatSession.user_id)
            .outerjoin(ChatMessage, ChatMessage.session_ref_id == ChatSession.id)
            .where(User.username == username)
            # Grouping by the primary key lets Postgres select the session's other columns.
            .group_by(ChatSession.id)
            .order_by(ChatSession.updated_at.desc())
        ).all()
        return [
            SessionSummary(row.session_id, dict(row.metadata_json or {}), row.updated_at, row.message_count)
            for row in rows
        ]

    def delete_session(self, username: str, session_id: str) -> bool:
        ref = self._session.scalar(
            select(ChatSession.id)
            .join(User, User.id == ChatSession.user_id)
            .where(User.username == username, ChatSession.session_id == session_id)
        )
        if ref is None:
            return False
        # The foreign key cascades to the messages in the database, so a long conversation
        # is removed without loading every message into the session first.
        self._session.execute(delete(ChatSession).where(ChatSession.id == ref))
        return True

    def _find_row(self, username: str, session_id: str) -> Row | None:
        return self._session.execute(
            select(ChatSession.id, ChatSession.session_id, ChatSession.metadata_json)
            .join(User, User.id == ChatSession.user_id)
            .where(User.username == username, ChatSession.session_id == session_id)
        ).first()


def _session(row) -> StoredSession:
    return StoredSession(id=row.id, session_id=row.session_id, metadata=dict(row.metadata_json or {}))


def _message(row) -> StoredMessage:
    return StoredMessage(
        id=row.id,
        message_type=row.message_type,
        content=row.content,
        timestamp=row.timestamp,
        rag_trace=row.rag_trace,
        attachment_id=row.attachment_id,
    )


__all__ = ["SqlAlchemyConversationRepository"]
