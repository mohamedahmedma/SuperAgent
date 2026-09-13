"""Persistence interfaces, declared by the code that uses them.

The arrangement follows `sis/application/ports/repositories.py`, for the reasons written
out there: the services own the shape of the storage they need; a Protocol lets a test
hand a service an in-memory fake that inherits nothing; and nothing here mentions a
session, a query or a transaction. Repositories stage work. Only the unit of work
commits (`backend/application/ports/unit_of_work.py`), so a service that writes several
things either lands all of them or none.

Repositories return the frozen records below, never ORM rows. A row carried out of a
repository is still attached to its session: reading a lazy attribute after the unit of
work has closed raises, and assigning to one is a change nobody will ever commit. A
record is only data.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


# -- conversations --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredSession:
    """A conversation: the id its messages hang off, and the metadata stored with it."""

    id: int
    session_id: str
    metadata: dict


@dataclass(frozen=True, slots=True)
class MessageHead:
    """A stored message without its body — what deciding how to save a turn needs."""

    id: int
    message_type: str
    timestamp: datetime
    rag_trace: dict | None


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: int
    message_type: str
    content: str
    timestamp: datetime
    rag_trace: dict | None


@dataclass(frozen=True, slots=True)
class NewMessage:
    message_type: str
    content: str
    timestamp: datetime
    rag_trace: dict | None


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    metadata: dict
    updated_at: datetime
    message_count: int


class ConversationRepository(Protocol):
    """A user's conversations and the messages in them, addressed by username."""

    def find_session(self, username: str, session_id: str) -> StoredSession | None:
        """The conversation, or None when the user or the conversation does not exist."""
        ...

    def open_session(self, username: str, session_id: str, metadata: dict) -> StoredSession | None:
        """The conversation, created with `metadata` if absent; None only for an unknown user."""
        ...

    def update_session(
        self, session: StoredSession, *, metadata: dict | None, updated_at: datetime
    ) -> None:
        """Replace the metadata when given, and always move `updated_at`."""
        ...

    def message_heads(self, session: StoredSession) -> Sequence[MessageHead]:
        """Every message, oldest first, without reading a single body."""
        ...

    def add_messages(self, session: StoredSession, messages: Sequence[NewMessage]) -> Sequence[int]:
        """Stage the messages in order and return the ids they were given, in that order."""
        ...

    def replace_trace(self, message_id: int, rag_trace: dict) -> None:
        ...

    def delete_messages(self, session: StoredSession) -> None:
        ...

    def messages(self, session: StoredSession) -> Sequence[StoredMessage]:
        """The whole conversation, oldest first."""
        ...

    def latest_messages(
        self, session: StoredSession, *, limit: int, before_id: int | None
    ) -> Sequence[StoredMessage]:
        """Up to `limit` messages older than `before_id` (or the newest), newest first."""
        ...

    def summaries(self, username: str) -> Sequence[SessionSummary]:
        """Every conversation the user has, most recently updated first, in one query."""
        ...

    def delete_session(self, username: str, session_id: str) -> bool:
        """Remove the conversation and its messages. False when there was nothing to remove."""
        ...
