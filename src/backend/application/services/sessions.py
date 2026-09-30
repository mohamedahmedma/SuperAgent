"""A user's conversations: list them, open one a batch at a time, delete one.

What the three session routes used to do inline, now callable without HTTP. The routes keep
only what is HTTP's: reading the request, and shaping the answer into the response models.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.application.ports.collaborators import (
    AssetRestorer,
    AttachmentLookup,
    ConversationStore,
)
from backend.application.ports.repositories import AttachmentRecord
from backend.domain.errors import NotFound, operation


@dataclass(frozen=True)
class SessionPage:
    """One batch of a stored conversation, ready to display.

    `messages` are the stored records with their images resolved to displayable references.
    `attachments` holds the voice notes those messages were spoken as, by id - looked up once
    for the batch, not once per message.
    """

    messages: list[dict]
    attachments: dict[str, AttachmentRecord]
    has_more: bool


class SessionService:
    """Conversations, for the account that owns them.

    Every method takes the user id first and never returns another account's data: storage
    is keyed on (user, session), and a voice note resolves only for its owner.
    """

    def __init__(
        self,
        *,
        conversations: ConversationStore,
        attachments: AttachmentLookup,
        restore_assets: AssetRestorer,
    ) -> None:
        self._conversations = conversations
        self._attachments = attachments
        self._restore_assets = restore_assets

    def list_sessions(self, user_id: str) -> list[dict]:
        """Every conversation the user has, most recently active first."""
        with operation():
            infos = self._conversations.list_session_infos(user_id)
            return sorted(infos, key=lambda item: item["updated_at"], reverse=True)

    def page(self, user_id: str, session_id: str, *, limit: int, before: int | None) -> SessionPage:
        """One batch of a conversation, with its images and voice notes made displayable.

        Batched rather than whole: opening a chat costs the last screenful of it, and
        scrolling back asks for the batch before the oldest message on screen by passing its
        id as `before`. Storage keeps assets as ids, so they are resolved here, once per
        batch; a voice note comes back the same way, owner-scoped.
        """
        with operation():
            stored = self._conversations.get_session_page(
                user_id, session_id, limit=limit, before_id=before
            )
            messages = self._restore_assets(stored["messages"])
            attachments = self._attachments.get_many(
                user_id, [message.get("attachment_id") or "" for message in messages]
            )
            return SessionPage(
                messages=messages, attachments=attachments, has_more=stored["has_more"]
            )

    def delete(self, user_id: str, session_id: str) -> None:
        """Delete a conversation. Raises `NotFound` when the user has no such conversation."""
        with operation():
            deleted = self._conversations.delete_session(user_id, session_id)
        if not deleted:
            raise NotFound("Session does not exist")


__all__ = ["SessionPage", "SessionService"]
