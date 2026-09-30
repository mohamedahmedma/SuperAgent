"""Ports for the collaborators a service is handed, as opposed to the repositories under it.

`repositories.py` describes persistence. This file describes the next layer up: the stores and
engines the backend already has - conversation storage, voice notes, the vector index, the
document loader, the job trackers - as seen by the services that use them. Each Protocol names
only the methods a service actually calls, so a test's fake is as small as the use it stands
in for.

They exist so a service never imports the concrete class. Those classes reach, through their
own imports, the database, Redis, Milvus and the graph framework, and `.importlinter` refuses
the application layer every one of those - directly and through any chain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from backend.application.ports.repositories import AttachmentRecord

# -- conversations ------------------------------------------------------------------


class ConversationStore(Protocol):
    """A user's stored conversations. Implemented by `agent.chat.storage.ConversationStorage`."""

    def list_session_infos(self, user_id: str) -> list[dict]: ...

    def get_session_page(
        self,
        user_id: str,
        session_id: str,
        limit: int | None = None,
        before_id: int | None = None,
    ) -> dict: ...

    def delete_session(self, user_id: str, session_id: str) -> bool: ...


class AttachmentLookup(Protocol):
    """A user's voice notes, looked up by id. Implemented by `agent.chat.attachments.ChatAttachments`."""

    def get_many(
        self, username: str, attachment_ids: Sequence[str]
    ) -> dict[str, AttachmentRecord]: ...


#: Stored messages in, the same messages with their asset ids resolved to displayable
#: references out. `agent.chat.assets_bridge.restore_session_assets`.
AssetRestorer = Callable[[list[dict]], list[dict]]


__all__ = [
    "AssetRestorer",
    "AttachmentLookup",
    "ConversationStore",
]
