"""A user's conversations over HTTP. Each route reads the request, calls `SessionService`, and
shapes the answer; the rules, and the errors, are the service's."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from backend.agent.chat.storage import ConversationStorage
from backend.agent.schemas import (
    MessageInfo,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)
from backend.api.deps import session_service
from backend.api.routes.attachments import attachment_info
from backend.application.ports import AttachmentRecord
from backend.application.services import SessionService
from backend.db.models import User
from backend.infra.auth import get_current_user

router = APIRouter(tags=["sessions"])


def _message_info(message: dict, attachments: dict[str, AttachmentRecord]) -> MessageInfo:
    attachment = attachments.get(message.get("attachment_id") or "")
    return MessageInfo(
        id=message.get("id"),
        type=message["type"],
        content=message["content"],
        timestamp=message["timestamp"],
        rag_trace=message.get("rag_trace"),
        attachment=attachment_info(attachment) if attachment is not None else None,
    )


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    limit: int = Query(
        default=ConversationStorage.DEFAULT_PAGE_SIZE,
        ge=1,
        le=ConversationStorage.MAX_PAGE_SIZE,
        description="How many messages to return, newest end first.",
    ),
    before: Optional[int] = Query(
        default=None,
        ge=1,
        description="Return the batch immediately older than this message id.",
    ),
    current_user: User = Depends(get_current_user),
    sessions: SessionService = Depends(session_service),
):
    """One batch of a stored conversation, with its images and voice notes made displayable again.

    Batched rather than whole: opening a chat costs the last screenful of it, and
    scrolling back asks for the batch before the oldest message on screen by passing its
    id as `before`. `has_more` says when there is nothing older left to ask for.

    Storage keeps assets as ids; a client needs renditions, so they are resolved here —
    once per batch, in a single lookup, not once per message. Capabilities are the
    browser defaults because this endpoint serves the web app; a client with other
    constraints reads the ids off the trace and calls POST /media/resolve with its own.
    A message spoken as a voice note carries the note the same way: one lookup per
    batch, owner-scoped, so the player comes back with the conversation.
    """
    page = sessions.page(current_user.username, session_id, limit=limit, before=before)
    return SessionMessagesResponse(
        messages=[_message_info(message, page.attachments) for message in page.messages],
        has_more=page.has_more,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    current_user: User = Depends(get_current_user),
    sessions: SessionService = Depends(session_service),
):
    infos = sessions.list_sessions(current_user.username)
    return SessionListResponse(sessions=[SessionInfo(**info) for info in infos])


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    sessions: SessionService = Depends(session_service),
):
    sessions.delete(current_user.username, session_id)
    return SessionDeleteResponse(session_id=session_id, message="Session deleted successfully")
