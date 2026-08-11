from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.chat.assets_bridge import restore_session_assets
from backend.chat.storage import ConversationStorage, storage
from backend.db.models import User
from backend.infra.auth import get_current_user
from backend.schemas import (
    MessageInfo,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)

router = APIRouter(tags=["sessions"])


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
):
    """One batch of a stored conversation, with its images made displayable again.

    Batched rather than whole: opening a chat costs the last screenful of it, and
    scrolling back asks for the batch before the oldest message on screen by passing its
    id as `before`. `has_more` says when there is nothing older left to ask for.

    Storage keeps assets as ids; a client needs renditions, so they are resolved here —
    once per batch, in a single lookup, not once per message. Capabilities are the
    browser defaults because this endpoint serves the web app; a client with other
    constraints reads the ids off the trace and calls POST /media/resolve with its own.
    """
    try:
        page = storage.get_session_page(
            current_user.username, session_id, limit=limit, before_id=before
        )
        messages = [
            MessageInfo(
                id=msg.get("id"),
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                rag_trace=msg.get("rag_trace"),
            )
            for msg in restore_session_assets(page["messages"])
        ]
        return SessionMessagesResponse(messages=messages, has_more=page["has_more"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    try:
        sessions = [SessionInfo(**item) for item in storage.list_session_infos(current_user.username)]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        deleted = storage.delete_session(current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session does not exist")
        return SessionDeleteResponse(session_id=session_id, message="Session deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
