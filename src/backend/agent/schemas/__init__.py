from backend.agent.schemas.chat import (
    AttachmentInfo,
    ChatRequest,
    ChatResponse,
    HitlResumeState,
    MessageInfo,
    PendingHitlState,
    RagSubTrace,
    RagTrace,
    RetrievedChunk,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)

# Auth request/response shapes are gone: login, registration and "who am I" are the
# identity service's routes now, and their schemas live with them in identity/.
# The document and asset-review models moved to backend/api/schemas/documents.py: nothing in
# the agent uses them, and the API layer is where the sibling services keep theirs.
__all__ = [
    "AttachmentInfo",
    "ChatRequest",
    "RetrievedChunk",
    "RagTrace",
    "RagSubTrace",
    "HitlResumeState",
    "PendingHitlState",
    "ChatResponse",
    "MessageInfo",
    "SessionMessagesResponse",
    "SessionInfo",
    "SessionListResponse",
    "SessionDeleteResponse",
]
