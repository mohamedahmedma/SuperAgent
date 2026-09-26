"""Voice notes over HTTP.

    POST /chat/attachments        the recording in, its record and transcript out
    GET  /chat/attachments/{id}   the bytes, for the owner

Under `/chat` rather than `/media`: `/media` is the knowledge base's images, which every
signed-in user may read, and a recording is one parent's and nobody else's. The prefix also
keeps both routes inside the paths the frontend's proxies already forward.

The upload answers only once the note is stored AND transcribed, because the transcript
is what the client sends as the message a moment later. That makes this the one request
in the chat that waits on a model call, and it is the right one: nothing can be asked
until the words are known, and the parent is watching a recording they just made turn
into text.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile

from backend.api.deps import get_services
from backend.application.ports.repositories import AttachmentRecord
from backend.chat.attachments import VoiceNoteRejected
from backend.composition import Services
from backend.infra.auth import AuthenticatedUser, get_current_user
from backend.schemas import AttachmentInfo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["attachments"])

#: Where a note's bytes are fetched from. One spelling, used by both routes below and by
#: the sessions page that restores a conversation's notes.
ATTACHMENT_PATH = "/chat/attachments/{attachment_id}"

#: The HTTP status each refusal maps to. 413 and 415 are what they say; the rest is 400.
_REJECTION_STATUS = {
    VoiceNoteRejected.TOO_LARGE: 413,
    VoiceNoteRejected.UNSUPPORTED_TYPE: 415,
}


def attachment_info(record: AttachmentRecord) -> AttachmentInfo:
    """`AttachmentRecord` as the API shows it. The one mapping, so the upload response
    and a reopened conversation describe a note the same way."""
    return AttachmentInfo(
        id=record.id,
        kind=record.kind,
        url=ATTACHMENT_PATH.format(attachment_id=record.id),
        content_type=record.content_type,
        byte_size=record.byte_size,
        duration_ms=record.duration_ms,
        transcript=record.transcript,
        transcript_status=record.transcript_status,
    )


@router.post("/chat/attachments", response_model=AttachmentInfo, status_code=201)
async def upload_attachment(
    file: UploadFile = File(...),
    duration_ms: int = Form(0),
    current_user: AuthenticatedUser = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> AttachmentInfo:
    """Store a voice note and transcribe it.

    Read to one byte past the limit and no further, so an oversized upload is refused
    without being held in memory whole. Storage, hashing and the transcription call all
    block, so they run on a worker thread: the event loop is streaming other parents'
    answers meanwhile.
    """
    limit = services.attachments.limits.max_bytes
    data = await file.read(limit + 1)
    try:
        record = await asyncio.to_thread(
            services.attachments.store_voice_note,
            current_user.username,
            data,
            file.content_type or "",
            duration_ms,
        )
    except VoiceNoteRejected as exc:
        raise HTTPException(
            status_code=_REJECTION_STATUS.get(exc.reason, 400),
            detail={"code": exc.reason, "message": str(exc)},
        ) from None
    except LookupError:
        # Authentication creates the row this needs, so this is a caller the backend has
        # never served — refused as such rather than reported as a server fault.
        raise HTTPException(status_code=403, detail="This account cannot send attachments.") from None
    return attachment_info(record)


@router.get(ATTACHMENT_PATH)
async def get_attachment_bytes(
    attachment_id: str,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Response:
    """The recording, for the account that sent it.

    Content-addressed and therefore immutable, so the digest is a strong ETag and the
    response is cached privately for as long as the client likes. Someone else's note is
    a 404, indistinguishable from a note that does not exist.
    """
    record = await asyncio.to_thread(services.attachments.get, current_user.username, attachment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    etag = f'"{record.sha256}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})

    try:
        data = await asyncio.to_thread(services.attachments.read_bytes, record)
    except FileNotFoundError:
        logger.error("Blob missing for attachment %s (%s)", attachment_id, record.storage_uri)
        raise HTTPException(status_code=404, detail="The recording is no longer available") from None

    return Response(
        content=data,
        media_type=record.content_type or "application/octet-stream",
        headers={
            "ETag": etag,
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "inline",
        },
    )


__all__ = ["ATTACHMENT_PATH", "attachment_info", "router"]
