"""Voice notes: stored once, transcribed, and played back on whatever device opens the chat.

A recording used to live only in the browser tab that made it — an object URL in memory —
while the backend received the literal text "[Voice recording attached]" and answered
that. The player vanished the moment the conversation was reopened, and the assistant
never heard a word.

`ChatAttachments` is the one place a recording is accepted. It checks what was sent, keeps
the bytes in the blob store under their sha256 (beside the knowledge base's images, and
for the same reasons: content-addressed, deduplicated, never in a Postgres row), asks the
transcriber what was said, and records the note against its owner. The transcript is what
the client sends as the message and what the assistant answers; the note is what the
parent hears back.

Ownership is checked on every read. A note's URL carries its id, and that id resolves
only for the account that sent it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence
from uuid import uuid4

from backend.application.ports.repositories import AttachmentRecord, NewAttachment
from backend.application.ports.unit_of_work import UnitOfWorkFactory
from backend.assets.blobs import BlobStore
from backend.assets.dossier import compute_sha256
from backend.chat.transcription import Transcriber

logger = logging.getLogger(__name__)


class VoiceNoteRejected(ValueError):
    """A recording this service will not keep. `reason` is one of the constants below and
    is what the client is told; the message is for a person reading a log."""

    TOO_LARGE = "too_large"
    TOO_LONG = "too_long"
    UNSUPPORTED_TYPE = "unsupported_type"
    EMPTY = "empty"

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class VoiceNoteLimits:
    """What a recording may be. Generous against what the web app records — a minute at
    32 kbit/s is under 300 KB — and firm against what it does not: nobody sends an hour
    of audio by mistake, and a limit is what stops the blob store being a file host."""

    max_bytes: int = 8 * 1024 * 1024
    max_duration_ms: int = 5 * 60 * 1000
    #: What browsers record, and nothing else. `audio/webm` (Chrome, Firefox), `audio/ogg`
    #: (Firefox), `audio/mp4` (Safari); the rest are what a native client would send.
    content_types: frozenset = frozenset({
        "audio/webm", "audio/ogg", "audio/mp4", "audio/x-m4a", "audio/aac", "audio/mpeg",
        "audio/wav", "audio/x-wav",
    })


class ChatAttachments:
    VOICE = "voice"

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWorkFactory,
        blob_store: BlobStore,
        transcriber: Transcriber,
        limits: VoiceNoteLimits | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._blobs = blob_store
        self._transcriber = transcriber
        self._limits = limits or VoiceNoteLimits()

    @property
    def limits(self) -> VoiceNoteLimits:
        """What this service accepts, for the route that reads the upload to that size."""
        return self._limits

    def store_voice_note(
        self, username: str, data: bytes, content_type: str, duration_ms: int = 0
    ) -> AttachmentRecord:
        """Keep a recording for `username` and say what it said.

        Raises `VoiceNoteRejected` for a recording outside the limits, and `LookupError`
        for a user this backend has no row for — which a signed-in caller never is, since
        authentication creates one.
        """
        content_type = _bare(content_type)
        self._check(data, content_type, duration_ms)

        digest = compute_sha256(data)
        # Idempotent on the digest: the same recording sent twice is stored once.
        uri = self._blobs.put(digest, data, content_type)
        transcript = self._transcriber.transcribe(data, content_type)

        note = NewAttachment(
            id=uuid4().hex,
            kind=self.VOICE,
            sha256=digest,
            storage_uri=uri,
            content_type=content_type,
            byte_size=len(data),
            duration_ms=max(0, int(duration_ms or 0)),
            transcript=transcript.text or None,
            transcript_status=transcript.status,
        )
        with self._unit_of_work() as uow:
            record = uow.attachments.add(username, note)
            if record is None:
                raise LookupError(f"no such user: {username!r}")
            uow.commit()
        return record

    def get(self, username: str, attachment_id: str) -> Optional[AttachmentRecord]:
        """The owner's note, or None — for a stranger's note as for a missing one."""
        if not attachment_id:
            return None
        with self._unit_of_work() as uow:
            return uow.attachments.get(username, attachment_id)

    def get_many(self, username: str, attachment_ids: Sequence[str]) -> dict[str, AttachmentRecord]:
        """The owner's notes among `attachment_ids`, by id. One query for a whole page."""
        wanted = [item for item in dict.fromkeys(attachment_ids) if item]
        if not wanted:
            return {}
        with self._unit_of_work() as uow:
            return {record.id: record for record in uow.attachments.get_many(username, wanted)}

    def read_bytes(self, record: AttachmentRecord) -> bytes:
        return self._blobs.get(record.storage_uri)

    def _check(self, data: bytes, content_type: str, duration_ms: int) -> None:
        limits = self._limits
        if not data:
            raise VoiceNoteRejected(VoiceNoteRejected.EMPTY, "the recording is empty")
        if len(data) > limits.max_bytes:
            raise VoiceNoteRejected(
                VoiceNoteRejected.TOO_LARGE, f"the recording is {len(data)} bytes; the limit is {limits.max_bytes}"
            )
        if content_type not in limits.content_types:
            raise VoiceNoteRejected(
                VoiceNoteRejected.UNSUPPORTED_TYPE, f"{content_type or 'no content type'} is not a recording format this accepts"
            )
        if duration_ms and duration_ms > limits.max_duration_ms:
            raise VoiceNoteRejected(
                VoiceNoteRejected.TOO_LONG, f"the recording is {duration_ms} ms; the limit is {limits.max_duration_ms}"
            )


def _bare(content_type: str) -> str:
    """`audio/webm;codecs=opus` -> `audio/webm`: the recorder names the codec too, and the
    stored type — and the one the limits are written against — is the container."""
    return (content_type or "").split(";", 1)[0].strip().lower()


__all__ = ["ChatAttachments", "VoiceNoteLimits", "VoiceNoteRejected"]
