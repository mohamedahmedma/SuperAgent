"""Files a parent sends with a message — today, voice notes."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.models._columns import timestamp_column
from backend.infra.database import Base


class ChatAttachment(Base):
    """One recording a parent sent, stored once and playable from any device.

    The bytes are NOT here. They live in the blob store under their sha256, beside the
    knowledge base's images (`backend/assets/blobs.py`), and this row keeps the pointer,
    what the file is, how long it plays, and the transcript the assistant answered. A
    conversation therefore grows by a row of text per voice note, never by the audio.

    Owned by a user, and read only by that user: the row carries the owner so the bytes
    route can refuse anyone else without loading the conversation the note belongs to.
    """

    __tablename__ = "chat_attachments"
    __table_args__ = (Index("ix_chat_attachments_user_created", "user_id", "created_at"),)

    #: A random id rather than the sha256: two parents who record the same silence must
    #: not share a row, and a URL that names one's note must not resolve for the other.
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    #: "voice" today. A string so a second kind is a value, not a migration.
    kind: Mapped[str] = mapped_column(String(20), nullable=False)

    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(80), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    #: As the recorder measured it. The player shows it before the bytes have loaded.
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: What the recording said, as the speech-to-text model heard it — the text the
    #: assistant answered. Null when transcription produced nothing.
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: "ok" | "empty" | "unavailable": whether there is a transcript, and if not, whether
    #: that is the recording's fault or the transcriber's.
    transcript_status: Mapped[str] = mapped_column(String(20), nullable=False)

    created_at: Mapped[datetime] = timestamp_column()


__all__ = ["ChatAttachment"]
