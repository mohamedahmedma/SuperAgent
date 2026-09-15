"""Chat sessions and the messages in them."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.models._columns import timestamp_column
from backend.infra.database import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_user_session"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    updated_at: Mapped[datetime] = timestamp_column(updates=True)
    created_at: Mapped[datetime] = timestamp_column()

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_ref_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = timestamp_column()
    rag_trace: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    #: The recording this message was spoken as, when it was. `content` holds the
    #: transcript — what the assistant answered and what history shows the model — and
    #: the attachment is what the parent hears back. SET NULL rather than CASCADE: a
    #: recording removed later leaves the words it carried in place.
    attachment_id: Mapped[str | None] = mapped_column(
        ForeignKey("chat_attachments.id", ondelete="SET NULL"), nullable=True
    )

    session = relationship("ChatSession", back_populates="messages")


__all__ = ["ChatMessage", "ChatSession"]
