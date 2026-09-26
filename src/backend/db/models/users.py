"""The accounts chat conversations belong to."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.models._columns import timestamp_column
from backend.infra.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    created_at: Mapped[datetime] = timestamp_column()

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


__all__ = ["User"]
