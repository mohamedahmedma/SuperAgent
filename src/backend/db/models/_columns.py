"""Column definitions every model shares."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import Mapped, mapped_column


def utcnow() -> datetime:
    """Every timestamp column is timestamptz, so every value written says it is UTC."""
    return datetime.now(UTC)


def timestamp_column(*, updates: bool = False) -> Mapped[datetime]:
    """A non-null timestamptz filled with the current UTC time, and refreshed on update if asked."""
    return mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow if updates else None,
        nullable=False,
    )


__all__ = ["timestamp_column", "utcnow"]
