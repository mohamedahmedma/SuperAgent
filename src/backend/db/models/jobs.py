"""Upload and delete job progress."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.models._columns import timestamp_column
from backend.infra.database import Base


class IngestJob(Base):
    """The progress of one document upload or delete, as the admin UI polls it.

    In the database rather than in process memory, so whichever worker answers the next
    poll can read the job a request started, and a restart leaves a record of what
    happened instead of turning every job into a 404.

    `steps` is the ordered list the progress bar is drawn from, one {key, label,
    percent, status, message} per step. JSON rather than a child table: it is written
    and read whole, and never queried inside.
    """

    __tablename__ = "ingest_jobs"
    __table_args__ = (
        # The list endpoint and the retention prune both ask "this kind, by age".
        Index("ix_ingest_jobs_kind_created", "kind", "created_at"),
    )

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    #: "upload" or "delete".
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Text, not String(255): a paired upload names both of its files here.
    filename: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    current_step: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    completion_step: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    steps: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = timestamp_column()
    # No `onupdate`: the job tracker owns the clock and writes this on every change. An
    # ORM default would overwrite its value whenever the new time equals the stored one,
    # and "has this job reported recently" would then be measured against two clocks.
    updated_at: Mapped[datetime] = timestamp_column()


__all__ = ["IngestJob"]
