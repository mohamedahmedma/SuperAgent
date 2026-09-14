"""Upload and delete job progress moves out of process memory into `ingest_jobs`.

The admin UI starts a job and polls it. With the progress held in one process's memory,
a poll answered by any other worker was a 404, and a restart erased every job, finished
or not. A table makes the job readable from any worker and lets it outlive the process
that ran it.

Purely additive, so the release before this one runs unchanged against it.

Revision ID: 0003
Revises: 0002
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingest_jobs",
        sa.Column("job_id", sa.String(length=32), primary_key=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("current_step", sa.String(length=64), nullable=False),
        sa.Column("completion_step", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("total_chunks", sa.Integer(), nullable=False),
        sa.Column("processed_chunks", sa.Integer(), nullable=False),
        sa.Column("steps", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ingest_jobs_kind_created", "ingest_jobs", ["kind", "created_at"])


def downgrade() -> None:
    # Loses the job history, which is only progress reporting; no document data is here.
    op.drop_index("ix_ingest_jobs_kind_created", table_name="ingest_jobs")
    op.drop_table("ingest_jobs")
