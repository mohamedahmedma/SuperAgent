"""Voice notes a parent sends are stored, transcribed, and played back on any device.

A recording used to exist only in the browser tab that made it: the backend received the
literal text "[Voice recording attached]", answered that, and the player vanished when the
conversation was reopened. `chat_attachments` records each note — its bytes are in the
blob store by sha256, so a conversation grows by a row of text and never by the audio —
and `chat_messages.attachment_id` says which message was spoken as which note.

Purely additive; the release before this one runs unchanged against it.

Revision ID: 0004
Revises: 0003
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_attachments",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(length=80), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("transcript", sa.Text(), nullable=True),
        sa.Column("transcript_status", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_attachments_user_id", "chat_attachments", ["user_id"])
    op.create_index("ix_chat_attachments_user_created", "chat_attachments", ["user_id", "created_at"])
    op.add_column(
        "chat_messages",
        sa.Column(
            "attachment_id",
            sa.String(length=32),
            sa.ForeignKey("chat_attachments.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    # The recordings' bytes stay in the blob store; only the rows that point at them go.
    op.drop_column("chat_messages", "attachment_id")
    op.drop_index("ix_chat_attachments_user_created", table_name="chat_attachments")
    op.drop_index("ix_chat_attachments_user_id", table_name="chat_attachments")
    op.drop_table("chat_attachments")
