"""Add edited_at and original_body columns for message editing.

Revision ID: 0040
Revises: 0039
"""
from alembic import op
import sqlalchemy as sa


revision = "0040"
down_revision = "0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("edited_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "chat_messages",
        sa.Column("original_body", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_messages", "original_body")
    op.drop_column("chat_messages", "edited_at")

