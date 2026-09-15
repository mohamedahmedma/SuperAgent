"""Add chat attachments plus per-recipient delivery and read receipts."""
from alembic import op
import sqlalchemy as sa


revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # An attachment-only message legitimately has an empty caption. The API enforces
    # that every message has either text or at least one persisted attachment.
    with op.batch_alter_table("chat_messages") as batch:
        batch.drop_constraint("ck_chat_messages_body_not_blank", type_="check")

    op.create_table(
        "chat_attachments",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=12), nullable=False),
        sa.Column("file_key", sa.String(length=128), nullable=False, unique=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], ondelete="CASCADE"),
        sa.CheckConstraint("kind IN ('image', 'file', 'audio')", name="ck_chat_attachments_kind"),
        sa.CheckConstraint("size_bytes > 0", name="ck_chat_attachments_size"),
    )
    op.create_index("ix_chat_attachments_message_id", "chat_attachments", ["message_id"])

    op.create_table(
        "chat_receipts",
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("message_id", "user_id"),
    )
    op.create_index("ix_chat_receipts_user_delivery", "chat_receipts", ["user_id", "delivered_at"])


def downgrade() -> None:
    op.drop_index("ix_chat_receipts_user_delivery", table_name="chat_receipts")
    op.drop_table("chat_receipts")
    op.drop_index("ix_chat_attachments_message_id", table_name="chat_attachments")
    op.drop_table("chat_attachments")
    op.execute(sa.text(
        "UPDATE chat_messages SET body = '[attachment removed]' WHERE length(trim(body)) = 0"
    ))
    with op.batch_alter_table("chat_messages") as batch:
        batch.create_check_constraint("ck_chat_messages_body_not_blank", "length(trim(body)) > 0")
