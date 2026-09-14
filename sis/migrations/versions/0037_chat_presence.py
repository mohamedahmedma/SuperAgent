"""Add cross-page chat presence and typing state."""
from alembic import op
import sqlalchemy as sa


revision = "0037"
down_revision = "0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_presence",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("school_id", sa.Integer(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("typing_conversation_id", sa.Integer(), nullable=True),
        sa.Column("typing_until", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["school_id"], ["schools.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["typing_conversation_id"], ["chat_conversations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index("ix_chat_presence_school_seen", "chat_presence", ["school_id", "last_seen_at"])
    op.create_index("ix_chat_presence_typing", "chat_presence", ["typing_conversation_id", "typing_until"])


def downgrade() -> None:
    op.drop_index("ix_chat_presence_typing", table_name="chat_presence")
    op.drop_index("ix_chat_presence_school_seen", table_name="chat_presence")
    op.drop_table("chat_presence")
