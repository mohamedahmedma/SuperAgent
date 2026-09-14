"""Add durable staff conversations with automatically-derived group membership."""
from alembic import op
import sqlalchemy as sa


revision = "0035"
down_revision = "0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_conversations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("school_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=12), nullable=False),
        sa.Column("group_key", sa.String(length=160), nullable=True),
        sa.Column("direct_user_one_id", sa.Integer(), nullable=True),
        sa.Column("direct_user_two_id", sa.Integer(), nullable=True),
        sa.Column("title_en", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("title_ar", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["school_id"], ["schools.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["direct_user_one_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["direct_user_two_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("school_id", "group_key", name="uq_chat_conversations_group"),
        sa.UniqueConstraint("school_id", "direct_user_one_id", "direct_user_two_id", name="uq_chat_conversations_direct_pair"),
        sa.CheckConstraint("kind IN ('group', 'direct')", name="ck_chat_conversations_kind"),
        sa.CheckConstraint(
            "(kind = 'group' AND group_key IS NOT NULL AND direct_user_one_id IS NULL AND direct_user_two_id IS NULL) OR "
            "(kind = 'direct' AND group_key IS NULL AND direct_user_one_id IS NOT NULL AND direct_user_two_id IS NOT NULL AND direct_user_one_id < direct_user_two_id)",
            name="ck_chat_conversations_shape",
        ),
    )
    op.create_index("ix_chat_conversations_school_id", "chat_conversations", ["school_id"])
    op.create_index("ix_chat_conversations_school_updated", "chat_conversations", ["school_id", "updated_at"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("sender_user_id", sa.Integer(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["conversation_id"], ["chat_conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["sender_user_id"], ["users.id"], ondelete="RESTRICT"),
        sa.CheckConstraint("length(trim(body)) > 0", name="ck_chat_messages_body_not_blank"),
        sa.CheckConstraint("length(body) <= 4000", name="ck_chat_messages_body_length"),
    )
    op.create_index("ix_chat_messages_conversation_id", "chat_messages", ["conversation_id", "id"])
    op.create_index("ix_chat_messages_sender_user_id", "chat_messages", ["sender_user_id"])

    op.create_table(
        "chat_reads",
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("last_read_message_id", sa.Integer(), nullable=True),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["conversation_id"], ["chat_conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["last_read_message_id"], ["chat_messages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("conversation_id", "user_id"),
    )

    permissions = (("chat.read", "Read staff chat"), ("chat.write", "Send staff chat messages"))
    roles = ("admin", "school_owner", "school_manager", "floor_supervisor", "attendance_supervisor", "teacher")
    for code, label in permissions:
        op.execute(sa.text(
            "INSERT INTO permissions (code, name_en, name_ar) SELECT :code, :label, :label "
            "WHERE NOT EXISTS (SELECT 1 FROM permissions WHERE code = :code)"
        ).bindparams(code=code, label=label))
        op.execute(sa.text(
            "INSERT INTO role_permissions (role_id, permission_id) "
            "SELECT r.id, p.id FROM roles r CROSS JOIN permissions p "
            "WHERE p.code = :code AND r.code IN :roles AND NOT EXISTS ("
            "SELECT 1 FROM role_permissions rp WHERE rp.role_id = r.id AND rp.permission_id = p.id)"
        ).bindparams(sa.bindparam("roles", value=roles, expanding=True), code=code))


def downgrade() -> None:
    op.drop_table("chat_reads")
    op.drop_index("ix_chat_messages_sender_user_id", table_name="chat_messages")
    op.drop_index("ix_chat_messages_conversation_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_chat_conversations_school_updated", table_name="chat_conversations")
    op.drop_index("ix_chat_conversations_school_id", table_name="chat_conversations")
    op.drop_table("chat_conversations")
    op.execute(sa.text(
        "DELETE FROM role_permissions WHERE permission_id IN "
        "(SELECT id FROM permissions WHERE code IN ('chat.read', 'chat.write'))"
    ))
    op.execute(sa.text("DELETE FROM permissions WHERE code IN ('chat.read', 'chat.write')"))
