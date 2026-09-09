"""Store sparse Admin-managed per-user permission exceptions."""
from alembic import op
import sqlalchemy as sa


revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_permission_overrides",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("permission_id", sa.Integer(), sa.ForeignKey("permissions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("effect", sa.String(length=8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "permission_id", name="uq_user_permission_overrides_user_permission"),
        sa.CheckConstraint("effect IN ('allow', 'deny')", name="ck_user_permission_overrides_effect"),
    )
    op.create_index("ix_user_permission_overrides_user", "user_permission_overrides", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_user_permission_overrides_user", table_name="user_permission_overrides")
    op.drop_table("user_permission_overrides")
