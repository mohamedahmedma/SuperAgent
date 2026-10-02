"""Reassert the permanent Admin permission set.

The runtime guard treats Admin as full access as a final safety net. This migration also
repairs the persisted catalogue so the permission editor and audit views show the same
truth, without changing any user or role ids.
"""
from alembic import op
import sqlalchemy as sa


revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text(
        "INSERT INTO role_permissions (role_id, permission_id) "
        "SELECT r.id, p.id FROM roles r CROSS JOIN permissions p "
        "WHERE r.code = 'admin' "
        "AND NOT EXISTS (SELECT 1 FROM role_permissions rp "
        "WHERE rp.role_id = r.id AND rp.permission_id = p.id)"
    ))


def downgrade() -> None:
    # Permissions granted to Admin are intentionally permanent.
    pass
