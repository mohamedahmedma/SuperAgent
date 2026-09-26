"""Allow School Managers to read daily class registers."""

from alembic import op
import sqlalchemy as sa


revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text(
        "INSERT INTO role_permissions (role_id, permission_id) "
        "SELECT r.id, p.id FROM roles r JOIN permissions p ON p.code='attendance.read' "
        "WHERE r.code='school_manager' AND NOT EXISTS ("
        "SELECT 1 FROM role_permissions rp WHERE rp.role_id=r.id AND rp.permission_id=p.id)"
    ))


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text(
        "DELETE FROM role_permissions WHERE role_id IN "
        "(SELECT id FROM roles WHERE code='school_manager') AND permission_id IN "
        "(SELECT id FROM permissions WHERE code='attendance.read')"
    ))
