"""Allow school managers to run promotions and end-term grade imports."""

from alembic import op
import sqlalchemy as sa


revision = "0041"
down_revision = "0040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    for permission in ("grades.write", "imports.run"):
        bind.execute(sa.text(
            "INSERT INTO role_permissions (role_id, permission_id) "
            "SELECT r.id, p.id FROM roles r JOIN permissions p ON p.code=:permission "
            "WHERE r.code='school_manager' AND NOT EXISTS ("
            "SELECT 1 FROM role_permissions rp WHERE rp.role_id=r.id AND rp.permission_id=p.id)"
        ), {"permission": permission})


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text(
        "DELETE FROM role_permissions WHERE role_id IN "
        "(SELECT id FROM roles WHERE code='school_manager') AND permission_id IN "
        "(SELECT id FROM permissions WHERE code IN ('grades.write', 'imports.run'))"
    ))
