"""Allow the School Owner to manage teaching staff only."""

from alembic import op
import sqlalchemy as sa


revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


STAFFING_WRITES = (
    "teachers.assign_subjects",
    "teachers.assign_classes",
    "roles.assign",
)


def upgrade() -> None:
    bind = op.get_bind()
    for code in STAFFING_WRITES:
        bind.execute(sa.text(
            "INSERT INTO role_permissions (role_id, permission_id) "
            "SELECT r.id, p.id FROM roles r JOIN permissions p ON p.code=:code "
            "WHERE r.code='school_owner' AND NOT EXISTS ("
            "SELECT 1 FROM role_permissions rp "
            "WHERE rp.role_id=r.id AND rp.permission_id=p.id)"
        ), {"code": code})


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text(
        "DELETE FROM role_permissions WHERE role_id IN "
        "(SELECT id FROM roles WHERE code='school_owner') AND permission_id IN "
        "(SELECT id FROM permissions WHERE code IN "
        "('teachers.assign_subjects', 'teachers.assign_classes', 'roles.assign'))"
    ))
