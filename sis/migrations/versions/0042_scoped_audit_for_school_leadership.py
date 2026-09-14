"""Give school leadership scoped access to operational audit events."""
from alembic import op
import sqlalchemy as sa

revision = "0042"
down_revision = "0041"
branch_labels = None
depends_on = None

ROLES = ("school_manager", "floor_supervisor", "attendance_supervisor")


def upgrade() -> None:
    connection = op.get_bind()
    permission_id = connection.execute(sa.text("SELECT id FROM permissions WHERE code='audit.read'" )).scalar()
    if permission_id is None:
        return
    for role_code in ROLES:
        role_id = connection.execute(sa.text("SELECT id FROM roles WHERE code=:code"), {"code": role_code}).scalar()
        if role_id is not None:
            connection.execute(sa.text(
                "INSERT INTO role_permissions(role_id, permission_id) "
                "SELECT :role_id, :permission_id WHERE NOT EXISTS ("
                "SELECT 1 FROM role_permissions WHERE role_id=:role_id AND permission_id=:permission_id)"
            ), {"role_id": role_id, "permission_id": permission_id})


def downgrade() -> None:
    connection = op.get_bind()
    connection.execute(sa.text(
        "DELETE FROM role_permissions WHERE permission_id=(SELECT id FROM permissions WHERE code='audit.read') "
        "AND role_id IN (SELECT id FROM roles WHERE code IN ('school_manager','floor_supervisor','attendance_supervisor'))"
    ))
