"""Make the persisted School Owner bundle read-only.

The built-in catalogue remains authoritative at runtime. This migration applies the same
boundary immediately for existing databases, before any owner can call an API manually.
"""
from alembic import op
import sqlalchemy as sa


revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


READ_ONLY = (
    "schools.read", "structure.read", "students.read", "teachers.read",
    "timetable.read", "guardians.read", "grades.read", "teacher_attendance.read",
    "users.read", "reports.read",
)


def upgrade() -> None:
    bind = op.get_bind()
    owner_id = bind.execute(sa.text("SELECT id FROM roles WHERE code='school_owner'")).scalar()
    if owner_id is None:
        return
    placeholders = ", ".join(f":p{index}" for index in range(len(READ_ONLY)))
    params = {f"p{index}": code for index, code in enumerate(READ_ONLY)}
    params["role_id"] = owner_id
    bind.execute(sa.text(
        "DELETE FROM role_permissions WHERE role_id=:role_id AND permission_id IN ("
        "SELECT id FROM permissions WHERE code NOT IN (" + placeholders + "))"
    ), params)


def downgrade() -> None:
    # Write permissions are deliberately never restored automatically.
    pass
