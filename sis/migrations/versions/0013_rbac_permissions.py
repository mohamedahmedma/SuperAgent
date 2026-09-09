"""Add the timetable permissions to the additive RBAC catalogue.

Revision ID: 0013
Revises: 0012

**Every statement here is written to be a no-op if the row is already there**, and that is
not defensive habit — it is the only way this migration can run at all.

The catalogue is *also* reconciled from code, by `ensure_catalogue` in
`sis/application/services/access.py`, the first time anybody signs in after a release. So a
database sitting at revision 0012 while running a build whose `Permission` enum already had
`timetable.read` has these rows before alembic ever sees it. A plain `bulk_insert` then
fails on the unique key, and the failure is an upgrade that stops halfway with the school's
service down — over two rows that were already correct.

The reverse case is a fresh database, where `permissions` is empty and this is the insert
that seeds it. Both have to work, so both are written as "insert unless present".

Deleting nothing on the way down, for the same reason in reverse: a downgrade that removed
`timetable.read` would take `role_permissions` rows with it, and the code that put them
there would put them straight back on the next sign-in. What downgrade *does* remove is the
grants this revision added to roles, which is the change it actually made.
"""
from alembic import op
import sqlalchemy as sa

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

PERMISSIONS = (
    ("timetable.read", "Read timetable"),
    ("timetable.write", "Write timetable"),
)

#: Which built-in roles carry each of them. Stated here as the catalogue stood at this
#: revision; the code's own definitions in `sis/domain/rbac.py` are what govern afterwards,
#: and `ensure_catalogue` reconciles any drift the day a release changes them.
ROLE_GRANTS = (
    ("timetable.read", ("system_admin", "school_owner", "principal", "year_supervisor", "teacher")),
    ("timetable.write", ("system_admin", "principal", "year_supervisor")),
)


def upgrade() -> None:
    for code, label in PERMISSIONS:
        op.execute(
            sa.text(
                "INSERT INTO permissions (code, name_en, name_ar) "
                "SELECT :code, :label, :label "
                "WHERE NOT EXISTS (SELECT 1 FROM permissions WHERE code = :code)"
            ).bindparams(code=code, label=label)
        )

    for code, roles in ROLE_GRANTS:
        op.execute(
            sa.text(
                "INSERT INTO role_permissions (role_id, permission_id) "
                "SELECT r.id, p.id FROM roles r CROSS JOIN permissions p "
                "WHERE p.code = :code AND r.code IN :roles "
                "  AND NOT EXISTS ("
                "    SELECT 1 FROM role_permissions rp "
                "    WHERE rp.role_id = r.id AND rp.permission_id = p.id)"
            ).bindparams(sa.bindparam("roles", value=roles, expanding=True), code=code)
        )


def downgrade() -> None:
    op.execute(
        sa.text(
            "DELETE FROM role_permissions WHERE permission_id IN "
            "(SELECT id FROM permissions WHERE code IN ('timetable.read', 'timetable.write'))"
        )
    )
    op.execute(
        sa.text(
            "DELETE FROM permissions WHERE code IN ('timetable.read', 'timetable.write')"
        )
    )
