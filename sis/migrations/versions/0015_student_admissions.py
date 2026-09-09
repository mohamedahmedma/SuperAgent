"""Add the create-only student admission permission.

Revision ID: 0015
Revises: 0014
"""
from alembic import op
import sqlalchemy as sa

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text(
        "INSERT INTO permissions (code, name_en, name_ar) "
        "SELECT 'students.create', 'Create complete student admissions', "
        "'Create complete student admissions' "
        "WHERE NOT EXISTS (SELECT 1 FROM permissions WHERE code='students.create')"
    ))
    op.execute(sa.text(
        "INSERT INTO role_permissions (role_id, permission_id) "
        "SELECT r.id, p.id FROM roles r CROSS JOIN permissions p "
        "WHERE r.code IN ('system_admin','principal') AND p.code='students.create' "
        "AND NOT EXISTS (SELECT 1 FROM role_permissions rp "
        "WHERE rp.role_id=r.id AND rp.permission_id=p.id)"
    ))


def downgrade() -> None:
    op.execute(sa.text(
        "DELETE FROM role_permissions WHERE permission_id IN "
        "(SELECT id FROM permissions WHERE code='students.create')"
    ))
    op.execute(sa.text("DELETE FROM permissions WHERE code='students.create'"))
