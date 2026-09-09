"""Add immutable department keys to the structural department table."""

from alembic import op
import sqlalchemy as sa

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Backfill from the existing structured kind, never from display labels.
    op.add_column(
        "educational_systems",
        sa.Column("department_key", sa.String(length=16), nullable=False, server_default="languages"),
    )
    op.execute(
        "UPDATE educational_systems SET department_key = "
        "CASE WHEN kind = 'arabic' THEN 'arabic' ELSE 'languages' END"
    )
    op.alter_column("educational_systems", "department_key", server_default=None)
    op.create_unique_constraint(
        "uq_educational_systems_school_department",
        "educational_systems",
        ["school_id", "department_key"],
    )
    op.create_check_constraint(
        "ck_educational_systems_department_key",
        "educational_systems",
        "department_key IN ('arabic', 'languages')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_educational_systems_department_key", "educational_systems", type_="check")
    op.drop_constraint("uq_educational_systems_school_department", "educational_systems", type_="unique")
    op.drop_column("educational_systems", "department_key")
