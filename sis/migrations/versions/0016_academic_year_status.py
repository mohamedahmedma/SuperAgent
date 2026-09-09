"""Add an explicit lifecycle status to academic years.

The existing ``is_current`` flag remains the compatibility projection of ``active``.
Existing current rows become active; every other existing row is conservatively marked
upcoming rather than guessing an administrative decision from the server clock.
"""
from alembic import op
import sqlalchemy as sa


revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Batch mode keeps this portable to the existing SQLite deployment while Alembic's
    # environment handles SQLite foreign-key suspension for the table rebuild.
    with op.batch_alter_table("academic_years") as batch:
        batch.add_column(
            sa.Column("status", sa.String(length=16), nullable=False, server_default="upcoming")
        )
        batch.create_check_constraint(
            "ck_academic_years_status",
            "status IN ('upcoming', 'active', 'completed')",
        )
    op.execute(
        sa.text("UPDATE academic_years SET status = 'active' WHERE is_current = 1")
    )


def downgrade() -> None:
    with op.batch_alter_table("academic_years") as batch:
        batch.drop_constraint("ck_academic_years_status", type_="check")
        batch.drop_column("status")
