"""School language, enabled levels, term count and working days.

Revision ID: 0008
Revises: 0007

The defaults preserve every existing school as a fully configured school. New school
creation sends explicit values, including zero for levels the operator did not select.
No year-level, term or timetable rows are created by this migration.
"""
from alembic import op
import sqlalchemy as sa

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("schools") as batch:
        batch.add_column(sa.Column("language_type", sa.String(16), nullable=False, server_default="both"))
        batch.add_column(sa.Column("kg_grade_count", sa.Integer(), nullable=False, server_default="3"))
        batch.add_column(sa.Column("primary_grade_count", sa.Integer(), nullable=False, server_default="6"))
        batch.add_column(sa.Column("preparatory_grade_count", sa.Integer(), nullable=False, server_default="3"))
        batch.add_column(sa.Column("secondary_grade_count", sa.Integer(), nullable=False, server_default="3"))
        batch.add_column(sa.Column("term_count", sa.Integer(), nullable=False, server_default="2"))
        batch.add_column(sa.Column(
            "working_days", sa.String(80), nullable=False,
            server_default="sunday,monday,tuesday,wednesday,thursday",
        ))
        batch.create_check_constraint("ck_schools_language_type", "language_type IN ('arabic','languages','both')")
        batch.create_check_constraint("ck_schools_kg_grades", "kg_grade_count BETWEEN 0 AND 3")
        batch.create_check_constraint("ck_schools_primary_grades", "primary_grade_count BETWEEN 0 AND 6")
        batch.create_check_constraint("ck_schools_preparatory_grades", "preparatory_grade_count BETWEEN 0 AND 3")
        batch.create_check_constraint("ck_schools_secondary_grades", "secondary_grade_count BETWEEN 0 AND 3")
        batch.create_check_constraint(
            "ck_schools_at_least_one_level",
            "kg_grade_count + primary_grade_count + preparatory_grade_count + secondary_grade_count > 0",
        )
        batch.create_check_constraint("ck_schools_term_count", "term_count BETWEEN 1 AND 3")
        batch.create_check_constraint("ck_schools_working_days", "length(working_days) > 0")


def downgrade() -> None:
    with op.batch_alter_table("schools") as batch:
        for name in (
            "ck_schools_working_days", "ck_schools_term_count",
            "ck_schools_at_least_one_level", "ck_schools_secondary_grades",
            "ck_schools_preparatory_grades", "ck_schools_primary_grades",
            "ck_schools_kg_grades", "ck_schools_language_type",
        ):
            batch.drop_constraint(name, type_="check")
        for column in (
            "working_days", "term_count", "secondary_grade_count",
            "preparatory_grade_count", "primary_grade_count", "kg_grade_count",
            "language_type",
        ):
            batch.drop_column(column)
