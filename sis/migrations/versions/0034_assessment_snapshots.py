"""Store the individual assessment sheets entered by teachers.

The class-mark entry screen writes the term mark and, separately, the assessment sheet
that produced it.  The latter tables were introduced with the route but not with a
schema revision, so every save reached the final INSERT and failed with a 500 in a
migrated production database.  These tables make that write durable.
"""
from alembic import op
import sqlalchemy as sa


revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assessments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("academic_year_code", sa.String(length=16), nullable=False),
        sa.Column("class_code", sa.String(length=32), nullable=False),
        sa.Column("subject_code", sa.String(length=32), nullable=False),
        sa.Column("term_code", sa.String(length=32), nullable=False),
        sa.Column("assessment_type", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("max_points", sa.Float(), nullable=True),
        sa.Column("recorded_by", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("academic_year_code", "class_code", "subject_code", "term_code", "assessment_type", "name", name="uq_assessments_identity"),
    )
    op.create_index("ix_assessments_class_lookup", "assessments", ["academic_year_code", "class_code", "subject_code", "term_code", "assessment_type"])

    op.create_table(
        "assessment_marks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("assessment_id", sa.Integer(), nullable=False),
        sa.Column("student_number", sa.String(length=64), nullable=False),
        sa.Column("points", sa.Float(), nullable=True),
        sa.Column("max_points", sa.Float(), nullable=True),
        sa.Column("percentage", sa.Float(), nullable=True),
        sa.Column("is_absent", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["assessment_id"], ["assessments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["student_number"], ["students.student_number"], ondelete="RESTRICT"),
        sa.UniqueConstraint("assessment_id", "student_number", name="uq_assessment_marks_student"),
        sa.CheckConstraint("percentage IS NULL OR (percentage >= 0 AND percentage <= 100)", name="ck_assessment_marks_percentage_range"),
    )
    op.create_index("ix_assessment_marks_assessment", "assessment_marks", ["assessment_id"])


def downgrade() -> None:
    op.drop_index("ix_assessment_marks_assessment", table_name="assessment_marks")
    op.drop_table("assessment_marks")
    op.drop_index("ix_assessments_class_lookup", table_name="assessments")
    op.drop_table("assessments")
