"""Teacher daily attendance and its explicit RBAC permissions.

Revision ID: 0014
Revises: 0013
"""
from alembic import op
import sqlalchemy as sa

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "teacher_attendance",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("teacher_id", sa.Integer(), nullable=False),
        sa.Column("school_id", sa.Integer(), nullable=False),
        sa.Column("on_date", sa.Date(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("note", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("recorded_by", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["teacher_id"], ["teachers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["school_id"], ["schools.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("teacher_id", "on_date", name="uq_teacher_attendance_teacher_date"),
    )
    op.create_index("ix_teacher_attendance_teacher_id", "teacher_attendance", ["teacher_id"])
    op.create_index("ix_teacher_attendance_school_id", "teacher_attendance", ["school_id"])
    op.create_index(
        "ix_teacher_attendance_school_date", "teacher_attendance", ["school_id", "on_date"]
    )


def downgrade() -> None:
    op.drop_index("ix_teacher_attendance_school_date", table_name="teacher_attendance")
    op.drop_index("ix_teacher_attendance_school_id", table_name="teacher_attendance")
    op.drop_index("ix_teacher_attendance_teacher_id", table_name="teacher_attendance")
    op.drop_table("teacher_attendance")
