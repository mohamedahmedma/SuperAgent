"""The weekly plan: a school's periods, and one lesson per class per slot.

Two tables, and the second one's constraints are the feature rather than paperwork.

`timetable_periods` is the school's day — period 1..N, optionally timed. Per school because
the bell is per building; nullable times because a school agrees how many periods it runs
long before it agrees when they ring, and a placeholder is indistinguishable afterwards
from a time somebody actually decided.

`timetable_entries` is the plan itself. Its two unique constraints are the conflict rules:

  uq_timetable_entries_slot          one class is in one place at one moment
  uq_timetable_entries_teacher_slot  and so is one teacher

The second is written now, in the stage before teachers are managed, because it costs
nothing today and would cost a migration later. It works because `teacher_id` is nullable
and SQL treats NULLs as distinct: every unassigned lesson may share a slot, and no named
teacher may be in two rooms at once.

Nothing here touches attendance. Per-lesson registers would need this table and are
deliberately not built on it yet.

**This revision creates and drops only its own two tables.** No existing row is read,
rewritten or referenced, so there is nothing for it to lose and the downgrade is a clean
drop of a plan that can be laid out again.

Revision ID: 0012
Revises: 0011
"""
from alembic import op
import sqlalchemy as sa

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "timetable_periods",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("school_id", sa.Integer(), nullable=False),
        sa.Column("period_number", sa.Integer(), nullable=False),
        sa.Column("name_en", sa.String(length=160), nullable=False, server_default=""),
        sa.Column("name_ar", sa.String(length=160), nullable=False, server_default=""),
        # `Time`, not `DateTime`: a bell rings at 09:05 every day of term, and stored as an
        # instant it would acquire a date and an offset.
        sa.Column("starts_at", sa.Time(), nullable=True),
        sa.Column("ends_at", sa.Time(), nullable=True),
        sa.Column("is_teaching", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        # RESTRICT, as every other reference to a school: a branch with a timetable is a
        # branch that ran, and the database should refuse to lose that quietly.
        sa.ForeignKeyConstraint(["school_id"], ["schools.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "school_id", "period_number", name="uq_timetable_periods_school_number"
        ),
        sa.CheckConstraint(
            "period_number >= 1 AND period_number <= 20",
            name="ck_timetable_periods_number_range",
        ),
        # NULL-safe: with either end absent the comparison is unknown, and an unknown CHECK
        # passes. Forbids an inverted stated range, permits an unstated one.
        sa.CheckConstraint("ends_at > starts_at", name="ck_timetable_periods_times_ordered"),
    )
    op.create_index("ix_timetable_periods_school_id", "timetable_periods", ["school_id"])
    op.create_index(
        "ix_timetable_periods_school_order", "timetable_periods", ["school_id", "period_number"]
    )

    op.create_table(
        "timetable_entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("class_section_id", sa.Integer(), nullable=False),
        sa.Column("academic_year_id", sa.Integer(), nullable=False),
        sa.Column("term_id", sa.Integer(), nullable=False),
        sa.Column("day_of_week", sa.String(length=16), nullable=False),
        sa.Column("period_number", sa.Integer(), nullable=False),
        # NULL is a stated free period, not an unplanned one: a slot with no *row* is one
        # nobody has planned yet, and the two must stay distinguishable on screen.
        sa.Column("subject_id", sa.Integer(), nullable=True),
        # Always NULL in this stage. The column is here so the teacher conflict rule below
        # is already law when teacher management arrives.
        sa.Column("teacher_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        # CASCADE on everything the plan describes: a timetable is an intention, and it
        # should follow its class, year, term or subject out rather than block the delete.
        sa.ForeignKeyConstraint(["class_section_id"], ["class_sections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["academic_year_id"], ["academic_years.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["term_id"], ["terms.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subject_id"], ["subjects.id"], ondelete="CASCADE"),
        # SET NULL, alone among them: a teacher leaving does not cancel the lesson, it
        # leaves it needing somebody.
        sa.ForeignKeyConstraint(["teacher_id"], ["teachers.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        # The conflict rules. See the module docstring.
        sa.UniqueConstraint(
            "class_section_id",
            "term_id",
            "day_of_week",
            "period_number",
            name="uq_timetable_entries_slot",
        ),
        sa.UniqueConstraint(
            "teacher_id",
            "term_id",
            "day_of_week",
            "period_number",
            name="uq_timetable_entries_teacher_slot",
        ),
        sa.CheckConstraint(
            "period_number >= 1 AND period_number <= 20",
            name="ck_timetable_entries_number_range",
        ),
    )
    op.create_index("ix_timetable_entries_term_id", "timetable_entries", ["term_id"])
    op.create_index("ix_timetable_entries_subject", "timetable_entries", ["subject_id"])
    # Serves "draw this class's week", ordered the way a grid is read.
    op.create_index(
        "ix_timetable_entries_class_week",
        "timetable_entries",
        ["class_section_id", "term_id", "day_of_week", "period_number"],
    )
    # Serves "the whole school's Tuesday" — how a clash is spotted by eye, and how a future
    # teacher-allocation screen will look for a free slot.
    op.create_index(
        "ix_timetable_entries_year_slot",
        "timetable_entries",
        ["academic_year_id", "day_of_week", "period_number"],
    )


def downgrade() -> None:
    # Entries first: they reference the schools whose periods the second drop removes, and
    # dropping a referenced table is what SQLite refuses.
    op.drop_index("ix_timetable_entries_year_slot", table_name="timetable_entries")
    op.drop_index("ix_timetable_entries_class_week", table_name="timetable_entries")
    op.drop_index("ix_timetable_entries_subject", table_name="timetable_entries")
    op.drop_index("ix_timetable_entries_term_id", table_name="timetable_entries")
    op.drop_table("timetable_entries")

    op.drop_index("ix_timetable_periods_school_order", table_name="timetable_periods")
    op.drop_index("ix_timetable_periods_school_id", table_name="timetable_periods")
    op.drop_table("timetable_periods")
