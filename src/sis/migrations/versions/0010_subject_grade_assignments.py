"""Assign year-scoped subjects to track-owned grades.

`subjects` says what a school teaches in a year. It has never said *where*, so every
screen that offered a subject offered all of them: a Primary marks sheet listed Physics
because Secondary sat it, and the Arabic and Languages sections of one school shared a
single undivided catalogue. This table is that missing sentence — one row per
(subject, rung) pair the school actually teaches.

The track falls out of the rung rather than being stored again here. A `year_level`
belongs to exactly one `educational_system` (revision 0007), so assigning Physics to the
Arabic section's Secondary 1 says nothing at all about the Languages section's, and a
column naming the track on this row could only ever disagree with the rung's own.

**The backfill is the point of this revision, not an afterthought.** Once anything reads
subjects through this table, a subject with no row here appears nowhere — so shipping the
empty table would silently empty every existing school's marks screens. The old implicit
rule was "every subject is taught on every rung of its school", and the loop below writes
exactly that rule down. It is not a guess about what a school wants; it is the behaviour
the school already had, made explicit so a registrar can now take rows *away*.

The pairing is per school and per year, never global: `subjects` reaches its school
through `academic_years`, and only that school's rungs are matched. A cross-branch pair
would be an assignment nobody could see on a board and nobody could delete from one.

Revision ID: 0010
Revises: 0009
"""
from alembic import op
import sqlalchemy as sa

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subject_year_levels",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("subject_id", sa.Integer(), nullable=False),
        sa.Column("year_level_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        # CASCADE on both sides, unlike the RESTRICT that guards `subject_grades`. This
        # row is a statement about a timetable, not a record of anything a child was
        # awarded, so it should follow whatever it describes out of the database rather
        # than block the delete.
        sa.ForeignKeyConstraint(["subject_id"], ["subjects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["year_level_id"], ["year_levels.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        # Requirement: no duplicate assignments. Enforced here rather than only in the
        # service, so two registrars dropping the same subject on the same rung at the
        # same moment cannot produce two rows.
        sa.UniqueConstraint(
            "subject_id", "year_level_id", name="uq_subject_year_levels_assignment"
        ),
    )
    op.create_index("ix_subject_year_levels_subject_id", "subject_year_levels", ["subject_id"])
    # Serves the question every reader actually asks: "what does this rung teach".
    op.create_index("ix_subject_year_levels_level", "subject_year_levels", ["year_level_id"])

    # Preserve what the schools already had. See the module docstring: before this table,
    # a subject was implicitly taught on every rung of its school, and that is what is
    # written out here. `NOT EXISTS` keeps the statement re-runnable rather than relying
    # on the unique constraint to reject a second attempt half way through.
    op.execute("""
        INSERT INTO subject_year_levels (subject_id, year_level_id, created_at)
        SELECT s.id, l.id, CURRENT_TIMESTAMP
        FROM subjects s
        JOIN academic_years y ON y.id = s.academic_year_id
        JOIN year_levels l ON l.school_id = y.school_id
        WHERE NOT EXISTS (
            SELECT 1 FROM subject_year_levels a
            WHERE a.subject_id = s.id AND a.year_level_id = l.id
        )
    """)


def downgrade() -> None:
    # Dropping the table restores the old implicit "every subject, every rung" reading,
    # so no data a school stated about its children is lost — only the narrowing it
    # stated about its timetable.
    op.drop_index("ix_subject_year_levels_level", table_name="subject_year_levels")
    op.drop_index("ix_subject_year_levels_subject_id", table_name="subject_year_levels")
    op.drop_table("subject_year_levels")
