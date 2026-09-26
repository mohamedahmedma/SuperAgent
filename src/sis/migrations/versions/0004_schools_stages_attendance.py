"""schools, stages, student demographics, and the daily register

Four changes, in one revision because the first three are one thought: this service held
exactly one school implicitly, and now holds several explicitly.

**Schools.** A `schools` table, and a `school_id` on `academic_years` and `year_levels`.
Everything else inherits its school through one of those two: a class section belongs to a
year and a rung, a term and a subject belong to a year, a mark belongs to a term and a
section. Students are deliberately *not* scoped — a child is a person, `student_number`
stays globally unique, and which school she attends follows from her placement. That keeps
the join key `records/` reads, the guardian tables and every stated mark untouched here, and
makes a child moving between branches a transfer rather than a duplicate.

**What each code is unique within**, because the two differ and the difference is load-bearing:

  academic_years.code   globally unique, unchanged. `2025-2026` names one year at one
                        school, so `?academic_year=` keeps identifying a year unambiguously
                        and no route, import or facade call has to learn about schools. The
                        cost is that two branches must not both use the literal code
                        `2025-2026`.
  year_levels.code      unique per school, changed. "Year 1" genuinely exists at every
                        branch, and a rung is only ever resolved alongside a year — which
                        names the school — so this stays unambiguous.

**The backfill invents one school**, because the existing rows belong to a school that was
never written down. It is created as `MAIN` and every year and rung on file is attached to
it. That is a real decision and not a technicality: after this revision a registrar sees a
school called "Main School" and should rename it (labels are renameable; the code is not).
The upgrade prints what it did.

**Stages.** `year_levels.stage` — garden / primary / preparatory / secondary — for grouping a
fourteen-rung ladder on screen. Every existing rung becomes `unspecified`, which is what
they all were, rather than being guessed at from their codes.

**Demographics.** `date_of_birth` (nullable) plus the child's own contact details. No age
column: an age is right for one year and wrong afterwards, so it is computed from the date.

**Attendance.** One mark per child per day, with `uq_attendance_student_day` making a second
mark for a day a correction rather than a contradiction. There is no state meaning
"unmarked" — a day with no row is a day nobody took the register, which is the only honest
way to keep "days recorded" usable as a denominator.

Every table this revision rebuilds is rebuilt with foreign keys disabled and verified
afterwards; `sis/migrations/env.py` owns that, and revision 0003's docstring explains why it
has to.

Revision ID: 0004
Revises: 0003
Created: 2026-08-20

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '0004'
down_revision: str | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The school the existing rows are attached to. Named rather than derived because there is
#: nothing in the database to derive it from — see the docstring.
_FALLBACK_SCHOOL_CODE = "MAIN"
_FALLBACK_SCHOOL_EN = "Main School"
_FALLBACK_SCHOOL_AR = "المدرسة الرئيسية"


def upgrade() -> None:
    bind = op.get_bind()

    # -- schools ---------------------------------------------------------------------
    op.create_table(
        "schools",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", sa.String(length=16), nullable=False),
        sa.Column("name_en", sa.String(length=160), nullable=False),
        sa.Column("name_ar", sa.String(length=160), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_schools")),
        sa.UniqueConstraint("code", name=op.f("uq_schools_code")),
    )

    years = bind.execute(sa.text("SELECT COUNT(*) FROM academic_years")).scalar_one()
    levels = bind.execute(sa.text("SELECT COUNT(*) FROM year_levels")).scalar_one()

    school_id: int | None = None
    if years or levels:
        bind.execute(
            sa.text(
                "INSERT INTO schools (code, name_en, name_ar, is_active, created_at) "
                "VALUES (:code, :en, :ar, 1, CURRENT_TIMESTAMP)"
            ),
            {
                "code": _FALLBACK_SCHOOL_CODE,
                "en": _FALLBACK_SCHOOL_EN,
                "ar": _FALLBACK_SCHOOL_AR,
            },
        )
        school_id = bind.execute(
            sa.text("SELECT id FROM schools WHERE code = :code"),
            {"code": _FALLBACK_SCHOOL_CODE},
        ).scalar_one()
        print(
            f"  [0004] {years} academic year(s) and {levels} year level(s) were already on "
            f"file with no school to belong to. They are now attached to a school created "
            f"as {_FALLBACK_SCHOOL_CODE!r} ({_FALLBACK_SCHOOL_EN!r}) — rename it to the "
            "real school; the labels are renameable and the code is not."
        )

    # -- academic_years.school_id ----------------------------------------------------
    #
    # Nullable, backfilled, then made NOT NULL. A non-null column with no server default
    # cannot be added to a populated table on Postgres, and on SQLite the batch rebuild
    # would have to carry a value it does not have yet.
    with op.batch_alter_table("academic_years", schema=None) as batch_op:
        batch_op.add_column(sa.Column("school_id", sa.Integer(), nullable=True))

    if school_id is not None:
        bind.execute(
            sa.text("UPDATE academic_years SET school_id = :school"), {"school": school_id}
        )

    with op.batch_alter_table("academic_years", schema=None) as batch_op:
        batch_op.alter_column("school_id", existing_type=sa.Integer(), nullable=False)
        batch_op.create_foreign_key(
            batch_op.f("fk_academic_years_school_id_schools"),
            "schools",
            ["school_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_index("ix_academic_years_school_id", ["school_id"], unique=False)

    # -- year_levels: school, stage, and the identity swap ---------------------------
    with op.batch_alter_table("year_levels", schema=None) as batch_op:
        batch_op.add_column(sa.Column("school_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "stage",
                sa.String(length=16),
                nullable=False,
                # Every existing rung was unclassified, which is exactly what this value
                # says. Guessing a stage from a code ("KG1 must be garden") would be right
                # often enough to be trusted and wrong often enough to matter.
                server_default="unspecified",
            )
        )

    if school_id is not None:
        bind.execute(
            sa.text("UPDATE year_levels SET school_id = :school"), {"school": school_id}
        )

    with op.batch_alter_table("year_levels", schema=None) as batch_op:
        batch_op.alter_column("school_id", existing_type=sa.Integer(), nullable=False)
        batch_op.drop_constraint("uq_year_levels_code", type_="unique")
        batch_op.create_foreign_key(
            batch_op.f("fk_year_levels_school_id_schools"),
            "schools",
            ["school_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_unique_constraint(
            "uq_year_levels_school_code", ["school_id", "code"]
        )
        batch_op.drop_index("ix_year_levels_order")
        batch_op.create_index("ix_year_levels_school_id", ["school_id"], unique=False)
        batch_op.create_index(
            "ix_year_levels_school_stage",
            ["school_id", "stage", "display_order"],
            unique=False,
        )

    # -- students: date of birth and her own contact details -------------------------
    with op.batch_alter_table("students", schema=None) as batch_op:
        batch_op.add_column(sa.Column("date_of_birth", sa.Date(), nullable=True))
        # `server_default=""` so the columns can be NOT NULL on a populated table without
        # leaving two spellings of empty — `None` on old rows and `""` on new ones — for
        # every reader to handle.
        batch_op.add_column(
            sa.Column("contact_phone", sa.String(length=16), nullable=False, server_default="")
        )
        batch_op.add_column(
            sa.Column("contact_email", sa.String(length=255), nullable=False, server_default="")
        )
        batch_op.add_column(
            sa.Column("address", sa.String(length=500), nullable=False, server_default="")
        )

    # -- attendance ------------------------------------------------------------------
    op.create_table(
        "attendance",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("class_section_id", sa.Integer(), nullable=False),
        sa.Column("on_date", sa.Date(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("note", sa.String(length=500), nullable=False),
        sa.Column("recorded_by", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["class_section_id"],
            ["class_sections.id"],
            name=op.f("fk_attendance_class_section_id_class_sections"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            ["students.id"],
            name=op.f("fk_attendance_student_id_students"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_attendance")),
        # The load-bearing one: a second mark for the same child on the same day is a
        # correction of the first, never a second opinion sitting beside it.
        sa.UniqueConstraint("student_id", "on_date", name="uq_attendance_student_day"),
    )
    with op.batch_alter_table("attendance", schema=None) as batch_op:
        batch_op.create_index(
            "ix_attendance_section_day", ["class_section_id", "on_date"], unique=False
        )
        batch_op.create_index(
            "ix_attendance_student_day", ["student_id", "on_date"], unique=False
        )


def downgrade() -> None:
    """Undo all four, and refuse rather than lose a school's worth of data.

    Two refusals, for the same reason 0003 refuses: a downgrade that cannot preserve what
    is on file has to stop and say which rows are in the way, not pick a survivor.
    """
    bind = op.get_bind()

    schools = bind.execute(sa.text("SELECT COUNT(*) FROM schools")).scalar_one()
    if schools > 1:
        codes = ", ".join(
            row[0] for row in bind.execute(sa.text("SELECT code FROM schools ORDER BY code"))
        )
        raise RuntimeError(
            f"Cannot go back to a single-school schema: {schools} schools are on file "
            f"({codes}). Dropping the column would merge their years and rungs into one "
            "namespace, and two branches each running a '3A' would silently become one "
            "class. Merge or remove the extra schools first."
        )

    clashes = bind.execute(
        sa.text("SELECT code, COUNT(*) AS n FROM year_levels GROUP BY code HAVING COUNT(*) > 1")
    ).all()
    if clashes:
        listed = ", ".join(f"{row.code} ({row.n} schools)" for row in clashes)
        raise RuntimeError(
            "Cannot restore the global unique on year_levels.code: these rung codes exist "
            f"in more than one school — {listed}. A single-school schema holds one row per "
            "code, so this would have to delete rungs that classes point at."
        )

    marks = bind.execute(sa.text("SELECT COUNT(*) FROM attendance")).scalar_one()
    if marks:
        print(
            f"  [0004] dropping the attendance table discards {marks} attendance mark(s). "
            "Nothing else in the schema holds them."
        )

    with op.batch_alter_table("attendance", schema=None) as batch_op:
        batch_op.drop_index("ix_attendance_student_day")
        batch_op.drop_index("ix_attendance_section_day")
    op.drop_table("attendance")

    with op.batch_alter_table("students", schema=None) as batch_op:
        batch_op.drop_column("address")
        batch_op.drop_column("contact_email")
        batch_op.drop_column("contact_phone")
        batch_op.drop_column("date_of_birth")

    with op.batch_alter_table("year_levels", schema=None) as batch_op:
        batch_op.drop_index("ix_year_levels_school_stage")
        batch_op.drop_index("ix_year_levels_school_id")
        batch_op.create_index("ix_year_levels_order", ["display_order"], unique=False)
        batch_op.drop_constraint("uq_year_levels_school_code", type_="unique")
        batch_op.drop_constraint(
            batch_op.f("fk_year_levels_school_id_schools"), type_="foreignkey"
        )
        batch_op.create_unique_constraint("uq_year_levels_code", ["code"])
        batch_op.drop_column("stage")
        batch_op.drop_column("school_id")

    with op.batch_alter_table("academic_years", schema=None) as batch_op:
        batch_op.drop_index("ix_academic_years_school_id")
        batch_op.drop_constraint(
            batch_op.f("fk_academic_years_school_id_schools"), type_="foreignkey"
        )
        batch_op.drop_column("school_id")

    op.drop_table("schools")
