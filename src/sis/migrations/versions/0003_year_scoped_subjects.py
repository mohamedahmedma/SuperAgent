"""subjects belong to an academic year

The subject catalogue was global: one `MATH` row for the whole school, `uq_subjects_code`
making the code unique across every year. It is per-year from here — `(academic_year_id,
code)` is the identity — because a school sets its own catalogue each year and the console
now asks for subjects one year at a time.

**What this costs, stated where it happens.** A child's `MATH` mark in 2025-2026 and her
`MATH` mark in 2026-2027 are now marks on two different subject rows. Nothing downstream
can treat them as the same subject, so no report card, average or trend can line them up
without matching on the code string and assuming the school meant the same thing by it
both times. The old model comment ("global across year levels so a child's marks stay
comparable over years") was describing exactly the property this revision gives up.

**No grade is touched and no grade needs to be.** `subject_grades.subject_id` points at a
row id, and this revision does not renumber, delete or re-key a single subject row — it
adds a column to the rows already there. Every existing mark still resolves to the subject
it was stated against, with the same name and the same order.

**The backfill.** Existing subjects are assigned to the year their marks were actually
stated in, taken from the term each grade belongs to. A subject with no marks yet has
nothing to infer from and goes to the current academic year, which is the only year a
registrar could have been working in when they created it. A subject whose marks span two
years cannot be split by a migration — that would mean inventing a second row and choosing
which grades to move — so it goes to the earliest year that references it and is reported
in the upgrade output for a human to finish by hand. On the data this revision was written
against no subject is in that position.

The whole upgrade refuses to run if there is no academic year to attach a subject to,
rather than inventing one: a database with subjects and no year is a state this service
cannot produce, and guessing at it would write rows nobody can explain later.

Revision ID: 0003
Revises: 0002
Created: 2026-08-20

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '0003'
down_revision: str | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()

    subject_count = bind.execute(sa.text("SELECT COUNT(*) FROM subjects")).scalar_one()
    year_rows = bind.execute(
        sa.text("SELECT id, code, is_current FROM academic_years ORDER BY code")
    ).all()

    if subject_count and not year_rows:
        raise RuntimeError(
            "There are subjects on file and no academic year to attach them to. This "
            "revision makes a subject belong to a year and will not invent one. Create "
            "the academic year these subjects are taught in, then run the upgrade again."
        )

    # The year a subject falls back to: the one marked current, else the earliest on file.
    fallback_year_id = None
    if year_rows:
        current = [row for row in year_rows if row.is_current]
        fallback_year_id = (current[0] if current else year_rows[0]).id

    # Which year each existing subject's marks were actually stated in. A subject appears
    # once per year that has a grade for it, so a second row for the same subject is the
    # spanning case the docstring describes.
    stated_in = bind.execute(
        sa.text(
            """
            SELECT DISTINCT g.subject_id AS subject_id,
                            t.academic_year_id AS academic_year_id,
                            y.code AS year_code
            FROM subject_grades g
            JOIN terms t ON t.id = g.term_id
            JOIN academic_years y ON y.id = t.academic_year_id
            ORDER BY g.subject_id, y.code
            """
        )
    ).all()

    resolved: dict[int, int] = {}
    spanning: dict[int, list[str]] = {}
    for row in stated_in:
        if row.subject_id in resolved:
            spanning.setdefault(row.subject_id, []).append(row.year_code)
            continue
        resolved[row.subject_id] = row.academic_year_id

    # Nullable first, backfilled, then made NOT NULL. Adding a non-null column with no
    # server default to a populated table fails outright on Postgres, and SQLite would
    # need a table rebuild it cannot do while a foreign key points at the old one.
    with op.batch_alter_table("subjects", schema=None) as batch_op:
        batch_op.add_column(sa.Column("academic_year_id", sa.Integer(), nullable=True))

    if subject_count:
        for subject_id, year_id in resolved.items():
            bind.execute(
                sa.text("UPDATE subjects SET academic_year_id = :year WHERE id = :id"),
                {"year": year_id, "id": subject_id},
            )
        bind.execute(
            sa.text(
                "UPDATE subjects SET academic_year_id = :year WHERE academic_year_id IS NULL"
            ),
            {"year": fallback_year_id},
        )

    if spanning:
        # Printed, not raised. The rows are consistent and every grade still resolves; what
        # a human has to decide is whether last year's marks want a subject row of their
        # own, and that is a data decision rather than a schema one.
        for subject_id, extra_years in spanning.items():
            code = bind.execute(
                sa.text("SELECT code FROM subjects WHERE id = :id"), {"id": subject_id}
            ).scalar_one()
            print(
                f"  [0003] subject {code!r} has marks in more than one year; it now "
                f"belongs to the earliest and its marks in {', '.join(extra_years)} are "
                "stated against that row. Split it by hand if those years need their own."
            )

    # The identity swap. The old constraint has to go before the new one can exist, or the
    # same code in a second year is unwritable — which is the entire point of the revision.
    with op.batch_alter_table("subjects", schema=None) as batch_op:
        batch_op.alter_column(
            "academic_year_id", existing_type=sa.Integer(), nullable=False
        )
        batch_op.drop_constraint("uq_subjects_code", type_="unique")
        batch_op.create_foreign_key(
            batch_op.f("fk_subjects_academic_year_id_academic_years"),
            "academic_years",
            ["academic_year_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_unique_constraint(
            "uq_subjects_year_code", ["academic_year_id", "code"]
        )
        batch_op.drop_index("ix_subjects_order")
        batch_op.create_index(
            "ix_subjects_year_order", ["academic_year_id", "display_order"], unique=False
        )


def downgrade() -> None:
    """Return the catalogue to one global list.

    Lossy, and refuses rather than guessing. Once two years each hold a `MATH`, a single
    global catalogue has no room for both, and picking one would silently repoint the other
    year's marks at a subject its school never taught. The downgrade only runs while every
    code is still unique across years — which is the case immediately after an upgrade, the
    only time a downgrade is a real prospect.
    """
    bind = op.get_bind()

    clashes = bind.execute(
        sa.text(
            "SELECT code, COUNT(*) AS n FROM subjects GROUP BY code HAVING COUNT(*) > 1"
        )
    ).all()
    if clashes:
        listed = ", ".join(f"{row.code} ({row.n} years)" for row in clashes)
        raise RuntimeError(
            "Cannot make the subject catalogue global again: these codes exist in more "
            f"than one academic year — {listed}. A global catalogue holds one row per "
            "code, so this downgrade would have to delete subjects that marks are stated "
            "against. Merge or rename them first."
        )

    with op.batch_alter_table("subjects", schema=None) as batch_op:
        batch_op.drop_index("ix_subjects_year_order")
        batch_op.create_index("ix_subjects_order", ["display_order"], unique=False)
        batch_op.drop_constraint("uq_subjects_year_code", type_="unique")
        batch_op.drop_constraint(
            batch_op.f("fk_subjects_academic_year_id_academic_years"), type_="foreignkey"
        )
        batch_op.create_unique_constraint("uq_subjects_code", ["code"])
        batch_op.drop_column("academic_year_id")
