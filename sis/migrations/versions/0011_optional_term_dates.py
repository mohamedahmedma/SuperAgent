"""Term dates become optional.

A school states how many terms it runs when it is created, and revision 0011 is what lets
the year's term sections be built from that number straight away — in June, months before
anybody has decided when the second term starts.

Under NOT NULL there were only two ways to do that, and both are worse than a nullable
column. Blocking the setup until the calendar is settled makes the term count useless at
the moment it is chosen. Writing a placeholder makes the column dishonest: nothing
downstream can tell a placeholder from a boundary a school actually agreed, and invariant 2
resolves a child's class against exactly these days — so a made-up date does not stay a
display problem, it decides which class a mark is filed under.

`NULL` means "not stated yet". A term that has one still orders, still holds marks and
still closes; what it cannot do alone is answer "which class was she in during it", and the
service answers that against the *year's* window instead. See `Term.resolution_window`.

**Nothing is lost and nothing is rewritten.** Widening NOT NULL to NULL cannot fail on
existing rows, and every term already on file keeps the dates it has. The downgrade is the
half that can fail, and it says so rather than inventing dates — see below.

Revision ID: 0011
Revises: 0010
"""
from alembic import op
import sqlalchemy as sa

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Batch mode because SQLite cannot ALTER a column's nullability in place; on
    # PostgreSQL this is a plain ALTER. The table is recreated under SQLite, so the
    # existing type has to be restated — `existing_type` is what carries the CHECK and the
    # two indexes across the rebuild rather than dropping them.
    with op.batch_alter_table("terms") as batch:
        batch.alter_column("starts_on", existing_type=sa.Date(), nullable=True)
        batch.alter_column("ends_on", existing_type=sa.Date(), nullable=True)


def downgrade() -> None:
    # A term with no dates cannot be narrowed back to NOT NULL without stating days the
    # school never did, so this refuses instead — the same rule 0003's downgrade follows
    # for subjects it cannot keep. Filling them from the year's window would be the worst
    # available option: it would look like a successful downgrade and would silently turn
    # "not decided" into a boundary that decides where marks are filed.
    undated = op.get_bind().execute(
        sa.text("SELECT code FROM terms WHERE starts_on IS NULL OR ends_on IS NULL")
    ).scalars().all()
    if undated:
        raise RuntimeError(
            "cannot restore NOT NULL term dates: "
            f"{len(undated)} term(s) have none on file ({', '.join(sorted(undated)[:10])}"
            f"{', …' if len(undated) > 10 else ''}). Give each of them a start and end "
            "date, or drop them, and run the downgrade again."
        )
    with op.batch_alter_table("terms") as batch:
        batch.alter_column("starts_on", existing_type=sa.Date(), nullable=False)
        batch.alter_column("ends_on", existing_type=sa.Date(), nullable=False)
