"""Gender on the child.

Revision ID: 0005
Revises: 0004

Added so a parent can say "my son" and be understood without being asked which child.
The chat service narrows a guardian's children by this field; a school that never fills
it in loses nothing except that narrowing, because every reader treats `unspecified` as
"the school has not said" rather than as a sex.

NOT NULL with a server default, not nullable. The table is populated — every child
already on file becomes `unspecified` here — and a nullable column would leave two
spellings of the same absence for every reader to handle.

Batch mode for the ALTER, as everywhere else in this history: SQLite cannot add a
constrained column in place and rebuilds the table instead.
"""
from alembic import op
import sqlalchemy as sa

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("students", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "gender",
                sa.String(length=16),
                nullable=False,
                server_default="unspecified",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("students", schema=None) as batch_op:
        batch_op.drop_column("gender")
