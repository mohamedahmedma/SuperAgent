"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Created: ${create_date}

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: str | None = ${repr(down_revision)}
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    # A downgrade that raises is a migration that can only go forwards, and the moment
    # it matters is a failed 02:00 deploy with a half-applied schema. Write the real
    # inverse here, even when it is a data-losing one -- say so in a comment instead.
    ${downgrades if downgrades else "pass"}
