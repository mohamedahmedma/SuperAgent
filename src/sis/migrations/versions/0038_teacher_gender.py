"""Store a teacher's recorded gender.

Revision ID: 0038
Revises: 0037
"""

import sqlalchemy as sa
from alembic import op

revision = "0038"
down_revision = "0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "teachers",
        sa.Column("gender", sa.String(length=16), nullable=False, server_default="unspecified"),
    )


def downgrade() -> None:
    op.drop_column("teachers", "gender")
