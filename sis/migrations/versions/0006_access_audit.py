"""Who was told about which child, and when.

Revision ID: 0006
Revises: 0005

The audit moves here from `records/`. It sat there while the facade held the guardian
tables and made the access decision; it holds neither now — the link is this service's,
and the parent-facing routes re-check it here on every read — so the record of the
decision follows the decision. See `sis/domain/access.py`.

Three indexes, and each answers a question somebody actually asks:

  student + time     "who saw my daughter's marks, and when" — the subject-access request
  guardian + time    "everything this parent was told" — the custody dispute
  allowed + time     "show me the refusals" — a run of them against one handle is somebody
                     probing, and it is invisible if only successes are indexed

No foreign keys. A row has to survive the deletion of the guardian it names, and a cascade
would erase exactly the history someone wants after an account is removed.

Created empty. The rows in `records/`'s own table are not migrated: they were written by a
service that is being taken out of the decision path, they key on a guardian id that means
the same thing but was never reconciled, and a partial history presented as a complete one
is worse than a clean start with a documented cut-over date.
"""
from alembic import op
import sqlalchemy as sa

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "access_audit",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("guardian_public_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("student_number", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("allowed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reason", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("actor", sa.String(length=16), nullable=False, server_default=""),
        sa.Column("request_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_access_audit_student_time", "access_audit", ["student_number", "created_at"]
    )
    op.create_index(
        "ix_access_audit_guardian_time",
        "access_audit",
        ["guardian_public_id", "created_at"],
    )
    op.create_index(
        "ix_access_audit_allowed_time", "access_audit", ["allowed", "created_at"]
    )
    op.create_index("ix_access_audit_request_id", "access_audit", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_access_audit_request_id", table_name="access_audit")
    op.drop_index("ix_access_audit_allowed_time", table_name="access_audit")
    op.drop_index("ix_access_audit_guardian_time", table_name="access_audit")
    op.drop_index("ix_access_audit_student_time", table_name="access_audit")
    op.drop_table("access_audit")
