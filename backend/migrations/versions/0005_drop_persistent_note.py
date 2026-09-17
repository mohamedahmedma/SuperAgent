"""Conversations stop carrying a persistent note.

The rolling conversation summary is gone. Query resolution carries a follow-up's conditions
and the session's child pin carries which child a conversation is about, so nothing reads
`persistent_note` or `persistent_note_guardian` out of `chat_sessions.metadata_json` any
more. Both keys are removed from every stored session rather than left there unread.

Compatible with the release before it, as DEVOPS.md requires of every migration: that
release reads a session without a note as one whose note has not been written yet.

The downgrade changes nothing. The summaries this deletes cannot be put back, and the
release before this one starts writing them again as conversations continue.

Revision ID: 0005
Revises: 0004
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE chat_sessions "
        "SET metadata_json = metadata_json - 'persistent_note' - 'persistent_note_guardian' "
        "WHERE metadata_json ?| array['persistent_note', 'persistent_note_guardian']"
    )


def downgrade() -> None:
    pass
