"""A stored message carries its writer's idempotency key, so a retried save cannot repeat it.

A turn's save runs in the background and is now retried after a transient failure
(RAG_FIX_PLAN item 21). A retry after a connection lost AT the commit cannot know whether
the first attempt landed; `chat_messages.client_key`, unique, lets the insert skip a
message that already did.

Purely additive: a nullable column and its index. Rows already stored keep NULL, which a
unique index never counts as equal, and the release before this one runs unchanged
against it. The index is built in the migration's transaction, which blocks writes to
the table while it is built. The column is NULL on every existing row, so that is
seconds even on a large table; build it CONCURRENTLY by hand first if that ever stops
being true.

Revision ID: 0006
Revises: 0005
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("client_key", sa.String(length=32), nullable=True))
    op.create_index("ix_chat_messages_client_key", "chat_messages", ["client_key"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_chat_messages_client_key", table_name="chat_messages")
    op.drop_column("chat_messages", "client_key")
