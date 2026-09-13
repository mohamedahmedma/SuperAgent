"""JSON becomes JSONB, and every timestamp says it is UTC.

Two type changes to columns revision 0001 created.

**JSON → JSONB.** JSON keeps the text it was given and re-parses it on every read; JSONB
keeps the parsed value, compares by content, and can be indexed. What JSON preserves and
JSONB drops — key order and duplicate keys — nothing in the backend relies on, so the
conversion is a plain cast.

**timestamp → timestamptz.** Every value was written as UTC without saying so, and a
reader that did not know the convention got it wrong: the chat sidebar's
`new Date("2026-09-13T10:00:00")` reads it as the browser's local time. The stored
wall-clock values are declared UTC during the cast, so no instant moves.

Compatible with the release before it, as DEVOPS.md requires of every migration: that
code sends JSON as text, which Postgres casts to JSONB on assignment, and naive datetimes,
which a timestamptz column interprets in the session time zone — UTC on the postgres
image this estate runs.

Each table is rewritten once, under an ACCESS EXCLUSIVE lock, with all of its columns in
a single ALTER. Quick at this estate's size; on a far larger database, schedule it.

Revision ID: 0002
Revises: 0001
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON_COLUMNS = {
    "chat_sessions": ("metadata_json",),
    "chat_messages": ("rag_trace",),
    "asset_extractions": ("payload",),
    "document_assets": ("dossier",),
    "parent_chunks": ("asset_ids",),
    "section_summaries": ("answers", "question_vectors", "topics"),
}

_TIMESTAMP_COLUMNS = {
    "users": ("created_at",),
    "chat_sessions": ("updated_at", "created_at"),
    "chat_messages": ("timestamp",),
    "asset_extractions": ("created_at", "updated_at"),
    "document_assets": ("created_at", "updated_at"),
    "entity_attributes": ("updated_at",),
    "parent_chunks": ("updated_at",),
    "document_pairs": ("created_at", "updated_at"),
    "corpus_digests": ("updated_at",),
    "section_summaries": ("created_at", "updated_at"),
}


def _alter(json_clause: str, timestamp_clause: str) -> None:
    for table in sorted(set(_JSON_COLUMNS) | set(_TIMESTAMP_COLUMNS)):
        clauses = [json_clause.format(c=column) for column in _JSON_COLUMNS.get(table, ())]
        clauses += [timestamp_clause.format(c=column) for column in _TIMESTAMP_COLUMNS.get(table, ())]
        op.execute(f'ALTER TABLE "{table}" ' + ", ".join(clauses))


def upgrade() -> None:
    _alter(
        'ALTER COLUMN "{c}" TYPE JSONB USING "{c}"::jsonb',
        'ALTER COLUMN "{c}" TYPE TIMESTAMP WITH TIME ZONE USING "{c}" AT TIME ZONE \'UTC\'',
    )


def downgrade() -> None:
    # Lossless: JSONB casts back to JSON, and each instant becomes its UTC wall-clock
    # time, which is what the columns held before.
    _alter(
        'ALTER COLUMN "{c}" TYPE JSON USING "{c}"::json',
        'ALTER COLUMN "{c}" TYPE TIMESTAMP WITHOUT TIME ZONE USING "{c}" AT TIME ZONE \'UTC\'',
    )
