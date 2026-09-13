"""Baseline: adopt the schema create_all() built, as revision 0001.

Until this revision the backend created its tables with `Base.metadata.create_all()` on
every boot, and patched in later columns with `python -m backend.db.migrate --apply`. A
database reaching this revision is therefore in one of three states, and all three must
end at the same schema:

  - **empty** — every table is created here;
  - **built by create_all()** — everything already exists, and nothing is changed;
  - **built by an older create_all()** — the tables, columns, indexes and constraints
    added since are missing, and exactly those are added. A NOT NULL column added to a
    table that holds rows is backfilled with the value the model defaulted to, and the
    database default is dropped again, so the result matches a fresh database.

The tables are written out here rather than imported from `backend.db.models`. A
revision describes the schema at one point in history; importing the live models would
make this revision mean something different every time a model changed, and a fresh
database would be built in tomorrow's shape before tomorrow's revisions ran on it.

Types are the ones create_all() used — JSON, and timestamps without a time zone.
Revision 0002 converts them.

Deciding what to create means inspecting the live database, so this revision cannot be
rendered offline. SQL for review starts after it:
`alembic -c backend/alembic.ini upgrade 0001:head --sql`.

Revision ID: 0001
Revises:
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The columns are timezone-less and their values are UTC, so a backfilled row gets the
# current UTC wall-clock time rather than the server's local one.
_NOW = "timezone('utc', now())"


def _column(name: str, type_, *args, backfill: str | None = None, **kwargs) -> sa.Column:
    """A column, plus the SQL literal existing rows get if it has to be added later."""
    return sa.Column(name, type_, *args, info={"backfill": backfill}, **kwargs)


def _schema() -> sa.MetaData:
    metadata = sa.MetaData()

    sa.Table(
        "users",
        metadata,
        _column("id", sa.Integer(), primary_key=True, index=True),
        _column("username", sa.String(100), unique=True, index=True, nullable=False),
        _column("password_hash", sa.String(255), nullable=False, backfill="''"),
        _column("role", sa.String(20), nullable=False, backfill="'user'"),
        _column("created_at", sa.DateTime(), nullable=False, backfill=_NOW),
    )
    sa.Table(
        "chat_sessions",
        metadata,
        _column("id", sa.Integer(), primary_key=True, index=True),
        _column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        _column("session_id", sa.String(120), nullable=False, index=True),
        _column("metadata_json", sa.JSON(), nullable=False, backfill="'{}'"),
        _column("updated_at", sa.DateTime(), nullable=False, backfill=_NOW),
        _column("created_at", sa.DateTime(), nullable=False, backfill=_NOW),
        sa.UniqueConstraint("user_id", "session_id", name="uq_user_session"),
    )
    sa.Table(
        "chat_messages",
        metadata,
        _column("id", sa.Integer(), primary_key=True, index=True),
        _column(
            "session_ref_id",
            sa.Integer(),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        _column("message_type", sa.String(20), nullable=False, backfill="''"),
        _column("content", sa.Text(), nullable=False, backfill="''"),
        _column("timestamp", sa.DateTime(), nullable=False, backfill=_NOW),
        _column("rag_trace", sa.JSON(), nullable=True),
    )
    sa.Table(
        "asset_extractions",
        metadata,
        _column("sha256", sa.String(64), primary_key=True),
        _column("profile", sa.String(64), primary_key=True),
        _column("dossier_version", sa.Integer(), primary_key=True),
        _column("payload", sa.JSON(), nullable=False, backfill="'{}'"),
        _column("model_used", sa.String(120), nullable=False, backfill="''"),
        _column("confidence", sa.Float(), nullable=False, backfill="0"),
        _column("needs_review", sa.Boolean(), nullable=False, backfill="false"),
        _column("created_at", sa.DateTime(), nullable=False, backfill=_NOW),
        _column("updated_at", sa.DateTime(), nullable=False, backfill=_NOW),
        sa.Index("ix_asset_extractions_version", "dossier_version"),
    )
    sa.Table(
        "document_assets",
        metadata,
        _column("asset_id", sa.String(512), primary_key=True),
        _column("sha256", sa.String(64), nullable=False, index=True),
        _column("profile", sa.String(64), nullable=False, backfill="'base'"),
        _column("dossier_version", sa.Integer(), nullable=False, backfill="1"),
        _column("filename", sa.String(255), nullable=False, index=True),
        _column("page_number", sa.Integer(), nullable=False, backfill="0"),
        _column("role", sa.String(20), nullable=False, backfill="'figure'"),
        _column("tier", sa.String(20), nullable=False, backfill="'simple'"),
        _column("status", sa.String(20), nullable=False, backfill="'pending'"),
        _column("storage_uri", sa.String(1024), nullable=False, backfill="''"),
        _column("content_type", sa.String(80), nullable=False, backfill="''"),
        _column("byte_size", sa.Integer(), nullable=False, backfill="0"),
        _column("width", sa.Integer(), nullable=False, backfill="0"),
        _column("height", sa.Integer(), nullable=False, backfill="0"),
        _column("dossier", sa.JSON(), nullable=False, backfill="'{}'"),
        _column("created_at", sa.DateTime(), nullable=False, backfill=_NOW),
        _column("updated_at", sa.DateTime(), nullable=False, backfill=_NOW),
        sa.Index("ix_document_assets_filename_page", "filename", "page_number"),
        sa.Index("ix_document_assets_status_version", "status", "dossier_version"),
    )
    sa.Table(
        "entity_attributes",
        metadata,
        _column("id", sa.Integer(), primary_key=True, autoincrement=True),
        _column("asset_id", sa.String(512), nullable=False, index=True),
        _column("profile", sa.String(64), nullable=False, backfill="'base'"),
        _column("name", sa.String(64), nullable=False),
        _column("value_key", sa.String(255), nullable=False, backfill="''"),
        _column("value_text", sa.String(255), nullable=True),
        _column("value_number", sa.Float(), nullable=True),
        _column("value_bool", sa.Boolean(), nullable=True),
        _column("updated_at", sa.DateTime(), nullable=False, backfill=_NOW),
        sa.Index("ix_entity_attributes_name_text", "name", "value_text"),
        sa.Index("ix_entity_attributes_name_number", "name", "value_number"),
        sa.UniqueConstraint("asset_id", "name", "value_key", name="uq_entity_attribute_value"),
    )
    sa.Table(
        "parent_chunks",
        metadata,
        _column("chunk_id", sa.String(512), primary_key=True),
        _column("text", sa.Text(), nullable=False, backfill="''"),
        _column("filename", sa.String(255), nullable=False, index=True, backfill="''"),
        _column("file_type", sa.String(50), nullable=False, backfill="''"),
        _column("file_path", sa.String(1024), nullable=False, backfill="''"),
        _column("page_number", sa.Integer(), nullable=False, backfill="0"),
        _column("parent_chunk_id", sa.String(512), nullable=False, backfill="''"),
        _column("root_chunk_id", sa.String(512), nullable=False, backfill="''"),
        _column("chunk_level", sa.Integer(), nullable=False, backfill="0"),
        _column("chunk_idx", sa.Integer(), nullable=False, backfill="0"),
        _column("modality", sa.String(20), nullable=False, backfill="'text'"),
        _column("asset_ids", sa.JSON(), nullable=False, backfill="'[]'"),
        _column("updated_at", sa.DateTime(), nullable=False, backfill=_NOW),
    )
    sa.Table(
        "document_pairs",
        metadata,
        _column("pair_id", sa.String(64), primary_key=True),
        _column("title", sa.String(255), nullable=False, backfill="''"),
        _column("filename_ar", sa.String(255), nullable=False, backfill="''"),
        _column("filename_en", sa.String(255), nullable=False, backfill="''"),
        _column("created_at", sa.DateTime(), nullable=False, backfill=_NOW),
        _column("updated_at", sa.DateTime(), nullable=False, backfill=_NOW),
        sa.Index("ix_document_pairs_ar", "filename_ar"),
        sa.Index("ix_document_pairs_en", "filename_en"),
    )
    sa.Table(
        "corpus_digests",
        metadata,
        _column("profile", sa.String(64), primary_key=True),
        _column("paragraph", sa.Text(), nullable=False, backfill="''"),
        _column("sections_sha256", sa.String(64), nullable=False, backfill="''"),
        _column("section_count", sa.Integer(), nullable=False, backfill="0"),
        _column("floor", sa.Float(), nullable=False, backfill="0"),
        _column("floor_sha256", sa.String(64), nullable=False, backfill="''"),
        _column("question_count", sa.Integer(), nullable=False, backfill="0"),
        _column("model_used", sa.String(120), nullable=False, backfill="''"),
        _column("updated_at", sa.DateTime(), nullable=False, backfill=_NOW),
    )
    sa.Table(
        "section_summaries",
        metadata,
        _column("chunk_id", sa.String(512), primary_key=True),
        _column("profile", sa.String(64), primary_key=True),
        _column("content_sha256", sa.String(64), nullable=False, index=True, backfill="''"),
        _column("filename", sa.String(255), nullable=False, backfill="''"),
        _column("chunk_level", sa.Integer(), nullable=False, backfill="0"),
        _column("summary", sa.Text(), nullable=False, backfill="''"),
        _column("answers", sa.JSON(), nullable=False, backfill="'[]'"),
        _column("question_vectors", sa.JSON(), nullable=False, backfill="'[]'"),
        _column("embedding_model", sa.String(120), nullable=False, backfill="''"),
        _column("topics", sa.JSON(), nullable=False, backfill="'[]'"),
        _column("model_used", sa.String(120), nullable=False, backfill="''"),
        _column("created_at", sa.DateTime(), nullable=False, backfill=_NOW),
        _column("updated_at", sa.DateTime(), nullable=False, backfill=_NOW),
        sa.Index("ix_section_summaries_profile", "profile"),
        sa.Index("ix_section_summaries_filename", "filename"),
    )
    return metadata


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    for table in _schema().sorted_tables:
        if table.name not in existing:
            table.create(bind)
            continue
        _add_missing_columns(inspector, table)
        _add_missing_indexes(bind, inspector, table)
        _add_missing_unique_constraints(inspector, table)


def _add_missing_columns(inspector, table: sa.Table) -> None:
    present = {column["name"] for column in inspector.get_columns(table.name)}
    for column in table.columns:
        if column.name in present:
            continue
        backfill = column.info.get("backfill")
        if not column.nullable and backfill is None:
            raise RuntimeError(
                f"{table.name}.{column.name} is missing and has no value to fill existing "
                "rows with. This database is older than anything revision 0001 can adopt."
            )
        op.add_column(
            table.name,
            sa.Column(
                column.name,
                column.type,
                nullable=column.nullable,
                server_default=sa.text(backfill) if backfill else None,
            ),
        )
        if backfill:
            # The default existed only to fill the rows already there. New rows get their
            # value from the application, exactly as on a database created fresh.
            op.alter_column(table.name, column.name, server_default=None)


def _add_missing_indexes(bind, inspector, table: sa.Table) -> None:
    present = {index["name"] for index in inspector.get_indexes(table.name)}
    for index in table.indexes:
        if index.name not in present:
            index.create(bind)


def _add_missing_unique_constraints(inspector, table: sa.Table) -> None:
    present = {constraint["name"] for constraint in inspector.get_unique_constraints(table.name)}
    for constraint in table.constraints:
        if isinstance(constraint, sa.UniqueConstraint) and constraint.name not in present:
            op.create_unique_constraint(
                constraint.name, table.name, [column.name for column in constraint.columns]
            )


def downgrade() -> None:
    # Loses every row. Before this revision nothing was under alembic's control, so the
    # only inverse is the empty database. Take a backup first.
    bind = op.get_bind()
    for table in reversed(_schema().sorted_tables):
        table.drop(bind, checkfirst=True)
