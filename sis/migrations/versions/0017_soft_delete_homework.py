"""Keep deleted homework records and attachments recoverable.

Homework predates the ORM table set, so some deployed databases have its table while
new installations do not.  The migration owns both cases instead of relying on a route
to create schema at runtime.
"""
from alembic import op
import sqlalchemy as sa


revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def _homework_table() -> None:
    op.create_table(
        "teacher_homework",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("academic_year_code", sa.String(length=32), nullable=False),
        sa.Column("class_code", sa.String(length=64), nullable=False),
        sa.Column("subject_code", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("details", sa.Text(), nullable=False, server_default=""),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("stored_filename", sa.String(length=255), nullable=True),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("uploaded_on", sa.Date(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("uploaded_by", sa.String(length=120), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_teacher_homework_visible_teacher_year",
        "teacher_homework",
        ["academic_year_code", "uploaded_by", "deleted_at"],
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("teacher_homework"):
        _homework_table()
        return

    existing = {column["name"] for column in inspector.get_columns("teacher_homework")}
    with op.batch_alter_table("teacher_homework") as batch:
        if "deleted_at" not in existing:
            batch.add_column(sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
        indexes = {index["name"] for index in inspector.get_indexes("teacher_homework")}
        if "ix_teacher_homework_visible_teacher_year" not in indexes:
            batch.create_index(
                "ix_teacher_homework_visible_teacher_year",
                ["academic_year_code", "uploaded_by", "deleted_at"],
            )


def downgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("teacher_homework"):
        return
    with op.batch_alter_table("teacher_homework") as batch:
        batch.drop_index("ix_teacher_homework_visible_teacher_year")
        batch.drop_column("deleted_at")
