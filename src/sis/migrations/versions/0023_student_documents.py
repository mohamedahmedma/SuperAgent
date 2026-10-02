"""Student record documents."""
from alembic import op
import sqlalchemy as sa
revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None
def upgrade():
    op.create_table("student_documents", sa.Column("id", sa.String(36), primary_key=True), sa.Column("student_id", sa.Integer(), sa.ForeignKey("students.id", ondelete="RESTRICT"), nullable=False), sa.Column("document_type", sa.String(64), nullable=False), sa.Column("file_key", sa.String(128), unique=True, nullable=False), sa.Column("original_filename", sa.String(255), nullable=False), sa.Column("mime_type", sa.String(128), nullable=False), sa.Column("size_bytes", sa.Integer(), nullable=False), sa.Column("uploaded_by", sa.String(64), nullable=False), sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False), sa.Column("expiry_date", sa.Date(), nullable=True), sa.Column("status", sa.String(24), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False), sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_student_documents_student_active", "student_documents", ["student_id", "deleted_at"])
def downgrade():
    op.drop_index("ix_student_documents_student_active", table_name="student_documents"); op.drop_table("student_documents")
