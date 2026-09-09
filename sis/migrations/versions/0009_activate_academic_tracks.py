"""Activate the educational-system rows introduced by revision 0007.

Revision ID: 0009
Revises: 0008
"""
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO educational_systems
            (school_id, code, kind, name_en, name_ar, display_order, is_active, created_at)
        SELECT id, 'AR', 'arabic', 'Arabic', 'العربية', 1, true, CURRENT_TIMESTAMP
        FROM schools WHERE language_type IN ('arabic', 'both')
          AND NOT EXISTS (
              SELECT 1 FROM educational_systems e
              WHERE e.school_id = schools.id AND e.code = 'AR'
          )
    """)
    op.execute("""
        INSERT INTO educational_systems
            (school_id, code, kind, name_en, name_ar, display_order, is_active, created_at)
        SELECT id, 'LANG', 'language', 'Languages', 'اللغات', 2, true, CURRENT_TIMESTAMP
        FROM schools WHERE language_type IN ('languages', 'both')
          AND NOT EXISTS (
              SELECT 1 FROM educational_systems e
              WHERE e.school_id = schools.id AND e.code = 'LANG'
          )
    """)
    # A legacy single-track school's unclassified rungs have only one possible owner.
    op.execute("""
        UPDATE year_levels SET educational_system_id = (
            SELECT e.id FROM educational_systems e
            WHERE e.school_id = year_levels.school_id AND e.is_active = true
        )
        WHERE educational_system_id IS NULL
          AND (SELECT COUNT(*) FROM educational_systems e
               WHERE e.school_id = year_levels.school_id AND e.is_active = true) = 1
    """)


def downgrade() -> None:
    # Rows may now be referenced by year levels. Stage 3 is therefore data-preserving on
    # downgrade; revision 0007 remains the owner of the table and columns.
    pass
