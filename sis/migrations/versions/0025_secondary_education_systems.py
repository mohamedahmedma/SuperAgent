"""Persist secondary education systems, tracks, and grade rules."""

from alembic import op
import sqlalchemy as sa

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "secondary_education_systems",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("school_id", sa.Integer(), sa.ForeignKey("schools.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.String(32), nullable=False),
        sa.Column("name_en", sa.String(160), nullable=False),
        sa.Column("name_ar", sa.String(160), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("school_id", "key", name="uq_secondary_systems_school_key"),
        sa.CheckConstraint("key IN ('general_secondary', 'egyptian_baccalaureate')", name="ck_secondary_system_key"),
    )
    op.create_table(
        "secondary_tracks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("education_system_id", sa.Integer(), sa.ForeignKey("secondary_education_systems.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.String(48), nullable=False),
        sa.Column("name_en", sa.String(160), nullable=False),
        sa.Column("name_ar", sa.String(160), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.UniqueConstraint("education_system_id", "key", name="uq_secondary_tracks_system_key"),
    )
    op.create_index("ix_secondary_tracks_education_system_id", "secondary_tracks", ["education_system_id"])
    op.create_table(
        "secondary_track_grades",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("secondary_track_id", sa.Integer(), sa.ForeignKey("secondary_tracks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("grade_number", sa.Integer(), nullable=False),
        sa.UniqueConstraint("secondary_track_id", "grade_number", name="uq_secondary_track_grade"),
        sa.CheckConstraint("grade_number IN (1, 2, 3)", name="ck_secondary_track_grade_number"),
    )
    op.create_index("ix_secondary_track_grades_secondary_track_id", "secondary_track_grades", ["secondary_track_id"])
    bind = op.get_bind()
    systems = sa.table(
        "secondary_education_systems", sa.column("id"), sa.column("school_id"), sa.column("key"),
        sa.column("name_en"), sa.column("name_ar"), sa.column("is_active"), sa.column("created_at"),
    )
    tracks = sa.table(
        "secondary_tracks", sa.column("id"), sa.column("education_system_id"), sa.column("key"),
        sa.column("name_en"), sa.column("name_ar"), sa.column("is_active"),
    )
    rules = sa.table("secondary_track_grades", sa.column("secondary_track_id"), sa.column("grade_number"))
    definitions = (
        ("general_secondary", "General Secondary", "الثانوي العام"),
        ("egyptian_baccalaureate", "Egyptian Baccalaureate", "البكالوريا المصرية"),
    )
    track_definitions = (
        ("general_secondary", "scientific", "Scientific", "علمي", (2,)),
        ("general_secondary", "literary", "Literary", "أدبي", (2, 3)),
        ("general_secondary", "science", "Science", "علوم", (3,)),
        ("general_secondary", "mathematics", "Mathematics", "رياضيات", (3,)),
        ("egyptian_baccalaureate", "medicine_life_sciences", "Medicine and Life Sciences", "الطب وعلوم الحياة", (1, 2, 3)),
        ("egyptian_baccalaureate", "engineering_computer_science", "Engineering and Computer Science", "الهندسة وعلوم الحاسب", (1, 2, 3)),
        ("egyptian_baccalaureate", "business", "Business", "الأعمال", (1, 2, 3)),
        ("egyptian_baccalaureate", "arts_humanities", "Arts and Humanities", "الآداب والعلوم الإنسانية", (1, 2, 3)),
    )
    now = sa.func.now()
    for (school_id,) in bind.execute(sa.text("SELECT id FROM schools")):
        bind.execute(systems.insert(), [
            {"school_id": school_id, "key": key, "name_en": en, "name_ar": ar,
             "is_active": True, "created_at": now}
            for key, en, ar in definitions
        ])
        system_ids = dict(bind.execute(
            sa.select(systems.c.key, systems.c.id).where(systems.c.school_id == school_id)
        ).all())
        bind.execute(tracks.insert(), [
            {"education_system_id": system_ids[system_key], "key": key, "name_en": en,
             "name_ar": ar, "is_active": True}
            for system_key, key, en, ar, _ in track_definitions
        ])
        track_ids = {
            (system_key, key): identifier
            for system_key, key, identifier in bind.execute(
                sa.select(systems.c.key, tracks.c.key, tracks.c.id)
                .join(tracks, tracks.c.education_system_id == systems.c.id)
                .where(systems.c.school_id == school_id)
            ).all()
        }
        bind.execute(rules.insert(), [
            {"secondary_track_id": track_ids[(system_key, key)], "grade_number": grade}
            for system_key, key, _, _, grades in track_definitions for grade in grades
        ])


def downgrade() -> None:
    op.drop_index("ix_secondary_track_grades_secondary_track_id", table_name="secondary_track_grades")
    op.drop_table("secondary_track_grades")
    op.drop_index("ix_secondary_tracks_education_system_id", table_name="secondary_tracks")
    op.drop_table("secondary_tracks")
    op.drop_table("secondary_education_systems")
