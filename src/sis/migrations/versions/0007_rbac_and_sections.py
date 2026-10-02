"""People who log in, what they may do, and which section of the school a rung belongs to.

Revision ID: 0007
Revises: 0006

Two changes in one revision, because they are one thought: the service is growing from a
registrar's console — one credential, one job — into a system a whole school signs in to,
and the roles that arrive with it need a structure they can be scoped against.

**The staff half.** `users`, `roles`, `permissions`, `role_permissions`, `user_roles`,
`user_sessions`, `teachers` and the three teacher-assignment tables. Nothing here touches
`api_keys`: machine callers (`records/`, the import jobs) keep authenticating exactly as
they did, and a person signing in is a second, separate door. That is deliberate — folding
the two together would mean either giving `records/` a user account or giving a teacher an
API key, and both are worse than two doors.

The grant table is `user_roles`, and it carries a scope. A plain `(user, role)` join table
cannot say "supervisor **of Grade 4**" or "teacher **of 3A and 3B**", and a school's roles
are almost all of that shape. `scope_type` names which table `scope_id` points into, which
is why there is no foreign key on it — see the model.

**The structural half.** `educational_systems`, plus `year_levels.educational_system_id`,
`year_levels.grade_number` and `class_sections.section_number`. A school runs an Arabic
section and a language section side by side; they count rungs differently and name rooms
differently, and the display name is generated from these columns rather than typed.

**Every added column is nullable and every added table starts empty**, so this revision
changes the behaviour of nothing that already exists. A rung with no
`educational_system_id` renders from its stored label exactly as it did before 0007, and a
service with no `users` rows authenticates precisely as it did at 0006. The seed
(`python -m sis.seed_demo`) is what puts the first administrator in — not this migration,
because a migration that creates a credential creates the same credential on every
database it is ever run against, and that credential is then in a git history.

SQLite rebuilds a table to add a column with a foreign key, so the two `add_column` calls
that carry one go through batch mode. `sis/migrations/env.py` already turns
`render_as_batch` on for SQLite and verifies foreign keys afterwards; revision 0003's
docstring explains why.
"""
from alembic import op
import sqlalchemy as sa

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- The section a rung belongs to ------------------------------------------------
    op.create_table(
        "educational_systems",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("school_id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=16), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False, server_default="unspecified"),
        sa.Column("name_en", sa.String(length=160), nullable=False, server_default=""),
        sa.Column("name_ar", sa.String(length=160), nullable=False, server_default=""),
        sa.Column("display_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["school_id"],
            ["schools.id"],
            name="fk_educational_systems_school_id_schools",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_educational_systems"),
        sa.UniqueConstraint("school_id", "code", name="uq_educational_systems_school_code"),
    )
    op.create_index(
        "ix_educational_systems_school_id", "educational_systems", ["school_id"]
    )

    with op.batch_alter_table("year_levels") as batch:
        batch.add_column(sa.Column("educational_system_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("grade_number", sa.Integer(), nullable=True))
        batch.create_foreign_key(
            "fk_year_levels_educational_system_id_educational_systems",
            "educational_systems",
            ["educational_system_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        "ix_year_levels_educational_system_id", "year_levels", ["educational_system_id"]
    )

    with op.batch_alter_table("class_sections") as batch:
        batch.add_column(sa.Column("section_number", sa.Integer(), nullable=True))

    # -- Accounts ---------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("full_name_en", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("full_name_ar", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("preferred_language", sa.String(length=8), nullable=False, server_default="en"),
        sa.Column("school_id", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("failed_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["school_id"], ["schools.id"], name="fk_users_school_id_schools", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_school", "users", ["school_id"])

    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name_en", sa.String(length=160), nullable=False, server_default=""),
        sa.Column("name_ar", sa.String(length=160), nullable=False, server_default=""),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("default_scope", sa.String(length=16), nullable=False, server_default="school"),
        sa.Column("is_builtin", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_roles"),
        sa.UniqueConstraint("code", name="uq_roles_code"),
    )

    op.create_table(
        "permissions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", sa.String(length=48), nullable=False),
        sa.Column("name_en", sa.String(length=160), nullable=False, server_default=""),
        sa.Column("name_ar", sa.String(length=160), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("id", name="pk_permissions"),
        sa.UniqueConstraint("code", name="uq_permissions_code"),
    )

    op.create_table(
        "role_permissions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("permission_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["role_id"], ["roles.id"], name="fk_role_permissions_role_id_roles", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["permission_id"],
            ["permissions.id"],
            name="fk_role_permissions_permission_id_permissions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_role_permissions"),
        sa.UniqueConstraint(
            "role_id", "permission_id", name="uq_role_permissions_role_permission"
        ),
    )
    op.create_index("ix_role_permissions_role_id", "role_permissions", ["role_id"])
    op.create_index("ix_role_permissions_permission_id", "role_permissions", ["permission_id"])

    op.create_table(
        "user_roles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("scope_type", sa.String(length=16), nullable=False, server_default="school"),
        # No foreign key: which table this points into depends on `scope_type`.
        sa.Column("scope_id", sa.Integer(), nullable=True),
        sa.Column("granted_by", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_user_roles_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["role_id"], ["roles.id"], name="fk_user_roles_role_id_roles", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_roles"),
        sa.UniqueConstraint(
            "user_id", "role_id", "scope_type", "scope_id", name="uq_user_roles_user_role_scope"
        ),
    )
    op.create_index("ix_user_roles_user", "user_roles", ["user_id"])
    op.create_index("ix_user_roles_role_id", "user_roles", ["role_id"])

    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("client_ip", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_user_sessions_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_sessions"),
        sa.UniqueConstraint("token_hash", name="uq_user_sessions_token_hash"),
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])

    # -- Teaching staff and their assignments -----------------------------------------
    op.create_table(
        "teachers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("staff_number", sa.String(length=32), nullable=False),
        sa.Column("school_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("full_name_en", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("full_name_ar", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("email", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("phone", sa.String(length=16), nullable=False, server_default=""),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["school_id"],
            ["schools.id"],
            name="fk_teachers_school_id_schools",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_teachers_user_id_users", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_teachers"),
        sa.UniqueConstraint(
            "school_id", "staff_number", name="uq_teachers_school_staff_number"
        ),
    )
    op.create_index("ix_teachers_school_id", "teachers", ["school_id"])
    op.create_index("ix_teachers_user", "teachers", ["user_id"])

    op.create_table(
        "teacher_subjects",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("teacher_id", sa.Integer(), nullable=False),
        sa.Column("subject_id", sa.Integer(), nullable=False),
        sa.Column("academic_year_id", sa.Integer(), nullable=False),
        sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["teacher_id"],
            ["teachers.id"],
            name="fk_teacher_subjects_teacher_id_teachers",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["subject_id"],
            ["subjects.id"],
            name="fk_teacher_subjects_subject_id_subjects",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["academic_year_id"],
            ["academic_years.id"],
            name="fk_teacher_subjects_academic_year_id_academic_years",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_teacher_subjects"),
        sa.UniqueConstraint(
            "teacher_id", "subject_id", name="uq_teacher_subjects_teacher_subject"
        ),
    )
    op.create_index("ix_teacher_subjects_teacher_id", "teacher_subjects", ["teacher_id"])
    op.create_index("ix_teacher_subjects_subject_id", "teacher_subjects", ["subject_id"])
    op.create_index(
        "ix_teacher_subjects_academic_year_id", "teacher_subjects", ["academic_year_id"]
    )

    op.create_table(
        "teacher_year_levels",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("teacher_id", sa.Integer(), nullable=False),
        sa.Column("year_level_id", sa.Integer(), nullable=False),
        sa.Column("subject_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["teacher_id"],
            ["teachers.id"],
            name="fk_teacher_year_levels_teacher_id_teachers",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["year_level_id"],
            ["year_levels.id"],
            name="fk_teacher_year_levels_year_level_id_year_levels",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["subject_id"],
            ["subjects.id"],
            name="fk_teacher_year_levels_subject_id_subjects",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_teacher_year_levels"),
        sa.UniqueConstraint(
            "teacher_id",
            "year_level_id",
            "subject_id",
            name="uq_teacher_year_levels_teacher_level_subject",
        ),
    )
    op.create_index("ix_teacher_year_levels_teacher_id", "teacher_year_levels", ["teacher_id"])
    op.create_index("ix_teacher_year_levels_level", "teacher_year_levels", ["year_level_id"])
    op.create_index("ix_teacher_year_levels_subject_id", "teacher_year_levels", ["subject_id"])

    op.create_table(
        "teacher_class_sections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("teacher_id", sa.Integer(), nullable=False),
        sa.Column("class_section_id", sa.Integer(), nullable=False),
        sa.Column("subject_id", sa.Integer(), nullable=False),
        sa.Column("assigned_by", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["teacher_id"],
            ["teachers.id"],
            name="fk_teacher_class_sections_teacher_id_teachers",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["class_section_id"],
            ["class_sections.id"],
            name="fk_teacher_class_sections_class_section_id_class_sections",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["subject_id"],
            ["subjects.id"],
            name="fk_teacher_class_sections_subject_id_subjects",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_teacher_class_sections"),
        sa.UniqueConstraint(
            "teacher_id",
            "class_section_id",
            "subject_id",
            name="uq_teacher_class_sections_teacher_class_subject",
        ),
    )
    op.create_index(
        "ix_teacher_class_sections_teacher_id", "teacher_class_sections", ["teacher_id"]
    )
    op.create_index(
        "ix_teacher_class_sections_class", "teacher_class_sections", ["class_section_id"]
    )
    op.create_index(
        "ix_teacher_class_sections_subject_id", "teacher_class_sections", ["subject_id"]
    )

    # -- Estate-wide switches ---------------------------------------------------------
    op.create_table(
        "system_settings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Text(), nullable=False, server_default=""),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.Column("updated_by", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_system_settings"),
        sa.UniqueConstraint("key", name="uq_system_settings_key"),
    )


def downgrade() -> None:
    """Drop everything 0007 added, in dependency order.

    The two structural columns go too. That loses which section a rung belonged to and
    nothing else — the rungs, classes, children and marks are all untouched, because this
    revision never wrote to them.
    """
    op.drop_table("system_settings")

    op.drop_index("ix_teacher_class_sections_subject_id", table_name="teacher_class_sections")
    op.drop_index("ix_teacher_class_sections_class", table_name="teacher_class_sections")
    op.drop_index("ix_teacher_class_sections_teacher_id", table_name="teacher_class_sections")
    op.drop_table("teacher_class_sections")

    op.drop_index("ix_teacher_year_levels_subject_id", table_name="teacher_year_levels")
    op.drop_index("ix_teacher_year_levels_level", table_name="teacher_year_levels")
    op.drop_index("ix_teacher_year_levels_teacher_id", table_name="teacher_year_levels")
    op.drop_table("teacher_year_levels")

    op.drop_index("ix_teacher_subjects_academic_year_id", table_name="teacher_subjects")
    op.drop_index("ix_teacher_subjects_subject_id", table_name="teacher_subjects")
    op.drop_index("ix_teacher_subjects_teacher_id", table_name="teacher_subjects")
    op.drop_table("teacher_subjects")

    op.drop_index("ix_teachers_user", table_name="teachers")
    op.drop_index("ix_teachers_school_id", table_name="teachers")
    op.drop_table("teachers")

    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
    op.drop_table("user_sessions")

    op.drop_index("ix_user_roles_role_id", table_name="user_roles")
    op.drop_index("ix_user_roles_user", table_name="user_roles")
    op.drop_table("user_roles")

    op.drop_index("ix_role_permissions_permission_id", table_name="role_permissions")
    op.drop_index("ix_role_permissions_role_id", table_name="role_permissions")
    op.drop_table("role_permissions")

    op.drop_table("permissions")
    op.drop_table("roles")

    op.drop_index("ix_users_school", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")

    with op.batch_alter_table("class_sections") as batch:
        batch.drop_column("section_number")

    op.drop_index("ix_year_levels_educational_system_id", table_name="year_levels")
    with op.batch_alter_table("year_levels") as batch:
        batch.drop_constraint(
            "fk_year_levels_educational_system_id_educational_systems", type_="foreignkey"
        )
        batch.drop_column("grade_number")
        batch.drop_column("educational_system_id")

    op.drop_index("ix_educational_systems_school_id", table_name="educational_systems")
    op.drop_table("educational_systems")
