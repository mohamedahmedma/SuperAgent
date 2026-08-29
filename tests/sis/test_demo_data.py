"""The development demo is coherent enough to exercise real SIS workflows."""
from sqlalchemy import func, select

from sis.demo import blueprint as bp
from sis.demo import seeder
from sis.domain.rbac import RoleCode
from sis.infrastructure.crypto import verify_password
from sis.infrastructure.db import models as m
from sis.infrastructure.db.session import get_sessionmaker


def _seed(session):
    seeder.sync_roles(session)
    counts = seeder.load(session)
    session.commit()
    return counts


def test_demo_loads_a_complete_mixed_school(database):
    with get_sessionmaker()() as session:
        counts = _seed(session)

        assert counts.schools == 1
        assert counts.sections == 2
        assert counts.class_sections == 30
        assert counts.students == 418
        assert counts.users == len(bp.STAFF) == 14
        assert counts.teachers == 8
        assert counts.attendance > 1_000
        assert counts.grades > 1_000

        kinds = set(
            session.scalars(select(m.EducationalSystem.kind)).all()
        )
        assert kinds == {"arabic", "language"}


def test_demo_accounts_have_valid_passwords_and_additive_roles(database):
    with get_sessionmaker()() as session:
        _seed(session)
        users = {
            user.username: user
            for user in session.scalars(select(m.User)).all()
        }
        assert all(verify_password(bp.DEMO_PASSWORD, user.password_hash) for user in users.values())

        science_roles = set(
            session.scalars(
                select(m.Role.code)
                .join(m.UserRole, m.UserRole.role_id == m.Role.id)
                .where(m.UserRole.user_id == users["t.science"].id)
            ).all()
        )
        assert science_roles == {RoleCode.TEACHER.value, RoleCode.YEAR_SUPERVISOR.value}

        social_roles = set(
            session.scalars(
                select(m.Role.code)
                .join(m.UserRole, m.UserRole.role_id == m.Role.id)
                .where(m.UserRole.user_id == users["t.social"].id)
            ).all()
        )
        assert social_roles == {
            RoleCode.TEACHER.value,
            RoleCode.ATTENDANCE_SUPERVISOR.value,
        }


def test_demo_kg_is_named_kg_in_both_languages_and_sync_is_non_destructive(database):
    with get_sessionmaker()() as session:
        _seed(session)
        before_students = session.scalar(select(func.count()).select_from(m.Student))
        touched = seeder.sync_demo(session)
        session.commit()
        after_students = session.scalar(select(func.count()).select_from(m.Student))

        levels = {
            level.code: (level.name_en, level.name_ar)
            for level in session.scalars(
                select(m.YearLevel).where(m.YearLevel.code.in_(("AR-KG1", "AR-KG2", "LG-KG1", "LG-KG2")))
            ).all()
        }
        assert touched == 1 + len(bp.SECTIONS) + len(bp.RUNGS)
        assert before_students == after_students == 418
        assert levels == {
            "AR-KG1": ("KG 1", "KG 1"),
            "AR-KG2": ("KG 2", "KG 2"),
            "LG-KG1": ("KG 1", "KG 1"),
            "LG-KG2": ("KG 2", "KG 2"),
        }


def test_demo_assignments_attendance_and_grades_are_connected(database):
    with get_sessionmaker()() as session:
        counts = _seed(session)
        assert session.scalar(select(func.count()).select_from(m.TeacherSubject)) == counts.teacher_subjects
        assert session.scalar(select(func.count()).select_from(m.TeacherYearLevel)) == counts.teacher_year_levels
        assert session.scalar(select(func.count()).select_from(m.TeacherClassSection)) == counts.teacher_class_sections
        assert session.scalar(select(func.count()).select_from(m.Attendance)) == counts.attendance
        assert session.scalar(select(func.count()).select_from(m.SubjectGrade)) == counts.grades

        unassigned = session.scalars(
            select(m.Teacher).where(m.Teacher.staff_number == "T-008")
        ).one()
        assert session.scalar(
            select(func.count()).select_from(m.TeacherYearLevel).where(
                m.TeacherYearLevel.teacher_id == unassigned.id
            )
        ) == 1
        assert session.scalar(
            select(func.count()).select_from(m.TeacherClassSection).where(
                m.TeacherClassSection.teacher_id == unassigned.id
            )
        ) == 0
