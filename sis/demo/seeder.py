"""Write the demo school into a database, and take it out again.

Four operations, and the third one is the reason this file is careful:

    load    insert everything `blueprint.py` describes
    sync    update mutable labels on an existing demo without deleting any rows
    reset   delete the demo school and its people, then load again
    status  report what is present without writing anything

**`reset` deletes by school code and by nothing else.** It finds the `DEMO` school, walks
down to its rows and removes them, and it removes the users whose `school_id` is that
school plus the demo accounts that belong to no school. A database holding both this demo
and a real school loses only the demo. There is no `DELETE FROM students` anywhere in this
file, and there should never be one: a seeder that truncates is a seeder that will one day
be run against the wrong `SIS_DATABASE_URL`.

**It refuses to run against anything that looks like production.** SQLite is taken as a
development database and needs no argument; anything else has to be opted into explicitly,
and an environment naming itself production is refused outright. See `guard_environment`.

**Reference data and demo data are different things.** The seven built-in roles and the
permission catalogue are not demo data — the service needs them to authorise anybody at
all — so `sync_roles` is separate, is idempotent, and is what a real deployment runs. It
is called by `load` for convenience and never undone by `reset`.

Everything is written through the ORM models rather than raw SQL so the foreign keys, the
check constraints and the naming convention all apply exactly as they do at runtime. That
is slower than a bulk insert and it is the point: a demo that only loads because it
bypassed a constraint is a demo that hides the constraint.
"""
from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from sis.config import get_settings
from sis.demo import blueprint as bp
from sis.demo.names import family_of, guardian_name, student_name
from sis.domain.attendance import AttendanceState
from sis.domain.naming import ClassCoordinates, render_class_title
from sis.domain.people import Gender
from sis.domain.rbac import BUILT_IN_ROLES, Permission, RoleCode, ScopeType, SystemStatus
from sis.infrastructure.crypto import hash_password
from sis.infrastructure.db import models as m
from sis.infrastructure.db.session import get_sessionmaker

log = logging.getLogger("sis.demo")

# The key `SystemSetting` keeps the estate's status under. One string, spelled once.
SYSTEM_STATUS_KEY = "system.status"

# The register is written backwards from here, which sits comfortably inside Term 1 of the
# demo year. A fixed anchor rather than "today" so two runs a week apart produce the same
# dates and a screenshot from last Tuesday still matches.
REGISTER_ANCHOR = date(2025, 11, 20)

# Egypt: the school week runs Sunday to Thursday. `date.weekday()` is Monday-zero, so
# Friday is 4 and Saturday is 5.
_WEEKEND = frozenset({4, 5})

# Which subjects each stage is marked in. Marking a kindergarten child in Computer Science
# would be data nobody recognises, and a demo whose numbers are obviously wrong is one
# people stop trusting for the numbers that are right.
_SUBJECTS_BY_DEPTH = {
    "junior": ("AR", "EN", "MA", "SC"),
    "senior": ("AR", "EN", "MA", "SC", "SS", "CS"),
}


class DemoRefused(RuntimeError):
    """The seeder declined to write. Carries the reason, which is always actionable."""


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

_PRODUCTION_NAMES = frozenset({"production", "prod", "live"})


def guard_environment(*, allow_remote: bool = False) -> None:
    """Refuse to seed anything that might be real. Raises `DemoRefused` with the reason.

    Two independent checks, because they catch different mistakes:

    **A named production environment is refused outright**, flag or no flag. If somebody
    has gone to the trouble of setting `SIS_ENV=production`, no command-line argument
    should be able to talk the seeder past it.

    **A non-SQLite database needs an explicit opt-in.** The default `sqlite:///./sis.db`
    is a file on a developer's laptop and is what this is for. A Postgres URL is a server,
    and a server is shared — so `--allow-remote` (or `SIS_ALLOW_DEMO_SEED=1`) has to say
    out loud that this particular one is a development server. That is not security; it is
    the pause between typing a command and running it against staging.
    """
    for variable in ("SIS_ENV", "APP_ENV", "ENVIRONMENT"):
        value = (os.getenv(variable) or "").strip().lower()
        if value in _PRODUCTION_NAMES:
            raise DemoRefused(
                f"{variable}={value!r}: this is fictional data and will not be written to "
                "a production environment. Unset it, or point SIS_DATABASE_URL at a "
                "development database."
            )

    settings = get_settings()
    if settings.is_sqlite:
        return

    opted_in = allow_remote or (os.getenv("SIS_ALLOW_DEMO_SEED") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not opted_in:
        raise DemoRefused(
            "SIS_DATABASE_URL does not point at a local SQLite file, so the seeder will "
            "not guess that it is safe to write four hundred fictional children into it.\n"
            "If it really is a development database, re-run with --allow-remote (or set "
            "SIS_ALLOW_DEMO_SEED=1)."
        )


# ---------------------------------------------------------------------------
# What a run produced, for the report at the end
# ---------------------------------------------------------------------------


@dataclass
class Counts:
    """Tallies, printed when a run finishes. Plain ints; nothing derived."""

    schools: int = 0
    sections: int = 0
    academic_years: int = 0
    terms: int = 0
    year_levels: int = 0
    class_sections: int = 0
    subjects: int = 0
    students: int = 0
    enrolments: int = 0
    guardians: int = 0
    users: int = 0
    role_grants: int = 0
    teachers: int = 0
    teacher_subjects: int = 0
    teacher_year_levels: int = 0
    teacher_class_sections: int = 0
    attendance: int = 0
    grades: int = 0

    def as_lines(self) -> list[str]:
        return [
            f"  {name.replace('_', ' '):<22} {value}"
            for name, value in vars(self).items()
            if value
        ]


# ---------------------------------------------------------------------------
# Reference data: the roles the service needs to authorise anybody
# ---------------------------------------------------------------------------


def sync_roles(session: Session) -> tuple[int, int]:
    """Insert or update the built-in roles and the permission catalogue. Idempotent.

    Not demo data — a real deployment runs this too, which is why it is a separate
    function and why `reset` never removes it. Existing rows are updated rather than
    replaced, so a role a school has granted to forty people keeps its id.

    Returns `(roles touched, permissions touched)`.
    """
    existing_permissions = {
        row.code: row for row in session.scalars(select(m.PermissionRow)).all()
    }
    for permission in Permission:
        row = existing_permissions.get(permission.value)
        # The label is derived from the code rather than kept in a second table: "noun.verb"
        # reads as "Verb noun" and that is a better label than most hand-written ones.
        noun, _, verb = permission.value.rpartition(".")
        label = f"{verb.replace('_', ' ').title()} {noun.replace('.', ' ')}"
        if row is None:
            session.add(
                m.PermissionRow(code=permission.value, name_en=label, name_ar=label)
            )
        else:
            row.name_en = row.name_en or label
    session.flush()

    permission_ids = {
        row.code: row.id for row in session.scalars(select(m.PermissionRow)).all()
    }

    roles_touched = 0
    for definition in BUILT_IN_ROLES:
        role = session.scalars(
            select(m.Role).where(m.Role.code == definition.code.value)
        ).one_or_none()
        if role is None:
            role = m.Role(code=definition.code.value)
            session.add(role)
        role.name_en = definition.name_en
        role.name_ar = definition.name_ar
        role.description = definition.description_en
        role.default_scope = definition.default_scope.value
        role.is_builtin = True
        session.flush()
        roles_touched += 1

        # Re-stated wholesale rather than diffed: a built-in role's permission set is
        # owned by the code, and a row somebody added by hand should not survive an
        # upgrade that removed the permission it grants.
        session.execute(
            delete(m.RolePermission).where(m.RolePermission.role_id == role.id)
        )
        for permission in definition.permissions:
            session.add(
                m.RolePermission(
                    role_id=role.id, permission_id=permission_ids[permission.value]
                )
            )

    session.flush()
    return roles_touched, len(permission_ids)


# ---------------------------------------------------------------------------
# Small helpers the loader leans on
# ---------------------------------------------------------------------------


def school_days(*, ending: date, count: int) -> list[date]:
    """The last `count` Sunday-to-Thursday days up to and including `ending`, ascending."""
    days: list[date] = []
    cursor = ending
    while len(days) < count:
        if cursor.weekday() not in _WEEKEND:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return sorted(days)


def _depth_for(grade_number: int, stage: str) -> str:
    """Which subject list a rung is marked in. KG and infants get the short one."""
    if stage in {"garden"}:
        return "junior"
    return "junior" if grade_number <= 4 else "senior"


def _dice(*parts: object) -> random.Random:
    """A generator seeded from what it is generating for.

    Deterministic per student, per class, per day — so the same seed run produces the same
    register twice, and a bug report naming a child names the same child tomorrow.
    """
    return random.Random("|".join(str(part) for part in parts))


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


def demo_school(session: Session) -> m.School | None:
    return session.scalars(
        select(m.School).where(m.School.code == bp.SCHOOL_CODE)
    ).one_or_none()


def sync_demo(session: Session) -> int:
    """Update mutable blueprint labels without deleting or recreating demo records.

    This is the safe path for an already-loaded development database. Stable codes and
    foreign keys stay untouched; only bilingual names and section metadata are restated.
    It deliberately does not try to grow a partial seed into a complete one because that
    would hide missing relationships. Use ``load`` for a clean database and ``status`` to
    detect an incomplete one.
    """
    school = demo_school(session)
    if school is None:
        raise DemoRefused(
            f"the demo school {bp.SCHOOL_CODE} is not present; run `python -m sis.demo load`"
        )

    touched = 0
    school.name_en = bp.SCHOOL_NAME_EN
    school.name_ar = bp.SCHOOL_NAME_AR
    touched += 1

    sections = {
        row.code: row
        for row in session.scalars(
            select(m.EducationalSystem).where(m.EducationalSystem.school_id == school.id)
        ).all()
    }
    for spec in bp.SECTIONS:
        row = sections.get(spec.code)
        if row is None:
            raise DemoRefused(f"demo section {spec.code} is missing; use `reset` to rebuild the demo")
        row.kind = spec.kind.value
        row.name_en = spec.name_en
        row.name_ar = spec.name_ar
        row.display_order = spec.display_order
        touched += 1

    levels = {
        row.code: row
        for row in session.scalars(
            select(m.YearLevel).where(m.YearLevel.school_id == school.id)
        ).all()
    }
    for spec in bp.RUNGS:
        row = levels.get(spec.code)
        if row is None:
            raise DemoRefused(f"demo level {spec.code} is missing; use `reset` to rebuild the demo")
        row.name_en = spec.name_en
        row.name_ar = spec.name_ar
        row.stage = spec.stage.value
        row.grade_number = spec.grade_number
        row.display_order = spec.display_order
        touched += 1

    session.flush()
    return touched


def remove(session: Session) -> int:
    """Delete the demo school and everything hanging off it. Returns rows removed.

    Ordered from the leaves inward, because every foreign key in this schema is RESTRICT
    or CASCADE and the RESTRICTs will refuse a parent that still has children — which is
    the correct behaviour and is why the order is written out rather than left to a
    cascade nobody has verified.

    A database with no demo school is a no-op, not an error: `reset` on a clean database
    should work.
    """
    school = demo_school(session)
    removed = 0

    # Accounts first: `users.school_id` RESTRICTs the school, and the demo administrator
    # belongs to no school at all so it has to be found by username.
    demo_usernames = [person.username for person in bp.STAFF]
    user_ids = [
        row
        for row in session.scalars(
            select(m.User.id).where(m.User.username.in_(demo_usernames))
        ).all()
    ]
    if user_ids:
        # Sessions and role grants cascade from `users`, but they are deleted explicitly:
        # a cascade that silently does nothing (SQLite with foreign keys off, say) would
        # leave orphan grants that still authorise.
        removed += session.execute(
            delete(m.UserSession).where(m.UserSession.user_id.in_(user_ids))
        ).rowcount or 0
        removed += session.execute(
            delete(m.UserRole).where(m.UserRole.user_id.in_(user_ids))
        ).rowcount or 0

    if school is None:
        if user_ids:
            removed += session.execute(
                delete(m.User).where(m.User.id.in_(user_ids))
            ).rowcount or 0
        return removed

    year_ids = list(
        session.scalars(
            select(m.AcademicYear.id).where(m.AcademicYear.school_id == school.id)
        ).all()
    )
    level_ids = list(
        session.scalars(
            select(m.YearLevel.id).where(m.YearLevel.school_id == school.id)
        ).all()
    )
    class_ids = (
        list(
            session.scalars(
                select(m.ClassSection.id).where(
                    m.ClassSection.academic_year_id.in_(year_ids)
                )
            ).all()
        )
        if year_ids
        else []
    )
    subject_ids = (
        list(
            session.scalars(
                select(m.Subject.id).where(m.Subject.academic_year_id.in_(year_ids))
            ).all()
        )
        if year_ids
        else []
    )
    term_ids = (
        list(
            session.scalars(
                select(m.Term.id).where(m.Term.academic_year_id.in_(year_ids))
            ).all()
        )
        if year_ids
        else []
    )
    student_ids = (
        list(
            session.scalars(
                select(m.ClassEnrolment.student_id)
                .where(m.ClassEnrolment.class_section_id.in_(class_ids))
                .distinct()
            ).all()
        )
        if class_ids
        else []
    )

    def wipe(statement) -> None:
        nonlocal removed
        removed += session.execute(statement).rowcount or 0

    teacher_ids = list(
        session.scalars(select(m.Teacher.id).where(m.Teacher.school_id == school.id)).all()
    )
    if teacher_ids:
        wipe(delete(m.TeacherClassSection).where(m.TeacherClassSection.teacher_id.in_(teacher_ids)))
        wipe(delete(m.TeacherYearLevel).where(m.TeacherYearLevel.teacher_id.in_(teacher_ids)))
        wipe(delete(m.TeacherSubject).where(m.TeacherSubject.teacher_id.in_(teacher_ids)))
        wipe(delete(m.Teacher).where(m.Teacher.id.in_(teacher_ids)))

    if student_ids:
        wipe(delete(m.SubjectGrade).where(m.SubjectGrade.student_id.in_(student_ids)))
        wipe(delete(m.Attendance).where(m.Attendance.student_id.in_(student_ids)))
        wipe(delete(m.ClassEnrolment).where(m.ClassEnrolment.student_id.in_(student_ids)))

        guardian_ids = list(
            session.scalars(
                select(m.StudentGuardian.guardian_id)
                .where(m.StudentGuardian.student_id.in_(student_ids))
                .distinct()
            ).all()
        )
        wipe(delete(m.StudentGuardian).where(m.StudentGuardian.student_id.in_(student_ids)))
        if guardian_ids:
            wipe(delete(m.GuardianPhone).where(m.GuardianPhone.guardian_id.in_(guardian_ids)))
            wipe(delete(m.Guardian).where(m.Guardian.id.in_(guardian_ids)))
        wipe(delete(m.Student).where(m.Student.id.in_(student_ids)))

    if class_ids:
        wipe(delete(m.ClassSection).where(m.ClassSection.id.in_(class_ids)))
    if subject_ids:
        wipe(delete(m.Subject).where(m.Subject.id.in_(subject_ids)))
    if term_ids:
        wipe(delete(m.Term).where(m.Term.id.in_(term_ids)))
    if level_ids:
        wipe(delete(m.YearLevel).where(m.YearLevel.id.in_(level_ids)))
    if year_ids:
        wipe(delete(m.AcademicYear).where(m.AcademicYear.id.in_(year_ids)))

    if user_ids:
        wipe(delete(m.User).where(m.User.id.in_(user_ids)))

    wipe(delete(m.EducationalSystem).where(m.EducationalSystem.school_id == school.id))
    wipe(delete(m.School).where(m.School.id == school.id))
    session.flush()
    return removed


# ---------------------------------------------------------------------------
# The load
# ---------------------------------------------------------------------------


@dataclass
class _Built:
    """Ids the loader resolves as it goes, keyed by the codes the blueprint uses."""

    school_id: int = 0
    year_id: int = 0
    sections: dict[str, int] = field(default_factory=dict)
    rungs: dict[str, int] = field(default_factory=dict)
    rooms: dict[str, int] = field(default_factory=dict)
    subjects: dict[str, int] = field(default_factory=dict)
    terms: dict[str, int] = field(default_factory=dict)
    # Which children sit in which room, in roll order, so registers and marks agree.
    roll: dict[str, list[int]] = field(default_factory=dict)
    users: dict[str, int] = field(default_factory=dict)
    teachers: dict[str, int] = field(default_factory=dict)


def load(session: Session, counts: Counts | None = None) -> Counts:
    """Write the whole demo. Assumes `remove` has run, or that the database is clean."""
    counts = counts or Counts()
    now = datetime.now(UTC)
    built = _Built()

    _load_school(session, built, counts, now)
    _load_structure(session, built, counts, now)
    _load_students(session, built, counts, now)
    _load_accounts(session, built, counts, now)
    _load_teaching(session, built, counts, now)
    _load_registers(session, built, counts, now)
    _load_marks(session, built, counts, now)
    _ensure_system_status(session, now)
    session.flush()
    return counts


def _load_school(session: Session, built: _Built, counts: Counts, now: datetime) -> None:
    school = m.School(
        code=bp.SCHOOL_CODE,
        name_en=bp.SCHOOL_NAME_EN,
        name_ar=bp.SCHOOL_NAME_AR,
        is_active=True,
        created_at=now,
    )
    session.add(school)
    session.flush()
    built.school_id = school.id
    counts.schools += 1

    for spec in bp.SECTIONS:
        row = m.EducationalSystem(
            school_id=school.id,
            code=spec.code,
            kind=spec.kind.value,
            name_en=spec.name_en,
            name_ar=spec.name_ar,
            display_order=spec.display_order,
            is_active=True,
            created_at=now,
        )
        session.add(row)
        session.flush()
        built.sections[spec.code] = row.id
        counts.sections += 1


def _load_structure(session: Session, built: _Built, counts: Counts, now: datetime) -> None:
    year = m.AcademicYear(
        code=bp.YEAR_CODE,
        school_id=built.school_id,
        name_en="2025 / 2026",
        name_ar="٢٠٢٥ / ٢٠٢٦",
        starts_on=bp.YEAR_STARTS,
        ends_on=bp.YEAR_ENDS,
        is_current=True,
        created_at=now,
        updated_at=now,
    )
    session.add(year)
    session.flush()
    built.year_id = year.id
    counts.academic_years += 1

    for spec in bp.TERMS:
        term = m.Term(
            code=spec.code,
            academic_year_id=year.id,
            name_en=spec.name_en,
            name_ar=spec.name_ar,
            starts_on=spec.starts_on,
            ends_on=spec.ends_on,
            sequence=spec.sequence,
            is_closed=False,
            created_at=now,
            updated_at=now,
        )
        session.add(term)
        session.flush()
        built.terms[spec.code] = term.id
        counts.terms += 1

    for spec in bp.SUBJECTS:
        subject = m.Subject(
            code=spec.code,
            academic_year_id=year.id,
            name_en=spec.name_en,
            name_ar=spec.name_ar,
            display_order=spec.display_order,
            is_active=True,
            created_at=now,
        )
        session.add(subject)
        session.flush()
        built.subjects[spec.code] = subject.id
        counts.subjects += 1

    for rung in bp.RUNGS:
        level = m.YearLevel(
            code=rung.code,
            school_id=built.school_id,
            name_en=rung.name_en,
            name_ar=rung.name_ar,
            display_order=rung.display_order,
            stage=rung.stage.value,
            educational_system_id=built.sections[rung.section],
            grade_number=rung.grade_number,
            created_at=now,
        )
        session.add(level)
        session.flush()
        built.rungs[rung.code] = level.id
        counts.year_levels += 1

        for index, (section_number, label_en, label_ar) in enumerate(rung.rooms, start=1):
            # The code is the machine identity and obeys `ClassCode` — uppercase, no
            # spaces, no slash. The written form a school recognises (`1/2 ب`, `Simba
            # Class`) lives in the labels, and the full title is generated from the
            # structured columns by `sis/domain/naming.py`.
            code = f"{rung.code}-{section_number or index}"
            room = m.ClassSection(
                academic_year_id=year.id,
                year_level_id=level.id,
                code=code,
                name_en=label_en,
                name_ar=label_ar,
                section_number=section_number,
                capacity=30,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            session.add(room)
            session.flush()
            built.rooms[bp.room_ref(rung.code, label_en)] = room.id
            counts.class_sections += 1


def _load_students(session: Session, built: _Built, counts: Counts, now: datetime) -> None:
    """One roll per room, with a placement open from the first day of the year.

    Placements start on the year's first day rather than today, so "which class was she in
    on 12 November" answers correctly for the registers written below — a placement that
    began after the register would make every one of those rows unreadable.
    """
    serial = 0
    for rung in bp.RUNGS:
        for _, label_en, _ in rung.rooms:
            key = bp.room_ref(rung.code, label_en)
            plan = bp.CLASS_PLANS.get(key, bp.DEFAULT_PLAN)
            room_id = built.rooms[key]
            roll: list[int] = []

            for seat in range(plan.students):
                serial += 1
                female = (serial % 2) == 0
                name_ar, name_en = student_name(serial, female=female)
                # A number a person can read: school, year, sequence. Uppercase and
                # hyphenated, which is what `StudentNumber` accepts.
                number = f"{bp.SCHOOL_CODE}-2026-{serial:05d}"
                dice = _dice("dob", number)
                # Age follows the rung, roughly: kindergarten at four, and a year per rung
                # after that. A demo whose Grade 12 children are five years old is one
                # nobody reads twice.
                born_year = 2025 - (4 + rung.display_order // 10 * 3 + (rung.grade_number or 1))
                student = m.Student(
                    student_number=number,
                    full_name_ar=name_ar,
                    full_name_en=name_en,
                    is_active=True,
                    date_of_birth=date(
                        born_year, dice.randint(1, 12), dice.randint(1, 28)
                    ),
                    gender=(Gender.FEMALE if female else Gender.MALE).value,
                    contact_phone="",
                    contact_email="",
                    address="",
                    created_at=now,
                    updated_at=now,
                )
                session.add(student)
                session.flush()
                counts.students += 1
                roll.append(student.id)

                session.add(
                    m.ClassEnrolment(
                        student_id=student.id,
                        class_section_id=room_id,
                        starts_on=bp.YEAR_STARTS,
                        ends_on=None,
                        reason="initial",
                        created_at=now,
                        updated_at=now,
                    )
                )
                counts.enrolments += 1

                # A guardian for the first four children of each room. Enough to exercise
                # the guardian screens and the parent-facing routes without doubling the
                # size of the seed — the flow is identical for the fifth child.
                if seat < 4:
                    _add_guardian(session, student.id, serial, now)
                    counts.guardians += 1

            built.roll[key] = roll


def _add_guardian(session: Session, student_id: int, serial: int, now: datetime) -> None:
    """One parent, one number, permitted to read records. The ordinary case."""
    family = family_of(serial)
    mother = (serial % 3) == 0
    name_ar, name_en = guardian_name(serial, female=mother, family=family)
    guardian = m.Guardian(
        public_id=f"G-DEMO-{serial:05d}",
        full_name_ar=name_ar,
        full_name_en=name_en,
        preferred_language="ar",
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    session.add(guardian)
    session.flush()
    # A number in Egypt's mobile range that cannot reach anybody: the 0100 000 xxxx block
    # is not allocated. Fictional data must not be dialable.
    session.add(
        m.GuardianPhone(
            guardian_id=guardian.id,
            phone=f"+2010000{serial:05d}",
            is_primary=True,
            created_at=now,
        )
    )
    session.add(
        m.StudentGuardian(
            student_id=student_id,
            guardian_id=guardian.id,
            relationship_type="mother" if mother else "father",
            relationship_label="",
            is_primary_contact=True,
            can_view_records=True,
            restriction_note="",
            created_at=now,
            updated_at=now,
        )
    )


def _load_accounts(session: Session, built: _Built, counts: Counts, now: datetime) -> None:
    """Every demo login, and the scoped role grants behind it.

    The password is hashed properly — the same PBKDF2 a real account gets — rather than
    written in plaintext with a comment promising to fix it. A demo that takes a shortcut
    through the credential path is a demo that cannot test the credential path.
    """
    roles = {row.code: row.id for row in session.scalars(select(m.Role)).all()}
    # Hashed once and reused: six hundred thousand rounds times fourteen accounts is
    # several seconds of a seed run, and every demo account shares the one password.
    shared_hash = hash_password(bp.DEMO_PASSWORD)

    for person in bp.STAFF:
        is_admin = any(grant.role is RoleCode.SYSTEM_ADMIN for grant in person.roles)
        user = m.User(
            username=person.username,
            password_hash=shared_hash,
            email=person.email,
            full_name_en=person.full_name_en,
            full_name_ar=person.full_name_ar,
            preferred_language=person.language,
            # The administrator belongs to no school. Everybody else is bound to this one,
            # which is a second wall behind the role scopes rather than a substitute.
            school_id=None if is_admin else built.school_id,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        session.add(user)
        session.flush()
        built.users[person.username] = user.id
        counts.users += 1

        for grant in person.roles:
            scope_id = _resolve_scope(built, grant.scope_type, grant.scope_ref)
            session.add(
                m.UserRole(
                    user_id=user.id,
                    role_id=roles[grant.role.value],
                    scope_type=grant.scope_type.value,
                    scope_id=scope_id,
                    granted_by="seed",
                    created_at=now,
                )
            )
            counts.role_grants += 1


def _resolve_scope(built: _Built, scope_type: ScopeType, ref: str | None) -> int | None:
    """Turn a blueprint's code into the id the grant is stored against."""
    if scope_type is ScopeType.GLOBAL:
        return None
    if scope_type is ScopeType.SCHOOL:
        return built.school_id
    if scope_type is ScopeType.YEAR_LEVEL:
        return built.rungs[str(ref)]
    if scope_type is ScopeType.CLASS_SECTION:
        return built.rooms[str(ref)]
    if scope_type is ScopeType.SUBJECT:
        return built.subjects[str(ref)]
    raise DemoRefused(f"cannot resolve a {scope_type} scope from {ref!r}")


def _load_teaching(session: Session, built: _Built, counts: Counts, now: datetime) -> None:
    """Teachers, and the two halves of their assignment.

    `teacher_subjects` + `teacher_year_levels` is what the principal decided;
    `teacher_class_sections` is what the year supervisor did with it. They are written
    separately here so the demo actually contains the half-finished state — `t.unassigned`
    has the first two and none of the third.
    """
    for person in bp.STAFF:
        if not person.staff_number:
            continue
        teacher = m.Teacher(
            staff_number=person.staff_number,
            school_id=built.school_id,
            user_id=built.users.get(person.username),
            full_name_en=person.full_name_en,
            full_name_ar=person.full_name_ar,
            email=person.email,
            phone="",
            is_active=True,
            created_at=now,
        )
        session.add(teacher)
        session.flush()
        built.teachers[person.username] = teacher.id
        counts.teachers += 1

        subject_id = built.subjects[person.subject]
        session.add(
            m.TeacherSubject(
                teacher_id=teacher.id,
                subject_id=subject_id,
                academic_year_id=built.year_id,
                is_primary=True,
                created_at=now,
            )
        )
        counts.teacher_subjects += 1

        for rung_code in person.rungs:
            session.add(
                m.TeacherYearLevel(
                    teacher_id=teacher.id,
                    year_level_id=built.rungs[rung_code],
                    subject_id=subject_id,
                    created_at=now,
                )
            )
            counts.teacher_year_levels += 1

        for room_key in person.rooms:
            session.add(
                m.TeacherClassSection(
                    teacher_id=teacher.id,
                    class_section_id=built.rooms[room_key],
                    subject_id=subject_id,
                    assigned_by="seed",
                    created_at=now,
                )
            )
            counts.teacher_class_sections += 1


def _load_registers(session: Session, built: _Built, counts: Counts, now: datetime) -> None:
    """Attendance for the rooms the blueprint marks, over real school days.

    The mix is deliberately not uniform. Most children are present most days; a few have a
    run of absences; one or two are late. A register where everybody is present tests the
    screen and nothing else — the interesting reads are "who was away last week" and "is
    this child's attendance a pattern".
    """
    for key, plan in bp.CLASS_PLANS.items():
        if not plan.register_days:
            continue
        room_id = built.rooms[key]
        roll = built.roll.get(key, [])
        for day in school_days(ending=REGISTER_ANCHOR, count=plan.register_days):
            for seat, student_id in enumerate(roll):
                dice = _dice("register", key, student_id, day.isoformat())
                draw = dice.random()
                # Roughly one child in eleven is not in the room on a given day, which is
                # about what an Egyptian school actually records.
                if seat % 11 == day.day % 11 and draw < 0.55:
                    state = AttendanceState.ABSENT
                elif draw < 0.03:
                    state = AttendanceState.EXCUSED
                elif draw < 0.09:
                    state = AttendanceState.LATE
                else:
                    state = AttendanceState.PRESENT
                session.add(
                    m.Attendance(
                        student_id=student_id,
                        class_section_id=room_id,
                        on_date=day,
                        state=state.value,
                        note="",
                        recorded_by="seed",
                        created_at=now,
                        updated_at=now,
                    )
                )
                counts.attendance += 1


def _load_marks(session: Session, built: _Built, counts: Counts, now: datetime) -> None:
    """Term 1 marks for the graded rooms.

    **Some cells are deliberately left empty**, and they are stored as no row at all
    rather than as a zero. That is the invariant the whole service is built around — a
    blank is not a mark of nought — and a demo that fills every cell cannot show a screen
    rendering the difference.
    """
    term_id = built.terms[bp.TERMS[0].code]
    rung_by_room = {
        bp.room_ref(rung.code, label_en): rung
        for rung in bp.RUNGS
        for _, label_en, _ in rung.rooms
    }

    for key, plan in bp.CLASS_PLANS.items():
        if not plan.graded:
            continue
        rung = rung_by_room[key]
        room_id = built.rooms[key]
        depth = _depth_for(rung.grade_number, rung.stage.value)
        for student_id in built.roll.get(key, []):
            for subject_code in _SUBJECTS_BY_DEPTH[depth]:
                dice = _dice("mark", key, student_id, subject_code)
                # One cell in fourteen is not yet marked. Left as no row, on purpose.
                if dice.random() < 0.07:
                    continue
                max_points = 100.0
                # A believable spread: most of a class between 55 and 95, a tail below.
                points = round(min(100.0, max(12.0, dice.gauss(76, 13))), 1)
                session.add(
                    m.SubjectGrade(
                        student_id=student_id,
                        subject_id=built.subjects[subject_code],
                        term_id=term_id,
                        class_section_id=room_id,
                        percentage=round(points / max_points * 100, 1),
                        points=points,
                        max_points=max_points,
                        remark="",
                        recorded_by="seed",
                        created_at=now,
                        updated_at=now,
                    )
                )
                counts.grades += 1


def _ensure_system_status(session: Session, now: datetime) -> None:
    """Put the estate in `active`, unless somebody has already set a status."""
    existing = session.scalars(
        select(m.SystemSetting).where(m.SystemSetting.key == SYSTEM_STATUS_KEY)
    ).one_or_none()
    if existing is not None:
        return
    session.add(
        m.SystemSetting(
            key=SYSTEM_STATUS_KEY,
            value=SystemStatus.ACTIVE.value,
            note="",
            updated_by="seed",
            updated_at=now,
        )
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def describe_classes(session: Session) -> Iterator[str]:
    """Every demo class with its generated title in both languages, for the report.

    Generated at read time from the stored columns rather than from a string, which is the
    point of the exercise: change `grade_number` and the title changes with it.
    """
    school = demo_school(session)
    if school is None:
        return
    rows = session.execute(
        select(m.ClassSection, m.YearLevel, m.EducationalSystem)
        .join(m.YearLevel, m.ClassSection.year_level_id == m.YearLevel.id)
        .join(
            m.EducationalSystem,
            m.YearLevel.educational_system_id == m.EducationalSystem.id,
            isouter=True,
        )
        .where(m.YearLevel.school_id == school.id)
        .order_by(m.YearLevel.display_order, m.ClassSection.code)
    ).all()
    from sis.domain.naming import EducationalSystemKind
    from sis.domain.structure import Stage

    for room, level, system in rows:
        coordinates = ClassCoordinates(
            stage=Stage(level.stage),
            grade_number=level.grade_number,
            section_number=room.section_number,
            kind=EducationalSystemKind(system.kind if system else "unspecified"),
            label_en=room.name_en,
            label_ar=room.name_ar,
        )
        yield (
            f"  {room.code:<14} {render_class_title(coordinates, 'en'):<38} "
            f"{render_class_title(coordinates, 'ar')}"
        )


def status(session: Session) -> list[str]:
    """What is in this database right now. Reads only."""
    school = demo_school(session)
    if school is None:
        return ["The demo school is not present in this database."]

    year_ids = list(
        session.scalars(
            select(m.AcademicYear.id).where(m.AcademicYear.school_id == school.id)
        ).all()
    )
    class_ids = (
        list(
            session.scalars(
                select(m.ClassSection.id).where(m.ClassSection.academic_year_id.in_(year_ids))
            ).all()
        )
        if year_ids
        else []
    )
    students = (
        len(
            set(
                session.scalars(
                    select(m.ClassEnrolment.student_id).where(
                        m.ClassEnrolment.class_section_id.in_(class_ids)
                    )
                ).all()
            )
        )
        if class_ids
        else 0
    )
    usernames = [person.username for person in bp.STAFF]
    users = len(
        session.scalars(select(m.User.id).where(m.User.username.in_(usernames))).all()
    )
    teachers = len(
        session.scalars(select(m.Teacher.id).where(m.Teacher.school_id == school.id)).all()
    )
    attendance = (
        len(
            session.scalars(
                select(m.Attendance.id).where(m.Attendance.class_section_id.in_(class_ids))
            ).all()
        )
        if class_ids
        else 0
    )
    grades = (
        len(
            session.scalars(
                select(m.SubjectGrade.id).where(m.SubjectGrade.class_section_id.in_(class_ids))
            ).all()
        )
        if class_ids
        else 0
    )
    return [
        f"school        {school.code} — {school.name_en}",
        f"classes       {len(class_ids)}",
        f"students      {students}",
        f"demo accounts {users} of {len(usernames)}",
        f"teachers      {teachers}",
        f"attendance    {attendance}",
        f"grades        {grades}",
    ]


def open_session(school_code: str | None = None) -> Session:
    """A session on whichever database the environment points at."""
    return get_sessionmaker(school_code)()


__all__ = [
    "Counts",
    "DemoRefused",
    "REGISTER_ANCHOR",
    "SYSTEM_STATUS_KEY",
    "describe_classes",
    "guard_environment",
    "load",
    "open_session",
    "remove",
    "school_days",
    "status",
    "sync_demo",
    "sync_roles",
]
