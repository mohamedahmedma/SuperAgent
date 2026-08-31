"""SQL persistence for teacher identities, accounts, and teaching assignments."""
from datetime import UTC, datetime
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from sis.application.ports.repositories import TeacherRecord, TeacherTeachingAssignment
from sis.domain.errors import DomainRuleViolation, UnknownReference, ValidationError
from sis.domain.staff import Teacher
from sis.domain.value_objects import AcademicYearCode, ClassCode, SchoolCode, SubjectCode, YearCode
from sis.infrastructure.db import models as m


class SqlAlchemyTeacherRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_school(
        self, school_code: SchoolCode, *, year_level_code: YearCode | None = None
    ) -> Sequence[TeacherRecord]:
        """The school's teachers, or only the ones who teach on one grade.

        The narrowed form is what a grade supervisor reads, and it narrows in *both*
        directions: which teachers come back, and how much of each teacher comes back.
        Filtering the rows alone would still hand a Grade 4 supervisor the Grade 9
        timetable of every teacher who works on both, which is the unrelated grade the
        role is defined not to see.
        """
        statement = (
            select(m.Teacher).join(m.School).where(m.School.code == str(school_code))
        )
        if year_level_code is not None:
            # `TeacherYearLevel` is per subject, so a teacher who teaches two subjects on
            # this grade joins twice. Distinct rather than a subquery because the id is
            # what duplicates and the row is otherwise identical.
            statement = statement.join(
                m.TeacherYearLevel, m.TeacherYearLevel.teacher_id == m.Teacher.id
            ).join(
                m.YearLevel, m.TeacherYearLevel.year_level_id == m.YearLevel.id
            ).where(m.YearLevel.code == str(year_level_code)).distinct()
        rows = self._session.scalars(
            statement.order_by(m.Teacher.staff_number)
        ).all()
        return [
            self._record(row, str(school_code), year_level_code=year_level_code)
            for row in rows
        ]

    def get(
        self,
        school_code: SchoolCode,
        staff_number: str,
        *,
        year_level_code: YearCode | None = None,
    ) -> TeacherRecord | None:
        """One teacher. With a grade named, one teacher *of that grade* and nothing else.

        A teacher who does not teach on the named grade answers `None` rather than a
        record with no assignments: to a grade supervisor that teacher is not a teacher
        they may read, and "exists but is empty" is still an answer about a person
        outside their grade.
        """
        statement = select(m.Teacher).join(m.School).where(
            m.School.code == str(school_code), m.Teacher.staff_number == staff_number.strip()
        )
        if year_level_code is not None:
            statement = statement.join(
                m.TeacherYearLevel, m.TeacherYearLevel.teacher_id == m.Teacher.id
            ).join(
                m.YearLevel, m.TeacherYearLevel.year_level_id == m.YearLevel.id
            ).where(m.YearLevel.code == str(year_level_code)).distinct()
        row = self._session.scalars(statement).first()
        return (
            None
            if row is None
            else self._record(row, str(school_code), year_level_code=year_level_code)
        )

    def save(
        self, *, school_code: SchoolCode, staff_number: str, full_name_en: str,
        full_name_ar: str, email: str, phone: str, is_active: bool,
        username: str | None, password_hash: str | None,
        assignments: Sequence[tuple[AcademicYearCode, SubjectCode, YearCode, Sequence[ClassCode]]],
        assigned_by: str,
    ) -> TeacherRecord:
        school = self._session.scalar(select(m.School).where(m.School.code == str(school_code)))
        if school is None:
            raise UnknownReference(f"no school {school_code}", field="school_code")
        staff_number = staff_number.strip()
        if not staff_number:
            raise ValidationError("a teacher needs a staff number", field="staff_number")

        teacher = self._session.scalar(select(m.Teacher).where(
            m.Teacher.school_id == school.id, m.Teacher.staff_number == staff_number
        ))
        user = None
        clean_username = (username or "").strip() or None
        if clean_username:
            user = self._session.scalar(select(m.User).where(m.User.username == clean_username))
            if user is not None and user.school_id != school.id:
                raise DomainRuleViolation("that username belongs to another school", field="username")
            if user is not None and teacher is not None and teacher.user_id not in (None, user.id):
                raise DomainRuleViolation("that account belongs to another teacher", field="username")
            linked = self._session.scalar(select(m.Teacher).where(m.Teacher.user_id == user.id)) if user else None
            if linked is not None and (teacher is None or linked.id != teacher.id):
                raise DomainRuleViolation("that account belongs to another teacher", field="username")
            if user is None:
                if password_hash is None:
                    raise ValidationError("a password is required for a new account", field="password")
                user = m.User(
                    username=clean_username, password_hash=password_hash, school_id=school.id,
                    full_name_en=full_name_en.strip(), full_name_ar=full_name_ar.strip(),
                    email=email.strip(), is_active=is_active,
                )
                self._session.add(user)
                self._session.flush()
            else:
                user.full_name_en, user.full_name_ar = full_name_en.strip(), full_name_ar.strip()
                user.email, user.is_active = email.strip(), is_active
                if password_hash is not None:
                    user.password_hash = password_hash

        if teacher is None:
            teacher = m.Teacher(staff_number=staff_number, school_id=school.id)
            self._session.add(teacher)
        teacher.user_id = user.id if user else None
        teacher.full_name_en, teacher.full_name_ar = full_name_en.strip(), full_name_ar.strip()
        teacher.email, teacher.phone, teacher.is_active = email.strip(), phone.strip(), is_active
        self._session.flush()

        prepared: list[tuple[m.Subject, m.YearLevel, list[m.ClassSection]]] = []
        seen: set[tuple[int, int]] = set()
        for year_code, subject_code, level_code, class_codes in assignments:
            year = self._session.scalar(select(m.AcademicYear).where(
                m.AcademicYear.code == str(year_code), m.AcademicYear.school_id == school.id
            ))
            if year is None:
                raise UnknownReference(f"no academic year {year_code} in {school_code}", field="academic_year_code")
            subject = self._session.scalar(select(m.Subject).where(
                m.Subject.code == str(subject_code), m.Subject.academic_year_id == year.id
            ))
            if subject is None:
                raise UnknownReference(f"no subject {subject_code} in {year_code}", field="subject_code")
            level = self._session.scalar(select(m.YearLevel).where(
                m.YearLevel.code == str(level_code), m.YearLevel.school_id == school.id
            ))
            if level is None:
                raise UnknownReference(f"no grade {level_code} in {school_code}", field="year_level_code")
            valid = self._session.scalar(select(m.SubjectYearLevel.id).where(
                m.SubjectYearLevel.subject_id == subject.id,
                m.SubjectYearLevel.year_level_id == level.id,
            ))
            if valid is None:
                raise DomainRuleViolation(
                    f"{subject_code} is not configured for {level_code}", field="subject_code"
                )
            key = (subject.id, level.id)
            if key in seen:
                raise ValidationError("each subject and grade may appear once", field="assignments")
            seen.add(key)
            classes = self._session.scalars(select(m.ClassSection).where(
                m.ClassSection.academic_year_id == year.id,
                m.ClassSection.year_level_id == level.id,
                m.ClassSection.code.in_([str(code) for code in class_codes] or [""]),
            )).all()
            found = {row.code for row in classes}
            missing = [str(code) for code in class_codes if str(code) not in found]
            if missing:
                raise DomainRuleViolation(
                    f"class(es) {', '.join(missing)} are not in {level_code} for {year_code}",
                    field="class_codes",
                )
            prepared.append((subject, level, classes))

        self._session.execute(delete(m.TeacherClassSection).where(m.TeacherClassSection.teacher_id == teacher.id))
        self._session.execute(delete(m.TeacherYearLevel).where(m.TeacherYearLevel.teacher_id == teacher.id))
        self._session.execute(delete(m.TeacherSubject).where(m.TeacherSubject.teacher_id == teacher.id))
        now = datetime.now(UTC)
        subject_ids: set[int] = set()
        for subject, level, classes in prepared:
            if subject.id not in subject_ids:
                self._session.add(m.TeacherSubject(
                    teacher_id=teacher.id, subject_id=subject.id,
                    academic_year_id=subject.academic_year_id, is_primary=not subject_ids,
                    created_at=now,
                ))
                subject_ids.add(subject.id)
            self._session.add(m.TeacherYearLevel(
                teacher_id=teacher.id, year_level_id=level.id, subject_id=subject.id,
                created_at=now,
            ))
            for section in classes:
                self._session.add(m.TeacherClassSection(
                    teacher_id=teacher.id, class_section_id=section.id,
                    subject_id=subject.id, assigned_by=assigned_by, created_at=now,
                ))
        self._session.flush()
        return self._record(teacher, str(school_code))

    def _record(
        self,
        teacher: m.Teacher,
        school_code: str,
        *,
        year_level_code: YearCode | None = None,
    ) -> TeacherRecord:
        statement = (
            select(m.TeacherYearLevel, m.Subject, m.AcademicYear, m.YearLevel, m.EducationalSystem)
            .join(m.Subject, m.TeacherYearLevel.subject_id == m.Subject.id)
            .join(m.AcademicYear, m.Subject.academic_year_id == m.AcademicYear.id)
            .join(m.YearLevel, m.TeacherYearLevel.year_level_id == m.YearLevel.id)
            .outerjoin(m.EducationalSystem, m.YearLevel.educational_system_id == m.EducationalSystem.id)
            .where(m.TeacherYearLevel.teacher_id == teacher.id)
        )
        if year_level_code is not None:
            statement = statement.where(m.YearLevel.code == str(year_level_code))
        rows = self._session.execute(
            statement.order_by(m.AcademicYear.code, m.YearLevel.display_order, m.Subject.code)
        ).all()
        assignments = []
        for link, subject, year, level, track in rows:
            classes = self._session.scalars(
                select(m.ClassSection.code).join(m.TeacherClassSection).where(
                    m.TeacherClassSection.teacher_id == teacher.id,
                    m.TeacherClassSection.subject_id == subject.id,
                    m.ClassSection.year_level_id == level.id,
                    m.ClassSection.academic_year_id == year.id,
                ).order_by(m.ClassSection.code)
            ).all()
            assignments.append(TeacherTeachingAssignment(
                academic_year_code=year.code, subject_code=subject.code,
                year_level_code=level.code, track_code=None if track is None else track.code,
                class_codes=tuple(classes),
            ))
        user = self._session.get(m.User, teacher.user_id) if teacher.user_id else None
        return TeacherRecord(
            teacher=Teacher(id=teacher.id, staff_number=teacher.staff_number,
                school_id=teacher.school_id, user_id=teacher.user_id,
                full_name_en=teacher.full_name_en, full_name_ar=teacher.full_name_ar,
                is_active=teacher.is_active),
            school_code=school_code, username=None if user is None else user.username,
            email=teacher.email, phone=teacher.phone, assignments=tuple(assignments),
        )


__all__ = ["SqlAlchemyTeacherRepository"]
