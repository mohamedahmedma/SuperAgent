"""Manage teaching staff without changing anybody's roles."""
from collections.abc import Callable, Sequence

from sis.application.ports.repositories import TeacherRecord
from sis.application.ports.unit_of_work import UnitOfWork
from sis.domain.errors import UnknownReference, ValidationError
from sis.domain.staff import PASSWORD_MIN_LENGTH
from sis.domain.value_objects import AcademicYearCode, ClassCode, SchoolCode, SubjectCode, YearCode
from sis.infrastructure.crypto import hash_password


class TeacherManagementService:
    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow_factory = uow_factory

    def list(
        self, school_code: SchoolCode, *, year_level_code: YearCode | None = None
    ) -> Sequence[TeacherRecord]:
        """The teaching staff a caller may read: the school's, or one grade's.

        The grade-scoped form is the grade supervisor's directory. It is the same read
        with a narrower boundary rather than a second method, because "who teaches here"
        is one question and two implementations of it would drift.
        """
        with self._uow_factory() as uow:
            if uow.schools.get(school_code) is None:
                raise UnknownReference(f"no school {school_code}", field="school_code")
            return uow.teachers.list_for_school(
                school_code, year_level_code=year_level_code
            )

    def get(
        self,
        school_code: SchoolCode,
        staff_number: str,
        *,
        year_level_code: YearCode | None = None,
    ) -> TeacherRecord:
        """One teacher. A teacher outside a named grade is reported as no such teacher.

        Deliberately the same `UnknownReference` a genuinely absent staff number raises.
        A grade supervisor who could tell "not in your grade" from "does not exist" could
        enumerate the school's staff list one number at a time, and those are people's
        names — the same reasoning the sign-in path uses for its single refusal message.
        """
        with self._uow_factory() as uow:
            record = uow.teachers.get(
                school_code, staff_number, year_level_code=year_level_code
            )
            if record is None:
                raise UnknownReference(f"no teacher {staff_number} in {school_code}", field="staff_number")
            return record

    def save(
        self, *, school_code: SchoolCode, staff_number: str, full_name_en: str,
        full_name_ar: str, email: str, phone: str, is_active: bool,
        username: str | None, password: str | None,
        assignments: Sequence[tuple[str, str, str, Sequence[str]]], assigned_by: str,
    ) -> TeacherRecord:
        if not full_name_en.strip() and not full_name_ar.strip():
            raise ValidationError("a teacher name is required", field="full_name_en")
        if password is not None and len(password) < PASSWORD_MIN_LENGTH:
            raise ValidationError(
                f"password must be at least {PASSWORD_MIN_LENGTH} characters", field="password"
            )
        with self._uow_factory() as uow:
            record = uow.teachers.save(
                school_code=school_code, staff_number=staff_number,
                full_name_en=full_name_en, full_name_ar=full_name_ar,
                email=email, phone=phone, is_active=is_active, username=username,
                password_hash=None if password is None else hash_password(password),
                assignments=[
                    (AcademicYearCode(year), SubjectCode(subject), YearCode(level),
                     tuple(ClassCode(code) for code in classes))
                    for year, subject, level, classes in assignments
                ],
                assigned_by=assigned_by,
            )
            uow.commit()
            return record


__all__ = ["TeacherManagementService"]
