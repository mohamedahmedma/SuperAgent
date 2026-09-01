"""Teacher identity, optional account, and valid teaching assignments."""
from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import delete, select

from sis.api.deps import (
    Principal,
    TodayDep,
    UowFactoryDep,
    get_teacher_management_service,
    require_permission,
)
from sis.api.routers import domain_errors, error_responses
from sis.application.ports.repositories import TeacherRecord
from sis.application.services.teachers import TeacherManagementService
from sis.domain.staff import PASSWORD_MIN_LENGTH, StaffAttendanceState
from sis.domain.rbac import Permission
from sis.domain.value_objects import SchoolCode, YearCode
from sis.infrastructure.db import models as m

router = APIRouter(prefix="/v1", tags=["teachers"])
Readers = Annotated[Principal, Depends(require_permission(Permission.TEACHERS_READ))]
Managers = Annotated[Principal, Depends(require_permission(Permission.TEACHERS_ASSIGN_SUBJECTS))]
AttendanceReaders = Annotated[
    Principal, Depends(require_permission(Permission.TEACHER_ATTENDANCE_READ))
]
AttendanceWriters = Annotated[
    Principal, Depends(require_permission(Permission.TEACHER_ATTENDANCE_WRITE))
]
ClassAssigners = Annotated[
    Principal, Depends(require_permission(Permission.TEACHERS_ASSIGN_CLASSES))
]
Teachers = Annotated[TeacherManagementService, Depends(get_teacher_management_service)]


def _sync_teacher_role_grants(session, teacher: m.Teacher, *, actor: str) -> None:  # noqa: ANN001
    """Make the Teacher role exactly follow the concrete classrooms they teach in."""
    if teacher.user_id is None:
        return
    role_id = session.scalar(select(m.Role.id).where(m.Role.code == "teacher"))
    if role_id is None:
        return
    desired = set(session.scalars(
        select(m.TeacherClassSection.class_section_id)
        .where(m.TeacherClassSection.teacher_id == teacher.id)
        .distinct()
    ).all())
    session.execute(delete(m.UserRole).where(
        m.UserRole.user_id == teacher.user_id,
        m.UserRole.role_id == role_id,
    ))
    session.add_all(
        m.UserRole(
            user_id=teacher.user_id,
            role_id=role_id,
            scope_type="class_section",
            scope_id=class_id,
            granted_by=actor,
        )
        for class_id in sorted(desired)
    )


class TeacherAssignmentIn(BaseModel):
    academic_year_code: str
    subject_code: str
    year_level_code: str = Field(description="The grade; its configured track is used automatically.")
    class_codes: list[str] = Field(
        default_factory=list,
        description="Optional concrete classes within this grade. May contain several sections.",
    )


class TeacherIn(BaseModel):
    full_name_en: str = ""
    full_name_ar: str = ""
    email: str = ""
    phone: str = ""
    is_active: bool = True
    username: str | None = Field(
        default=None, description="Optional existing or new account. This does not grant a role."
    )
    password: str | None = Field(
        default=None, min_length=PASSWORD_MIN_LENGTH,
        description="Required only when creating a new account; omitted values never reset it.",
    )
    assignments: list[TeacherAssignmentIn] = Field(default_factory=list)


class TeacherAssignmentOut(BaseModel):
    academic_year_code: str
    subject_code: str
    year_level_code: str
    track_code: str | None
    class_codes: list[str]


class TeacherOut(BaseModel):
    staff_number: str
    school_code: str
    user_id: int | None
    username: str | None
    full_name_en: str
    full_name_ar: str
    email: str
    phone: str
    is_active: bool
    assignments: list[TeacherAssignmentOut]

    @classmethod
    def of(cls, record: TeacherRecord) -> "TeacherOut":
        return cls(
            staff_number=record.teacher.staff_number, school_code=record.school_code,
            user_id=record.teacher.user_id, username=record.username,
            full_name_en=record.teacher.full_name_en, full_name_ar=record.teacher.full_name_ar,
            email=record.email, phone=record.phone, is_active=record.teacher.is_active,
            assignments=[TeacherAssignmentOut(
                academic_year_code=row.academic_year_code, subject_code=row.subject_code,
                year_level_code=row.year_level_code, track_code=row.track_code,
                class_codes=list(row.class_codes),
            ) for row in record.assignments],
        )


class TeacherAttendanceIn(BaseModel):
    state: StaffAttendanceState
    note: str = Field(default="", max_length=500)


class TeacherAttendanceOut(BaseModel):
    staff_number: str
    full_name_en: str
    full_name_ar: str
    on_date: date
    state: StaffAttendanceState
    note: str
    recorded_by: str
    updated_at: datetime


class EligibleTeacherOut(BaseModel):
    staff_number: str
    full_name_en: str
    full_name_ar: str
    assigned_class_codes: list[str] = Field(default_factory=list)


class GradeAssignmentOptionsOut(BaseModel):
    school_code: str
    academic_year_code: str
    year_level_code: str
    year_level_name_en: str
    year_level_name_ar: str
    subjects: list[dict[str, str]]
    classes: list[dict[str, str]]
    available_classes: list[dict[str, str]] = Field(default_factory=list)
    eligible_teachers: list[EligibleTeacherOut]


class GradeClassAssignmentIn(BaseModel):
    academic_year_code: str
    subject_code: str
    staff_number: str
    class_codes: list[str] = Field(min_length=1)


def _grade_context(session, school_code: str, year_code: str, level_code: str):  # noqa: ANN001, ANN202
    row = session.execute(
        select(m.School, m.AcademicYear, m.YearLevel)
        .join(m.AcademicYear, m.AcademicYear.school_id == m.School.id)
        .join(m.YearLevel, m.YearLevel.school_id == m.School.id)
        .where(
            m.School.code == school_code,
            m.AcademicYear.code == year_code,
            m.YearLevel.code == level_code,
        )
    ).one_or_none()
    if row is None:
        from sis.domain.errors import UnknownReference
        raise UnknownReference(f"no grade {level_code} in {school_code}", field="year_level_code")
    return row


@router.get(
    "/schools/{school_code}/grades/{year_level_code}/teacher-assignment-options",
    response_model=GradeAssignmentOptionsOut,
    responses=error_responses(401, 403, 404, 422),
)
def grade_assignment_options(
    school_code: str, year_level_code: str, caller: ClassAssigners,
    uow_factory: UowFactoryDep,
    academic_year: Annotated[str, Query()], subject: Annotated[str | None, Query()] = None,
) -> GradeAssignmentOptionsOut:
    caller.narrow(
        Permission.TEACHERS_ASSIGN_CLASSES,
        lambda scopes: scopes.for_year_level_in_school(
            school_code=school_code, year_level_code=year_level_code
        ),
    )
    with uow_factory() as uow:
        session = uow._session
        school, year, level = _grade_context(
            session, school_code, academic_year, year_level_code
        )
        subjects = session.execute(
            select(m.Subject.code, m.Subject.name_en, m.Subject.name_ar)
            .join(m.SubjectYearLevel, m.SubjectYearLevel.subject_id == m.Subject.id)
            .where(
                m.Subject.academic_year_id == year.id,
                m.SubjectYearLevel.year_level_id == level.id,
                m.Subject.is_active.is_(True),
            ).order_by(m.Subject.display_order, m.Subject.code)
        ).all()
        classes = session.execute(
            select(m.ClassSection.code, m.ClassSection.name_en, m.ClassSection.name_ar)
            .where(
                m.ClassSection.academic_year_id == year.id,
                m.ClassSection.year_level_id == level.id,
            ).order_by(m.ClassSection.code)
        ).all()
        eligible: list[EligibleTeacherOut] = []
        available_classes = [
            {"code": r.code, "name_en": r.name_en, "name_ar": r.name_ar} for r in classes
        ]
        if subject:
            subject_row = session.scalar(select(m.Subject).where(
                m.Subject.academic_year_id == year.id, m.Subject.code == subject
            ))
            compatible = None if subject_row is None else session.scalar(
                select(m.SubjectYearLevel.id).where(
                    m.SubjectYearLevel.subject_id == subject_row.id,
                    m.SubjectYearLevel.year_level_id == level.id,
                )
            )
            if compatible is None:
                from sis.domain.errors import DomainRuleViolation
                raise DomainRuleViolation(
                    f"{subject} is not configured for {year_level_code}",
                    field="subject_code",
                )
            if subject_row is not None:
                occupied = set(session.scalars(
                    select(m.ClassSection.code)
                    .join(m.TeacherClassSection)
                    .where(
                        m.TeacherClassSection.subject_id == subject_row.id,
                        m.ClassSection.academic_year_id == year.id,
                        m.ClassSection.year_level_id == level.id,
                    )
                ).all())
                available_classes = [row for row in available_classes if row["code"] not in occupied]
                teachers = session.scalars(
                    select(m.Teacher)
                    .join(m.TeacherYearLevel)
                    .where(
                        m.Teacher.school_id == school.id,
                        m.Teacher.is_active.is_(True),
                        m.TeacherYearLevel.year_level_id == level.id,
                        m.TeacherYearLevel.subject_id == subject_row.id,
                    ).order_by(m.Teacher.staff_number)
                ).all()
                for teacher in teachers:
                    assigned = session.scalars(
                        select(m.ClassSection.code)
                        .join(m.TeacherClassSection)
                        .where(
                            m.TeacherClassSection.teacher_id == teacher.id,
                            m.TeacherClassSection.subject_id == subject_row.id,
                            m.ClassSection.academic_year_id == year.id,
                            m.ClassSection.year_level_id == level.id,
                        ).order_by(m.ClassSection.code)
                    ).all()
                    eligible.append(EligibleTeacherOut(
                        staff_number=teacher.staff_number,
                        full_name_en=teacher.full_name_en, full_name_ar=teacher.full_name_ar,
                        assigned_class_codes=list(assigned),
                    ))
        return GradeAssignmentOptionsOut(
            school_code=school.code, academic_year_code=year.code,
            year_level_code=level.code, year_level_name_en=level.name_en,
            year_level_name_ar=level.name_ar,
            subjects=[{"code": r.code, "name_en": r.name_en, "name_ar": r.name_ar} for r in subjects],
            classes=[{"code": r.code, "name_en": r.name_en, "name_ar": r.name_ar} for r in classes],
            available_classes=available_classes,
            eligible_teachers=eligible,
        )


@router.put(
    "/schools/{school_code}/grades/{year_level_code}/teacher-class-assignments",
    response_model=GradeAssignmentOptionsOut,
    responses=error_responses(401, 403, 404, 409, 422),
)
def assign_teacher_classes(
    school_code: str, year_level_code: str, body: GradeClassAssignmentIn,
    caller: ClassAssigners, uow_factory: UowFactoryDep,
) -> GradeAssignmentOptionsOut:
    caller.narrow(
        Permission.TEACHERS_ASSIGN_CLASSES,
        lambda scopes: scopes.for_year_level_in_school(
            school_code=school_code, year_level_code=year_level_code
        ),
    )
    with uow_factory() as uow:
        session = uow._session
        school, year, level = _grade_context(
            session, school_code, body.academic_year_code, year_level_code
        )
        subject = session.scalar(select(m.Subject).where(
            m.Subject.academic_year_id == year.id, m.Subject.code == body.subject_code
        ))
        teacher = session.scalar(select(m.Teacher).where(
            m.Teacher.school_id == school.id, m.Teacher.staff_number == body.staff_number
        ))
        compatible = None if subject is None else session.scalar(
            select(m.SubjectYearLevel.id).where(
                m.SubjectYearLevel.subject_id == subject.id,
                m.SubjectYearLevel.year_level_id == level.id,
            )
        )
        eligible = None if subject is None or teacher is None else session.scalar(
            select(m.TeacherYearLevel.id).where(
                m.TeacherYearLevel.teacher_id == teacher.id,
                m.TeacherYearLevel.subject_id == subject.id,
                m.TeacherYearLevel.year_level_id == level.id,
            )
        )
        if compatible is None:
            from sis.domain.errors import DomainRuleViolation
            raise DomainRuleViolation(
                f"{body.subject_code} is not configured for {year_level_code}",
                field="subject_code",
            )
        if teacher is not None and not teacher.is_active:
            from sis.domain.errors import DomainRuleViolation
            raise DomainRuleViolation("that teacher is inactive", field="staff_number")
        if eligible is None:
            from sis.domain.errors import DomainRuleViolation
            raise DomainRuleViolation(
                "that teacher is not eligible for this subject and grade", field="staff_number"
            )
        classes = session.scalars(select(m.ClassSection).where(
            m.ClassSection.academic_year_id == year.id,
            m.ClassSection.year_level_id == level.id,
            m.ClassSection.code.in_(body.class_codes),
        )).all()
        if {row.code for row in classes} != set(body.class_codes):
            from sis.domain.errors import DomainRuleViolation
            raise DomainRuleViolation("one or more classes are outside this grade", field="class_codes")
        conflict = session.execute(
            select(m.ClassSection.code, m.Teacher.staff_number)
            .join(m.TeacherClassSection, m.TeacherClassSection.class_section_id == m.ClassSection.id)
            .join(m.Teacher, m.Teacher.id == m.TeacherClassSection.teacher_id)
            .where(
                m.TeacherClassSection.subject_id == subject.id,
                m.TeacherClassSection.teacher_id != teacher.id,
                m.ClassSection.id.in_([row.id for row in classes]),
            )
        ).first()
        if conflict is not None:
            from sis.domain.errors import DomainRuleViolation
            raise DomainRuleViolation(
                f"{conflict.code} already has a {body.subject_code} teacher ({conflict.staff_number})",
                field="class_codes",
            )
        existing_ids = session.scalars(select(m.TeacherClassSection.id)
            .join(m.ClassSection)
            .where(
                m.TeacherClassSection.teacher_id == teacher.id,
                m.TeacherClassSection.subject_id == subject.id,
                m.ClassSection.academic_year_id == year.id,
                m.ClassSection.year_level_id == level.id,
            )).all()
        if existing_ids:
            session.execute(delete(m.TeacherClassSection).where(m.TeacherClassSection.id.in_(existing_ids)))
        session.add_all(m.TeacherClassSection(
            teacher_id=teacher.id, class_section_id=section.id, subject_id=subject.id,
            assigned_by=caller.username,
        ) for section in classes)
        session.flush()
        _sync_teacher_role_grants(session, teacher, actor=caller.username)
        uow.commit()
    return grade_assignment_options(
        school_code, year_level_code, caller, uow_factory,
        academic_year=body.academic_year_code, subject=body.subject_code,
    )


@router.get(
    "/schools/{school_code}/teachers/attendance",
    response_model=list[TeacherAttendanceOut],
    responses=error_responses(401, 403, 404, 422),
)
def list_teacher_attendance(
    school_code: str,
    caller: AttendanceReaders,
    uow_factory: UowFactoryDep,
    from_date: Annotated[date | None, Query(alias="from")] = None,
    to_date: Annotated[date | None, Query(alias="to")] = None,
) -> list[TeacherAttendanceOut]:
    caller.narrow(
        Permission.TEACHER_ATTENDANCE_READ, lambda scopes: scopes.for_school(school_code)
    )
    with uow_factory() as uow:
        school = uow._session.scalar(select(m.School).where(m.School.code == school_code))
        if school is None:
            from sis.domain.errors import UnknownReference
            raise UnknownReference("school", school_code)
        statement = (
            select(m.TeacherAttendance, m.Teacher)
            .join(m.Teacher, m.TeacherAttendance.teacher_id == m.Teacher.id)
            .where(m.TeacherAttendance.school_id == school.id)
            .order_by(m.TeacherAttendance.on_date.desc(), m.Teacher.staff_number)
        )
        if from_date is not None:
            statement = statement.where(m.TeacherAttendance.on_date >= from_date)
        if to_date is not None:
            statement = statement.where(m.TeacherAttendance.on_date <= to_date)
        rows = uow._session.execute(statement).all()
        return [TeacherAttendanceOut(
            staff_number=teacher.staff_number, full_name_en=teacher.full_name_en,
            full_name_ar=teacher.full_name_ar, on_date=row.on_date,
            state=StaffAttendanceState(row.state), note=row.note,
            recorded_by=row.recorded_by, updated_at=row.updated_at,
        ) for row, teacher in rows]


@router.put(
    "/schools/{school_code}/teachers/{staff_number}/attendance/{on_date}",
    response_model=TeacherAttendanceOut,
    responses=error_responses(401, 403, 404, 422),
)
def record_teacher_attendance(
    school_code: str, staff_number: str, on_date: date, body: TeacherAttendanceIn,
    caller: AttendanceWriters, uow_factory: UowFactoryDep, today: TodayDep,
) -> TeacherAttendanceOut:
    caller.narrow(
        Permission.TEACHER_ATTENDANCE_WRITE, lambda scopes: scopes.for_school(school_code)
    )
    # `today` is injected for the same reason the child register injects it: a refusal
    # judged against the wall clock is one no suite can state without dating its own
    # fixtures to the afternoon it was written.
    if on_date > today:
        from sis.domain.errors import DomainRuleViolation
        raise DomainRuleViolation(
            "teacher attendance cannot be recorded for a future day",
            field="on_date",
        )
    with uow_factory() as uow:
        teacher = uow._session.scalar(
            select(m.Teacher).join(m.School).where(
                m.School.code == school_code, m.Teacher.staff_number == staff_number
            )
        )
        if teacher is None:
            from sis.domain.errors import UnknownReference
            raise UnknownReference("teacher", staff_number)
        row = uow._session.scalar(select(m.TeacherAttendance).where(
            m.TeacherAttendance.teacher_id == teacher.id,
            m.TeacherAttendance.on_date == on_date,
        ))
        if row is None:
            row = m.TeacherAttendance(
                teacher_id=teacher.id, school_id=teacher.school_id, on_date=on_date
            )
            uow._session.add(row)
        row.state, row.note = body.state.value, body.note.strip()
        row.recorded_by, row.updated_at = caller.username, datetime.now(UTC)
        uow._session.flush()
        result = TeacherAttendanceOut(
            staff_number=teacher.staff_number, full_name_en=teacher.full_name_en,
            full_name_ar=teacher.full_name_ar, on_date=row.on_date, state=body.state,
            note=row.note, recorded_by=row.recorded_by, updated_at=row.updated_at,
        )
        uow.commit()
        return result


@router.get("/schools/{school_code}/teachers", response_model=list[TeacherOut],
    summary="The teaching staff a caller may read",
    description="Without `year_level` this is the school's whole directory, and it needs "
    "a school-wide grant.\n\n"
    "With `year_level` it is one grade's teaching staff, which is what a grade supervisor "
    "holds `teachers.read` for. The narrowing runs on both halves of the answer: only "
    "teachers assigned to that grade come back, and each one carries only their "
    "assignments on it — a teacher who also works two grades up does not bring that grade "
    "with them.",
    responses=error_responses(401, 403, 404, 422))
def list_teachers(
    school_code: str,
    service: Teachers,
    caller: Readers,
    year_level: Annotated[
        str | None,
        Query(description="Restrict to one grade. Required of a grade-scoped caller."),
    ] = None,
) -> list[TeacherOut]:
    if year_level is None:
        caller.narrow(Permission.TEACHERS_READ, lambda scopes: scopes.for_school(school_code))
    else:
        caller.narrow(
            Permission.TEACHERS_READ,
            lambda scopes: scopes.for_year_level_in_school(
                school_code=school_code, year_level_code=year_level
            ),
        )
    with domain_errors():
        rows = service.list(
            SchoolCode(school_code),
            year_level_code=None if year_level is None else YearCode(year_level),
        )
    return [TeacherOut.of(row) for row in rows]


@router.get("/schools/{school_code}/teachers/{staff_number}", response_model=TeacherOut,
    summary="One teacher",
    description="`year_level` narrows this the same way it narrows the directory: the "
    "record comes back holding only that grade's assignments, and a teacher who does not "
    "teach on that grade answers 404 rather than 403 — a supervisor able to tell the two "
    "apart could enumerate the school's staff numbers.",
    responses=error_responses(401, 403, 404, 422))
def get_teacher(
    school_code: str,
    staff_number: str,
    service: Teachers,
    caller: Readers,
    year_level: Annotated[
        str | None,
        Query(description="Restrict to one grade. Required of a grade-scoped caller."),
    ] = None,
) -> TeacherOut:
    if year_level is None:
        caller.narrow(Permission.TEACHERS_READ, lambda scopes: scopes.for_school(school_code))
    else:
        caller.narrow(
            Permission.TEACHERS_READ,
            lambda scopes: scopes.for_year_level_in_school(
                school_code=school_code, year_level_code=year_level
            ),
        )
    with domain_errors():
        row = service.get(
            SchoolCode(school_code),
            staff_number,
            year_level_code=None if year_level is None else YearCode(year_level),
        )
    return TeacherOut.of(row)


@router.put("/schools/{school_code}/teachers/{staff_number}", response_model=TeacherOut,
    summary="Create or replace a teacher and their teaching assignments",
    description="School-manager operation. Assignments replace the teacher's current subject, grade, and class scope atomically. The selected grade determines the academic track. No role is granted or promoted.",
    responses=error_responses(401, 403, 404, 409, 422))
def save_teacher(school_code: str, staff_number: str, body: TeacherIn,
    service: Teachers, caller: Managers, uow_factory: UowFactoryDep) -> TeacherOut:
    caller.narrow(
        Permission.TEACHERS_ASSIGN_SUBJECTS, lambda scopes: scopes.for_school(school_code)
    )
    with domain_errors():
        row = service.save(
            school_code=SchoolCode(school_code), staff_number=staff_number,
            full_name_en=body.full_name_en, full_name_ar=body.full_name_ar,
            email=body.email, phone=body.phone, is_active=body.is_active,
            username=body.username, password=body.password,
            assignments=[(a.academic_year_code, a.subject_code, a.year_level_code, a.class_codes)
                         for a in body.assignments], assigned_by=str(caller),
        )
    with uow_factory() as uow:
        teacher = uow._session.scalar(select(m.Teacher).join(m.School).where(
            m.School.code == school_code, m.Teacher.staff_number == staff_number
        ))
        if teacher is not None:
            _sync_teacher_role_grants(uow._session, teacher, actor=caller.username)
            uow.commit()
    return TeacherOut.of(row)
