"""The daily register over HTTP: read a class's day, record it, read a child's back.

`GET /v1/classes/{class_code}/attendance` returns **every child placed in the class that
day**, marked or not, and the unmarked ones carry `state: null`. That is the whole shape of
this router's contract and it is worth stating in the file: a response built from the marks
alone would show four children on a morning somebody marked four and stopped, and a screen
rendering it would report a half-taken register as a small class with perfect attendance.

`state: null` is a third value beside present and absent, and a client that renders it as
either is claiming a fact the school never stated. It is the same distinction the grade
routes make between a mark of `0` and a mark of `null`, one column over — and for the same
reason: `absent` accuses a child, `present` flatters the school, and neither is what "nobody
took the register" means.

**No rates.** Every count here is a count, and `recorded` is reported beside them so a
caller that wants a percentage divides by a number it can see. This service does not hold a
school calendar, so it cannot tell an unmarked Tuesday from a holiday and will not compute a
figure that pretends otherwise.
"""
from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy import func, or_, select
from pydantic import BaseModel, Field

from sis.api.deps import (
    AttendanceServiceDep,
    UowFactoryDep,
    Caller,
    RequestId,
    require_read_access,
    require_registrar,
    Principal,
    require_permission,
)
from sis.domain.rbac import Permission, Target
from sis.api.routers import domain_errors, error_responses
from sis.application.services.attendance import ClassRegister, StudentAttendance
from sis.domain.attendance import AttendanceState, AttendanceTally
from sis.domain.errors import UnknownReference
from sis.domain.value_objects import AcademicYearCode, ClassCode, StudentNumber
from sis.infrastructure.db import models as m

router = APIRouter(prefix="/v1", tags=["attendance"])

Reader = Annotated[Principal, Depends(require_permission(Permission.ATTENDANCE_READ))]
Registrar = Annotated[Principal, Depends(require_permission(Permission.ATTENDANCE_WRITE))]


class TallyOut(BaseModel):
    """Counts, and the number of days they were counted over. Never a rate."""

    present: int
    absent: int
    late: int
    excused: int
    recorded: int = Field(
        description="Days a mark exists for — the only honest denominator this service "
        "has. It is not the number of school days: no calendar is held here, so an "
        "unmarked Tuesday cannot be told from a holiday."
    )
    in_the_room: int = Field(description="Present plus late: days she was actually there.")
    away: int = Field(description="Absent plus excused: days she was not.")

    @classmethod
    def of(cls, counts: AttendanceTally) -> "TallyOut":
        return cls(
            present=counts.present,
            absent=counts.absent,
            late=counts.late,
            excused=counts.excused,
            recorded=counts.recorded,
            in_the_room=counts.in_the_room,
            away=counts.away,
        )


class RegisterLineOut(BaseModel):
    """One child on a day's register."""

    student_number: str
    full_name_ar: str = ""
    full_name_en: str = ""
    state: str | None = Field(
        default=None,
        description="present / absent / late / excused, or **null** when nobody marked "
        "her. Null is not absent: rendering it as either states a fact the school did not.",
    )
    note: str = ""
    is_marked: bool


class ClassRegisterOut(BaseModel):
    academic_year_code: str
    class_code: str
    on_date: date = Field(
        description="The day this register is a statement about. Echoed because `on` "
        "defaults to today, and a caller who meant last Friday must be able to see that it "
        "did not."
    )
    size: int = Field(description="Children placed in this class on that day.")
    unmarked: int = Field(
        description="How many of them nobody marked. Reported separately and never folded "
        "into the counts, so an unfinished register cannot read as a finished one."
    )
    is_complete: bool = Field(
        description="Whether every child has a mark — not whether they were present."
    )
    counts: TallyOut
    students: list[RegisterLineOut]

    @classmethod
    def of(cls, register: ClassRegister) -> "ClassRegisterOut":
        return cls(
            academic_year_code=register.academic_year_code,
            class_code=register.class_code,
            on_date=register.on_date,
            size=register.size,
            unmarked=register.unmarked,
            is_complete=register.is_complete,
            counts=TallyOut.of(register.counts),
            students=[
                RegisterLineOut(
                    student_number=entry.student_number,
                    full_name_ar=entry.student.full_name_ar if entry.student else "",
                    full_name_en=entry.student.full_name_en if entry.student else "",
                    state=None if entry.mark is None else str(entry.mark.state),
                    note="" if entry.mark is None else entry.mark.note,
                    is_marked=entry.is_marked,
                )
                for entry in register.entries
            ],
        )


class AttendanceEntryIn(BaseModel):
    """One child's state for the day, and why if it needs one."""

    student_number: str = Field(examples=["10432"])
    state: str = Field(
        description="present / absent / late / excused.",
        examples=["present"],
    )
    note: str = Field(
        default="",
        description="Required for `excused`: an excused absence with no reason on file "
        "cannot be told apart from an ordinary one marked by mistake.",
    )


class TakeRegisterIn(BaseModel):
    """A day's register for one class.

    Children left out of `entries` are **left alone**, not marked. Taking the register for
    the twelve children present so far must not silently mark the other twenty-eight absent,
    and a second save later in the morning is a correction of the names it carries and a
    no-op for the rest.

    `absent_unlisted` is how a caller says the pass is finished — see the field.
    """

    entries: list[AttendanceEntryIn] = Field(default_factory=list)
    absent_unlisted: bool = Field(
        default=False,
        description=(
            "Close the register: every child still blank afterwards is recorded `absent`. "
            "This is the supervisor's workflow — name the children in the room, and the "
            "rest are away.\n\n"
            "Off by default, because an unnamed child means two different things and only "
            "the caller knows which: 'not here' or 'not reached yet'. Turning it on is a "
            "caller stating the first.\n\n"
            "It fills blanks and overwrites nothing: a child already marked excused stays "
            "excused. Closing a register is a statement about the children nobody reached, "
            "not a re-statement about the ones somebody did."
        ),
    )


class StudentDayOut(BaseModel):
    on_date: date
    state: str
    note: str = ""
    class_code: str = Field(
        description="The class she was in that day — stored on the mark, not resolved now, "
        "so a transfer in March leaves October's register saying 3A."
    )


class StudentAttendanceOut(BaseModel):
    student_number: str
    from_date: date | None
    to_date: date | None = Field(
        description="Both bounds inclusive, and echoed: a count of absences means nothing "
        "without the window it was counted over."
    )
    counts: TallyOut
    days: list[StudentDayOut]

    @classmethod
    def of(cls, record: StudentAttendance) -> "StudentAttendanceOut":
        return cls(
            student_number=record.student_number,
            from_date=record.from_date,
            to_date=record.to_date,
            counts=TallyOut.of(record.counts),
            days=[
                StudentDayOut(
                    on_date=mark.on_date,
                    state=str(mark.state),
                    note=mark.note,
                    class_code=str(mark.class_code),
                )
                for mark in record.marks
            ],
        )


class RegisterClassOut(BaseModel):
    """One class this caller may take, and how far its register has got on the day asked."""

    class_code: str
    class_name_en: str = ""
    class_name_ar: str = ""
    year_level_code: str
    year_level_name_en: str = ""
    year_level_name_ar: str = ""
    track_code: str | None = None
    may_record: bool = Field(
        description="Whether this caller may write this register, as opposed to only read "
        "it. Both kinds appear in the list: a supervisor who may read 3B and record 3A "
        "should see both and be told which is which."
    )
    size: int = Field(description="Children placed in this class on that day.")
    marked: int = Field(description="How many of them already have a mark.")
    is_complete: bool = Field(
        description="Every child marked. What stops a register being taken twice by "
        "somebody who could not otherwise tell it had been taken at all."
    )


class RegisterClassesOut(BaseModel):
    academic_year_code: str
    on_date: date
    classes: list[RegisterClassOut]


@router.get(
    "/attendance/classes",
    response_model=RegisterClassesOut,
    summary="The classes this caller may take the register for",
    description="Step one of taking a register, and the only read here that answers from "
    "the caller's own grants rather than from something they named.\n\n"
    "Every other listing narrows a target the caller supplies, which a class-scoped "
    "attendance supervisor cannot use: they hold four rooms and no authority over the "
    "grade or the year those rooms sit in, so `GET /v1/structure/classes` refuses them "
    "however they ask. Without this route the workflow's first step — pick a grade, pick a "
    "class — is only possible for somebody who already knows the class code.\n\n"
    "The answer is the union over every grant, so a school-wide registrar sees the school "
    "and a supervisor sees their rooms, through one route. Each row carries its grade, so "
    "a client groups by grade without a second call, and the day's progress, so a register "
    "already taken is visible before it is opened again.",
    responses=error_responses(401, 403, 422),
)
def list_registerable_classes(
    caller: Reader,
    uow_factory: UowFactoryDep,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    on: Annotated[
        date | None,
        Query(description="The day to report progress for. Defaults to today."),
    ] = None,
) -> RegisterClassesOut:
    on_date = on or datetime.now(UTC).date()
    with uow_factory() as uow:
        session = uow._session
        year = session.scalar(
            select(m.AcademicYear).where(m.AcademicYear.code == academic_year)
        )
        if year is None:
            raise UnknownReference(
                f"no academic year {academic_year}", field="academic_year"
            )

        rows = session.execute(
            select(m.ClassSection, m.YearLevel, m.EducationalSystem)
            .join(m.YearLevel, m.ClassSection.year_level_id == m.YearLevel.id)
            .outerjoin(
                m.EducationalSystem,
                m.YearLevel.educational_system_id == m.EducationalSystem.id,
            )
            .where(m.ClassSection.academic_year_id == year.id)
            .order_by(m.YearLevel.display_order, m.YearLevel.code, m.ClassSection.code)
        ).all()

        # One aggregate rather than a count per class: a school-wide registrar has thirty
        # rooms and this route is the first thing their screen calls.
        marked_by_class = dict(
            session.execute(
                select(m.Attendance.class_section_id, func.count())
                .where(
                    m.Attendance.on_date == on_date,
                    m.Attendance.class_section_id.in_([row[0].id for row in rows] or [0]),
                )
                .group_by(m.Attendance.class_section_id)
            ).all()
        )
        sizes = dict(
            session.execute(
                select(m.ClassEnrolment.class_section_id, func.count())
                .where(
                    m.ClassEnrolment.class_section_id.in_([row[0].id for row in rows] or [0]),
                    m.ClassEnrolment.starts_on <= on_date,
                    or_(
                        m.ClassEnrolment.ends_on.is_(None),
                        m.ClassEnrolment.ends_on >= on_date,
                    ),
                )
                .group_by(m.ClassEnrolment.class_section_id)
            ).all()
        )

        listed: list[RegisterClassOut] = []
        for section, level, track in rows:
            # The target names everything above the room, so a grant at any rung of the
            # ladder matches — this is the same `Target` every narrowed route builds, and
            # building it by hand here is what lets one loop answer for all five scopes.
            where = Target(
                school_id=year.school_id,
                track_id=level.educational_system_id,
                year_level_id=level.id,
                class_section_id=section.id,
            )
            if not caller.allows(Permission.ATTENDANCE_READ, where):
                continue
            size = int(sizes.get(section.id, 0))
            marked = int(marked_by_class.get(section.id, 0))
            listed.append(
                RegisterClassOut(
                    class_code=section.code,
                    class_name_en=section.name_en,
                    class_name_ar=section.name_ar,
                    year_level_code=level.code,
                    year_level_name_en=level.name_en,
                    year_level_name_ar=level.name_ar,
                    track_code=None if track is None else track.code,
                    may_record=caller.allows(Permission.ATTENDANCE_WRITE, where),
                    size=size,
                    marked=marked,
                    is_complete=size > 0 and marked >= size,
                )
            )

    return RegisterClassesOut(
        academic_year_code=academic_year, on_date=on_date, classes=listed
    )


@router.get(
    "/classes/{class_code}/attendance",
    response_model=ClassRegisterOut,
    summary="The register of one class on one day",
    description="Every child placed in the class that day, marked or not. `state` is null "
    "for a child nobody marked, which is a third value beside present and absent — a client "
    "that renders it as either is stating a fact the school did not.\n\n"
    "`academic_year` is required: a class code names a different room of children each "
    "September. `on` defaults to today and is echoed as `on_date`.",
    responses=error_responses(401, 403, 404, 422),
)
def read_class_register(
    class_code: str,
    attendance: AttendanceServiceDep,
    caller: Reader,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    on: Annotated[
        date | None,
        Query(description="The day to answer for. Defaults to today, echoed as `on_date`."),
    ] = None,
) -> ClassRegisterOut:
    on_date = on or datetime.now(UTC).date()
    # Narrowed before the read, not after: a refusal that arrives with the register
    # already assembled has still had the register assembled, and a class-scoped grant
    # exists precisely so that this person cannot see this room.
    caller.narrow(
        Permission.ATTENDANCE_READ,
        lambda scopes: scopes.for_class(
            academic_year_code=academic_year, class_code=class_code
        ),
    )
    with domain_errors():
        register = attendance.register_for_class(
            AcademicYearCode(academic_year), ClassCode(class_code), on_date
        )
    return ClassRegisterOut.of(register)


@router.put(
    "/classes/{class_code}/attendance",
    response_model=ClassRegisterOut,
    summary="Record the register for one class on one day",
    description="Idempotent: saving the same morning twice corrects the marks rather than "
    "writing a second set beside them, because a day holds one statement per child. The "
    "database holds one row per `(child, day)`, so a double-submitted register cannot "
    "become two.\n\n"
    "PUT rather than POST for that reason — the request states what the register *is* for "
    "that day, and repeating it changes nothing. Children absent from `entries` are left "
    "untouched, so a partial register can be saved and finished later — unless "
    "`absent_unlisted` closes it, which records every child still blank as absent.\n\n"
    "A child who was not in this class on this day is refused with her number in the "
    "message: that is a stale screen or the wrong class, and writing it would file her "
    "attendance under a room she had already left.",
    responses=error_responses(401, 403, 404, 422),
)
def take_register(
    class_code: str,
    body: Annotated[TakeRegisterIn, Body()],
    attendance: AttendanceServiceDep,
    caller: Registrar,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    on: Annotated[
        date | None,
        Query(description="The day being recorded. Defaults to today."),
    ] = None,
) -> ClassRegisterOut:
    on_date = on or datetime.now(UTC).date()
    # The check this whole scope model is for: an attendance supervisor is given rooms,
    # and taking the register of a room they were not given is the thing being refused.
    caller.narrow(
        Permission.ATTENDANCE_WRITE,
        lambda scopes: scopes.for_class(
            academic_year_code=academic_year, class_code=class_code
        ),
    )
    with domain_errors():
        register = attendance.take_register(
            AcademicYearCode(academic_year),
            ClassCode(class_code),
            on_date,
            states={entry.student_number: entry.state for entry in body.entries},
            notes={
                entry.student_number: entry.note for entry in body.entries if entry.note
            },
            actor=caller.prefix,
            absent_unlisted=body.absent_unlisted,
        )
    return ClassRegisterOut.of(register)


@router.get(
    "/guardians/by-id/{public_id}/students/{student_number}/attendance",
    response_model=StudentAttendanceOut,
    summary="A child's attendance, read by one of her guardians",
    description="The same record as the route below, for a caller who holds a guardian "
    "handle rather than a registrar's authority. The guardian-to-child link is re-checked "
    "on this request: a caller that names a child who is not hers, or whose access has "
    "been restricted, gets the same 404 as one naming a child who does not exist.\n\n"
    "This is the route a parent-facing service should use. It never needs the parent's "
    "phone number, and it does not rely on that service having filtered correctly before "
    "asking — which matters because the caller is a language model's tool, reading text a "
    "stranger wrote.",
    responses=error_responses(401, 403, 404, 422),
)
def read_guardian_student_attendance(
    public_id: str,
    student_number: str,
    attendance: AttendanceServiceDep,
    caller: Reader,
    request_id: RequestId,
    from_: Annotated[
        date | None, Query(alias="from", description="First day, inclusive.")
    ] = None,
    to: Annotated[date | None, Query(description="Last day, inclusive.")] = None,
) -> StudentAttendanceOut:
    with domain_errors():
        record = attendance.for_guardian_student(
            public_id,
            StudentNumber(student_number),
            from_date=from_,
            to_date=to,
            actor=caller.prefix,
            request_id=request_id,
        )
    return StudentAttendanceOut.of(record)


@router.get(
    "/students/{student_number}/attendance",
    response_model=StudentAttendanceOut,
    summary="One child's attendance over a range",
    description="Oldest first, with counts. Both `from` and `to` are inclusive and both are "
    "optional; omitting them returns everything on file.\n\n"
    "The counts carry `recorded` — the number of days a mark exists for — and no rate. A "
    "percentage needs a denominator, and the only one this service can state honestly is "
    "the days somebody actually took the register.",
    responses=error_responses(401, 403, 404, 422),
)
def read_student_attendance(
    student_number: str,
    attendance: AttendanceServiceDep,
    caller: Reader,
    from_: Annotated[
        date | None, Query(alias="from", description="First day, inclusive.")
    ] = None,
    to: Annotated[date | None, Query(description="Last day, inclusive.")] = None,
) -> StudentAttendanceOut:
    with domain_errors():
        record = attendance.for_student(
            StudentNumber(student_number), from_date=from_, to_date=to
        )
    return StudentAttendanceOut.of(record)
