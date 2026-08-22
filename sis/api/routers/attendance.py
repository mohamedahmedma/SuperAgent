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
from pydantic import BaseModel, Field

from sis.api.deps import (
    AttendanceServiceDep,
    Caller,
    require_read_access,
    require_registrar,
)
from sis.api.routers import domain_errors, error_responses
from sis.application.services.attendance import ClassRegister, StudentAttendance
from sis.domain.attendance import AttendanceState, AttendanceTally
from sis.domain.value_objects import AcademicYearCode, ClassCode, StudentNumber

router = APIRouter(prefix="/v1", tags=["attendance"])

Reader = Annotated[Caller, Depends(require_read_access)]
Registrar = Annotated[Caller, Depends(require_registrar)]


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
    """

    entries: list[AttendanceEntryIn] = Field(min_length=1)


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
    "writing a second set beside them, because a day holds one statement per child.\n\n"
    "PUT rather than POST for that reason — the request states what the register *is* for "
    "that day, and repeating it changes nothing. Children absent from `entries` are left "
    "untouched, so a partial register can be saved and finished later.\n\n"
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
        )
    return ClassRegisterOut.of(register)


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
