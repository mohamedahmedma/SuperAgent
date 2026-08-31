"""What a teacher teaches, and the mark sheet they are allowed to fill in.

Two routes and one rule, and the rule is the whole point of the module:

**A teacher records their own subject, in their own rooms, and nothing else.** The scope
model cannot express that — a teacher's grant is a *classroom*, and `grades.write` on 4/1
covers every question that names 4/1, Physics and Arabic alike. So every write here runs
two checks in order, and both have to pass:

    caller.narrow(GRADES_WRITE, ...)   the scope: is this room yours at all
    teaching.may_record(...)           the assignment: is this subject yours in it

The first is the same check every other route makes and settles a registrar in memory. The
second reads `teacher_class_sections` and only ever *narrows* — a caller with no teacher
record is unaffected by it, because "which subject do you teach" is a question about
teaching staff and meaningless asked of the office. See `sis/application/services/teaching.py`.

`GET /v1/teaching/assignments` is the other half, and it exists for the same reason the
Stage 12 and Stage 13 listings do: a teacher holds classrooms and nothing above them, so
every listing that narrows a grade or a year refuses them and they have no way to discover
what they teach. It answers from their own grants and assignments, grouped by grade, so a
teacher of four rooms across three rungs and two sections sees all of it in one call.
"""
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from sis.api.deps import (
    Principal,
    get_mark_sheet_service,
    get_teaching_service,
    require_permission,
)
from sis.api.routers import domain_errors, error_responses
from sis.application.services.marks import MarkSheet, MarkSheetService, StatedMark
from sis.application.services.teaching import TeachingAssignment, TeachingService
from sis.domain.rbac import Permission
from sis.domain.value_objects import AcademicYearCode, ClassCode, SubjectCode, TermCode

router = APIRouter(prefix="/v1", tags=["teaching"])

Reader = Annotated[Principal, Depends(require_permission(Permission.GRADES_READ))]
Recorder = Annotated[Principal, Depends(require_permission(Permission.GRADES_WRITE))]
Teaching = Annotated[TeachingService, Depends(get_teaching_service)]
Sheets = Annotated[MarkSheetService, Depends(get_mark_sheet_service)]


class TeachingAssignmentOut(BaseModel):
    """One room, one subject, and the rung it sits on."""

    class_code: str
    class_name_en: str = ""
    class_name_ar: str = ""
    subject_code: str
    subject_name_en: str = ""
    subject_name_ar: str = ""
    year_level_code: str
    year_level_name_en: str = ""
    year_level_name_ar: str = ""
    track_code: str | None = None
    academic_year_code: str

    @classmethod
    def of(cls, row: TeachingAssignment) -> "TeachingAssignmentOut":
        return cls(
            class_code=row.class_code,
            class_name_en=row.class_name_en,
            class_name_ar=row.class_name_ar,
            subject_code=row.subject_code,
            subject_name_en=row.subject_name_en,
            subject_name_ar=row.subject_name_ar,
            year_level_code=row.year_level_code,
            year_level_name_en=row.year_level_name_en,
            year_level_name_ar=row.year_level_name_ar,
            track_code=row.track_code,
            academic_year_code=row.academic_year_code,
        )


class TeachingAssignmentsOut(BaseModel):
    is_teaching_staff: bool = Field(
        description="Whether a teacher record is linked to this account. `false` with an "
        "empty list is a registrar, who is bounded by scope alone; `true` with an empty "
        "list is a teacher nobody has given a class yet, who may record nothing."
    )
    assignments: list[TeachingAssignmentOut]


class MarkSheetLineOut(BaseModel):
    student_number: str
    full_name_ar: str = ""
    full_name_en: str = ""
    percentage: float | None = Field(
        default=None,
        description="The stated figure, or **null** when nobody has marked her. Null is "
        "not zero: zero is a mark a child earned and null is the absence of one.",
    )
    points: float | None = None
    max_points: float | None = None
    is_graded: bool


class MarkSheetOut(BaseModel):
    academic_year_code: str
    class_code: str
    subject_code: str
    subject_name_en: str = ""
    subject_name_ar: str = ""
    term_code: str
    term_is_closed: bool
    may_record: bool = Field(
        description="Whether this caller may write this sheet — scope *and* teaching "
        "assignment. A teacher opening a colleague's subject in their own room reads it "
        "and gets `false`, which is what the screen renders read-only."
    )
    size: int
    graded: int
    ungraded: int = Field(
        description="Children with no figure on file. Reported separately and never folded "
        "into a total, so an unfinished sheet cannot read as a finished one."
    )
    students: list[MarkSheetLineOut]

    @classmethod
    def of(cls, sheet: MarkSheet, *, may_record: bool) -> "MarkSheetOut":
        return cls(
            academic_year_code=sheet.academic_year_code,
            class_code=sheet.class_code,
            subject_code=sheet.subject_code,
            subject_name_en=sheet.subject.name_en if sheet.subject else "",
            subject_name_ar=sheet.subject.name_ar if sheet.subject else "",
            term_code=sheet.term_code,
            term_is_closed=sheet.term_is_closed,
            may_record=may_record,
            size=sheet.size,
            graded=sheet.graded,
            ungraded=sheet.ungraded,
            students=[
                MarkSheetLineOut(
                    student_number=line.student_number,
                    full_name_ar=line.student.full_name_ar if line.student else "",
                    full_name_en=line.student.full_name_en if line.student else "",
                    percentage=(
                        None
                        if line.grade is None or line.grade.percentage is None
                        else float(line.grade.percentage.value)
                    ),
                    points=None if line.grade is None else line.grade.points,
                    max_points=None if line.grade is None else line.grade.max_points,
                    is_graded=line.is_graded,
                )
                for line in sheet.lines
            ],
        )


class StatedMarkIn(BaseModel):
    """One figure being stated for one child."""

    student_number: str = Field(examples=["10432"])
    percentage: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="The mark out of 100. Omit it with `clear: true` to erase what is on "
        "file; omitting it without `clear` records no figure rather than a zero.",
    )
    points: float | None = Field(
        default=None, description="The raw figure, when the teacher marked out of something else."
    )
    max_points: float | None = Field(
        default=None, description="What `points` was out of. Required whenever points is sent."
    )
    clear: bool = Field(
        default=False,
        description="Erase the mark on file, returning this child to not-yet-marked. A "
        "separate flag rather than a null percentage, so a screen sending every visible "
        "row cannot wipe the marks it merely failed to load.",
    )


class RecordMarksIn(BaseModel):
    term_code: str = Field(examples=["2026-T1"])
    subject_code: str = Field(examples=["MATH"])
    marks: list[StatedMarkIn] = Field(min_length=1)


@router.get(
    "/teaching/assignments",
    response_model=TeachingAssignmentsOut,
    summary="The classes and subjects this teacher is assigned",
    description="What a teacher may record, answered from their own assignments rather "
    "than from something they named — the same shape as the Stage 12 and Stage 13 "
    "listings, and for the same reason: a teacher holds classrooms and nothing above "
    "them, so every listing that narrows a grade refuses them and they cannot discover "
    "their own timetable.\n\n"
    "Each row is one room and one subject, carrying its grade and track, so a teacher "
    "working several classes across several grades and both sections gets all of it in "
    "one call and a client groups it without a second.",
    responses=error_responses(401, 403, 422),
)
def list_my_teaching(
    caller: Reader,
    teaching: Teaching,
    academic_year: Annotated[str | None, Query(examples=["2025-2026"])] = None,
) -> TeachingAssignmentsOut:
    user_id = None if caller.profile is None else caller.profile.user_id
    return TeachingAssignmentsOut(
        is_teaching_staff=teaching.is_teaching_staff(user_id),
        assignments=[
            TeachingAssignmentOut.of(row)
            for row in teaching.assignments_for_user(
                user_id, academic_year_code=academic_year
            )
        ],
    )


@router.get(
    "/classes/{class_code}/grades",
    response_model=MarkSheetOut,
    summary="The mark sheet of one class, for one subject and term",
    description="Every child who sat in the class **for that term**, with her figure for "
    "this subject or null where nobody has marked her. Built from the enrolments rather "
    "than from the marks, so a sheet four children into a class of thirty shows thirty "
    "rows and twenty-six blanks instead of four children who all did well.\n\n"
    "The class is the one she sat in for the term, not her current one, so a child who "
    "transferred in March still appears on the Term 1 sheet of the room she earned those "
    "marks in.\n\n"
    "`may_record` reports whether this caller may write it. A teacher may read the sheet "
    "of a colleague's subject in their own room and gets `false` — reading a class's marks "
    "is part of teaching it; stating a figure for somebody else's subject is not.",
    responses=error_responses(401, 403, 404, 422),
)
def read_mark_sheet(
    class_code: str,
    caller: Reader,
    sheets: Sheets,
    teaching: Teaching,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    term: Annotated[str, Query(examples=["2026-T1"])],
    subject: Annotated[str, Query(examples=["MATH"])],
) -> MarkSheetOut:
    caller.narrow(
        Permission.GRADES_READ,
        lambda scopes: scopes.for_class(
            academic_year_code=academic_year, class_code=class_code
        ),
    )
    if not _may_record(caller, teaching, academic_year, class_code, subject):
        raise _assignment_forbidden(subject, class_code)
    with domain_errors():
        sheet = sheets.sheet(
            AcademicYearCode(academic_year),
            ClassCode(class_code),
            SubjectCode(subject),
            TermCode(term),
        )
    return MarkSheetOut.of(
        sheet, may_record=_may_record(caller, teaching, academic_year, class_code, subject)
    )


@router.put(
    "/classes/{class_code}/grades",
    response_model=MarkSheetOut,
    summary="Record marks for one class, subject and term",
    description="The teacher's own write path. Idempotent by `(child, subject, term)` — "
    "which is what the table is unique on — so saving the same sheet twice corrects it "
    "rather than filing a second set of figures beside the first.\n\n"
    "**A teacher may only record their own subject.** The scope check says whether this "
    "room is theirs at all; the assignment check says whether this subject in it is. Both "
    "run, in that order, and a teacher of Arabic in 4/1 recording its Mathematics is "
    "refused by the second with 403 — the room is genuinely theirs, and the subject is "
    "not.\n\n"
    "A child who was not in this class for this term is refused by number rather than "
    "skipped: a silently dropped mark is one the teacher believes they entered.\n\n"
    "Omitting `percentage` records no figure; it does not record a zero. Erasing a mark "
    "is `clear: true`.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def record_marks(
    class_code: str,
    body: Annotated[RecordMarksIn, Body()],
    caller: Recorder,
    sheets: Sheets,
    teaching: Teaching,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
) -> MarkSheetOut:
    # First the scope: is this room yours at all. Settled from memory for anybody holding
    # a school-wide grant, and it is what refuses a teacher another teacher's classroom.
    caller.narrow(
        Permission.GRADES_WRITE,
        lambda scopes: scopes.for_class(
            academic_year_code=academic_year, class_code=class_code
        ),
    )
    # Then the assignment: is this subject yours in it. The check the scope model cannot
    # make, and the one this stage exists for.
    if not _may_record(caller, teaching, academic_year, class_code, body.subject_code):
        raise _assignment_forbidden(body.subject_code, class_code)

    with domain_errors():
        sheet = sheets.record(
            AcademicYearCode(academic_year),
            ClassCode(class_code),
            SubjectCode(body.subject_code),
            TermCode(body.term_code),
            [
                StatedMark(
                    student_number=mark.student_number,
                    percentage=mark.percentage,
                    points=mark.points,
                    max_points=mark.max_points,
                    clear=mark.clear,
                )
                for mark in body.marks
            ],
        )
    return MarkSheetOut.of(sheet, may_record=True)


def _may_record(
    caller: Principal,
    teaching: TeachingService,
    academic_year: str,
    class_code: str,
    subject_code: str,
) -> bool:
    """Whether this caller may state a figure for this subject in this room.

    Resolves the pair to surrogate ids and asks `TeachingService`. A caller with no teacher
    record answers `True` here and is bounded by the scope check that has already run;
    this only ever narrows a teacher.
    """
    if caller.profile is None:
        return True
    return teaching.may_record_by_code(
        caller.profile.user_id,
        academic_year_code=academic_year,
        class_code=class_code,
        subject_code=subject_code,
    )


def _assignment_forbidden(subject_code: str, class_code: str) -> HTTPException:
    """An assignment failure is an authorization refusal, not a grade conflict."""
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "code": "not_authorized",
            "message": f"You are not assigned to teach {subject_code} in {class_code}.",
            "field": "subject_code",
        },
    )
