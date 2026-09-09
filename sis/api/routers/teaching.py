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
from typing import Annotated, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import text

from sis.api.deps import (
    Principal,
    UowFactoryDep,
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
    is_absent: bool = False


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
                    is_absent=False,
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
    absent: bool = Field(
        default=False,
        description="Student was absent from this specific assessment. Distinct from blank and zero.",
    )


class RecordMarksIn(BaseModel):
    term_code: str = Field(examples=["2026-T1"])
    subject_code: str = Field(examples=["MATH"])
    assessment_type: Literal["exam", "assignment"] | None = None
    assessment_name: str | None = Field(default=None, min_length=1, max_length=160)
    marks: list[StatedMarkIn] = Field(min_length=1)


class AssessmentSummaryOut(BaseModel):
    id: int
    assessment_type: str
    name: str
    max_points: float | None = None

class AssessmentListOut(BaseModel):
    assessments: list[AssessmentSummaryOut]

class AssessmentSheetOut(BaseModel):
    id: int
    class_code: str
    subject_code: str
    term_code: str
    assessment_type: str
    name: str
    max_points: float | None = None
    students: list[MarkSheetLineOut]


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
    if (
        caller.profile is not None
        and caller.profile.has_role("teacher")
        and not caller.profile.has_role("floor_supervisor")
        and not _may_record(caller, teaching, academic_year, class_code, subject)
    ):
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


@router.get("/classes/{class_code}/assessments", response_model=AssessmentListOut)
def list_class_assessments(class_code: str, caller: Reader, uow_factory: UowFactoryDep,
    academic_year: Annotated[str, Query()], term: Annotated[str, Query()],
    subject: Annotated[str, Query()], assessment_type: Annotated[Literal["exam", "assignment"], Query()]) -> AssessmentListOut:
    caller.narrow(Permission.GRADES_READ, lambda scopes: scopes.for_class(academic_year_code=academic_year, class_code=class_code))
    with uow_factory() as uow:
        rows=uow._session.execute(text("SELECT id,assessment_type,name,max_points FROM assessments WHERE academic_year_code=:y AND class_code=:c AND subject_code=:s AND term_code=:t AND assessment_type=:a ORDER BY created_at DESC,id DESC"),
            {"y":academic_year,"c":class_code,"s":subject,"t":term,"a":assessment_type}).mappings().all()
    return AssessmentListOut(assessments=[AssessmentSummaryOut(**dict(x)) for x in rows])

@router.get("/classes/{class_code}/assessments/{assessment_id}", response_model=AssessmentSheetOut)
def read_class_assessment(class_code: str, assessment_id: int, caller: Reader, sheets: Sheets,
    uow_factory: UowFactoryDep, academic_year: Annotated[str, Query()]) -> AssessmentSheetOut:
    caller.narrow(Permission.GRADES_READ, lambda scopes: scopes.for_class(academic_year_code=academic_year, class_code=class_code))
    with uow_factory() as uow:
        head=uow._session.execute(text("SELECT id,class_code,subject_code,term_code,assessment_type,name,max_points FROM assessments WHERE id=:id AND academic_year_code=:y AND class_code=:c"),
            {"id":assessment_id,"y":academic_year,"c":class_code}).mappings().first()
        if head is None: raise HTTPException(status_code=404, detail="Assessment not found.")
        saved={x["student_number"]:x for x in uow._session.execute(text("SELECT student_number,points,max_points,percentage,is_absent FROM assessment_marks WHERE assessment_id=:id"),{"id":assessment_id}).mappings().all()}
    with domain_errors():
        roster=sheets.sheet(AcademicYearCode(academic_year),ClassCode(class_code),SubjectCode(head["subject_code"]),TermCode(head["term_code"]))
    base=MarkSheetOut.of(roster,may_record=False)
    students=[]
    for line in base.students:
        mark=saved.get(line.student_number)
        students.append(MarkSheetLineOut(student_number=line.student_number,full_name_ar=line.full_name_ar,full_name_en=line.full_name_en,
            percentage=None if mark is None else mark["percentage"],points=None if mark is None else mark["points"],
            max_points=None if mark is None else mark["max_points"],
            is_graded=mark is not None and not bool(mark["is_absent"]),
            is_absent=False if mark is None else bool(mark["is_absent"])))
    return AssessmentSheetOut(**dict(head),students=students)


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
    uow_factory: UowFactoryDep,
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
        numeric_marks = [mark for mark in body.marks if not mark.absent]
        if numeric_marks:
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
                    for mark in numeric_marks
                ],
            )
        else:
            sheet = sheets.sheet(
                AcademicYearCode(academic_year),
                ClassCode(class_code),
                SubjectCode(body.subject_code),
                TermCode(body.term_code),
            )
    if body.assessment_type and body.assessment_name:
        _save_assessment_snapshot(uow_factory, class_code, academic_year, body, caller.prefix)
    return MarkSheetOut.of(sheet, may_record=True)


def _save_assessment_snapshot(uow_factory, class_code: str, academic_year: str, body: RecordMarksIn, actor: str) -> None:
    maximum=max([float(x.max_points) for x in body.marks if x.max_points is not None] or [0.0]) or None
    with uow_factory() as uow:
        s=uow._session
        s.execute(text("INSERT INTO assessments(academic_year_code,class_code,subject_code,term_code,assessment_type,name,max_points,recorded_by) VALUES(:y,:c,:s,:t,:a,:n,:m,:r) ON CONFLICT(academic_year_code,class_code,subject_code,term_code,assessment_type,name) DO UPDATE SET max_points=excluded.max_points,recorded_by=excluded.recorded_by,updated_at=CURRENT_TIMESTAMP"),
            {"y":academic_year,"c":class_code,"s":body.subject_code,"t":body.term_code,"a":body.assessment_type,"n":body.assessment_name.strip(),"m":maximum,"r":actor})
        aid=s.execute(text("SELECT id FROM assessments WHERE academic_year_code=:y AND class_code=:c AND subject_code=:s AND term_code=:t AND assessment_type=:a AND name=:n"),
            {"y":academic_year,"c":class_code,"s":body.subject_code,"t":body.term_code,"a":body.assessment_type,"n":body.assessment_name.strip()}).scalar_one()
        for mark in body.marks:
            if mark.clear:
                s.execute(text("DELETE FROM assessment_marks WHERE assessment_id=:a AND student_number=:n"),{"a":aid,"n":mark.student_number})
                continue
            if mark.absent:
                s.execute(text("INSERT INTO assessment_marks(assessment_id,student_number,points,max_points,percentage,is_absent) VALUES(:a,:n,NULL,NULL,NULL,1) ON CONFLICT(assessment_id,student_number) DO UPDATE SET points=NULL,max_points=NULL,percentage=NULL,is_absent=1,updated_at=CURRENT_TIMESTAMP"),{"a":aid,"n":mark.student_number})
                continue
            if mark.points is None and mark.percentage is None: continue
            pct=mark.percentage
            if pct is None and mark.points is not None and mark.max_points:
                pct=(float(mark.points)/float(mark.max_points))*100.0
            s.execute(text("INSERT INTO assessment_marks(assessment_id,student_number,points,max_points,percentage,is_absent) VALUES(:a,:n,:p,:m,:pct,0) ON CONFLICT(assessment_id,student_number) DO UPDATE SET points=excluded.points,max_points=excluded.max_points,percentage=excluded.percentage,is_absent=0,updated_at=CURRENT_TIMESTAMP"),{"a":aid,"n":mark.student_number,"p":mark.points,"m":mark.max_points,"pct":pct})
        uow.commit()



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
