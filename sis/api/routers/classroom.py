"""A child's room over HTTP: its name, its subject board, and who teaches it.

One route, and it answers three questions a parent asks separately:

  "what class is my daughter in"      -> `class_code` and `class_name_*`
  "what subjects does she study"      -> `subjects`
  "who are her teachers"              -> `teachers`, one entry per teacher per subject

**One route rather than three, because all three hang off one fact.** Which room she was
placed in for the term is resolved once, inside one transaction — see
`sis/application/services/classroom.py`. Three routes would resolve it three times and a
placement edited in between would answer with one room's subjects and another room's staff.
The parent-facing facade in `records/` is free to project this into three endpoints, and
does; what must not be split is the read.

**Guardian-scoped, and the class is resolved here.** Like the guardian-scoped grades,
attendance and timetable routes, the caller names a STUDENT and a TERM and never a class:
a parent has no class code, and a service that cached one would eventually ask about a room
the child has left. The guardian-to-child link is re-checked on this request from the
registrar's own data, so a caller naming a child who is not theirs gets the same 404 as one
naming a child who does not exist.

**What the term pins, stated because it is easy to over-promise.** The term resolves the
ROOM — 3A for Term 1, forever, even after a March move to 3B. It does **not** resolve the
STAFFING: `teacher_class_sections` carries no term and no validity window, so `teachers` is
who teaches that room *now*. Asking about last term gives last term's room and today's
teachers for it, and the field description says so rather than implying a history the
schema cannot keep.

**Three empties, three meanings.** `class_code: null` says no placement covered the term —
a fact about the child. A class with `subjects: []` says nobody has curated that rung's
board; with `teachers: []`, nobody has entered the room's staffing. Both of those are facts
about the school, and a client that rendered all three as "she has no teachers" would tell a
parent something false in two cases out of three.

Nothing here exposes a teacher's email, phone, username or user id. The response model is
the privacy boundary rather than a filter someone has to remember downstream, which is why
it is a lean shape of its own and never `TeacherOut`.
"""
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from sis.api.deps import (
    Principal,
    RequestId,
    get_classroom_service,
    get_query_service,
    require_permission,
)
from sis.api.routers import domain_errors, error_responses
from sis.application.services import ClassroomService, QueryService, StudentClassroom
from sis.domain.rbac import Permission
from sis.domain.value_objects import StudentNumber, TermCode

router = APIRouter(prefix="/v1", tags=["classroom"])

# `TEACHERS_READ` because staff is the most sensitive thing on the payload; the subject
# board and the class name are structural and readable by anyone who may read teachers.
Reader = Annotated[Principal, Depends(require_permission(Permission.TEACHERS_READ))]
Classrooms = Annotated[ClassroomService, Depends(get_classroom_service)]
Queries = Annotated[QueryService, Depends(get_query_service)]


class ClassSubjectOut(BaseModel):
    """One subject on the rung's board."""

    code: str = Field(examples=["MATH"])
    name_ar: str
    name_en: str


class ClassTeacherOut(BaseModel):
    """One teacher in the room, and the subject they teach in it.

    One entry per (teacher, subject): a teacher taking two subjects for the class appears
    twice, and a subject taught by two teachers has two entries. The table permits both,
    so a client must render a list rather than assume a single name per subject.

    Carries a name and a subject and nothing else. No staff number, no email, no phone, no
    username — a parent needs to know who teaches their child, not how to reach a member of
    staff directly, and the school's contact routes are the school's to publish.
    """

    full_name_ar: str = Field(
        description="Empty when the school recorded only one spelling; the other is then "
        "populated. Never both empty for an active teacher."
    )
    full_name_en: str
    subject_code: str = Field(examples=["MATH"])
    subject_name_ar: str
    subject_name_en: str


class StudentClassroomOut(BaseModel):
    """A child's room for one term: its name, its subjects, and its staff."""

    student_number: str
    term_code: str
    academic_year_code: str | None = Field(
        default=None, description="`null` when no placement covered the term."
    )
    class_code: str | None = Field(
        default=None,
        description="The school's internal code for the room — `3A`, `P1-01`. The class "
        "she was in **for this term**, not her current one. `null` means no placement "
        "covered the term: she had left, or had not yet joined, and neither is a missing "
        "record.",
    )
    class_name_ar: str | None = Field(
        default=None,
        description="What the school calls the room — `Primary 1 Class 1`, `3/1`. This is "
        "the answer to 'which class is my child in'; the code above is an internal key a "
        "family has no reason to see.",
    )
    class_name_en: str | None = None
    year_level_code: str | None = Field(
        default=None, description="The rung the room sits on."
    )
    year_level_name_ar: str | None = Field(
        default=None,
        description="The rung's human name — `Year 3`. `null` only when the school's "
        "ladder is missing the rung its own class points at, which is broken data rather "
        "than a normal state; the rest of the answer is still returned.",
    )
    year_level_name_en: str | None = None
    subjects: list[ClassSubjectOut] = Field(
        default_factory=list,
        description="The subjects assigned to this rung, in the school's own display "
        "order. Retired subjects are excluded — a dropped subject is not one she studies, "
        "though it still resolves for marks already stated against it.\n\n"
        "Empty with a class present means nobody has curated this rung's board for the "
        "year. That is a fact about the school's admin and must not be reported as the "
        "child studying nothing.",
    )
    teachers: list[ClassTeacherOut] = Field(
        default_factory=list,
        description="Ordered by subject display order, then subject code, then teacher. "
        "Inactive teachers are excluded: someone who has left is not the answer to 'who "
        "teaches my daughter maths'.\n\n"
        "**Who teaches the room now, not who taught it in the term asked about.** The "
        "assignment table carries no term, so this cannot be reported historically.\n\n"
        "Empty with a class present means nobody has entered the room's staffing — again "
        "about the school, not the child.",
    )

    @classmethod
    def of(cls, room: StudentClassroom) -> "StudentClassroomOut":
        section = room.class_section
        if section is None:
            return cls(
                student_number=room.student_number, term_code=room.term_code
            )
        level = room.year_level
        return cls(
            student_number=room.student_number,
            term_code=room.term_code,
            academic_year_code=str(section.academic_year_code),
            class_code=str(section.code),
            class_name_ar=section.name_ar,
            class_name_en=section.name_en,
            year_level_code=str(section.year_level_code),
            year_level_name_ar=None if level is None else level.name_ar,
            year_level_name_en=None if level is None else level.name_en,
            subjects=[
                ClassSubjectOut(
                    code=str(subject.code),
                    name_ar=subject.name_ar,
                    name_en=subject.name_en,
                )
                for subject in room.subjects
            ],
            teachers=[
                ClassTeacherOut(
                    full_name_ar=teacher.full_name_ar,
                    full_name_en=teacher.full_name_en,
                    subject_code=teacher.subject_code,
                    subject_name_ar=teacher.subject_name_ar,
                    subject_name_en=teacher.subject_name_en,
                )
                for teacher in room.teachers
            ],
        )


@router.get(
    "/guardians/by-id/{public_id}/students/{student_number}/classroom",
    response_model=StudentClassroomOut,
    summary="A child's class, its subjects and its teachers, read by one of her guardians",
    description="Everything about the room a child sits in that a family may be told, in "
    "one request: what the class is called, which subjects that rung studies, and who "
    "teaches them.\n\n"
    "**One request rather than three, because three can disagree.** The placement that "
    "decides the room is resolved once, in one transaction. Asked separately, a placement "
    "edited in between would answer with one room's subjects and another room's staff — the "
    "same reason `/timetable/week` serves the grid and the lessons together.\n\n"
    "**The class is resolved here, not supplied.** A parent has no class code and a "
    "parent-facing service has no business holding one: it changes mid-year, and a cached "
    "one eventually names a room the child has left.\n\n"
    "**`term` pins the room, not the staffing.** She gets the class she sat in for that "
    "term — 3A for Term 1 even after a March move to 3B — but `teachers` is who teaches "
    "that room *now*, because the assignment table carries no term.\n\n"
    "The guardian-to-child link is re-checked on this request: a caller naming a child who "
    "is not hers, or whose access a court order has restricted, gets the same 404 as one "
    "naming a child who does not exist.\n\n"
    "A child with no placement covering the term answers 200 with `class_code: null` rather "
    "than 404 — she was enrolled for September, or has left, and neither is a missing "
    "record.",
    responses=error_responses(401, 403, 404, 422),
)
def read_guardian_student_classroom(
    public_id: str,
    student_number: str,
    classrooms: Classrooms,
    queries: Queries,
    caller: Reader,
    request_id: RequestId,
    term: Annotated[
        str,
        Query(
            description="Term code. Required, for the reason the guardian grades route "
            "gives: a bare 'this term' would be answered from a clock, and which room a "
            "child sat in is a statement about a named stretch of the year.",
            examples=["2026-T1"],
        ),
    ],
) -> StudentClassroomOut:
    # No `caller.narrow`, the same asymmetry the guardian-scoped grades, attendance and
    # timetable routes make: a registrar's key is bounded by the rooms they hold, while
    # this caller is a service acting for one family, and what bounds it is the link
    # re-checked below from the registrar's own data on this request. Narrowing by scope
    # would refuse it for holding no class at all.
    with domain_errors():
        queries.require_guardian_may_see(
            public_id,
            StudentNumber(student_number),
            actor=caller.prefix,
            request_id=request_id,
        )
        room = classrooms.for_student(StudentNumber(student_number), TermCode(term))
    return StudentClassroomOut.of(room)
