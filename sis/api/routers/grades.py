"""A child's marks for one term — the route `records/` reads to answer a parent.

Two rules survive or die at this serialiser, so both are stated here rather than trusted
to whoever writes the next client.

**A blank grade is `null`, never `0`** (invariant 1). `percentage` is `float | None` and
there is no default anywhere on the way out: no `or 0.0`, no `.get(col, 0)`, no numeric
placeholder for an unmarked subject. Zero is a mark a child can earn, so a zero on a wire
that also means "not marked yet" is indistinguishable from one a teacher awarded — and the
parent-facing consequence is a school telling a family their daughter scored 0% in a
subject nobody has graded. `is_graded` travels beside every line so a client never has to
re-derive the distinction as a truth test, which is where `if grade.percentage:` silently
turns a legitimate zero into a blank.

**Nothing is aggregated** (invariant 5). There is no average, no total, no GPA and no
rank in this response, because every one of those encodes a policy — which subjects count,
what weight each carries, what an ungraded subject does to the denominator — and a policy
invented in a serialiser is one no human ever approved. `graded_count` is a count of
stated figures, not a score.

The class on the response is resolved *for the term* rather than read from the child's
current placement (invariant 2). A Term 1 report keeps printing 3A after a March move to
3B, because that is the room those marks were earned in.
"""
from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from sis.api.deps import (
    Principal,
    RequestId,
    get_query_service,
    get_teaching_service,
    require_permission,
)
from sis.domain.rbac import Permission
from sis.api.routers import domain_errors, error_responses
from sis.application.services import QueryService
from sis.application.services.teaching import TeachingService
from sis.application.services.queries import GradeLine, StudentTermGrades
from sis.domain.value_objects import StudentNumber, TermCode

router = APIRouter(prefix="/v1", tags=["grades"])

# `require_read_access`, not `require_reader`: `Scope.permits` is exact equality, so a
# reader-only check refuses the registrar's own key — and the registrar UI printing a
# report card is the first caller of this route.
Reader = Annotated[Principal, Depends(require_permission(Permission.GRADES_READ))]
Queries = Annotated[QueryService, Depends(get_query_service)]
Teaching = Annotated[TeachingService, Depends(get_teaching_service)]


class GradeLineOut(BaseModel):
    """One line of a report card: a subject, and the figure the school stated for it."""

    subject_code: str
    subject_name_ar: str = Field(
        description="Empty when the subject row could not be loaded. The line is still "
        "returned — a report card that quietly drops a line is one nobody can tell is "
        "incomplete."
    )
    subject_name_en: str
    is_graded: bool = Field(
        description="The supported way to ask whether a mark exists. `false` means the "
        "subject has not been marked yet; it does not mean zero."
    )
    percentage: float | None = Field(
        default=None,
        description="`null` when not graded yet — never 0. A stated 0 is a mark the child "
        "earned and is returned as `0.0` with `is_graded: true`.",
    )
    points: float | None = Field(
        default=None,
        description="The raw figure the teacher wrote, when the sheet stated one. Never "
        "recomputed from `percentage`, and `percentage` is never recomputed from it.",
    )
    max_points: float | None = None
    class_code: str = Field(
        description="The class this mark was earned in, carried on the grade itself so a "
        "later transfer cannot reprint it under a room the child had not yet entered."
    )

    @classmethod
    def of(cls, line: GradeLine) -> "GradeLineOut":
        grade = line.grade
        return cls(
            subject_code=str(grade.subject_code),
            subject_name_ar=line.subject.name_ar if line.subject else "",
            subject_name_en=line.subject.name_en if line.subject else "",
            is_graded=line.is_graded,
            # `.value` only when a figure was stated. There is deliberately no fallback
            # branch here for the ungraded case to acquire a number in.
            percentage=grade.percentage.value if grade.percentage is not None else None,
            points=grade.points,
            max_points=grade.max_points,
            class_code=str(grade.class_code),
        )


class StudentTermGradesOut(BaseModel):
    """One child, one term, the marks as stated. No aggregate of any kind."""

    student_number: str
    full_name_ar: str
    full_name_en: str
    term_code: str
    term_name_ar: str
    term_name_en: str
    term_is_closed: bool
    class_code: str | None = Field(
        default=None,
        description="The class she was in *for this term*. `null` means no placement "
        "covered it — she had left, or had not yet joined — and is deliberately "
        "distinguishable from her current class.",
    )
    class_name_ar: str | None = None
    class_name_en: str | None = None
    subject_count: int = Field(description="Subjects with a row this term, graded or not.")
    graded_count: int = Field(
        description="Subjects carrying a stated figure. The remainder are awaiting a mark, "
        "not scoring zero. This is a count, not a score."
    )
    grades: list[GradeLineOut]

    @classmethod
    def of(cls, report: StudentTermGrades) -> "StudentTermGradesOut":
        section = report.class_section
        return cls(
            student_number=str(report.student.student_number),
            full_name_ar=report.student.full_name_ar,
            full_name_en=report.student.full_name_en,
            term_code=str(report.term.code),
            term_name_ar=report.term.name_ar,
            term_name_en=report.term.name_en,
            term_is_closed=report.term.is_closed,
            class_code=None if section is None else str(section.code),
            class_name_ar=None if section is None else section.name_ar,
            class_name_en=None if section is None else section.name_en,
            subject_count=len(report.lines),
            graded_count=report.graded_count,
            grades=[GradeLineOut.of(line) for line in report.lines],
        )


@router.get(
    "/guardians/by-id/{public_id}/students/{student_number}/grades",
    response_model=StudentTermGradesOut,
    summary="A child's marks, read by one of her guardians",
    description="The same figures as the route below, for a caller who holds a guardian "
    "handle rather than a registrar's authority. The guardian-to-child link is re-checked "
    "on this request: a caller that names a child who is not hers, or whose access has "
    "been restricted, gets the same 404 as one naming a child who does not exist.\n\n"
    "This is the route a parent-facing chat service should use. It never needs the "
    "parent's phone number, and it does not rely on that service having filtered "
    "correctly before asking.",
    responses=error_responses(401, 403, 404, 422),
)
def read_guardian_student_term_grades(
    public_id: str,
    student_number: str,
    queries: Queries,
    caller: Reader,
    request_id: RequestId,
    term: Annotated[
        str,
        Query(
            description="Term code. Required, for the reason the registrar route gives: "
            "a bare 'this term' would be answered from a clock.",
            examples=["2026-T1"],
        ),
    ],
) -> StudentTermGradesOut:
    with domain_errors():
        report = queries.guardian_student_term_grades(
            public_id,
            StudentNumber(student_number),
            TermCode(term),
            actor=caller.prefix,
            request_id=request_id,
        )
    return StudentTermGradesOut.of(report)


@router.get(
    "/students/{student_number}/grades",
    response_model=StudentTermGradesOut,
    summary="A child's stated marks for one term",
    description="Returns the figures exactly as recorded, in subject order. An unmarked "
    "subject is `percentage: null` with `is_graded: false` — never 0. Nothing is averaged, "
    "weighted or ranked. The class is the one she was in for this term, not her current one.",
    responses=error_responses(401, 403, 404, 422),
)
def read_student_term_grades(
    student_number: str,
    queries: Queries,
    caller: Reader,
    teaching: Teaching,
    term: Annotated[
        str,
        Query(
            description="Term code. Required: a bare 'this term' would be answered from a "
            "clock, and a report is a statement about a named period.",
            examples=["2026-T1"],
        ),
    ],
) -> StudentTermGradesOut:
    # A teacher holds `grades.read` on the rooms they stand in, so a report card is
    # answered only when the child was in one of them that year. Without this the class
    # scope would bound nothing on the one screen it most obviously should.
    caller.narrow(
        Permission.GRADES_READ,
        lambda scopes: scopes.for_student_in_term(
            term_code=term, student_number=student_number
        ),
    )
    with domain_errors():
        report = queries.student_term_grades(
            StudentNumber(student_number), TermCode(term)
        )
    # Class access does not imply access to every colleague's subject. A teacher sees
    # only the academic lines assigned to them in the child's class for this term.
    # Non-teaching roles return ``None`` and retain their ordinary scope-based view.
    if caller.profile is not None and report.class_section is not None:
        allowed = teaching.subject_codes_for_section(
            caller.profile.user_id,
            academic_year_code=str(report.class_section.academic_year_code),
            year_level_code=str(report.class_section.year_level_code),
            class_code=str(report.class_section.code),
        )
        if allowed is not None:
            report = replace(
                report,
                lines=tuple(
                    line
                    for line in report.lines
                    if str(line.grade.subject_code) in allowed
                ),
            )
    return StudentTermGradesOut.of(report)
