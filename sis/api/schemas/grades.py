"""Wire shapes for marks — including the report the `records/` adapter reads.

One rule dominates this module: **a blank grade serialises as JSON `null`, never `0`**
(invariant 1). Zero is a mark a child can earn, so a payload that renders "not marked yet"
as `0` tells a parent her daughter scored nothing in a subject nobody has marked. Three
things enforce it here and all three are load-bearing:

* `percentage` is typed `float | None` with **no zero default**, so a missing value is
  structurally impossible to spell as a number;
* `is_graded` is a *computed* field rather than one a caller passes, so it can never
  contradict the percentage beside it;
* the conversion from `Percentage | None` happens once, in `SubjectGradeOut.from_line`,
  rather than at each call site — because a call site is where somebody eventually writes
  `line.grade.percentage.value if ... else 0.0`.

The fourth guard is not code: a route serving these models must never set
`response_model_exclude_none=True`. Dropping the key is as bad as sending zero — clients
read an absent key as nothing-there and then default it.

Nothing here totals, averages or ranks (invariant 5). A caller wanting an average is
stating a policy about how a school judges children, and it must state it somewhere a
human can see, not receive one from this service as if it were a fact.
"""
from typing import Annotated, Self

from pydantic import Field, computed_field

from sis.api.schemas.common import CodeStr, RequestModel, ResponseModel
from sis.application.services.queries import GradeLine, StudentTermGrades

__all__ = [
    "GradeCommitRequest",
    "GradePreviewRequest",
    "StudentGradesResponse",
    "SubjectGradeOut",
]

#: A stated mark out of 100, both ends inclusive. Constrained inside the union rather than
#: on the field, so the bounds apply to the number and `None` stays untouched by them.
PercentageValue = Annotated[float, Field(ge=0, le=100)]


class GradePreviewRequest(RequestModel):
    """The non-file half of a marks upload. Parses and validates; writes no grades."""

    term_code: CodeStr = Field(
        description="Term the marks belong to. Required — a bare mark with no term is unfileable.",
        examples=["2026-T1"],
    )
    subject_code: CodeStr | None = Field(
        default=None,
        description="Target subject when the file covers one. Null means each row names its own.",
        examples=["MATH"],
    )
    class_code: CodeStr | None = Field(
        default=None,
        description=(
            "Narrows matching to one class, so a file uploaded against the wrong class is "
            "rejected loudly instead of quietly matching students across the whole school."
        ),
        examples=["3A"],
    )


class GradeCommitRequest(RequestModel):
    """Apply a previewed marks batch. See `RosterCommitRequest` for why it names a batch."""

    batch_id: str = Field(
        description="Batch id returned by the preview.",
        examples=["c7a1e93b6d284f5a8b2c0e1f3a4d5b62"],
    )


class SubjectGradeOut(ResponseModel):
    """One line of a report card: a subject, and the figure the school stated for it.

    `percentage` and `points`/`max_points` sit side by side and neither is derived from the
    other. A school reports "78%" or "17 out of 20"; rescaling one into the other would be
    this service inventing a figure, and a grade it returns must be one a teacher can
    recognise in the sheet they uploaded (invariant 5).

    `class_code` is the class the mark was *earned* in, not the child's current class. A
    Term 1 mark stays under 3A after a March move to 3B (invariant 2).
    """

    subject_code: CodeStr = Field(description="Subject this mark is for.", examples=["MATH"])
    subject_name_en: str | None = Field(
        default=None, description="English subject label, or null if the subject row could not be loaded."
    )
    subject_name_ar: str | None = Field(
        default=None, description="Arabic subject label, or null if the subject row could not be loaded."
    )
    percentage: PercentageValue | None = Field(
        default=None,
        description=(
            "The stated mark out of 100, or **null when no mark has been given yet**. Null is "
            "not zero: 0 is a mark a child can earn, and reporting an unmarked subject as 0 "
            "tells a parent something false."
        ),
        examples=[78.5, None],
    )
    points: float | None = Field(
        default=None, description="Raw points, when the school stated them ('17 out of 20'). Null if not."
    )
    max_points: float | None = Field(
        default=None, description="The maximum those points were out of. Null if not stated."
    )
    class_code: CodeStr = Field(
        description="The class this mark was earned in, resolved for the term it belongs to.",
        examples=["3A"],
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_graded(self) -> bool:
        """Whether the school has stated anything at all for this subject.

        Computed rather than accepted from a caller so it can never disagree with the
        figures beside it. Clients should branch on this instead of on `percentage`, which
        is how `percentage || 0` gets written.
        """
        return self.percentage is not None or self.points is not None

    @classmethod
    def from_line(cls, line: GradeLine) -> Self:
        """The one place `Percentage | None` becomes `float | None`. Keep it the only one."""
        grade = line.grade
        return cls(
            subject_code=str(grade.subject_code),
            subject_name_en=line.subject.name_en if line.subject else None,
            subject_name_ar=line.subject.name_ar if line.subject else None,
            # `.value` only when a Percentage exists. No `or 0.0`, no `float(... or 0)`:
            # the absent case stays absent all the way to the JSON.
            percentage=grade.percentage.value if grade.percentage is not None else None,
            points=grade.points,
            max_points=grade.max_points,
            class_code=str(grade.class_code),
        )


class StudentGradesResponse(ResponseModel):
    """A child's marks for one term — the payload the `records/` adapter consumes.

    `class_code` is the class she was in *for this term*, resolved from the term's dates,
    not read from her current placement. That is what keeps a Term 1 report printing 3A
    after a March move to 3B. Null means no placement covered the term at all — she had
    left, or had not yet joined — and is deliberately distinguishable from "her current
    class", which a client must not substitute.

    Subjects with no mark are present with `percentage: null`, not omitted. A report card
    that silently drops its ungraded lines is a report card nobody can tell is incomplete.
    """

    student_number: CodeStr = Field(description="The join key `records/` matches on.", examples=["10432"])
    full_name_ar: str = Field(description="Full name in Arabic.")
    full_name_en: str = Field(description="Full name in English.")
    term_code: CodeStr = Field(description="Term reported on.", examples=["2026-T1"])
    term_name_en: str = Field(description="English term label.")
    term_name_ar: str = Field(description="Arabic term label.")
    academic_year_code: CodeStr = Field(description="Year the term belongs to.")
    class_code: CodeStr | None = Field(
        default=None, description="Class she sat in for this term, or null if none covered it."
    )
    class_name_en: str | None = Field(default=None, description="English class label, when placed.")
    class_name_ar: str | None = Field(default=None, description="Arabic class label, when placed.")
    grades: list[SubjectGradeOut] = Field(
        default_factory=list,
        description="One line per subject on file, in report-card order. Ungraded subjects included.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def subject_count(self) -> int:
        """Subjects with a row this term, graded or not."""
        return len(self.grades)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def graded_count(self) -> int:
        """Subjects with a stated figure. The remainder are awaiting a mark, not scoring zero."""
        return sum(1 for grade in self.grades if grade.is_graded)

    @classmethod
    def from_dto(cls, report: StudentTermGrades) -> Self:
        """Flatten the read model for the wire. No aggregation happens here, by design."""
        section = report.class_section
        return cls(
            student_number=str(report.student.student_number),
            full_name_ar=report.student.full_name_ar,
            full_name_en=report.student.full_name_en,
            term_code=str(report.term.code),
            term_name_en=report.term.name_en,
            term_name_ar=report.term.name_ar,
            academic_year_code=str(report.term.academic_year_code),
            class_code=str(section.code) if section else None,
            class_name_en=section.name_en if section else None,
            class_name_ar=section.name_ar if section else None,
            grades=[SubjectGradeOut.from_line(line) for line in report.lines],
        )
