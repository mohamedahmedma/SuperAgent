"""Wire shapes for children and their class placements.

The shape that matters is that a placement is a *window*, not a column on the student
(invariant 2). `ClassEnrolmentOut` therefore carries `starts_on` and `ends_on` and a child
appears in the register of whichever class covers the day being asked about — which is the
only reason "which class was she in for Term 1" is still answerable in June, after a March
move from 3A to 3B. A response that flattened this to `student.class_code` would destroy
that at the boundary, and no amount of care underneath could bring it back.

Uploads are multipart: the file arrives as `UploadFile` and the options below as form
fields. They are modelled anyway so their descriptions reach OpenAPI and so a JSON-bodied
variant of the endpoint cannot drift from the multipart one.
"""
from datetime import date

from pydantic import Field, computed_field

from sis.api.schemas.common import (
    CodeStr,
    PageResponse,
    RequestModel,
    ResponseModel,
)

__all__ = [
    "ClassEnrolmentOut",
    "ClassRosterEntryOut",
    "ClassRosterPage",
    "ClassRosterResponse",
    "RosterCommitRequest",
    "RosterPreviewRequest",
    "StudentOut",
]


class RosterPreviewRequest(RequestModel):
    """The non-file half of a roster upload. Parses and validates; writes no students.

    `class_code` is optional because both upload styles are real: one file per class, where
    the target comes from this request, and one file for the whole school, where each row
    names its own class. A row that names none and has no target here is rejected rather
    than guessed at — guessing puts a child in a room she was never in.
    """

    academic_year_code: CodeStr = Field(
        description="Year the placements belong to.", examples=["2025-2026"]
    )
    class_code: CodeStr | None = Field(
        default=None,
        description="Target class when the file covers exactly one. Null means each row must name its own.",
        examples=["3A"],
    )
    default_starts_on: date | None = Field(
        default=None,
        description=(
            "Placement start for rows that omit one. Defaults to the academic year's first "
            "day, never to today — importing in November must not record every child as "
            "having joined in November."
        ),
    )


class RosterCommitRequest(RequestModel):
    """Apply a previewed roster batch.

    Names the batch rather than re-sending the file, so what is written is what the
    registrar actually reviewed. Re-uploading at commit time would let a different file be
    applied than the one on screen, which is the substitution the two-step flow exists to
    prevent. The actor is taken from the authenticated caller and is deliberately not a
    field here — a body-supplied actor is an audit trail anyone can forge.
    """

    batch_id: str = Field(
        description="Batch id returned by the preview.",
        examples=["b0f2c1d84e7a4b0d9a3f5c6e7d8a9b01"],
    )


class StudentOut(ResponseModel):
    """A child, identified by the number the school itself issued.

    `student_number` is the join key `records/` matches on and is immutable (invariant 6):
    re-issuing one detaches every grade and every enrolment from the child they belong to,
    and not one of those rows reports an error when it happens. It stays text, because
    leading zeros are significant and `int("0071")` is a different child, or no child.
    """

    student_number: CodeStr = Field(description="The school's own identifier.", examples=["10432"])
    full_name_ar: str = Field(description="Full name in Arabic.")
    full_name_en: str = Field(description="Full name in English.")
    is_active: bool = Field(default=True, description="False once the child has left the school.")


class ClassEnrolmentOut(ResponseModel):
    """One time-bounded membership of one class.

    The class is identified by year *and* code, never code alone: `3A` names a different
    room of children every September, so a client resolving a placement from the code by
    itself will eventually resolve it into the wrong year.
    """

    student_number: CodeStr = Field(description="The child placed.")
    academic_year_code: CodeStr = Field(description="Year the placement sits in.")
    class_code: CodeStr = Field(description="Class the child was placed in.", examples=["3A"])
    starts_on: date = Field(description="First day of the placement.")
    ends_on: date | None = Field(
        default=None,
        description=(
            "Last day, inclusive, or null when no end has been decided. Null is 'undecided', "
            "not 'forever' — a sentinel like 9999-12-31 sorts like a real answer and is "
            "eventually printed to somebody as one."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_open(self) -> bool:
        """Whether the placement is still running. Derived, so it cannot contradict `ends_on`."""
        return self.ends_on is None


class ClassRosterEntryOut(ResponseModel):
    """One line of a register: the placement, and the child it belongs to when she loads.

    `student` is nullable so a placement whose student row cannot be read still appears,
    carrying its number. Dropping it would shorten the register silently, and a register
    that is quietly one child short is worse than one showing a number without a name.
    """

    student_number: CodeStr = Field(description="Present even when `student` is null.")
    enrolment: ClassEnrolmentOut = Field(description="The placement this line is listed under.")
    student: StudentOut | None = Field(
        default=None, description="The child's record, or null if it could not be loaded."
    )


class ClassRosterResponse(ResponseModel):
    """Who was in a class on a given day.

    `on_date` is echoed because a register is a statement about a day. A client printing
    last term's sheet and a client printing today's get different answers from the same
    URL, and the date on the payload is what tells them apart on paper.
    """

    academic_year_code: CodeStr = Field(description="Year asked about.")
    class_code: CodeStr = Field(description="Class asked about.", examples=["3A"])
    on_date: date = Field(description="The day this register is a statement about.")
    entries: list[ClassRosterEntryOut] = Field(
        default_factory=list, description="Children placed in the class on that day."
    )


class ClassRosterPage(PageResponse[ClassRosterEntryOut]):
    """A paged register, for year groups too large to send whole. Same echo of the day asked about."""

    academic_year_code: CodeStr = Field(description="Year asked about.")
    class_code: CodeStr = Field(description="Class asked about.")
    on_date: date = Field(description="The day this register is a statement about.")
