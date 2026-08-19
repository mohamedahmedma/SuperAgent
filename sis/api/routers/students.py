"""The class register: who was in a room on a given day.

`on` is a query parameter and the answer echoes it back as `as_of`. A register is a
statement about a day rather than about a class — invariant 2 means membership is
time-bounded, so a child who moved 3A->3B in March is genuinely on both registers,
each for its own dates. Defaulting to today is a convenience the API layer is allowed
(it may read a clock; the services below it may not), but answering "today" to a
caller who meant last term without saying so is how a printed Term 1 attendance sheet
quietly acquires this term's children. Hence the echo: every response states the day it
answered for.

An unknown class is a 404, never an empty register. "No such class" and "a class with
nobody in it" render identically on screen, and the first is a typo the caller fixes in
seconds while the second sends a registrar looking for children who were never missing.
That refusal is `QueryService`'s, not this router's — it is a rule, and rules do not live
here.
"""
from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from sis.api.deps import Caller, get_query_service, require_read_access
from sis.api.routers import domain_errors, error_responses
from sis.application.services import QueryService
from sis.application.services.queries import ClassRosterEntry
from sis.domain.value_objects import AcademicYearCode, ClassCode

router = APIRouter(prefix="/v1", tags=["students"])

# Both scopes, spelled out: scope comparison is exact equality, so a reader-only check
# would refuse the registrar reading her own register.
Reader = Annotated[Caller, Depends(require_read_access)]
Queries = Annotated[QueryService, Depends(get_query_service)]


class RosterEntryOut(BaseModel):
    """One child on the register, with the placement window she is listed under."""

    student_number: str
    full_name_ar: str = Field(
        description="Empty when the student row could not be loaded. The placement is "
        "still listed: a register that is quietly one child short is worse than one "
        "showing a number without a name."
    )
    full_name_en: str
    starts_on: date
    ends_on: date | None = Field(
        default=None,
        description="Her **last day** in the class, not the day after. `null` means the "
        "placement is open — she is in this class now and no end has been decided.",
    )
    is_open: bool

    @classmethod
    def of(cls, entry: ClassRosterEntry) -> "RosterEntryOut":
        return cls(
            student_number=entry.student_number,
            full_name_ar=entry.display_name_ar,
            full_name_en=entry.display_name_en,
            starts_on=entry.enrolment.starts_on,
            ends_on=entry.enrolment.ends_on,
            is_open=entry.enrolment.is_open,
        )


class ClassRosterOut(BaseModel):
    academic_year_code: str
    class_code: str
    as_of: date = Field(
        description="The day this register answers for. Echoed because `on` defaults to "
        "today, and a caller who meant last term must be able to see that it did."
    )
    count: int
    students: list[RosterEntryOut]


@router.get(
    "/classes/{class_code}/students",
    response_model=ClassRosterOut,
    summary="The register of one class on one day",
    description="`academic_year` is required: a class code is unique within a year, so "
    "`3A` alone names a different room of children every September. Unknown class or "
    "year is a 404, never an empty list.",
    responses=error_responses(401, 403, 404, 422),
)
def read_class_roster(
    class_code: str,
    queries: Queries,
    caller: Reader,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    on: Annotated[
        date | None,
        Query(description="The day to answer for. Defaults to today, echoed as `as_of`."),
    ] = None,
) -> ClassRosterOut:
    on_date = on or datetime.now(UTC).date()
    with domain_errors():
        entries = queries.class_roster(
            AcademicYearCode(academic_year), ClassCode(class_code), on_date
        )
    return ClassRosterOut(
        academic_year_code=academic_year,
        class_code=class_code,
        as_of=on_date,
        count=len(entries),
        students=[RosterEntryOut.of(entry) for entry in entries],
    )
