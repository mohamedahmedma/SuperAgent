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
from uuid import uuid4
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from sis.api.deps import (
    Caller,
    StudentDesk,
    get_query_service,
    get_student_desk,
    require_read_access,
    require_registrar,
    Principal,
    require_permission,
    UowFactoryDep,
)
from sis.domain.rbac import Permission
from sis.api.routers import domain_errors, error_responses
from sis.application.services import QueryService
from sis.application.services.queries import ClassRosterEntry
from sis.domain.errors import UnknownReference
from sis.domain.guardians import Guardian, RelationshipType, StudentGuardian
from sis.domain.people import ClassEnrolment, Gender, Student
from sis.domain.value_objects import AcademicYearCode, ClassCode, StudentNumber
from sis.infrastructure.db import models as m

router = APIRouter(prefix="/v1", tags=["students"])

# Both scopes, spelled out: scope comparison is exact equality, so a reader-only check
# would refuse the registrar reading her own register.
Reader = Annotated[Principal, Depends(require_permission(Permission.STUDENTS_READ))]
Registrar = Annotated[Principal, Depends(require_permission(Permission.STUDENTS_WRITE))]
AdmissionsManager = Annotated[
    Principal, Depends(require_permission(Permission.STUDENTS_CREATE))
]
Queries = Annotated[QueryService, Depends(get_query_service)]
Desk = Annotated[StudentDesk, Depends(get_student_desk)]


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
    caller.narrow(
        Permission.STUDENTS_READ,
        lambda scopes: scopes.for_class(
            academic_year_code=academic_year, class_code=class_code
        ),
    )
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


# ---------------------------------------------------------------------------
# The student record itself, and placement as an act rather than an import
#
# Until these existed, the only way to add or correct one child was to build a one-row
# spreadsheet, preview it and commit a batch. That is the right ceremony for September's
# nine hundred rows and absurd for a misspelt name, and the absurdity had a cost: a
# registrar who will not do it leaves the name wrong, and the wrong name is what a parent
# reads on a report card.
#
# The import is not replaced. It still owns anything touching more than one child, because
# that is where a per-row report earns its keep. These routes own the single child.
#
# Placement is still a dated membership (invariant 2). There is no route here that moves a
# child by rewriting a class code, because that is the history rewrite the whole schema is
# shaped to prevent -- a transfer is one placement ended and another opened, in one
# transaction, and `POST /transfer` is that pair.
# ---------------------------------------------------------------------------


class StudentIn(BaseModel):
    """A child's own record: her number and how her name is written."""

    student_number: str = Field(
        examples=["10432"],
        description="The school's own identifier, and the key every mark and guardian "
        "link points at. Immutable — correcting a mistyped number is a new student plus a "
        "roster fix, not an edit, because the old number may already carry marks.",
    )
    full_name_ar: str = Field(default="", description="Full name in Arabic.")
    full_name_en: str = Field(default="", description="Full name in English.")
    is_active: bool = Field(
        default=True,
        description="False once the child has left the school. She keeps her marks, her "
        "placements and her guardians; she stops appearing in the pickers.",
    )
    date_of_birth: date | None = Field(
        default=None,
        description="Stated, never derived — and there is no age field anywhere in this "
        "API, because an age is right for one year and silently wrong afterwards. A date in "
        "the future is refused.",
    )
    contact_phone: str = Field(
        default="",
        description="The child's own number, which is **not** her guardian's. Guardian "
        "numbers live under `/students/{student_number}/guardians`, where the permission to "
        "be told about her marks lives with them.",
    )
    contact_email: str = Field(default="")
    address: str = Field(default="")


class StudentPatch(BaseModel):
    """A correction. Every field optional; omitted means leave it alone.

    `student_number` is absent by design. It is in the path, and it is identity.
    """

    full_name_ar: str | None = None
    full_name_en: str | None = None
    is_active: bool | None = None
    date_of_birth: date | None = None
    contact_phone: str | None = None
    contact_email: str | None = None
    address: str | None = None


class StudentOut(BaseModel):
    student_number: str
    full_name_ar: str
    full_name_en: str
    is_active: bool
    gender: Gender = Gender.UNSPECIFIED
    date_of_birth: date | None = None
    age: int | None = Field(
        default=None,
        description="Her age in whole years **as of the day this response was built**, "
        "computed from `date_of_birth` and never stored. `null` when no birth date is on "
        "file — which is not the same as nought, exactly as a blank mark is not a zero.",
    )
    contact_phone: str = ""
    contact_email: str = ""
    address: str = ""

    @classmethod
    def of(cls, student: Student) -> "StudentOut":
        # `age_on` takes the day rather than reading a clock, so the API layer supplies it:
        # this is the one layer allowed to know what today is.
        return cls(
            student_number=str(student.student_number),
            full_name_ar=student.full_name_ar,
            full_name_en=student.full_name_en,
            is_active=student.is_active,
            gender=student.gender,
            date_of_birth=student.date_of_birth,
            age=student.age_on(datetime.now(UTC).date()),
            contact_phone=student.contact_phone,
            contact_email=student.contact_email,
            address=student.address,
        )


class StudentSearchOut(BaseModel):
    query: str = Field(description="Echoed, so a stale response is recognisable as stale.")
    count: int
    students: list[StudentOut]


class PlacementIn(BaseModel):
    """Put a child in a class from a date."""

    academic_year_code: str = Field(examples=["2025-2026"])
    class_code: str = Field(examples=["3A"])
    starts_on: date = Field(description="Her first day in this class.")
    ends_on: date | None = Field(
        default=None,
        description="Her **last day** in the class, not the day after. `null` opens the "
        "placement — she is in this class now and no end has been decided.",
    )


class TransferIn(BaseModel):
    """Move a child to another class in the same year, from a date."""

    academic_year_code: str = Field(examples=["2025-2026"])
    to_class_code: str = Field(examples=["3B"])
    on_date: date = Field(
        description="Her first day in the new class. The old placement is closed the day "
        "before, so no single day belongs to two classes."
    )


class PlacementEndIn(BaseModel):
    ends_on: date = Field(
        description="Her last day in the class she is currently in — not the day after."
    )


class PlacementOut(BaseModel):
    student_number: str
    academic_year_code: str
    class_code: str
    starts_on: date
    ends_on: date | None
    is_open: bool

    @classmethod
    def of(cls, enrolment: ClassEnrolment) -> "PlacementOut":
        return cls(
            student_number=str(enrolment.student_number),
            academic_year_code=str(enrolment.academic_year_code),
            class_code=str(enrolment.class_code),
            starts_on=enrolment.starts_on,
            ends_on=enrolment.ends_on,
            is_open=enrolment.is_open,
        )


class TransferOut(BaseModel):
    closed: PlacementOut | None = Field(
        default=None,
        description="The placement that ended, or `null` when the child had none open — "
        "which is a first placement rather than a transfer, and is not an error.",
    )
    opened: PlacementOut


class PlacementHistoryOut(BaseModel):
    student_number: str
    count: int
    placements: list[PlacementOut]


class StudentAdmissionIn(BaseModel):
    """Every fact required to admit one child; no partial records are accepted."""

    full_name_ar: str = Field(min_length=1)
    full_name_en: str = Field(min_length=1)
    gender: Gender
    date_of_birth: date
    contact_phone: str = Field(min_length=1)
    contact_email: str = Field(min_length=1)
    address: str = Field(min_length=1)
    guardian_full_name_ar: str = Field(min_length=1)
    guardian_full_name_en: str = Field(min_length=1)
    guardian_phone: str = Field(min_length=1)
    relationship_type: RelationshipType
    relationship_label: str = Field(min_length=1)
    academic_year_code: str = Field(min_length=1)
    class_code: str = Field(min_length=1)
    starts_on: date

    @field_validator(
        "full_name_ar", "full_name_en", "contact_phone",
        "contact_email", "address", "guardian_full_name_ar",
        "guardian_full_name_en", "guardian_phone", "relationship_label",
        "academic_year_code", "class_code",
    )
    @classmethod
    def no_blank_fields(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("this field is required")
        return cleaned


class StudentAdmissionOut(BaseModel):
    student: StudentOut
    placement: PlacementOut
    guardian_phone: str


@router.get(
    "/students",
    response_model=StudentSearchOut,
    summary="Find a child by number or name",
    description="Type-ahead over the student number and both spellings of the name. A "
    "blank `q` returns nothing rather than the whole school. Children who have left are "
    "excluded unless `include_inactive` asks for them.",
    responses=error_responses(401, 403, 422),
)
def search_students(
    queries: Queries,
    caller: Reader,
    uow_factory: UowFactoryDep,
    q: Annotated[str, Query(description="Number or part of a name.", examples=["ahmed"])] = "",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    include_inactive: bool = False,
    academic_year: Annotated[str | None, Query()] = None,
    year_level: Annotated[str | None, Query()] = None,
) -> StudentSearchOut:
    if academic_year and year_level:
        caller.narrow(
            Permission.STUDENTS_READ,
            lambda scopes: scopes.for_year_level(
                school_id=caller.school_id, year_level_code=year_level
            ),
        )
    else:
        caller.narrow(
            Permission.STUDENTS_READ,
            lambda scopes: scopes.for_year(academic_year),
        )
    with domain_errors():
        found = queries.search_students(q, limit=limit, include_inactive=include_inactive)
    if academic_year and year_level and found:
        numbers = [str(student.student_number) for student in found]
        with uow_factory() as uow:
            allowed = set(uow._session.scalars(
                select(m.Student.student_number)
                .join(m.ClassEnrolment)
                .join(m.ClassSection)
                .join(m.AcademicYear)
                .join(m.YearLevel, m.ClassSection.year_level_id == m.YearLevel.id)
                .where(
                    m.Student.student_number.in_(numbers),
                    m.AcademicYear.code == academic_year,
                    m.YearLevel.code == year_level,
                )
            ).all())
        found = [student for student in found if str(student.student_number) in allowed]
    return StudentSearchOut(
        query=q, count=len(found), students=[StudentOut.of(s) for s in found]
    )


@router.post(
    "/students/admissions",
    response_model=StudentAdmissionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Admit one child with a complete family record",
    responses=error_responses(401, 403, 404, 409, 422),
)
def admit_student(
    body: StudentAdmissionIn, desk: Desk, caller: AdmissionsManager
) -> StudentAdmissionOut:
    caller.narrow(
        Permission.STUDENTS_CREATE,
        lambda scopes: scopes.for_class(
            academic_year_code=body.academic_year_code, class_code=body.class_code
        ),
    )
    with domain_errors():
        # A student number is an internal, immutable reference.  It is minted here so a
        # manager never has to guess the next number or coordinate with another desk.
        student_number = f"S-{uuid4().hex[:12].upper()}"
        student = Student(
            student_number=student_number,
            full_name_ar=body.full_name_ar,
            full_name_en=body.full_name_en,
            gender=body.gender,
            date_of_birth=body.date_of_birth,
            contact_phone=body.contact_phone,
            contact_email=body.contact_email,
            address=body.address,
        )
        guardian = Guardian(
            phones=(body.guardian_phone,),
            full_name_ar=body.guardian_full_name_ar,
            full_name_en=body.guardian_full_name_en,
        )
        link = StudentGuardian(
            student_number=student.student_number,
            guardian_phone=guardian.primary_phone,
            relationship_type=body.relationship_type,
            relationship_label=body.relationship_label,
            is_primary_contact=True,
            can_view_records=True,
        )
        placement = ClassEnrolment(
            student_number=student.student_number,
            academic_year_code=body.academic_year_code,
            class_code=body.class_code,
            starts_on=body.starts_on,
        )
        desk.create_family(student, guardian, link, placement)
    return StudentAdmissionOut(
        student=StudentOut.of(student),
        placement=PlacementOut.of(placement),
        guardian_phone=str(guardian.primary_phone),
    )


@router.get(
    "/students/{student_number}",
    response_model=StudentOut,
    summary="One child's record",
    description="Her number and her name, and deliberately not her class: placement is a "
    "dated membership with a different answer per term, and there is no current-class "
    "field to read. Ask `/students/{student_number}/placements` for the history.",
    responses=error_responses(401, 403, 404, 422),
)
def read_student(
    student_number: str, queries: Queries, caller: Reader,
    academic_year: Annotated[str | None, Query()] = None,
) -> StudentOut:
    caller.narrow(
        Permission.STUDENTS_READ,
        lambda scopes: scopes.for_student(
            academic_year_code=academic_year or "", student_number=student_number
        ),
    )
    with domain_errors():
        student = queries.get_student(StudentNumber(student_number))
    return StudentOut.of(student)


@router.post(
    "/students",
    response_model=StudentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add or correct one child",
    description="201 when the child is new, 200 when this number was already on file — in "
    "which case her names are corrected and not one mark, placement or guardian link is "
    "touched. Enrolling her in a class is a separate act: "
    "`POST /v1/students/{student_number}/placements`.",
    responses=error_responses(401, 403, 409, 422),
)
def save_student(
    body: StudentIn, desk: Desk, caller: Registrar, response: Response
) -> StudentOut:
    with domain_errors():
        student = Student(
            student_number=body.student_number,
            full_name_ar=body.full_name_ar,
            full_name_en=body.full_name_en,
            is_active=body.is_active,
            date_of_birth=body.date_of_birth,
            contact_phone=body.contact_phone,
            contact_email=body.contact_email,
            address=body.address,
        )
        created = desk.save_student(student)
    if not created:
        response.status_code = status.HTTP_200_OK
    return StudentOut.of(student)


@router.patch(
    "/students/{student_number}",
    response_model=StudentOut,
    summary="Correct one child's name, or mark her as having left",
    description="Omitted fields are left alone. Setting `is_active: false` is how a child "
    "who has left the school is recorded — there is no delete, because her marks and her "
    "placements are still true statements about terms that happened.",
    responses=error_responses(401, 403, 404, 422),
)
def update_student(
    student_number: str,
    body: StudentPatch,
    queries: Queries,
    desk: Desk,
    caller: Registrar,
) -> StudentOut:
    with domain_errors():
        number = StudentNumber(student_number)
        # Read first, so a PATCH of one field does not blank the others. The 404 for an
        # unknown child comes from here rather than from the write, which would otherwise
        # cheerfully create her.
        current = queries.get_student(number)
        def kept(new: object, old: object) -> object:
            """Omitted means "leave it alone", never "blank it"."""
            return old if new is None else new

        updated = Student(
            student_number=number,
            full_name_ar=kept(body.full_name_ar, current.full_name_ar),
            full_name_en=kept(body.full_name_en, current.full_name_en),
            is_active=kept(body.is_active, current.is_active),
            # A caller who wants to *clear* a birth date cannot express it here, and that is
            # the right trade: omitting the field is overwhelmingly the common case, and
            # silently erasing a date because a form did not include it is worse than
            # needing a separate act to remove one.
            date_of_birth=kept(body.date_of_birth, current.date_of_birth),
            contact_phone=kept(body.contact_phone, current.contact_phone),
            contact_email=kept(body.contact_email, current.contact_email),
            address=kept(body.address, current.address),
        )
        desk.save_student(updated)
    return StudentOut.of(updated)


@router.get(
    "/students/{student_number}/placements",
    response_model=PlacementHistoryOut,
    summary="Every class this child has been in",
    description="The whole history, not just the open placement. A child who moved 3A to "
    "3B in March has two rows here and both are true — which is what makes her Term 1 "
    "report card say 3A and her Term 2 report card say 3B.",
    responses=error_responses(401, 403, 404, 422),
)
def read_student_placements(
    student_number: str, queries: Queries, caller: Reader
) -> PlacementHistoryOut:
    with domain_errors():
        placements = queries.student_placements(StudentNumber(student_number))
    caller.narrow_all(
        Permission.STUDENTS_READ,
        lambda scopes: (
            scopes.for_class(
                academic_year_code=str(row.academic_year_code), class_code=str(row.class_code)
            ) for row in placements
        ),
    )
    return PlacementHistoryOut(
        student_number=student_number,
        count=len(placements),
        placements=[PlacementOut.of(p) for p in placements],
    )


@router.post(
    "/students/{student_number}/placements",
    response_model=PlacementOut,
    status_code=status.HTTP_201_CREATED,
    summary="Place one child in a class",
    description="Opens a placement from `starts_on`. A child may hold only one *open* "
    "placement at a time and the database enforces it, so placing a child who is already "
    "in a class is refused rather than silently leaving her in two — use "
    "`POST /transfer`, which closes the old one in the same transaction.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def place_student(
    student_number: str, body: PlacementIn, desk: Desk, caller: Registrar
) -> PlacementOut:
    with domain_errors():
        enrolment = ClassEnrolment(
            student_number=student_number,
            academic_year_code=body.academic_year_code,
            class_code=body.class_code,
            starts_on=body.starts_on,
            ends_on=body.ends_on,
        )
        desk.place_student(enrolment)
    return PlacementOut.of(enrolment)


@router.post(
    "/students/{student_number}/transfer",
    response_model=TransferOut,
    summary="Move one child to another class",
    description="One transaction: the open placement is closed the day before `on_date` "
    "and a new one opens on it. Two separate calls would leave a window in which the child "
    "is in no class at all, and a marks upload landing in that window rejects every one of "
    "her rows for having no placement. Her marks in the old class stay filed under the old "
    "class — that is the point of the invariant.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def transfer_student(
    student_number: str, body: TransferIn, desk: Desk, caller: Registrar
) -> TransferOut:
    with domain_errors():
        closed, opened = desk.transfer_student(
            StudentNumber(student_number),
            academic_year_code=AcademicYearCode(body.academic_year_code),
            to_class=ClassCode(body.to_class_code),
            on_date=body.on_date,
        )
    return TransferOut(
        closed=None if closed is None else PlacementOut.of(closed),
        opened=PlacementOut.of(opened),
    )


@router.patch(
    "/students/{student_number}/placements/current",
    response_model=PlacementOut,
    summary="End this child's current placement",
    description="`ends_on` is her **last day** in the class, not the day after. She then "
    "holds no open placement, which is the correct state for a child who has left mid-year "
    "and is not the same as her never having been there. 404 when she has no open "
    "placement to end.",
    responses=error_responses(401, 403, 404, 422),
)
def end_current_placement(
    student_number: str,
    body: Annotated[PlacementEndIn, Body()],
    desk: Desk,
    caller: Registrar,
) -> PlacementOut:
    with domain_errors():
        closed = desk.end_placement(StudentNumber(student_number), ends_on=body.ends_on)
    if closed is None:
        # A 404 rather than a no-op success. "She had no class to leave" is either a typo
        # in the number or a registrar looking at a stale screen, and both want telling.
        raise UnknownReference(
            f"student {student_number} has no open placement to end",
            field="student_number",
        )
    return PlacementOut.of(closed)
