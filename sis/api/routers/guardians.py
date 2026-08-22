"""Who a child's guardians are, and which of them may read her records.

Two directions, because two callers ask opposite questions. A registrar asks about a
child — "who do we call about Layla" — and a parent-facing service asks about a number:
"which children may this phone see". Both are one query here; neither is derivable from
the other without loading a whole family.

An unknown child or an unregistered number is a 404, never an empty list. "No such
student" and "a student with no guardians on file" render identically on screen, and only
one of them is a typo the caller fixes in seconds — the other is the normal state of every
child between the roster upload and the guardians upload. That refusal is `QueryService`'s
and not this router's, because it is a rule.

The `PATCH` route exists so a custody change does not require a spreadsheet. When a court
order arrives, the office needs to revoke one parent's access in the next minute, and an
answer of "re-upload the guardians file" is not one.
"""
from typing import Annotated

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, Field

from sis.api.deps import (
    Caller,
    get_query_service,
    get_unit_of_work_factory,
    require_read_access,
    require_registrar,
)
from sis.api.routers import domain_errors, error_responses
from sis.application.ports.unit_of_work import UnitOfWork
from sis.application.services import QueryService
from sis.application.services.queries import GuardianIdentity, GuardianLink
from sis.domain.errors import UnknownReference
from sis.domain.people import Gender
from sis.domain.guardians import RelationshipType
from sis.domain.value_objects import Phone, StudentNumber

router = APIRouter(prefix="/v1", tags=["guardians"])

# Reads are open to both scopes; writes are the registrar's alone. Scope comparison is
# exact equality, so a reader-only check would refuse the registrar reading her own list.
Reader = Annotated[Caller, Depends(require_read_access)]
Registrar = Annotated[Caller, Depends(require_registrar)]
Queries = Annotated[QueryService, Depends(get_query_service)]
UnitOfWorkFactory = Annotated[object, Depends(get_unit_of_work_factory)]


class GuardianOut(BaseModel):
    """One adult on a child's contact list, and what she is to that child."""

    phone: str = Field(
        description="Her identifying number, in E.164. Normalised on the way in, so the "
        "same parent typed three ways in three uploads appears once."
    )
    phones: list[str] = Field(
        default_factory=list,
        description="Every number that reaches her, this one first.",
    )
    full_name_ar: str = Field(
        description="Empty when the guardian row could not be loaded. The link is still "
        "listed: a contact list that is quietly one adult short is worse than one showing "
        "a number without a name — the number still reaches somebody."
    )
    full_name_en: str
    relationship_type: RelationshipType = Field(
        description="The closed vocabulary, for counting and filtering."
    )
    relationship_label: str = Field(
        default="",
        description="What the registrar actually typed — 'big brother' — kept verbatim "
        "beside the bucketed type.",
    )
    is_primary_contact: bool
    can_view_records: bool = Field(
        description="Whether this adult may read the child's academic records. Separate "
        "from the link existing: an emergency contact barred by a court order stays on "
        "file and reads nothing."
    )
    restriction_note: str = Field(
        default="",
        description="Why access was restricted. For the registrar and an audit; never "
        "rendered to a parent.",
    )

    @classmethod
    def of(cls, entry: GuardianLink) -> "GuardianOut":
        return cls(
            phone=entry.phone,
            phones=list(entry.phones),
            full_name_ar=entry.display_name_ar,
            full_name_en=entry.display_name_en,
            relationship_type=entry.link.relationship_type,
            relationship_label=entry.link.relationship_label,
            is_primary_contact=entry.link.is_primary_contact,
            can_view_records=entry.link.can_view_records,
            restriction_note=entry.link.restriction_note,
        )


class StudentGuardiansOut(BaseModel):
    student_number: str
    count: int
    guardians: list[GuardianOut]


class GuardianChildOut(BaseModel):
    """One child a guardian is responsible for."""

    student_number: str
    full_name_ar: str
    full_name_en: str
    #: The child's own sex, so a parent writing "my son" can be understood without being
    #: asked which child. `unspecified` is the honest answer for every child until a
    #: registrar uploads it, and a reader must treat it as "not said" rather than as a
    #: default — see `sis.domain.people.Gender`.
    gender: Gender = Gender.UNSPECIFIED
    relationship_type: RelationshipType
    relationship_label: str = ""
    can_view_records: bool

    @classmethod
    def of(cls, entry: GuardianLink) -> "GuardianChildOut":
        return cls(
            student_number=entry.student_number,
            full_name_ar=entry.student.full_name_ar if entry.student else "",
            full_name_en=entry.student.full_name_en if entry.student else "",
            gender=entry.student.gender if entry.student else Gender.UNSPECIFIED,
            relationship_type=entry.link.relationship_type,
            relationship_label=entry.link.relationship_label,
            can_view_records=entry.link.can_view_records,
        )


class GuardianChildrenOut(BaseModel):
    phone: str
    full_name_ar: str
    full_name_en: str
    count: int
    students: list[GuardianChildOut]


class AccessIn(BaseModel):
    """A custody decision, applied to one link."""

    can_view_records: bool
    restriction_note: str = Field(
        default="",
        description="Why. Recorded whichever way the flag went, so a later reader can "
        "tell a deliberate restriction from a default that was never revisited.",
    )


class LinkOut(BaseModel):
    student_number: str
    phone: str
    can_view_records: bool


class ResolveIn(BaseModel):
    """A number to look up. In a body, never a path — see the route's own docstring."""

    phone: str = Field(
        description="The number in international form, e.g. +201001234567, or in the "
        "school's national form. Normalised here, so a caller holding whatever a parent "
        "typed does not have to know the rules.",
        examples=["+201001234567", "01001234567"],
    )
    default_country_code: str = Field(
        default="+20",
        description="Applied only to a number given in national form. A number carrying "
        "its own + prefix ignores this entirely.",
    )


class GuardianRefOut(BaseModel):
    """A guardian named by her stable handle. Deliberately carries no phone number back."""

    public_id: str = Field(
        description="Opaque and permanent. This is what another service stores to refer "
        "to her later, so that it never has to keep her phone number."
    )
    full_name_ar: str = ""
    full_name_en: str = ""
    preferred_language: str = "ar"

    @classmethod
    def of(cls, found: GuardianIdentity) -> "GuardianRefOut":
        return cls(
            public_id=found.public_id,
            full_name_ar=found.full_name_ar,
            full_name_en=found.full_name_en,
            preferred_language=found.preferred_language,
        )


@router.post(
    "/guardians/resolve",
    response_model=GuardianRefOut,
    summary="Which guardian does this number reach",
    description="For a service that has just proved somebody controls a number and needs "
    "to know whether the school has that number on file. Returns her stable `public_id`, "
    "which is what the caller should store — never the number itself. "
    "A number that reaches nobody is a 404. That is an ordinary answer here, not an "
    "error: most numbers in the world are not this school's parents.",
    responses=error_responses(401, 403, 404, 422),
)
def resolve_guardian(
    body: Annotated[ResolveIn, Body()], queries: Queries, caller: Reader
) -> GuardianRefOut:
    """POST, and the method is the point.

    The number goes in a body rather than a path so it stays out of access logs, browser
    history and referrer headers — the same reasoning that put `public_id` on the guardians
    table in the first place. A GET with the number in the URL would undo that on the very
    request whose purpose is to stop holding numbers.
    """
    with domain_errors():
        phone = Phone.parse(
            body.phone, default_country_code=body.default_country_code
        )
        found = queries.resolve_guardian(phone)
    if found is None:
        # The same shape as every other "nothing on file" in this service. It says nothing
        # about *why* — a caller learns that this number is not a parent here, and not
        # whether some other number would have been.
        raise UnknownReference("no guardian is on file for that number", field="phone")
    return GuardianRefOut.of(found)


@router.get(
    "/students/{student_number}/guardians",
    response_model=StudentGuardiansOut,
    summary="Every guardian on file for one child",
    description="Primary contact first. An unknown student is a 404, never an empty "
    "list — a child with no guardians recorded yet is a real and common answer, and the "
    "two must stay distinguishable.",
    responses=error_responses(401, 403, 404, 422),
)
def read_student_guardians(
    student_number: str, queries: Queries, caller: Reader
) -> StudentGuardiansOut:
    with domain_errors():
        entries = queries.student_guardians(StudentNumber(student_number))
    return StudentGuardiansOut(
        student_number=student_number,
        count=len(entries),
        guardians=[GuardianOut.of(entry) for entry in entries],
    )


@router.get(
    "/guardians/{phone}/students",
    response_model=GuardianChildrenOut,
    summary="Which children one number may ask about",
    description="The parent-facing question. Resolves through **any** of her numbers, so "
    "a parent who verifies the second line she gave the school sees the same children as "
    "one who verifies the first. Restricted links are excluded unless `include_restricted` "
    "is set, which is a registrar's view and not a parent's.",
    responses=error_responses(401, 403, 404, 422),
)
def read_guardian_students(
    phone: str,
    queries: Queries,
    caller: Reader,
    include_restricted: bool = False,
) -> GuardianChildrenOut:
    with domain_errors():
        # Already-E.164 in the path: this is a machine-to-machine identifier taken from a
        # previous response, not something a human types, so it is validated rather than
        # normalised. A national-format number here would silently resolve to nobody.
        parsed = Phone(phone)
        entries = queries.guardian_students(parsed, viewable_only=not include_restricted)
    first = entries[0].guardian if entries else None
    return GuardianChildrenOut(
        phone=str(parsed),
        full_name_ar=first.full_name_ar if first else "",
        full_name_en=first.full_name_en if first else "",
        count=len(entries),
        students=[GuardianChildOut.of(entry) for entry in entries],
    )


@router.get(
    "/guardians/by-id/{public_id}/students",
    response_model=GuardianChildrenOut,
    summary="Which children this guardian handle may ask about",
    description="The same question as the by-phone route, asked with the opaque handle a "
    "token carries. This is the one a parent-facing service should use: it never has to "
    "hold, log or transmit a parent's phone number to find out which children are hers.",
    responses=error_responses(401, 403, 404, 422),
)
def read_guardian_students_by_id(
    public_id: str,
    queries: Queries,
    caller: Reader,
    include_restricted: bool = False,
) -> GuardianChildrenOut:
    with domain_errors():
        entries = queries.guardian_students_by_id(
            public_id, viewable_only=not include_restricted
        )
    first = entries[0].guardian if entries else None
    return GuardianChildrenOut(
        # The handle is echoed rather than the number. A caller that only ever knew the
        # handle must not learn a phone number by asking this question.
        phone="",
        full_name_ar=first.full_name_ar if first else "",
        full_name_en=first.full_name_en if first else "",
        count=len(entries),
        students=[GuardianChildOut.of(entry) for entry in entries],
    )


@router.patch(
    "/students/{student_number}/guardians/{phone}",
    response_model=LinkOut,
    summary="Grant or revoke one guardian's access to one child's records",
    description="The urgent custody path, deliberately not requiring an upload: when a "
    "court order arrives the office has to act now, and 're-upload the guardians file' is "
    "not an answer. The link itself is untouched — the adult stays on the contact list.",
    responses=error_responses(401, 403, 404, 422),
)
def set_records_access(
    student_number: str,
    phone: str,
    body: Annotated[AccessIn, Body()],
    caller: Registrar,
    uow_factory: UnitOfWorkFactory,
) -> LinkOut:
    with domain_errors():
        number = StudentNumber(student_number)
        parsed = Phone(phone)
        with uow_factory() as uow:  # type: ignore[operator]  # factory of UnitOfWork
            link = _require_link(uow, number, parsed)
            uow.student_guardians.upsert_many(
                [
                    _replaced(
                        link,
                        can_view_records=body.can_view_records,
                        restriction_note=body.restriction_note,
                    )
                ]
            )
            uow.commit()
    return LinkOut(
        student_number=str(number),
        phone=str(parsed),
        can_view_records=body.can_view_records,
    )


@router.delete(
    "/students/{student_number}/guardians/{phone}",
    status_code=204,
    summary="Remove one guardian from one child",
    description="For a link created in error. Ending a *correct* relationship is the "
    "PATCH above, which keeps the contact and removes only the reading.",
    responses=error_responses(401, 403, 404, 422),
)
def unlink_guardian(
    student_number: str,
    phone: str,
    caller: Registrar,
    uow_factory: UnitOfWorkFactory,
) -> None:
    with domain_errors():
        number = StudentNumber(student_number)
        parsed = Phone(phone)
        with uow_factory() as uow:  # type: ignore[operator]  # factory of UnitOfWork
            if not uow.student_guardians.unlink(number, parsed):
                raise UnknownReference(
                    f"{parsed} is not a guardian of student {number}", field="phone"
                )
            uow.commit()


def _require_link(uow: UnitOfWork, number: StudentNumber, phone: Phone):
    """The one link this request names, or `UnknownReference`.

    Resolved through the child rather than the guardian so the 404 can say which half was
    missing, and matched on *any* of her numbers — a registrar acting on a court order
    will copy whichever number is on the paperwork.
    """
    guardian = uow.guardians.get(phone)
    links = uow.student_guardians.list_for_student(number)
    if guardian is not None:
        for link in links:
            if guardian.reachable_on(link.guardian_phone):
                return link
    raise UnknownReference(
        f"{phone} is not a guardian of student {number}", field="phone"
    )


def _replaced(link, *, can_view_records: bool, restriction_note: str):
    """A copy of the link carrying a new access decision. Frozen dataclass, so a rebuild."""
    from dataclasses import replace

    return replace(
        link,
        can_view_records=can_view_records,
        restriction_note=restriction_note,
    )
