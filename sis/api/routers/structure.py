"""The skeleton a school hangs its children on: year levels, classes, terms, subjects.

Generation is idempotent (invariant 3), so `POST /structure/generate` is safe to click
twice and answers `created: false` for everything already on file rather than 409. The
response lists every item the run *touched*, not just the new ones, because the question
a registrar is actually asking after a second click is "did I just double the school".

`GET /structure/years` returns the academic years and the year levels in one body. Two
routes would mean two round trips for one screen, and between them another registrar can
add a rung — so the two dropdowns a registrar picks from could disagree, and the class
list she then requests would be for a year level the year select never offered.

Writes here are upserts keyed on the code, which is why `POST /terms` and `POST /subjects`
answer 201 or 200 rather than 201 or 409. Invariant 6 is what makes that safe: the code is
identity, the names are labels, and re-posting a term with a corrected Arabic name renames
it without detaching one grade.
"""
from datetime import date
from typing import Annotated, Literal, Protocol

from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import BaseModel, Field

from sis.api.deps import (
    Caller,
    get_query_service,
    get_structure_catalogue,
    get_structure_service,
    require_read_access,
    require_registrar,
)
from sis.api.routers import domain_errors, error_responses
from sis.application.dto import GenerateStructureCommand
from sis.application.services import QueryService, StructureGenerationService
from sis.domain.structure import (
    AcademicYear,
    ClassSection,
    Subject,
    Term,
    YearLevel,
)
from sis.domain.value_objects import AcademicYearCode, YearCode

router = APIRouter(prefix="/v1", tags=["structure"])


class StructureCatalogue(Protocol):
    """Creating or relabelling one term or subject, stated by the routes that need it.

    Returns whether the row was created, so this layer can answer 201 or 200 without a
    second read. Insert-or-update rather than insert-only because a school corrects the
    spelling of "الفصل الأول" the week after setting it up, and forcing that through a
    separate PATCH is how registrars end up creating `T1-FIXED`.
    """

    def create_term(self, term: Term) -> bool:
        """Store the term; `True` when this call created it."""

    def create_subject(self, subject: Subject) -> bool:
        """Store the subject; `True` when this call created it."""

    def create_academic_year(self, year: AcademicYear, *, make_current: bool) -> bool:
        """Store the year; `True` when this call created it."""


# Reads take either scope, named out loud; writes take `registrar` and nothing else.
# Scope comparison is exact equality, so a reader-only check here would lock the
# registrar out of the very dropdowns she generates the structure from.
Reader = Annotated[Caller, Depends(require_read_access)]
Registrar = Annotated[Caller, Depends(require_registrar)]
Queries = Annotated[QueryService, Depends(get_query_service)]
Structure = Annotated[StructureGenerationService, Depends(get_structure_service)]
Catalogue = Annotated[StructureCatalogue, Depends(get_structure_catalogue)]


# -- Wire shapes ------------------------------------------------------------


class AcademicYearOut(BaseModel):
    code: str
    name_ar: str
    name_en: str
    starts_on: date
    ends_on: date
    is_current: bool

    @classmethod
    def of(cls, year: AcademicYear) -> "AcademicYearOut":
        return cls(
            code=str(year.code),
            name_ar=year.name_ar,
            name_en=year.name_en,
            starts_on=year.starts_on,
            ends_on=year.ends_on,
            is_current=year.is_current,
        )


class AcademicYearIn(BaseModel):
    """Create or relabel one school year.

    Upsert rather than insert-only, for the same reason as terms and subjects: `code` is
    identity and the names are labels, so correcting a spelling must not create a second
    year beside the one every class already points at.
    """

    code: str = Field(description="Academic year code.", examples=["2025-2026"])
    name_en: str = Field(default="", description="English label.")
    name_ar: str = Field(default="", description="Arabic label.")
    starts_on: date = Field(description="First day of the year.")
    ends_on: date = Field(description="Last day of the year, inclusive.")
    is_current: bool = Field(
        default=True,
        description="Make this the working year. Defaults true: a registrar creating a "
        "year almost always means the one they are about to build.",
    )


class YearLevelOut(BaseModel):
    code: str
    name_ar: str
    name_en: str
    display_order: int = Field(
        description="Stated order, not code order: `Y10` sorts before `Y9` as text, so a "
        "school with ten rungs prints its ladder scrambled without this."
    )

    @classmethod
    def of(cls, level: YearLevel) -> "YearLevelOut":
        return cls(
            code=str(level.code),
            name_ar=level.name_ar,
            name_en=level.name_en,
            display_order=level.display_order,
        )


class StructureYearsOut(BaseModel):
    """Both ladders at once — see the module note on why this is one route."""

    academic_years: list[AcademicYearOut]
    year_levels: list[YearLevelOut]
    current_academic_year_code: str | None = Field(
        default=None,
        description="`null` until a registrar has marked one current; never guessed from "
        "today's date.",
    )


class ClassSectionOut(BaseModel):
    code: str
    academic_year_code: str
    year_level_code: str
    name_ar: str
    name_en: str
    capacity: int | None = Field(
        default=None,
        description="`null` means no stated limit. It is not 0 — a capacity of 0 is a "
        "section that admits nobody, which a registrar closing a class may mean.",
    )

    @classmethod
    def of(cls, section: ClassSection) -> "ClassSectionOut":
        return cls(
            code=str(section.code),
            academic_year_code=str(section.academic_year_code),
            year_level_code=str(section.year_level_code),
            name_ar=section.name_ar,
            name_en=section.name_en,
            capacity=section.capacity,
        )


class GenerateStructureIn(BaseModel):
    """Either uniform (`year_count` + `classes_per_year`) or explicit (`classes_by_year`).

    Every template is optional and omitted fields keep the command's own defaults; sending
    an explicit `null` is treated as "not stated" for the same reason, so a client that
    serialises its whole form does not blank a template into a crash.
    """

    academic_year_code: str = Field(examples=["2025-2026"])
    year_count: int | None = Field(default=None, ge=1)
    classes_per_year: int | None = Field(default=None, ge=0)
    classes_by_year: dict[str, int] | None = Field(
        default=None,
        description="Year level code to section count, e.g. `{\"Y1\": 3, \"Y2\": 5}`. "
        "Key order is creation order.",
    )
    year_code_template: str | None = Field(default=None, examples=["Y{n}"])
    year_name_en_template: str | None = None
    year_name_ar_template: str | None = None
    class_code_template: str | None = Field(default=None, examples=["{year}{suffix}"])
    class_name_en_template: str | None = None
    class_name_ar_template: str | None = None
    class_suffixes: list[str] | None = None
    allow_new_convention: bool = Field(
        default=False,
        description="Set only to build a second ladder beside the codes already on file. "
        "The default refusal exists because a mismatched template generates a full set of "
        "empty classes that reports as a success and strands every existing enrolment.",
    )

    def to_command(self) -> GenerateStructureCommand:
        """Unset fields fall through to the command's defaults rather than to `None`."""
        stated = self.model_dump(
            exclude={"academic_year_code", "allow_new_convention"}, exclude_none=True
        )
        suffixes = stated.pop("class_suffixes", None)
        if suffixes is not None:
            stated["class_suffixes"] = tuple(suffixes)
        return GenerateStructureCommand(
            academic_year_code=AcademicYearCode(self.academic_year_code), **stated
        )


class GeneratedItemOut(BaseModel):
    kind: Literal["year_level", "class_section"]
    code: str
    name_ar: str
    name_en: str
    created: bool = Field(
        description="`false` is a success, not a skip: generation is idempotent, so a "
        "re-run reports what was already there."
    )
    year_level_code: str | None = None


class GenerateStructureOut(BaseModel):
    academic_year_code: str
    created_count: int
    existing_count: int
    items: list[GeneratedItemOut]


class TermIn(BaseModel):
    code: str = Field(examples=["2026-T1"])
    academic_year_code: str = Field(examples=["2025-2026"])
    name_ar: str = ""
    name_en: str = ""
    starts_on: date
    ends_on: date
    sequence: int = Field(
        default=1,
        description="Chronological order without parsing the code, which schools format "
        "freely.",
    )
    is_closed: bool = Field(
        default=False,
        description="Stated by a human, never derived from `ends_on`: a school enters "
        "late marks for a week after a term ends.",
    )


class TermOut(BaseModel):
    code: str
    academic_year_code: str
    name_ar: str
    name_en: str
    starts_on: date
    ends_on: date
    sequence: int
    is_closed: bool

    @classmethod
    def of(cls, term: Term) -> "TermOut":
        return cls(
            code=str(term.code),
            academic_year_code=str(term.academic_year_code),
            name_ar=term.name_ar,
            name_en=term.name_en,
            starts_on=term.starts_on,
            ends_on=term.ends_on,
            sequence=term.sequence,
            is_closed=term.is_closed,
        )


class SubjectIn(BaseModel):
    code: str = Field(examples=["MATH"])
    name_ar: str = ""
    name_en: str = ""
    display_order: int = 0
    is_active: bool = Field(
        default=True,
        description="Retire a subject with `false` rather than deleting it; marks already "
        "stated against it must keep resolving, or last year's report card loses a heading.",
    )


class SubjectOut(BaseModel):
    code: str
    name_ar: str
    name_en: str
    display_order: int
    is_active: bool

    @classmethod
    def of(cls, subject: Subject) -> "SubjectOut":
        return cls(
            code=str(subject.code),
            name_ar=subject.name_ar,
            name_en=subject.name_en,
            display_order=subject.display_order,
            is_active=subject.is_active,
        )


# -- Routes -----------------------------------------------------------------


@router.post(
    "/structure/generate",
    response_model=GenerateStructureOut,
    summary="Generate a year's levels and class sections",
    description="Idempotent and additive: it creates only what is missing, renames "
    "nothing, and deletes nothing. Asking for fewer classes than exist removes none.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def generate_structure(
    body: GenerateStructureIn, structure: Structure, caller: Registrar
) -> GenerateStructureOut:
    with domain_errors():
        result = structure.generate(
            body.to_command(), allow_new_convention=body.allow_new_convention
        )
    return GenerateStructureOut(
        academic_year_code=result.academic_year_code,
        created_count=result.created_count,
        existing_count=result.existing_count,
        items=[
            GeneratedItemOut(
                kind=item.kind,
                code=item.code,
                name_ar=item.name_ar,
                name_en=item.name_en,
                created=item.created,
                year_level_code=item.year_level_code,
            )
            for item in result.items
        ],
    )


@router.get(
    "/structure/years",
    response_model=StructureYearsOut,
    summary="Academic years and year levels",
    description="Both lists in one body so the two selects on a structure screen cannot "
    "be fetched a moment apart and disagree.",
    responses=error_responses(401, 403),
)
def list_years(queries: Queries, caller: Reader) -> StructureYearsOut:
    with domain_errors():
        academic_years = queries.list_academic_years()
        year_levels = queries.list_year_levels()
        current = queries.current_academic_year()
    return StructureYearsOut(
        academic_years=[AcademicYearOut.of(year) for year in academic_years],
        year_levels=[YearLevelOut.of(level) for level in year_levels],
        current_academic_year_code=None if current is None else str(current.code),
    )


@router.get(
    "/structure/classes",
    response_model=list[ClassSectionOut],
    summary="Class sections of one academic year",
    description="`academic_year` is required and never defaulted: `3A` names a different "
    "room of children every September, so a missing year would be answered about whichever "
    "one happens to be flagged current.",
    responses=error_responses(401, 403, 404, 422),
)
def list_classes(
    queries: Queries,
    caller: Reader,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    year_level: Annotated[str | None, Query(examples=["Y3"])] = None,
) -> list[ClassSectionOut]:
    with domain_errors():
        sections = queries.list_classes(
            AcademicYearCode(academic_year),
            year_level_code=None if year_level is None else YearCode(year_level),
        )
    return [ClassSectionOut.of(section) for section in sections]


@router.post(
    "/terms",
    response_model=TermOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create or relabel a term",
    description="201 when the term is new, 200 when a term with this code already existed "
    "and its labels or dates were updated. The code is identity and is never rewritten.",
    responses=error_responses(401, 403, 409, 422),
)
def create_term(
    body: TermIn, catalogue: Catalogue, caller: Registrar, response: Response
) -> TermOut:
    with domain_errors():
        term = Term(
            code=body.code,
            academic_year_code=body.academic_year_code,
            name_ar=body.name_ar,
            name_en=body.name_en,
            starts_on=body.starts_on,
            ends_on=body.ends_on,
            sequence=body.sequence,
            is_closed=body.is_closed,
        )
        created = catalogue.create_term(term)
    if not created:
        response.status_code = status.HTTP_200_OK
    return TermOut.of(term)


@router.post(
    "/academic-years",
    response_model=AcademicYearOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create or relabel an academic year",
    description="201 when the year is new, 200 when a year with this code already existed "
    "and its labels or dates were updated. Every other structural row hangs off a year, "
    "so this is the first call when setting a school up.",
    responses=error_responses(401, 403, 409, 422),
)
def create_academic_year(
    body: AcademicYearIn, catalogue: Catalogue, caller: Registrar, response: Response
) -> AcademicYearOut:
    with domain_errors():
        year = AcademicYear(
            code=AcademicYearCode(body.code),
            name_ar=body.name_ar,
            name_en=body.name_en,
            starts_on=body.starts_on,
            ends_on=body.ends_on,
            is_current=body.is_current,
        )
        created = catalogue.create_academic_year(year, make_current=body.is_current)
    if not created:
        response.status_code = status.HTTP_200_OK
    return AcademicYearOut.of(year)


@router.get(
    "/terms",
    response_model=list[TermOut],
    summary="Terms of one academic year",
    description="In `sequence` order, which is chronological without parsing codes.",
    responses=error_responses(401, 403, 404, 422),
)
def list_terms(
    queries: Queries,
    caller: Reader,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
) -> list[TermOut]:
    with domain_errors():
        terms = queries.list_terms(AcademicYearCode(academic_year))
    return [TermOut.of(term) for term in terms]


@router.post(
    "/subjects",
    response_model=SubjectOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create or relabel a subject",
    description="201 when the subject is new, 200 when one with this code already existed. "
    "Retire a subject with `is_active: false`; there is no delete.",
    responses=error_responses(401, 403, 409, 422),
)
def create_subject(
    body: SubjectIn, catalogue: Catalogue, caller: Registrar, response: Response
) -> SubjectOut:
    with domain_errors():
        subject = Subject(
            code=body.code,
            name_ar=body.name_ar,
            name_en=body.name_en,
            display_order=body.display_order,
            is_active=body.is_active,
        )
        created = catalogue.create_subject(subject)
    if not created:
        response.status_code = status.HTTP_200_OK
    return SubjectOut.of(subject)


@router.get(
    "/subjects",
    response_model=list[SubjectOut],
    summary="Subjects, in report-card order",
    description="Retired subjects are omitted unless asked for by name.",
    responses=error_responses(401, 403),
)
def list_subjects(
    queries: Queries, caller: Reader, include_inactive: bool = False
) -> list[SubjectOut]:
    with domain_errors():
        subjects = queries.list_subjects(include_inactive=include_inactive)
    return [SubjectOut.of(subject) for subject in subjects]
