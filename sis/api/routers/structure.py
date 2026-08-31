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
from collections.abc import Sequence
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
    Principal,
    require_permission,
)
from sis.domain.rbac import Permission
from sis.api.routers import domain_errors, error_responses
from sis.application.dto import GenerateStructureCommand, TermPlan
from sis.application.services import QueryService, StructureGenerationService
from sis.domain.structure import (
    AcademicYear,
    ClassSection,
    School,
    SchoolLanguage,
    Stage,
    Subject,
    Term,
    WorkingDay,
    YearLevel,
)
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    SchoolCode,
    SubjectCode,
    YearCode,
)

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
Reader = Annotated[Principal, Depends(require_permission(Permission.STRUCTURE_READ))]
Registrar = Annotated[Principal, Depends(require_permission(Permission.STRUCTURE_WRITE))]
Queries = Annotated[QueryService, Depends(get_query_service)]
Structure = Annotated[StructureGenerationService, Depends(get_structure_service)]
Catalogue = Annotated[StructureCatalogue, Depends(get_structure_catalogue)]


# -- Wire shapes ------------------------------------------------------------


class AcademicYearOut(BaseModel):
    code: str
    school_code: str = Field(
        description="The school this year belongs to. Year codes are globally unique, so a "
        "code names one year at one school — this says which."
    )
    name_ar: str
    name_en: str
    starts_on: date
    ends_on: date
    is_current: bool
    terms: "TermPlanOut | None" = Field(
        default=None,
        description="Present only on the write that created or re-synced this year: what "
        "happened to its term sections. Absent in listings, where the terms themselves are "
        "the answer and a plan would be a stale record of an older call.",
    )

    @classmethod
    def of(cls, year: AcademicYear, *, terms: TermPlan | None = None) -> "AcademicYearOut":
        return cls(
            code=str(year.code),
            school_code=str(year.school_code),
            name_ar=year.name_ar,
            name_en=year.name_en,
            starts_on=year.starts_on,
            ends_on=year.ends_on,
            is_current=year.is_current,
            terms=None if terms is None else TermPlanOut.of(terms),
        )


class AcademicYearIn(BaseModel):
    """Create or relabel one school year.

    Upsert rather than insert-only, for the same reason as terms and subjects: `code` is
    identity and the names are labels, so correcting a spelling must not create a second
    year beside the one every class already points at.
    """

    code: str = Field(
        description="Academic year code. Globally unique, so it names one year at one "
        "school — two branches cannot both use `2025-2026` and must distinguish them.",
        examples=["2025-2026"],
    )
    school_code: str = Field(
        description="The school this year belongs to. Must already exist.",
        examples=["MAIN"],
    )
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
    school_code: str = Field(
        description="Rung codes are unique *within* a school, because 'Year 1' exists at "
        "every branch. This is the half of the identity the code does not carry."
    )
    stage: str = Field(
        description="garden / primary / preparatory / secondary, or `unspecified`. A label "
        "for grouping a long ladder on screen; it carries no rules."
    )
    track_code: str | None = Field(
        default=None, description="Arabic/Languages academic track owning this rung."
    )
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
            school_code=str(level.school_code),
            stage=str(level.stage),
            track_code=level.track_code,
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


#: What both term date fields say, written once so the request and the response cannot
#: describe the same optionality differently.
_TERM_DATE_NOTE = (
    "Optional. `null` means the school has not fixed this boundary yet — a year's term "
    "sections are created from the school's term count long before the calendar is "
    "settled. It is never a placeholder: nothing downstream can tell an invented date "
    "from a real one, and these days decide which class a mark is filed under."
)


class TermIn(BaseModel):
    code: str = Field(examples=["2026-T1"])
    academic_year_code: str = Field(examples=["2025-2026"])
    name_ar: str = ""
    name_en: str = ""
    starts_on: date | None = Field(default=None, description=f"First day of term. {_TERM_DATE_NOTE}")
    ends_on: date | None = Field(
        default=None, description=f"Last day of term, inclusive. {_TERM_DATE_NOTE}"
    )
    sequence: int = Field(
        default=1,
        description="Chronological order without parsing the code, which schools format "
        "freely. This is what orders terms on every screen — the dates never did, and now "
        "may not be there to.",
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
    starts_on: date | None = Field(default=None, description=_TERM_DATE_NOTE)
    ends_on: date | None = Field(default=None, description=_TERM_DATE_NOTE)
    sequence: int
    is_closed: bool
    is_dated: bool = Field(
        default=False,
        description="Whether both boundaries are on file. The supported way to ask — one "
        "date alone is not a window, and a client testing `starts_on` alone would treat a "
        "half-filled term as dated.",
    )

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
            is_dated=term.is_dated,
        )


class TermPlanOut(BaseModel):
    """What creating or re-syncing a year did to its term sections."""

    academic_year_code: str
    term_count: int = Field(description="The school's stated number of terms.")
    created: list[str] = Field(
        default_factory=list, description="Term codes this call created, undated."
    )
    removed: list[str] = Field(
        default_factory=list,
        description="Surplus terms deleted because nothing was stated against them.",
    )
    kept: list[str] = Field(
        default_factory=list,
        description="Surplus terms left in place because they hold marks. The year still "
        "shows them, and this is why — dropping a term count never deletes a grade.",
    )

    @classmethod
    def of(cls, plan: TermPlan) -> "TermPlanOut":
        return cls(
            academic_year_code=plan.academic_year_code,
            term_count=plan.term_count,
            created=list(plan.created),
            removed=list(plan.removed),
            kept=list(plan.kept),
        )


class SubjectIn(BaseModel):
    """One subject as one academic year teaches it.

    `academic_year_code` is required and is half of the subject's identity. The same code
    in two years is two subjects, so posting `MATH` for 2026-2027 creates a row rather than
    colliding with the `MATH` of 2025-2026 — and a mark stated against one of them is not a
    mark on the other.
    """

    code: str = Field(examples=["MATH"])
    academic_year_code: str = Field(
        examples=["2025-2026"],
        description="The year that teaches this subject. Part of its identity, not a tag.",
    )
    name_ar: str = ""
    name_en: str = ""
    display_order: int = 0
    is_active: bool = Field(
        default=True,
        description="Retire a subject with `false` rather than deleting it; marks already "
        "stated against it must keep resolving, or that year's report card loses a heading.",
    )


class SubjectOut(BaseModel):
    code: str
    academic_year_code: str
    name_ar: str
    name_en: str
    display_order: int
    is_active: bool

    @classmethod
    def of(cls, subject: Subject) -> "SubjectOut":
        return cls(
            code=str(subject.code),
            academic_year_code=str(subject.academic_year_code),
            name_ar=subject.name_ar,
            name_en=subject.name_en,
            display_order=subject.display_order,
            is_active=subject.is_active,
        )


class SubjectAssignmentIn(BaseModel):
    """One statement of the form "this rung teaches this subject", or stops teaching it."""

    academic_year_code: str = Field(examples=["2025-2026"])
    subject_code: str = Field(examples=["PHYS"])
    year_level_code: str = Field(
        examples=["AR-SEC1"],
        description="The rung, which belongs to exactly one academic track. Assigning to "
        "the Arabic section's Secondary 1 says nothing about the Languages section's.",
    )
    assigned: bool = Field(
        default=True,
        description="`false` removes the assignment. The subject row and every mark "
        "already stated against it survive — this is a statement about the timetable, "
        "not a retraction of a child's grade.",
    )


class GradeSubjectAssignmentsOut(BaseModel):
    """One rung and everything it teaches. Rungs with no assignment are omitted."""

    year_level_code: str
    track_code: str | None = Field(
        default=None, description="The academic track owning the rung, when it has one."
    )
    subjects: list[SubjectOut]


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
    description="Both lists in one body so the two selects on a structure screen cannot be "
    "fetched a moment apart and disagree.\n\n"
    "`school` narrows both to one school and is optional: without it the years of every "
    "school are returned and `year_levels` is empty, because a ladder only means something "
    "inside a school and merging several would put two branches' `Y1` side by side with "
    "nothing to tell them apart. The school picker itself is the caller that omits it.",
    responses=error_responses(401, 403, 404, 422),
)
def list_years(
    queries: Queries,
    caller: Reader,
    school: Annotated[
        str | None,
        Query(description="School code. Omit to list the years of every school."),
    ] = None,
) -> StructureYearsOut:
    with domain_errors():
        school_code = None if school is None else SchoolCode(school)
        academic_years = queries.list_academic_years(school_code)
        # Empty rather than "every school's rungs" when no school is named. See the route
        # description: a merged ladder is a list a registrar cannot act on.
        year_levels = (
            queries.list_year_levels(school_code) if school_code is not None else ()
        )
        current = queries.current_academic_year(school_code)
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
    if year_level is None:
        caller.narrow(Permission.STRUCTURE_READ, lambda scopes: scopes.for_year(academic_year))
    else:
        caller.narrow(
            Permission.STRUCTURE_READ,
            lambda scopes: scopes.for_year_level(
                school_id=caller.school_id, year_level_code=year_level
            ),
        )
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
    "and its labels or dates were updated. The code is identity and is never rewritten."
    "\n\nA year's terms are normally created for it, from the school's term count, when "
    "the year is created — this route is how their labels and dates are filled in "
    "afterwards, and how a school that wants a term the count did not produce adds one. "
    "Both dates are optional; sending `null` for one clears it.",
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
    summary="Create or relabel an academic year, and build its terms",
    description="201 when the year is new, 200 when a year with this code already existed "
    "and its labels or dates were updated. Every other structural row hangs off a year, "
    "so this is the first call when setting a school up.\n\n"
    "**The year's term sections are created here, from the school's own `term_count`.** "
    "One selected term produces one term section, two produce two, three produce three — "
    "the school answered that question when it was created and is not asked it again. The "
    "new terms carry no dates: those are optional and are filled in later, per term, when "
    "the calendar is settled. `terms` on the response says what this call did, including "
    "any surplus term it kept because marks are already stated against it.\n\n"
    "Re-posting a year to correct a label does not disturb the terms underneath it.",
    responses=error_responses(401, 403, 409, 422),
)
def create_academic_year(
    body: AcademicYearIn, catalogue: Catalogue, caller: Registrar, response: Response
) -> AcademicYearOut:
    with domain_errors():
        year = AcademicYear(
            code=AcademicYearCode(body.code),
            school_code=SchoolCode(body.school_code),
            name_ar=body.name_ar,
            name_en=body.name_en,
            starts_on=body.starts_on,
            ends_on=body.ends_on,
            is_current=body.is_current,
        )
        created, plan = catalogue.create_academic_year(year, make_current=body.is_current)
    if not created:
        response.status_code = status.HTTP_200_OK
    return AcademicYearOut.of(year, terms=plan)


@router.post(
    "/academic-years/{code}/terms/sync",
    response_model=TermPlanOut,
    summary="Bring a year's term sections back in line with its school",
    description="Idempotent, and normally unnecessary: the sync runs when a year is "
    "created and again for every year of a school whose term count changes. This route is "
    "for the case those two miss — a year built before the count was corrected, or a "
    "registrar who wants to see the effect without re-saving anything.\n\n"
    "It never deletes a term that holds marks. A surplus term with grades stated against "
    "it is reported in `kept` and left exactly where it is.",
    responses=error_responses(401, 403, 404, 422),
)
def sync_year_terms(code: str, catalogue: Catalogue, caller: Registrar) -> TermPlanOut:
    with domain_errors():
        return TermPlanOut.of(catalogue.sync_year_terms(AcademicYearCode(code)))


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
            academic_year_code=body.academic_year_code,
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
    summary="One year's subjects, in report-card order",
    description="`academic_year` is required: the catalogue is per-year, so `MATH` alone "
    "names a different subject each September and 'the subjects' is not a question with an "
    "answer. Retired subjects are omitted unless asked for by name. Unknown year is a 404, "
    "never an empty list — a typo and a year with no catalogue read identically on screen."
    "\n\n`year_level` narrows the answer to the subjects **assigned** to that rung, which "
    "is what a marks screen for one class should ask for: a school teaches Physics, but "
    "only Secondary sits it. Rungs belong to one academic track, so this is also how the "
    "Arabic and Languages sections of a bilingual school get different answers.",
    responses=error_responses(401, 403, 404, 422),
)
def list_subjects(
    queries: Queries,
    caller: Reader,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    include_inactive: bool = False,
    year_level: Annotated[str | None, Query(examples=["SEC1"])] = None,
) -> list[SubjectOut]:
    if year_level is None:
        caller.narrow(Permission.STRUCTURE_READ, lambda scopes: scopes.for_year(academic_year))
    else:
        caller.narrow(
            Permission.STRUCTURE_READ,
            lambda scopes: scopes.for_year_level(
                school_id=caller.school_id, year_level_code=year_level
            ),
        )
    with domain_errors():
        subjects = queries.list_subjects(
            AcademicYearCode(academic_year),
            include_inactive=include_inactive,
            year_level_code=YearCode(year_level) if year_level else None,
        )
    return [SubjectOut.of(subject) for subject in subjects]


@router.get(
    "/subject-assignments",
    response_model=list[GradeSubjectAssignmentsOut],
    summary="Which rungs teach which subjects, for one year",
    description="The whole board in one read, so the assignment screen does not issue a "
    "request per rung. Ordered by rung, then by each subject's report-card order.\n\n"
    "A rung with no assignment is absent rather than present-and-empty: the caller is "
    "drawing rungs it already has, and an absent key and an empty list mean the same "
    "thing to it. Unknown year is a 404.",
    responses=error_responses(401, 403, 404, 422),
)
def list_subject_assignments(
    catalogue: Catalogue,
    caller: Reader,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
    year_level: Annotated[str | None, Query(examples=["Y3"])] = None,
) -> list[GradeSubjectAssignmentsOut]:
    if year_level is None:
        caller.narrow(Permission.STRUCTURE_READ, lambda scopes: scopes.for_year(academic_year))
    else:
        caller.narrow(
            Permission.STRUCTURE_READ,
            lambda scopes: scopes.for_year_level(
                school_id=caller.school_id, year_level_code=year_level
            ),
        )
    with domain_errors():
        rows = catalogue.subject_assignments(AcademicYearCode(academic_year))
    if year_level is not None:
        rows = [row for row in rows if str(row.year_level_code) == year_level]
    return [
        GradeSubjectAssignmentsOut(
            year_level_code=row.year_level_code,
            track_code=row.track_code,
            subjects=[SubjectOut.of(subject) for subject in row.subjects],
        )
        for row in rows
    ]


@router.put(
    "/subject-assignments",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Assign one subject to one rung, or take it back off",
    description="Idempotent in both directions, which is what a drag-and-drop board needs: "
    "dropping a subject onto a rung that already has it is a no-op, not a 409, and a "
    "duplicate is impossible in the database besides "
    "(`uq_subject_year_levels_assignment`).\n\n"
    "The subject and the rung must belong to the same school, which the year names. A "
    "rung from another branch is a 404 rather than an assignment nobody can see.",
    responses=error_responses(401, 403, 404, 422),
)
def set_subject_assignment(
    body: SubjectAssignmentIn, catalogue: Catalogue, caller: Registrar
) -> None:
    with domain_errors():
        catalogue.set_subject_assignment(
            AcademicYearCode(body.academic_year_code),
            SubjectCode(body.subject_code),
            YearCode(body.year_level_code),
            assigned=body.assigned,
        )


# -- One class at a time ----------------------------------------------------
#
# The generator above builds a whole ladder and is the right tool for September. These two
# are the rest of the year: the extra section a school opens in November, and the label a
# registrar corrects. Expressing either through the generator would mean asking it to
# rebuild every level to add one room, and reading back "forty-one already present".


class ClassSectionIn(BaseModel):
    """One class section, in one academic year, on one year level."""

    code: str = Field(
        examples=["3C"],
        description="Unique within the academic year, and immutable once created. "
        "Renaming is a label change through PATCH; the code is identity.",
    )
    academic_year_code: str = Field(examples=["2025-2026"])
    year_level_code: str = Field(
        examples=["Y3"], description="The rung this section sits on. Must already exist."
    )
    name_ar: str = ""
    name_en: str = ""
    capacity: int | None = Field(
        default=None,
        ge=0,
        description="Places in the room. `null` means nobody has stated one, which is not "
        "the same as `0` — a capacity of zero is a section that admits nobody, and a "
        "registrar can legitimately mean it.",
    )


class ClassSectionPatch(BaseModel):
    """Labels only. There is deliberately no `code` and no `year_level_code` here.

    Moving a class to another rung or renaming its code would carry every enrolment and
    every mark with it, under a class the children were never in. Both are new sections
    plus a roster change, and neither is an edit.
    """

    name_ar: str | None = None
    name_en: str | None = None


@router.post(
    "/structure/classes",
    response_model=ClassSectionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add one class section to a year",
    description="201 when the section is new, 200 when one with this code already existed "
    "in this year — in which case its labels are corrected and no enrolment is touched. "
    "An unknown year or year level is a 404: this route will not invent the rung a class "
    "sits on.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def create_class_section(
    body: ClassSectionIn, catalogue: Catalogue, caller: Registrar, response: Response
) -> ClassSectionOut:
    with domain_errors():
        section = ClassSection(
            code=body.code,
            academic_year_code=body.academic_year_code,
            year_level_code=body.year_level_code,
            name_ar=body.name_ar,
            name_en=body.name_en,
            capacity=body.capacity,
        )
        created = catalogue.create_class_section(section)
    if not created:
        response.status_code = status.HTTP_200_OK
    return ClassSectionOut.of(section)

@router.patch(
    "/structure/classes/{class_code}",
    response_model=ClassSectionOut,
    summary="Relabel one class section",
    description="Renaming `3A` to `Falcons` changes a label and detaches no student and no "
    "grade — the code is what everything points at, and this route cannot reach it. "
    "`academic_year` is required, because a class code names a different room each year.",
    responses=error_responses(401, 403, 404, 422),
)
def rename_class_section(
    class_code: str,
    body: ClassSectionPatch,
    catalogue: Catalogue,
    caller: Registrar,
    academic_year: Annotated[str, Query(examples=["2025-2026"])],
) -> ClassSectionOut:
    with domain_errors():
        section = catalogue.rename_class_section(
            AcademicYearCode(academic_year),
            ClassCode(class_code),
            name_en=body.name_en,
            name_ar=body.name_ar,
        )
    return ClassSectionOut.of(section)

# -- Schools ----------------------------------------------------------------
#
# The outermost scope, and the newest. Everything above belongs to one school, reached
# through a year or a rung; a student does not, because a child is a person rather than a
# school's property, and a child moving between branches is a transfer.


class SchoolIn(BaseModel):
    """Create or relabel one school."""

    code: str = Field(
        description="Immutable identity. Every year and rung in the school points at it.",
        examples=["MAIN"],
    )
    name_en: str = Field(min_length=1, description="English school name.")
    name_ar: str = Field(min_length=1, description="Arabic school name.")
    language_type: Literal["arabic", "languages", "both"] = "both"
    kg_grade_count: int = Field(default=3, ge=0, le=3)
    primary_grade_count: int = Field(default=6, ge=0, le=6)
    preparatory_grade_count: int = Field(default=3, ge=0, le=3)
    secondary_grade_count: int = Field(
        default=3, ge=0, le=3,
        description="The current curriculum configuration defines three secondary grades.",
    )
    term_count: Literal[1, 2, 3] = 2
    working_days: list[Literal[
        "saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"
    ]] = Field(default_factory=lambda: ["sunday", "monday", "tuesday", "wednesday", "thursday"], min_length=1)
    is_active: bool = Field(
        default=True,
        description="Set false to close a branch. Nothing is deleted: the registers taken "
        "and marks stated in the years it ran are still true, and the database refuses a "
        "delete while any year or rung points at it.",
    )


class SchoolOut(BaseModel):
    code: str
    name_ar: str
    name_en: str
    is_active: bool
    language_type: str
    kg_grade_count: int
    primary_grade_count: int
    preparatory_grade_count: int
    secondary_grade_count: int
    term_count: int
    working_days: list[str]
    terms: list["TermPlanOut"] = Field(
        default_factory=list,
        description="Present only when this write changed `term_count`: one entry per "
        "academic year of the school whose term sections were brought back in line. Empty "
        "otherwise — an ordinary rename does not touch a single year.",
    )

    @classmethod
    def of(cls, school: School, *, terms: Sequence[TermPlan] = ()) -> "SchoolOut":
        return cls(
            code=str(school.code),
            name_ar=school.name_ar,
            name_en=school.name_en,
            is_active=school.is_active,
            language_type=school.language_type.value,
            kg_grade_count=school.kg_grade_count,
            primary_grade_count=school.primary_grade_count,
            preparatory_grade_count=school.preparatory_grade_count,
            secondary_grade_count=school.secondary_grade_count,
            term_count=school.term_count,
            working_days=[day.value for day in school.working_days],
            terms=[TermPlanOut.of(plan) for plan in terms],
        )


class YearTrackOut(BaseModel):
    """One academic track's share of a year: its rungs, and how many classes each holds."""

    track_code: str | None = Field(
        description="`null` groups the rungs that belong to no track — a school's rungs "
        "from before it declared its sections. They are shown, not hidden."
    )
    name_en: str = ""
    name_ar: str = ""
    year_levels: list[YearLevelOut] = Field(default_factory=list)
    class_count: int = Field(
        default=0, description="Classes this track's rungs hold in this year."
    )


class AcademicYearDetailOut(BaseModel):
    """One year and everything it hangs off, so a screen can draw it from one read."""

    year: AcademicYearOut
    school: SchoolOut = Field(
        description="The school that runs this year, including the `term_count` its term "
        "sections were built from."
    )
    terms: list[TermOut] = Field(
        description="In `sequence` order. One entry per term the school runs; the dates "
        "may be `null` until someone fills them in."
    )
    tracks: list[YearTrackOut] = Field(
        description="The year's ladder, grouped by academic track. A single-track school "
        "gets one group; the terms above are shared by all of them."
    )
    class_count: int = Field(description="Every class in the year, across all tracks.")


@router.get(
    "/academic-years/{code}",
    response_model=AcademicYearDetailOut,
    summary="One year, its school, its terms and its ladder",
    description="Everything an academic year is attached to, in one body: the school that "
    "runs it, the term sections built from that school's term count, and the grades and "
    "classes it carries grouped by academic track.\n\n"
    "One route rather than four because the four can disagree. A screen that fetches the "
    "year, then its terms, then its rungs, then its classes can have a rung added under it "
    "between the second and third call, and will then draw a school that existed at no "
    "instant. Unknown year is a 404.",
    responses=error_responses(401, 403, 404, 422),
)
def read_academic_year(
    code: str, catalogue: Catalogue, caller: Reader
) -> AcademicYearDetailOut:
    with domain_errors():
        detail = catalogue.academic_year_detail(AcademicYearCode(code))
    return AcademicYearDetailOut(
        year=AcademicYearOut.of(detail["year"]),
        school=SchoolOut.of(detail["school"]),
        terms=[TermOut.of(term) for term in detail["terms"]],
        tracks=[
            YearTrackOut(
                track_code=group["track_code"],
                name_en="" if group["track"] is None else group["track"].name_en,
                name_ar="" if group["track"] is None else group["track"].name_ar,
                year_levels=[YearLevelOut.of(level) for level in group["levels"]],
                class_count=sum(
                    group["classes_per_level"].get(str(level.code), 0)
                    for level in group["levels"]
                ),
            )
            for group in detail["tracks"]
        ],
        class_count=detail["class_count"],
    )


class YearLevelIn(BaseModel):
    """Add or relabel one rung of one school's ladder."""

    code: str = Field(
        description="Unique within the school, not globally: `Y1` exists at every branch.",
        examples=["Y1"],
    )
    school_code: str = Field(examples=["MAIN"])
    track_code: str | None = Field(
        default=None,
        description="AR or LANG. Optional only when the school has one academic track.",
    )
    name_en: str = Field(default="", description="English label.")
    name_ar: str = Field(default="", description="Arabic label.")
    display_order: int = Field(
        default=0,
        description="Order within the stage. Stated, not derived from the code: `Y10` sorts "
        "before `Y9` as text.",
    )
    stage: str = Field(
        default="unspecified",
        description="garden / primary / preparatory / secondary, or `unspecified`. Grouping "
        "only — no rule anywhere depends on it. An unrecognised value is refused rather "
        "than silently treated as unspecified.",
        examples=["primary"],
    )


@router.get(
    "/schools",
    response_model=list[SchoolOut],
    summary="Every school",
    description="Closed branches are omitted unless `include_inactive` asks for them.",
    responses=error_responses(401, 403),
)
def list_schools(
    queries: Queries, caller: Reader, include_inactive: bool = False
) -> list[SchoolOut]:
    with domain_errors():
        schools = queries.list_schools(include_inactive=include_inactive)
    return [SchoolOut.of(school) for school in schools]


@router.post(
    "/schools",
    response_model=SchoolOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create or relabel a school",
    description="201 when the school is new, 200 when this code already existed and its "
    "labels were corrected — which detaches nothing, because every year and rung points at "
    "the row rather than at the name. There is no delete; close a branch with "
    "`is_active: false`. On an existing school, a field left out of the body keeps the "
    "value already on file: the defaults below are what a NEW school gets, not what an "
    "omission resets an existing one to.",
    responses=error_responses(401, 403, 409, 422),
)
def create_school(
    body: SchoolIn, catalogue: Catalogue, caller: Registrar, response: Response
) -> SchoolOut:
    with domain_errors():
        school = School(
            code=body.code,
            name_ar=body.name_ar,
            name_en=body.name_en,
            is_active=body.is_active,
            language_type=SchoolLanguage(body.language_type),
            kg_grade_count=body.kg_grade_count,
            primary_grade_count=body.primary_grade_count,
            preparatory_grade_count=body.preparatory_grade_count,
            secondary_grade_count=body.secondary_grade_count,
            term_count=body.term_count,
            working_days=tuple(WorkingDay(day) for day in body.working_days),
        )
        # `model_fields_set` is the only thing that can tell "the caller asked for two
        # terms" apart from "the caller said nothing and Pydantic filled in two".
        stored, created, plans = catalogue.create_school(
            school, stated=body.model_fields_set
        )
    if not created:
        response.status_code = status.HTTP_200_OK
    return SchoolOut.of(stored, terms=plans)


class AcademicTrackOut(BaseModel):
    code: str
    school_code: str
    language_type: str
    name_en: str
    name_ar: str
    display_order: int


class ConfiguredGradeOut(BaseModel):
    code: str
    stage: str
    name_en: str
    name_ar: str
    display_order: int


class ConfiguredClassesIn(BaseModel):
    academic_year_code: str
    track_code: str
    mode: Literal["same", "custom"]
    class_count: int | None = Field(default=None, ge=0, le=60)
    classes_by_grade: dict[str, int] | None = None
    sequence: Literal["numeric", "alphabetic"]


@router.get("/schools/{school_code}/tracks/{track_code}/configured-grades",
    response_model=list[ConfiguredGradeOut])
def configured_grades(school_code: str, track_code: str, catalogue: Catalogue, caller: Reader):
    with domain_errors():
        return catalogue.configured_grades(SchoolCode(school_code), track_code)


@router.post("/structure/configured-classes", response_model=list[ClassSectionOut])
def create_configured_classes(body: ConfiguredClassesIn, catalogue: Catalogue, caller: Registrar):
    with domain_errors():
        if body.mode == "same":
            if body.class_count is None:
                raise ValidationError("class_count is required in same mode", field="class_count")
        _, sections = catalogue.create_configured_classes(
            AcademicYearCode(body.academic_year_code), body.track_code,
            body.classes_by_grade if body.mode == "custom" else None, body.sequence,
            same_count=body.class_count if body.mode == "same" else None,
        )
    return [ClassSectionOut.of(section) for section in sections]


@router.get(
    "/schools/{school_code}/tracks",
    response_model=list[AcademicTrackOut],
    summary="The school's Arabic/Languages academic tracks",
    responses=error_responses(401, 403, 404, 422),
)
def list_school_tracks(
    school_code: str, catalogue: Catalogue, caller: Reader
) -> list[AcademicTrackOut]:
    with domain_errors():
        tracks = catalogue.list_school_tracks(SchoolCode(school_code))
    return [
        AcademicTrackOut(
            code=track.code,
            school_code=str(track.school_code),
            language_type=track.language_type.value,
            name_en=track.name_en,
            name_ar=track.name_ar,
            display_order=track.display_order,
        )
        for track in tracks
    ]


@router.get(
    "/schools/{school_code}/levels",
    response_model=list[YearLevelOut],
    summary="One school's ladder, grouped by stage",
    description="Ordered by stage — garden, primary, preparatory, secondary, then anything "
    "unclassified — and by `display_order` within each. Unknown school is a 404, never an "
    "empty ladder.",
    responses=error_responses(401, 403, 404, 422),
)
def list_school_levels(
    school_code: str, queries: Queries, caller: Reader
) -> list[YearLevelOut]:
    with domain_errors():
        levels = queries.list_year_levels(SchoolCode(school_code))
    return [YearLevelOut.of(level) for level in levels]


@router.post(
    "/structure/levels",
    response_model=YearLevelOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add or relabel one rung",
    description="201 when the rung is new in this school, 200 when it already existed and "
    "its labels, order or stage were corrected — none of which detaches a class or a mark, "
    "because those point at the rung's row. The generator builds a whole ladder at once; "
    "this is for the single rung a school adds afterwards.",
    responses=error_responses(401, 403, 404, 409, 422),
)
def create_year_level(
    body: YearLevelIn, catalogue: Catalogue, caller: Registrar, response: Response
) -> YearLevelOut:
    with domain_errors():
        level = YearLevel(
            code=body.code,
            school_code=body.school_code,
            track_code=body.track_code,
            name_ar=body.name_ar,
            name_en=body.name_en,
            display_order=body.display_order,
            stage=body.stage,
        )
        created = catalogue.create_year_level(level)
    if not created:
        response.status_code = status.HTTP_200_OK
    return YearLevelOut.of(level)
