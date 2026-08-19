"""Wire shapes for the school's skeleton: years, rungs, sections, terms, subjects.

The generate request is the interesting one. It carries two mutually exclusive shapes —
"five years of eight classes" and "year 1 has 3, year 2 has 5" — because a registrar asks
for both in the same breath and they are one operation with a different count per rung
(invariant 3). Two endpoints would mean two generation paths, and the second is the one
that never receives the idempotency fix.

The ceilings and the exactly-one-of rule are checked here *as well as* in
`GenerateStructureCommand`, and that duplication is deliberate: the command raises a
domain `ValidationError`, which reaches a client as a 4xx with no field-level detail, while
a check here becomes a 422 that names `classes_by_year` and shows in OpenAPI before anyone
presses the button. The command stays the authority — this layer is the courtesy.
"""
from datetime import date
from typing import Annotated, Literal

from pydantic import Field, model_validator

from sis.api.schemas.common import CodeStr, NamedOut, RequestModel, ResponseModel
from sis.application.dto.structure import GenerateStructureCommand

__all__ = [
    "AcademicYearOut",
    "ClassSectionOut",
    "GenerateStructureRequest",
    "GenerateStructureResponse",
    "GeneratedItemOut",
    "SubjectOut",
    "TermOut",
    "YearLevelOut",
]

_MAX_YEARS = GenerateStructureCommand.MAX_YEARS
_MAX_CLASSES = GenerateStructureCommand.MAX_CLASSES_PER_YEAR

#: Section counts are bounded in the type rather than in each field, so the uniform form
#: and the per-year form cannot end up with different ceilings after an edit.
ClassCount = Annotated[int, Field(ge=0, le=_MAX_CLASSES)]


class GenerateStructureRequest(RequestModel):
    """Generate one academic year's year levels and class sections. Safe to re-run.

    Send **either** `year_count` with `classes_per_year`, **or** `classes_by_year`. Sending
    both, or neither, is a 422: a request that silently preferred one field would build a
    structure nobody asked for, and the registrar would meet it weeks later as a teacher
    who cannot find a section.

    Re-running an identical request creates nothing and reports every item as
    `created: false` (invariant 3), so a double-clicked button does not double the school.
    """

    academic_year_code: CodeStr = Field(
        description="The academic year to build into, e.g. `2025-2026`. Must already exist.",
        examples=["2025-2026"],
    )
    year_count: int | None = Field(
        default=None,
        ge=1,
        le=_MAX_YEARS,
        description=(
            "Uniform form only: how many year levels to create, numbered from 1. Omit when "
            "sending `classes_by_year`, whose keys already say which rungs exist."
        ),
        examples=[5],
    )
    classes_per_year: ClassCount | None = Field(
        default=None,
        description=(
            "Uniform form only: sections per year level. 0 is allowed — creating the rungs "
            "now and their sections once enrolment numbers are known is a real workflow."
        ),
        examples=[8],
    )
    classes_by_year: dict[str, ClassCount] | None = Field(
        default=None,
        description=(
            "Per-year form: year level code to section count. Key order is creation order, "
            "so two runs of the same request produce the same sequence."
        ),
        examples=[{"Y1": 3, "Y2": 5, "Y3": 5}],
    )
    year_code_template: str = Field(
        default="Y{n}",
        description="Year level code template. Must contain `{n}`, or every rung gets the same code.",
    )
    year_name_en_template: str = Field(default="Year {n}", description="English year level name template.")
    year_name_ar_template: str = Field(default="السنة {n}", description="Arabic year level name template.")
    class_code_template: str = Field(
        default="{year}{suffix}",
        description="Class code template. Must contain both `{year}` and `{suffix}`.",
    )
    class_name_en_template: str = Field(
        default="{year_name} {suffix}", description="English class name template."
    )
    class_name_ar_template: str = Field(
        default="{year_name} {suffix}", description="Arabic class name template."
    )
    class_suffixes: list[str] | None = Field(
        default=None,
        description="Section suffixes in order. Defaults to A, B, C… Latin letters, because this is half of a code.",
        examples=[["A", "B", "C", "D"]],
    )

    # Both shapes are shown as whole-body examples because the field descriptions alone
    # cannot express "these two are alternatives" — a caller reading one field at a time
    # reliably sends both halves and gets a 422 on their first attempt.
    model_config = RequestModel.model_config | {
        "json_schema_extra": {
            "examples": [
                {"academic_year_code": "2025-2026", "year_count": 5, "classes_per_year": 8},
                {"academic_year_code": "2025-2026", "classes_by_year": {"Y1": 3, "Y2": 5}},
            ]
        }
    }

    @model_validator(mode="after")
    def _exactly_one_shape(self) -> "GenerateStructureRequest":
        uniform = self.classes_per_year is not None
        explicit = self.classes_by_year is not None
        if uniform == explicit:
            raise ValueError(
                "send either year_count with classes_per_year (uniform) or classes_by_year "
                "(one count per year level) — not both, and not neither"
            )
        if uniform and self.year_count is None:
            raise ValueError("year_count is required alongside classes_per_year")
        if explicit:
            if self.year_count is not None:
                raise ValueError("year_count is implied by the keys of classes_by_year; remove it")
            if not self.classes_by_year:
                raise ValueError("classes_by_year names no year levels")
            if len(self.classes_by_year) > _MAX_YEARS:
                raise ValueError(f"classes_by_year names more than {_MAX_YEARS} year levels")
        return self


class GeneratedItemOut(ResponseModel):
    """One year level or class section the run touched, and whether it is new.

    `created: false` is a success and not a skip — see `GenerateStructureRequest`. The
    registrar reads this list to answer "did my second click double the school", so the
    flag has to be legible at a glance rather than inferred from a count.
    """

    kind: Literal["year_level", "class_section"] = Field(description="What was generated.")
    code: CodeStr = Field(description="Code of the created or already-present item.", examples=["3A"])
    name_en: str = Field(description="English label, from the name template.")
    name_ar: str = Field(description="Arabic label, from the name template.")
    created: bool = Field(
        description="True if this run wrote it; false if it was already there and was left alone."
    )
    year_level_code: CodeStr | None = Field(
        default=None, description="Owning rung, for class sections. Null on year levels themselves."
    )


class GenerateStructureResponse(ResponseModel):
    """Everything the run touched, with the created/already-present split counted."""

    academic_year_code: CodeStr = Field(description="The year that was built into.")
    created_count: int = Field(description="Items this run wrote.")
    existing_count: int = Field(description="Items already present and left untouched.")
    items: list[GeneratedItemOut] = Field(
        default_factory=list, description="Every item touched, in creation order."
    )


class AcademicYearOut(NamedOut):
    """One school year. Everything else in the service hangs off `code`."""

    code: CodeStr = Field(description="Academic year code.", examples=["2025-2026"])
    starts_on: date = Field(description="First day of the year.")
    ends_on: date = Field(description="Last day of the year, inclusive.")
    is_current: bool = Field(
        default=False, description="Whether the registrar has marked this the working year."
    )


class YearLevelOut(NamedOut):
    """A rung of the ladder — `Y3`, `G10` — not a year group within one year."""

    code: CodeStr = Field(description="Year level code.", examples=["Y3"])
    display_order: int = Field(
        description="Sort position. Sort by this, never by code: `Y10` sorts before `Y9` as text."
    )


class ClassSectionOut(NamedOut):
    """One class section within one academic year.

    `code` is unique per academic year, not globally: `3A` exists every September and is a
    different room of children each time, so a client caching by code alone will mix years.
    """

    code: CodeStr = Field(description="Class code, unique within its academic year.", examples=["3A"])
    academic_year_code: CodeStr = Field(description="Year this section belongs to.")
    year_level_code: CodeStr = Field(description="Rung this section sits on.")
    capacity: int | None = Field(
        default=None,
        description="Stated seat limit. Null means no limit was stated — 0 means a section that admits nobody.",
    )


class TermOut(NamedOut):
    """A reporting period. Its dates are what resolve a child's class for the term (invariant 2)."""

    code: CodeStr = Field(description="Term code, carrying its year.", examples=["2026-T1"])
    academic_year_code: CodeStr = Field(description="Year this term belongs to.")
    starts_on: date = Field(description="First day of term.")
    ends_on: date = Field(description="Last day of term, inclusive.")
    sequence: int = Field(default=1, description="Chronological position; sort by this, not by code.")
    is_closed: bool = Field(default=False, description="Whether marks for this term are final.")


class SubjectOut(NamedOut):
    """A subject, global across year levels so a child's maths marks compare across years."""

    code: CodeStr = Field(description="Subject code.", examples=["MATH"])
    display_order: int = Field(default=0, description="Report card ordering.")
    is_active: bool = Field(
        default=True,
        description="False for a retired subject. Retired, never deleted: its historical grades stay attached.",
    )
