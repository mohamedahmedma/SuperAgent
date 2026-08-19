"""The wire vocabulary every endpoint shares: errors, import outcomes, paging, names.

Why a second set of shapes when `application/dto` already describes these: a DTO is the
argument a use case is called with, and a response model is a promise made to a browser
and to the `records/` adapter. Renaming a DTO field is a refactor; renaming a field here
is a released breaking change. Keeping them as separate types is what lets a service grow
a field without it appearing on the wire, and what gives OpenAPI somewhere to put the
`description=` text a registrar's UI is built from.

**Nothing here may be serialised with `exclude_none`.** A route declared
`response_model_exclude_none=True` drops `"percentage": null` from a report card, and a
missing key is read by every client as "nothing there" — which the next line of
JavaScript writes as `0`. Invariant 1 is that an ungraded subject is null; null has to
arrive as a key whose value is `null`, not as an absence.

Pydantic is imported in `sis/api/` and nowhere else in this service.
"""
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, computed_field

from sis.application.dto.common import PageRequest, RowCode

__all__ = [
    "CodeStr",
    "ErrorDetail",
    "ErrorResponse",
    "ImportCommitResponse",
    "ImportPreviewResponse",
    "ImportRowsPage",
    "NamedOut",
    "PageParams",
    "PageResponse",
    "RequestModel",
    "ResponseModel",
    "RowCode",
    "RowOutcomeOut",
]


def _wire_code(value: object) -> object:
    """Accept the domain's code value objects wherever the wire says `str`.

    `QueryService` hands back entities whose `code` is a `ClassCode`, not a `str`, and a
    response model that insisted on `str` would 500 while serialising a perfectly good
    class list. `str()` is each code's canonical, already-normalised spelling, so this
    narrows rather than converts.
    """
    return str(value) if value is not None and not isinstance(value, str) else value


#: A code as it appears on the wire: the normalised uppercase string, never the object.
CodeStr = Annotated[str, BeforeValidator(_wire_code)]


class RequestModel(BaseModel):
    """Base for anything a client sends.

    `extra="forbid"` because the alternative is silence: a registrar who posts
    `classesPerYear` instead of `classes_per_year` would otherwise get a 200 and a
    structure generated from the defaults, and nothing on screen would say her setting was
    ignored. A 422 naming the unknown field costs her ten seconds instead of an afternoon.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResponseModel(BaseModel):
    """Base for anything the service returns.

    `from_attributes=True` so a route can call `Model.model_validate(dto)` on a frozen
    dataclass and have properties (`ok_count`, `is_open`) read as fields. That keeps the
    mapping from DTO to wire in the field names themselves rather than in a hand-written
    converter per endpoint, where a field added to the DTO and forgotten in the converter
    is invisible until somebody notices the UI never shows it.
    """

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class NamedOut(ResponseModel):
    """Both scripts, on everything nameable (invariant 7).

    Neither is optional and neither is a fallback for the other: a UI that renders Arabic
    should never quietly show an English name because the Arabic one was omitted from the
    payload, since that reads as a data-entry gap the school does not have.
    """

    name_en: str = Field(description="Name in English.")
    name_ar: str = Field(description="Name in Arabic.")


class ErrorDetail(ResponseModel):
    """The body of every failure: a stable code, a human sentence, and the field at fault.

    `code` is what a client branches on and `message` is what a human reads — never the
    other way round. A UI that switches on the message breaks the day the wording is
    improved or translated, and this school reads Arabic.
    """

    code: str = Field(
        description="Stable machine-readable error code, e.g. `unknown_reference`.",
        examples=["unknown_reference"],
    )
    message: str = Field(
        description="Human-readable explanation. Wording is not part of the contract.",
        examples=["no class 3A in 2025-2026"],
    )
    field: str | None = Field(
        default=None,
        description="The request field or spreadsheet column at fault, when there is a single one.",
        examples=["class_code"],
    )


class ErrorResponse(ResponseModel):
    """The envelope FastAPI puts an `HTTPException` detail into. Documented so OpenAPI shows it."""

    detail: ErrorDetail


class RowOutcomeOut(ResponseModel):
    """What happened to one row of one uploaded file.

    `line` is the number in Excel's own gutter — 1-based and counting the header — so
    "row 12 is wrong" points at row 12 on the registrar's screen. Reporting a zero-based
    data index instead is how she ends up editing the wrong child's marks.
    """

    line: int = Field(
        description="1-based line number as shown in the spreadsheet, header included.",
        examples=[14],
    )
    code: RowCode = Field(
        description=(
            "Closed vocabulary of row outcomes. `ok` is the only non-failure; the rest are "
            "what the UI counts and filters by."
        ),
        examples=[RowCode.UNKNOWN_CLASS],
    )
    message: str = Field(
        default="",
        description="Human sentence for this row. Empty for accepted rows — there is nothing to say.",
    )
    field: str | None = Field(
        default=None,
        description="Column at fault, when a single one is, so the UI can highlight a cell.",
        examples=["student_number"],
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The row's identifying cells, so the table can be rendered without holding the "
            "uploaded file. Shape varies by importer and is not part of the contract."
        ),
        examples=[{"student_number": "10432", "class_code": "3A"}],
    )


# Totals are typed by the enum so the JSON object's keys are documented rather than being
# "some strings". Codes that did not occur are absent, not zero: a file with two bad rows
# renders two filter chips, not fifteen of which thirteen read 0.
_TOTALS_FIELD = Field(
    default_factory=dict,
    description=(
        "Row count per outcome code. Codes that did not occur are omitted. Renders as the "
        "filter chips above the row table."
    ),
    examples=[{"ok": 1186, "unknown_class": 9, "invalid_grade": 5}],
)


class _ImportResultOut(ResponseModel):
    """The counts and rows shared by preview and commit."""

    batch_id: str = Field(
        description="Names this upload. Pass it to the commit endpoint to apply what was reviewed.",
        examples=["b0f2c1d84e7a4b0d9a3f5c6e7d8a9b01"],
    )
    total_rows: int = Field(description="Rows processed, accepted and rejected together.")
    ok_count: int = Field(description="Rows that will be written, or were.")
    rejected_count: int = Field(
        description="Rows that will not be written. One bad row never discards the good ones."
    )
    totals: dict[RowCode, int] = _TOTALS_FIELD
    rows: list[RowOutcomeOut] = Field(
        default_factory=list,
        description=(
            "Per-row outcomes. May be large; the paged endpoint returning `ImportRowsPage` "
            "exists for files a UI should not load whole."
        ),
    )


class ImportPreviewResponse(_ImportResultOut):
    """What the registrar is shown before anything is written.

    `expires_at` is on the payload rather than left to be discovered as a 409, because a
    preview validated against last week's class list must not be committable today — and
    finding that out *after* reviewing 1200 rows is the worst moment to find it out.
    """

    expires_at: datetime | None = Field(
        default=None,
        description="When this preview stops being committable. Null means no deadline was set.",
    )


class ImportCommitResponse(_ImportResultOut):
    """What was actually written, row by row.

    Deliberately not assumed to equal the preview. A row that previewed `ok` can commit as
    `changed_since_preview` — another registrar enrolled the child in the meantime — so
    `ok_count` here is the number of rows that reached the database, and only that.
    """

    committed_at: datetime | None = Field(
        default=None,
        description="When the batch was applied. Null if nothing was written.",
    )


class PageParams(RequestModel):
    """A window over a listing. Out-of-range values are clamped, not refused.

    Clamping mirrors `PageRequest` exactly, and the reason is the same: a request for
    10000 students is a UI bug or a curious URL, and answering it with 200 rows serves the
    caller better than a 422 — while answering it in full is how one request pulls an
    entire school into memory. The response echoes the limit that was actually used, so a
    clamped client can see what it got.
    """

    limit: int = Field(
        default=PageRequest.DEFAULT_LIMIT,
        description=(
            f"Rows to return. Values above {PageRequest.MAX_LIMIT} are clamped to it, and "
            f"values below 1 fall back to {PageRequest.DEFAULT_LIMIT}."
        ),
        examples=[50],
    )
    offset: int = Field(
        default=0, description="Rows to skip. Negative values are treated as 0.", examples=[0]
    )

    def to_request(self) -> PageRequest:
        """The application's own paging type, which applies the clamp. One clamp, one place."""
        return PageRequest(limit=self.limit, offset=self.offset)


class PageResponse[ItemT](ResponseModel):
    """One window of results plus the size of the whole match.

    `total` counts everything that matched, not what is in `items`. Without it a UI can
    offer "next" and never "page 7 of 40", and a registrar hunting one child through a year
    group of 600 cannot tell whether she is near the end.
    """

    items: list[ItemT] = Field(default_factory=list, description="This window's rows.")
    total: int = Field(description="Total rows matching the query, across all pages.")
    limit: int = Field(
        description="Rows requested for this window, after clamping. Echoed so a clamped client sees it."
    )
    offset: int = Field(description="Rows skipped before this window.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Rows in this window. Derived, so it can never disagree with `items`."""
        return len(self.items)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_more(self) -> bool:
        """Whether another window follows."""
        return self.offset + len(self.items) < self.total

    @classmethod
    def of(cls, items: list[ItemT], total: int, page: PageRequest | PageParams) -> Self:
        """Build from the paging object the query was run with, so the echo cannot drift."""
        return cls(items=items, total=total, limit=page.limit, offset=page.offset)


class ImportRowsPage(PageResponse[RowOutcomeOut]):
    """A batch's row outcomes, paged, with the batch's totals alongside.

    The totals travel with every page on purpose. They are the filter chips, and chips
    computed from the visible page would report "9 unknown classes" on page one and a
    different number on page two, which reads as the file changing while it is being read.
    """

    batch_id: str = Field(description="The batch these rows belong to.")
    totals: dict[RowCode, int] = _TOTALS_FIELD

    @classmethod
    def for_batch(
        cls,
        batch_id: str,
        rows: list[RowOutcomeOut],
        total: int,
        totals: Mapping[RowCode, int],
        page: PageRequest | PageParams,
    ) -> Self:
        return cls(
            batch_id=batch_id,
            items=rows,
            total=total,
            limit=page.limit,
            offset=page.offset,
            totals=dict(totals),
        )
