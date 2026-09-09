"""Roster and grade uploads: preview first, commit second, per-row outcomes throughout.

A preview writes an `ImportBatch` and not one student, placement or mark. The commit names
that batch rather than re-uploading the file, so what is written is what a human reviewed —
re-reading the upload at commit is exactly how a different file gets applied than the one
that was approved. One bad row is a value, never an exception: 299 good rows still import
and the 300th comes back as one red line the registrar can act on (invariant 4).

**Uploads are never written to disk.** This is why these handlers take a bare `Request` and
parse the multipart body themselves instead of declaring `UploadFile`. Starlette spools a
form part above a megabyte into a `SpooledTemporaryFile`, which for a school-sized roster
means a file of children's full names materialises in the machine's temp directory and
stays there for whatever cleans it up — a copy of the data nobody audited, nobody
encrypted, and nobody knows exists. The body here lives in a `bytearray` for the duration
of one request and is handed straight to the parser as `bytes`.

**The size ceiling is enforced before the body is buffered**, in two passes for two
different lies. `Content-Length` is checked first, so an obviously oversized upload is
refused having read nothing at all; the streaming loop then re-checks as bytes arrive,
because a chunked request declares no length and a hostile one can declare whatever it
likes. Discovering the size inside the parser would mean the memory the limit exists to
protect has already been spent.

`actor` on every batch is the caller's key prefix, not a name from the request body. An
import that records who it says it is records nothing; a prefix is the handle an operator
can revoke, and it is the one identity this layer actually knows.
"""
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field as dataclass_field
from datetime import UTC, date, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Final, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from sis.api.deps import (
    Caller,
    get_grade_import_service,
    get_guardian_import_service,
    get_import_reports,
    get_roster_import_service,
    require_registrar,
    Principal,
    require_permission,
)
from sis.domain.rbac import Permission
from sis.api.routers import domain_errors, error_responses
from sis.application.dto import (
    GradeCommitCommand,
    GradePreviewCommand,
    GuardianCommitCommand,
    GuardianPreviewCommand,
    ImportCommitResult,
    ImportPreviewResult,
    Page,
    PageRequest,
    RosterCommitCommand,
    RosterPreviewCommand,
    RowCode,
)
# The DTO's `RowOutcome` is the per-row diagnostic returned by preview and commit; the
# domain's is the write verb stored on a row. Both are needed here, so the diagnostic is
# aliased and the enum keeps the name it carries in the repository contract.
from sis.application.dto.common import RowOutcome as RowDiagnostic
from sis.application.services import (
    GradeImportService,
    GuardianImportService,
    RosterImportService,
)
from sis.config import Settings, get_settings
from sis.domain.imports import ImportBatch, ImportKind, ImportRow, ImportStatus, RowOutcome
from sis.domain.value_objects import AcademicYearCode, ClassCode, SubjectCode, TermCode
# One source of truth for what the door accepts, so the extension check here and the
# parser's own refusal cannot drift into accepting a file the layer below rejects.
from sis.infrastructure.parsers import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/v1/imports", tags=["imports"])


class ImportReportReader(Protocol):
    """Reading back a batch and its rows, stated by the route that needs it.

    One method returning both halves, on purpose: the counts and the visible page must
    come from a single transaction. Fetched separately, a registrar paging through a
    commit that is landing concurrently sees a summary saying two rows were rejected
    beside a page listing three, and spends the afternoon deciding which is lying.
    """

    def report(
        self,
        batch_id: str,
        *,
        page: PageRequest,
        outcomes: Collection[RowOutcome] | None = None,
    ) -> tuple[ImportBatch, Page[ImportRow]]:
        """The batch and one window of its rows. Raises `ImportBatchNotFound`."""


# Every route in this module writes, or reads a write's audit trail. `registrar` only,
# and scope is compared by exact equality — a `reader` key reaches none of them.
Registrar = Annotated[Principal, Depends(require_permission(Permission.IMPORTS_RUN))]
Config = Annotated[Settings, Depends(get_settings)]
RosterImports = Annotated[RosterImportService, Depends(get_roster_import_service)]
GradeImports = Annotated[GradeImportService, Depends(get_grade_import_service)]
GuardianImports = Annotated[GuardianImportService, Depends(get_guardian_import_service)]
Reports = Annotated[ImportReportReader, Depends(get_import_reports)]


# -- Reading the upload -----------------------------------------------------

_CRLF: Final[bytes] = b"\r\n"
_HEADER_END: Final[bytes] = b"\r\n\r\n"

# Parameters of a `Content-Disposition` header. A regex rather than a split on ';'
# because a filename is quoted and may legitimately contain one.
_DISPOSITION_PARAM: Final[re.Pattern[bytes]] = re.compile(
    rb'(?P<key>[a-zA-Z_*-]+)\s*=\s*(?:"(?P<quoted>[^"]*)"|(?P<bare>[^";]+))'
)


def _refuse(
    status_code: int, code: str, message: str, field: str | None = None
) -> HTTPException:
    """The same envelope `domain_errors()` produces, for refusals this layer makes itself."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "field": field},
    )


@dataclass(frozen=True, slots=True)
class _Upload:
    """One parsed multipart body: the file, and the text fields sent beside it."""

    filename: str
    content: bytes
    fields: Mapping[str, str] = dataclass_field(default_factory=dict)

    def optional(self, name: str) -> str | None:
        """A blank field is absent, not empty: a browser sends "" for an untouched input."""
        value = self.fields.get(name, "").strip()
        return value or None

    def require(self, name: str) -> str:
        value = self.optional(name)
        if value is None:
            raise _refuse(422, "invalid_value", f"{name} is required", name)
        return value

    def optional_date(self, name: str) -> date | None:
        value = self.optional(name)
        if value is None:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise _refuse(
                422, "invalid_value", f"{name} must be an ISO date (YYYY-MM-DD)", name
            ) from None


def _boundary_of(content_type: str) -> bytes:
    """The multipart boundary, or a refusal naming what the route actually accepts."""
    media, _, parameters = content_type.partition(";")
    if media.strip().lower() != "multipart/form-data":
        raise _refuse(
            415,
            "unsupported_file_type",
            "this route takes a multipart/form-data upload",
            "content-type",
        )
    for parameter in parameters.split(";"):
        key, _, value = parameter.partition("=")
        if key.strip().lower() == "boundary":
            boundary = value.strip().strip('"')
            if boundary:
                return boundary.encode("ascii", "ignore")
    raise _refuse(
        400, "unreadable_file", "the multipart body states no boundary", "content-type"
    )


def _disposition(head: bytes) -> tuple[str | None, str | None]:
    """A part's `name` and `filename`; either may be absent."""
    for line in head.split(_CRLF):
        key, separator, value = line.partition(b":")
        if not separator or key.strip().lower() != b"content-disposition":
            continue
        found = {
            match.group("key").decode("ascii", "replace").lower(): (
                match.group("quoted")
                if match.group("quoted") is not None
                else match.group("bare").strip()
            ).decode("utf-8", "replace")
            for match in _DISPOSITION_PARAM.finditer(value)
        }
        return found.get("name"), found.get("filename")
    return None, None


def _parse_multipart(body: bytes, boundary: bytes) -> _Upload:
    """Split an in-memory multipart body into its file part and its text fields.

    Only the first file part is kept. A second one is a client sending something this
    service has no route for, and quietly importing the last of them is how the wrong
    file gets applied without anybody choosing it.
    """
    fields: dict[str, str] = {}
    filename: str | None = None
    content: bytes | None = None

    for part in body.split(b"--" + boundary)[1:]:
        if part.startswith(b"--"):  # closing delimiter; the epilogue is not a part
            break
        chunk = part[len(_CRLF) :] if part.startswith(_CRLF) else part
        head, separator, payload = chunk.partition(_HEADER_END)
        if not separator:
            continue
        if payload.endswith(_CRLF):
            payload = payload[: -len(_CRLF)]
        name, part_filename = _disposition(head)
        if name is None:
            continue
        if part_filename is None:
            fields[name] = payload.decode("utf-8", "replace").strip()
        elif content is None:
            filename, content = part_filename, payload

    if content is None or not filename:
        raise _refuse(400, "unreadable_file", "the request carries no file part", "file")
    return _Upload(filename=filename, content=content, fields=fields)


def _check_extension(filename: str) -> None:
    """Refuse by extension at the door, tolerating a browser that sent a full path."""
    suffix = PurePosixPath(filename.replace("\\", "/").strip()).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise _refuse(
            415,
            "unsupported_file_type",
            f"{filename!r} is not a readable spreadsheet; save it as "
            f"{' or '.join(SUPPORTED_EXTENSIONS)}",
            "file",
        )


async def _read_upload(request: Request, *, max_bytes: int) -> _Upload:
    """Buffer a capped multipart body in memory. See the module docstring for both whys."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise _refuse(
            413,
            "upload_too_large",
            f"the upload declares {declared} bytes; the limit is {max_bytes}",
            "file",
        )
    boundary = _boundary_of(request.headers.get("content-type", ""))

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            # Abandoned mid-stream: the remainder is never read and never allocated.
            raise _refuse(
                413,
                "upload_too_large",
                f"the upload exceeds the {max_bytes} byte limit",
                "file",
            )

    upload = _parse_multipart(bytes(body), boundary)
    _check_extension(upload.filename)
    return upload


# -- Wire shapes ------------------------------------------------------------


class RowOutcomeOut(BaseModel):
    """What happened, or would happen, to one line of the uploaded file."""

    line: int = Field(
        description="The number in Excel's own gutter, 1-based and counting the header, so "
        "'row 12 is wrong' points at row 12 on the registrar's screen."
    )
    code: RowCode = Field(
        description="A closed vocabulary, safe to filter and count on. `ok` is the only "
        "non-failure."
    )
    message: str = Field(
        default="",
        description="Prose for a human. Nothing may branch on its wording — it is "
        "translated.",
    )
    field: str | None = Field(
        default=None, description="Which column failed, when a single one did."
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="The row's identifying cells, so a table can be rendered without "
        "holding the original file.",
    )

    @classmethod
    def of(cls, row: RowDiagnostic) -> "RowOutcomeOut":
        return cls(
            line=row.line,
            code=row.code,
            message=row.message,
            field=row.field,
            payload=dict(row.payload),
        )


class ImportPreviewOut(BaseModel):
    """What the registrar reads *before* anything is written."""

    batch_id: str = Field(description="Name this in the commit call; do not re-upload.")
    expires_at: datetime | None = Field(
        default=None,
        description="After this the batch cannot be committed and the file must be "
        "uploaded again — a preview taken against last week's class list must not be "
        "applied to today's.",
    )
    total_rows: int
    ok_count: int
    rejected_count: int
    totals: dict[str, int] = Field(
        description="Counts per row code, omitting codes that did not occur. These are the "
        "filter chips above the table."
    )
    rows: list[RowOutcomeOut]

    @classmethod
    def of(cls, result: ImportPreviewResult) -> "ImportPreviewOut":
        return cls(
            batch_id=result.batch_id,
            expires_at=result.expires_at,
            total_rows=result.total_rows,
            ok_count=result.ok_count,
            rejected_count=result.rejected_count,
            totals={str(code): count for code, count in result.totals.items()},
            rows=[RowOutcomeOut.of(row) for row in result.rows],
        )


class ImportCommitOut(BaseModel):
    """What was actually written, row by row — not assumed to match the preview.

    A row that previewed `ok` can commit as `changed_since_preview` because somebody else
    edited the same child in between, so this is a fresh account rather than an echo.
    """

    batch_id: str
    committed_at: datetime | None = None
    total_rows: int
    ok_count: int = Field(description="Rows that reached the database.")
    rejected_count: int
    totals: dict[str, int]
    rows: list[RowOutcomeOut]

    @classmethod
    def of(cls, result: ImportCommitResult) -> "ImportCommitOut":
        return cls(
            batch_id=result.batch_id,
            committed_at=result.committed_at,
            total_rows=result.total_rows,
            ok_count=result.ok_count,
            rejected_count=result.rejected_count,
            totals={str(code): count for code, count in result.totals.items()},
            rows=[RowOutcomeOut.of(row) for row in result.rows],
        )


class ImportRowOut(BaseModel):
    """One stored row of a batch: its fate, its diagnosis, and the cells it came from."""

    line: int
    outcome: RowOutcome = Field(description="Whether anything was, or would be, written.")
    code: str = Field(description="Why. Finer-grained than `outcome` and stable on the wire.")
    message: str | None = None
    field: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def of(cls, row: ImportRow) -> "ImportRowOut":
        return cls(
            line=row.line_number,
            outcome=row.outcome,
            code=row.code,
            message=row.message,
            field=row.field,
            payload=dict(row.payload),
        )


class ImportReportOut(BaseModel):
    """A batch and one page of its rows."""

    batch_id: str
    kind: ImportKind
    status: ImportStatus = Field(
        description="Derived against the clock on read, so a preview past its TTL reports "
        "`expired` rather than offering a commit that will fail."
    )
    actor: str = Field(description="The prefix of the key that uploaded the file.")
    created_at: datetime
    expires_at: datetime
    committed_at: datetime | None = None
    counts: dict[str, int] = Field(
        description="Rows per outcome, with every outcome present so a zero is explicit."
    )
    total: int = Field(description="Rows matching the filter, not the size of this page.")
    limit: int = Field(
        description="The window actually applied. A larger request is clamped rather than "
        "refused, so this may be smaller than the one asked for."
    )
    offset: int
    has_more: bool
    rows: list[ImportRowOut]

    @classmethod
    def of(
        cls, batch: ImportBatch, page: Page[ImportRow], *, now: datetime
    ) -> "ImportReportOut":
        return cls(
            batch_id=batch.batch_id,
            kind=batch.kind,
            status=batch.status_at(now),
            actor=batch.actor,
            created_at=batch.created_at,
            expires_at=batch.expires_at,
            committed_at=batch.committed_at,
            counts=dict(batch.counts),
            total=page.total,
            limit=page.request.limit,
            offset=page.request.offset,
            has_more=page.has_more,
            rows=[ImportRowOut.of(row) for row in page.items],
        )


# -- Routes -----------------------------------------------------------------

_PREVIEW_ERRORS = error_responses(400, 401, 403, 404, 413, 415, 422)
# 410 is listed because an expired preview is the commonest commit failure and is not a
# 404: the batch was real, and the registrar's next action is to upload the file again.
_COMMIT_ERRORS = error_responses(401, 403, 404, 409, 410, 422)


def _multipart_body(
    *, required: Mapping[str, str], optional: Mapping[str, str]
) -> dict[str, Any]:
    """Describe the upload form by hand, because the handler deliberately cannot.

    FastAPI infers a request body from the parameters it parses, and these routes parse
    none — they read the raw stream so the file never reaches a temp directory (see the
    module docstring). Written out here rather than left blank: an integrator reading the
    generated contract would otherwise see an upload route with no documented fields and
    guess at the names, and a mistyped `academic_year` instead of `academic_year_code`
    fails as "required" with nothing on the page to correct it against.
    """
    properties: dict[str, Any] = {
        "file": {
            "type": "string",
            "format": "binary",
            "description": f"The sheet itself: {' or '.join(SUPPORTED_EXTENSIONS)}.",
        }
    }
    for name, description in {**required, **optional}.items():
        properties[name] = {"type": "string", "description": description}
    return {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": ["file", *required],
                    }
                }
            },
        }
    }


@router.post(
    "/roster/preview",
    response_model=ImportPreviewOut,
    summary="Preview a roster upload",
    description="Multipart. Send the spreadsheet as `file` and `academic_year_code` as a "
    "form field; `class_code` when the file is one class per upload, and "
    "`default_starts_on` to date the placements it creates. **Nothing is written**: this "
    "returns a per-row account and a `batch_id` to commit.",
    responses=_PREVIEW_ERRORS,
    openapi_extra=_multipart_body(
        required={"academic_year_code": "The year these placements belong to."},
        optional={
            "class_code": "Target class, when the file is one class per upload. Omitted "
            "means every row must name its own class.",
            "default_starts_on": "ISO date for placements whose row states none. "
            "Defaults to the academic year's start, never to today.",
        },
    ),
)
async def preview_roster(
    request: Request, imports: RosterImports, settings: Config, caller: Registrar
) -> ImportPreviewOut:
    upload = await _read_upload(request, max_bytes=settings.max_upload_bytes)
    class_code = upload.optional("class_code")
    with domain_errors():
        result = imports.preview(
            RosterPreviewCommand(
                academic_year_code=AcademicYearCode(upload.require("academic_year_code")),
                filename=upload.filename,
                content=upload.content,
                actor=caller.prefix,
                class_code=None if class_code is None else ClassCode(class_code),
                # Absent means "the academic year's start date", decided by the service.
                # Defaulting to today here would record a November import as every child
                # having joined in November.
                default_starts_on=upload.optional_date("default_starts_on"),
            )
        )
    return ImportPreviewOut.of(result)


@router.post(
    "/roster/{batch_id}/commit",
    response_model=ImportCommitOut,
    summary="Apply a previewed roster batch",
    description="Every row is judged again against current data, so a child enrolled by "
    "somebody else since the preview is reported rather than overwritten. A second commit "
    "of the same batch is refused, which is what makes a double-clicked button safe.",
    responses=_COMMIT_ERRORS,
)
def commit_roster(
    batch_id: str, imports: RosterImports, caller: Registrar
) -> ImportCommitOut:
    with domain_errors():
        result = imports.commit(
            RosterCommitCommand(batch_id=batch_id, actor=caller.prefix)
        )
    return ImportCommitOut.of(result)


@router.post(
    "/guardians/preview",
    response_model=ImportPreviewOut,
    summary="Preview a guardians upload",
    description="Multipart. Send the spreadsheet as `file` and nothing else — every row "
    "names its own child and its own guardian, so this upload has no target to scope it "
    "to. Columns: `student_number`, `phone` (required), a guardian name in either script, "
    "and optionally `alt_phone`, `relationship`, `can_view_records`. **Nothing is "
    "written**: this returns a per-row account and a `batch_id` to commit.",
    responses=_PREVIEW_ERRORS,
    # No form fields at all, which is the concrete expression of "this upload names no
    # target". Roster takes an academic year and grades take a term; a guardians file
    # takes neither, so there is nothing here for a mis-set field to scope wrongly.
    openapi_extra=_multipart_body(required={}, optional={}),
)
async def preview_guardians(
    request: Request, imports: GuardianImports, settings: Config, caller: Registrar
) -> ImportPreviewOut:
    upload = await _read_upload(request, max_bytes=settings.max_upload_bytes)
    with domain_errors():
        result = imports.preview(
            GuardianPreviewCommand(
                filename=upload.filename,
                content=upload.content,
                actor=caller.prefix,
            )
        )
    return ImportPreviewOut.of(result)


@router.post(
    "/guardians/{batch_id}/commit",
    response_model=ImportCommitOut,
    summary="Apply a previewed guardians batch",
    description="Every row is judged again against current data, so an adult linked by "
    "somebody else since the preview is reported rather than overwritten. A second commit "
    "of the same batch is refused, which is what makes a double-clicked button safe.",
    responses=_COMMIT_ERRORS,
)
def commit_guardians(
    batch_id: str, imports: GuardianImports, caller: Registrar
) -> ImportCommitOut:
    with domain_errors():
        result = imports.commit(
            GuardianCommitCommand(batch_id=batch_id, actor=caller.prefix)
        )
    return ImportCommitOut.of(result)


@router.post(
    "/grades/preview",
    response_model=ImportPreviewOut,
    summary="Preview a marks upload",
    description="Multipart. Send the sheet as `file` and `term_code` as a form field; "
    "`subject_code` and `class_code` narrow the upload so a file sent against the wrong "
    "class is refused loudly instead of matched across the whole school. **Nothing is "
    "written.** A blank cell previews as not-graded and never as zero.",
    responses=_PREVIEW_ERRORS,
    openapi_extra=_multipart_body(
        required={"term_code": "The term these marks belong to, e.g. `2026-T1`."},
        optional={
            "subject_code": "Refuse rows naming any other subject.",
            "class_code": "Refuse rows for children who were in another class this term.",
        },
    ),
)
async def preview_grades(
    request: Request, imports: GradeImports, settings: Config, caller: Registrar
) -> ImportPreviewOut:
    upload = await _read_upload(request, max_bytes=settings.max_upload_bytes)
    subject_code = upload.optional("subject_code")
    class_code = upload.optional("class_code")
    with domain_errors():
        result = imports.preview(
            GradePreviewCommand(
                term_code=TermCode(upload.require("term_code")),
                filename=upload.filename,
                content=upload.content,
                actor=caller.prefix,
                subject_code=None if subject_code is None else SubjectCode(subject_code),
                class_code=None if class_code is None else ClassCode(class_code),
            )
        )
    return ImportPreviewOut.of(result)


@router.post(
    "/grades/{batch_id}/commit",
    response_model=ImportCommitOut,
    summary="Apply a previewed marks batch",
    description="Re-validated row by row: a term closed, a subject retired or a child "
    "transferred since the preview is reported as changed rather than written. Marks are "
    "stored exactly as stated — nothing is averaged, weighted or rescaled.",
    responses=_COMMIT_ERRORS,
)
def commit_grades(
    batch_id: str, imports: GradeImports, caller: Registrar
) -> ImportCommitOut:
    with domain_errors():
        result = imports.commit(GradeCommitCommand(batch_id=batch_id, actor=caller.prefix))
    return ImportCommitOut.of(result)


@router.get(
    "/{batch_id}",
    response_model=ImportReportOut,
    summary="A batch and one page of its per-row outcomes",
    description="Filter with `outcome` (repeatable) to read the fourteen rejected rows of "
    "a twelve-hundred-row file without fetching the rest. `limit` above the ceiling is "
    "clamped, and the applied window is echoed back.",
    responses=error_responses(401, 403, 404, 422),
)
def read_import_report(
    batch_id: str,
    reports: Reports,
    caller: Registrar,
    limit: Annotated[int, Query(ge=1)] = PageRequest.DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
    outcome: Annotated[
        list[RowOutcome] | None,
        Query(description="Repeatable, e.g. `?outcome=rejected&outcome=updated`."),
    ] = None,
) -> ImportReportOut:
    page_request = PageRequest(limit=limit, offset=offset)
    with domain_errors():
        batch, page = reports.report(
            batch_id,
            page=page_request,
            outcomes=tuple(outcome) if outcome else None,
        )
    return ImportReportOut.of(batch, page, now=datetime.now(UTC))
