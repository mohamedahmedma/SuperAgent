"""Roster import: preview what a file would do, then apply exactly that.

Two entry points and one evaluator between them. `preview()` reads the file, judges every
row against the structure currently on file, writes an `ImportBatch` and its rows, and
**creates no student and no placement**. `commit()` loads that batch back and runs the
*same* evaluator again, against state it re-reads, then writes.

Re-running the evaluator rather than replaying the preview's verdict is the design
decision worth the words. Between the two HTTP requests another registrar can create the
child, move her to another class, or delete the section the file names; a commit that
trusted the preview would write against a world that no longer exists and report success
for it. So the preview's verdict is treated as a *proposal*, never as an instruction: the
only thing carried forward is the row's parsed payload — what the human actually read on
screen — and the resolved placement start, frozen so that committing in November does not
silently record every child as having joined in November. Everything identity-bearing is
looked up again.

Invariant 2 lives in `_evaluate`. A child already placed elsewhere is never given a second
open placement; her current one is closed the day before the new one begins and a new row
is opened. Overwriting the old row's class instead would be a history rewrite: her Term 1
marks were earned in 3A, and after a March transfer a single mutable column can only say
3A or 3B, never both truthfully.

Invariant 4 is why a bad row is a value. Only an `ImportFailure` — an unreadable file, an
expired batch — kills a batch; a row that names an unknown class becomes one red line and
the other 299 still import.

Ports only. No sqlalchemy, no fastapi, no `sis.config`: the constructor takes a unit of
work factory, a parser, a clock and a TTL, so every rule here is testable with fakes and
no database.
"""
import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from sis.application.dto import (
    ImportCommitResult,
    ImportPreviewResult,
    ParsedRosterRow,
    RosterCommitCommand,
    RosterPreviewCommand,
    RowCode,
    RowOutcome,
)
from sis.application.ports.parsers import RosterFileParser
from sis.application.ports.repositories import ClassSectionKey
from sis.application.ports.unit_of_work import UnitOfWork
from sis.domain.errors import (
    ImportBatchAlreadyCommitted,
    ImportBatchExpired,
    ImportBatchNotFound,
    ImportContentMismatch,
    ImportFailure,
    SisError,
    UnknownReference,
    UploadTooLarge,
)
from sis.domain.imports import (
    ImportBatch,
    ImportKind,
    ImportRow,
    # The domain's `RowOutcome` is the *write verb* — created, updated, unchanged — while
    # the DTO's is the row report the API returns. Both names are correct in their own
    # module and neither may be renamed from here, so the storage one is aliased and the
    # reported one keeps the name it has in the public return types.
    RowOutcome as StoredOutcome,
    tally,
)
from sis.domain.people import ClassEnrolment, Gender, Student
from sis.domain.value_objects import AcademicYearCode, ClassCode, StudentNumber

__all__ = ["BATCH_FAILURE_ROW_CODE", "RosterImportService"]


# The three batch-level failures also exist in the row vocabulary, so an API layer can
# render "nothing was written, and here is why" with the one table renderer it already has
# for row outcomes instead of a second code path for HTTP errors.
BATCH_FAILURE_ROW_CODE: Mapping[type[ImportFailure], RowCode] = {
    ImportBatchExpired: RowCode.BATCH_EXPIRED,
    ImportBatchAlreadyCommitted: RowCode.BATCH_ALREADY_COMMITTED,
    ImportContentMismatch: RowCode.CHANGED_SINCE_PREVIEW,
}

# `ClassEnrolment.ends_on` is the child's **last day** in the class, not the day after, so
# a placement that begins on the 12th closes the previous one on the 11th. Ending it on
# the 12th would put her in two classes on that one day — and a report cut on exactly that
# date is where the resulting ambiguity surfaces.
_ONE_DAY = timedelta(days=1)

# Payload keys. Named once because `commit()` reads back what `preview()` wrote, and a
# typo in one of the two halves is a batch that previews perfectly and commits nothing.
_K_NUMBER = "student_number"
_K_NAME_AR = "full_name_ar"
_K_NAME_EN = "full_name_en"
_K_YEAR = "academic_year_code"
_K_CLASS = "class_code"
_K_STARTS = "starts_on"
_K_GENDER = "gender"


@dataclass(frozen=True, slots=True)
class _Assertion:
    """What one row claims, resolved but not yet judged.

    Built from a parsed row at preview and from a stored payload at commit, which is what
    lets one evaluator serve both. `class_code` is already resolved from the row or the
    upload's target, and `starts_on` from the academic year, so neither path can resolve
    them differently.
    """

    line: int
    number: StudentNumber
    name_ar: str
    name_en: str
    year_code: AcademicYearCode
    class_code: ClassCode
    starts_on: date
    #: Defaulted, so a caller written against the previous shape still builds — and gets
    #: the value that means "the school has not said", which is the truthful answer for
    #: any row whose sheet had no such column.
    gender: Gender = Gender.UNSPECIFIED

    @property
    def payload(self) -> dict[str, object]:
        """The row as the registrar sees it, and as commit reads it back."""
        return {
            _K_NUMBER: str(self.number),
            _K_NAME_AR: self.name_ar,
            _K_NAME_EN: self.name_en,
            _K_YEAR: str(self.year_code),
            _K_CLASS: str(self.class_code),
            _K_STARTS: self.starts_on.isoformat(),
            # In the payload, not just on the row, because commit rebuilds the assertion
            # from this dict rather than from the file: a field missing here is a field
            # that previews correctly and is silently dropped on the way to being stored.
            _K_GENDER: str(self.gender),
        }


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """Everything the evaluator is allowed to know about stored state.

    Loaded in three bulk queries for the whole file, never per row: a 600-child roster that
    asks the database once per line is 1800 round trips and a registrar watching a spinner.
    Holding it as a value also means the evaluator is a pure function of (assertion,
    snapshot), so a unit test of "a transfer closes the old placement" builds two dicts.
    """

    sections: Mapping[ClassSectionKey, object]
    students: Mapping[str, Student]
    enrolments: Mapping[str, Sequence[ClassEnrolment]]


@dataclass(frozen=True, slots=True)
class _Plan:
    """One row's verdict *and* the writes it implies — decided once, applied later.

    Preview keeps the verdict and throws the writes away; commit keeps both. That is the
    whole of "no student is created at preview": the two paths differ by whether `_apply`
    is called, not by which rules ran.
    """

    line: int
    payload: Mapping[str, object]
    code: RowCode
    outcome: StoredOutcome
    message: str = ""
    field: str | None = None
    student: Student | None = None
    enrolment: ClassEnrolment | None = None
    #: Last day of the placement being superseded, when this row is a transfer.
    close_open_on: date | None = None
    #: Set when the previewed verdict and the re-derived one disagree on identity.
    superseded: bool = False


def _norm_name(raw: str) -> str:
    """Collapse a name for comparison only. Never stored in this form."""
    return " ".join(raw.split()).casefold()


def _names_collide(stated: _Assertion, stored: Student) -> bool:
    """Does this number already belong to a *different* child?

    Compared only where both sides state a name in the same script: an Arabic-first export
    with no English column must not read as a conflict against a record that has both.

    Refusing the row rather than overwriting is deliberate, and it does cost the registrar
    a rejected line when she meant to fix a spelling. That trade is the right way round.
    A rejected row is visible and recoverable in one edit; a silent overwrite reassigns a
    student number — the key every grade, enrolment and `records/` guardian link hangs
    from — to a different human being, and nothing anywhere reports it. The message names
    what is on file so the fix takes a glance.
    """
    pairs = ((stated.name_ar, stored.full_name_ar), (stated.name_en, stored.full_name_en))
    return any(a and b and _norm_name(a) != _norm_name(b) for a, b in pairs)


def _stored_row(plan: _Plan) -> ImportRow:
    """Persist the fine-grained `RowCode` in `code` and the write verb in `outcome`.

    Two vocabularies on purpose. `outcome` answers "was anything written", which is what
    `ImportBatch.counts` tallies; `code` answers "why", which is what the registrar filters
    on — and `UnknownReference` alone cannot say whether the missing thing was a class or a
    student, so the code the importer chose has to survive the round trip. Constructed
    directly rather than through `ImportRow.accepted`, which would overwrite `code` with
    the verb and lose the diagnosis.
    """
    return ImportRow(
        line_number=plan.line,
        payload=plan.payload,
        outcome=plan.outcome,
        code=plan.code.value,
        message=plan.message or None,
        field=plan.field,
    )


def _reported(plan: _Plan) -> RowOutcome:
    return RowOutcome(
        line=plan.line,
        code=plan.code,
        message=plan.message,
        payload=plan.payload,
        field=plan.field,
    )


def _restore(row: ImportRow) -> RowOutcome:
    """Re-report a stored row without re-judging it — how a rejection survives to commit."""
    try:
        code = RowCode(row.code)
    except ValueError:
        # A code this build has no member for means the batch predates the current
        # vocabulary. Reporting it as `OK` would claim a write that never happened.
        code = RowCode.CHANGED_SINCE_PREVIEW
    return RowOutcome(
        line=row.line_number,
        code=code,
        message=row.message or "",
        payload=row.payload,
        field=row.field,
    )


def _diagnostic_row(report: RowOutcome) -> ImportRow:
    """Store a parser diagnostic beside the rows it sits between, in file order."""
    return ImportRow(
        line_number=report.line,
        payload=report.payload,
        outcome=StoredOutcome.REJECTED,
        code=report.code.value,
        message=report.message or None,
        field=report.field,
    )


class RosterImportService:
    """Preview and commit roster uploads. Depends on ports only."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        parser: RosterFileParser,
        *,
        preview_ttl: timedelta,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_upload_bytes: int | None = None,
        batch_ids: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        """A *factory* of units of work, because preview and commit are two transactions.

        The service outlives both: it is composed once and handles many requests, so a
        single injected unit of work would share one transaction between two registrars.
        The clock and the TTL arrive here rather than being read from `sis.config`, which
        no service may import — a use case that reads the environment cannot be tested
        without arranging it, and expiry cannot be tested at all without sleeping.
        """
        self._uow_factory = uow_factory
        self._parser = parser
        self._preview_ttl = preview_ttl
        self._clock = clock
        self._max_upload_bytes = max_upload_bytes
        self._batch_ids = batch_ids

    # -- preview ------------------------------------------------------------

    def preview(self, command: RosterPreviewCommand) -> ImportPreviewResult:
        """Judge every row and write a batch. Not one student is created here.

        Raises `UploadTooLarge`, `UnsupportedFileType` or `UnreadableImportFile` when there
        was never a file worth reporting on, and `UnknownReference` when the academic year
        itself is missing — a whole-batch fact that would otherwise render as every single
        row failing for the same reason.
        """
        if self._max_upload_bytes is not None and len(command.content) > self._max_upload_bytes:
            raise UploadTooLarge(
                f"{command.filename} is larger than the {self._max_upload_bytes} byte limit",
                field="content",
            )
        parsed = self._parser.parse(command.content, command.filename)
        now = self._clock()

        with self._uow_factory() as uow:
            year = uow.academic_years.get(command.academic_year_code)
            if year is None:
                raise UnknownReference(
                    f"academic year {command.academic_year_code} is not on file",
                    field="academic_year_code",
                )
            default_start = command.default_starts_on or year.starts_on
            assertions, unresolved = self._assert_rows(parsed.rows, command, default_start)
            snapshot = self._load(uow, assertions)

            seen: dict[str, int] = {}
            plans = [self._evaluate(a, snapshot, seen) for a in assertions]

            reports = [*parsed.diagnostics, *unresolved]
            reports.extend(_reported(plan) for plan in plans)
            stored = [
                *(_diagnostic_row(r) for r in parsed.diagnostics),
                *(_diagnostic_row(r) for r in unresolved),
                *(_stored_row(plan) for plan in plans),
            ]
            stored.sort(key=lambda row: row.line_number)
            reports.sort(key=lambda report: report.line)

            batch = ImportBatch(
                batch_id=self._batch_ids(),
                kind=ImportKind.ROSTER,
                content_hash=hashlib.sha256(command.content).hexdigest(),
                actor=command.actor,
                created_at=now,
                expires_at=now + self._preview_ttl,
                counts=tally(stored),
            )
            uow.imports.add(batch, stored)
            uow.commit()

        return ImportPreviewResult.from_rows(
            batch.batch_id, reports, expires_at=batch.expires_at
        )

    def _assert_rows(
        self,
        rows: Iterable[ParsedRosterRow],
        command: RosterPreviewCommand,
        default_start: date,
    ) -> tuple[list[_Assertion], list[RowOutcome]]:
        """Resolve each row's class and start date, or reject it for naming neither.

        A row that names no class in a file uploaded without a target class is rejected
        rather than guessed at: guessing puts a real child in a real class that nobody
        chose, and every layer below reports that as a correct row.
        """
        assertions: list[_Assertion] = []
        unresolved: list[RowOutcome] = []
        for row in rows:
            payload: dict[str, object] = {
                _K_NUMBER: str(row.student_number),
                _K_NAME_AR: row.full_name_ar,
                _K_NAME_EN: row.full_name_en,
            }
            if not row.has_name:
                unresolved.append(
                    RowOutcome(
                        line=row.line_number,
                        code=RowCode.INVALID_NAME,
                        message="a student needs a name in Arabic or English",
                        payload=payload,
                        field=_K_NAME_AR,
                    )
                )
                continue
            class_code = row.class_code or command.class_code
            if class_code is None:
                unresolved.append(
                    RowOutcome(
                        line=row.line_number,
                        code=RowCode.UNKNOWN_CLASS,
                        message="this row names no class and the upload names no target class",
                        payload=payload,
                        field=_K_CLASS,
                    )
                )
                continue
            assertions.append(
                _Assertion(
                    line=row.line_number,
                    number=row.student_number,
                    name_ar=row.full_name_ar,
                    name_en=row.full_name_en,
                    year_code=command.academic_year_code,
                    class_code=class_code,
                    starts_on=row.starts_on or default_start,
                    gender=row.gender,
                )
            )
        return assertions, unresolved

    # -- commit -------------------------------------------------------------

    def commit(self, command: RosterCommitCommand) -> ImportCommitResult:
        """Re-judge the previewed rows against current state, then write them.

        Raises `ImportBatchNotFound`, `ImportBatchExpired` or `ImportBatchAlreadyCommitted`
        — three distinct codes, because the registrar's next action differs for each and a
        single "cannot commit" tells her none of them. The second commit of a batch is
        refused rather than re-applied, which is what makes a double-clicked button safe:
        the first call's outcome stands and the second is told so.

        Every write in this method lands in one transaction. A commit-per-row loop that
        failed on row 140 would leave 139 placements written and 61 missing, a half-imported
        roster that no screen reports as broken because each row it did write looks correct.
        """
        now = self._clock()
        batch_id = str(command.batch_id)

        with self._uow_factory() as uow:
            batch = uow.imports.get(batch_id)
            if batch is None:
                raise ImportBatchNotFound(f"no import batch {batch_id}", field="batch_id")
            if batch.kind is not ImportKind.ROSTER:
                raise ImportContentMismatch(
                    f"batch {batch_id} is a {batch.kind.value} import, not a roster",
                    field="batch_id",
                )
            # The hash is passed back unchanged: commit names a batch rather than
            # re-uploading, so there is no second file to disagree with the first. The
            # guard that matters here is re-evaluation below, not the hash.
            batch.assert_committable(now, batch.content_hash)

            previewed = uow.imports.list_rows(batch_id)
            assertions, replayed = self._replay(previewed)
            approved = {
                row.line_number: row.outcome for row in previewed if row.is_written
            }
            snapshot = self._load(uow, assertions)

            seen: dict[str, int] = {}
            plans = [
                self._reconcile(self._evaluate(a, snapshot, seen), approved)
                for a in assertions
            ]
            self._apply(uow, plans)

            stored = [*replayed, *(_stored_row(plan) for plan in plans)]
            stored.sort(key=lambda row: row.line_number)
            uow.imports.replace_rows(batch_id, stored)
            uow.imports.save(batch.committed(now))
            uow.commit()

        reports = sorted((_restore(row) for row in stored), key=lambda r: r.line)
        return ImportCommitResult.from_rows(batch_id, reports, committed_at=now)

    def _replay(
        self, rows: Sequence[ImportRow]
    ) -> tuple[list[_Assertion], list[ImportRow]]:
        """Rebuild an assertion per writable row; carry every other row through untouched.

        Rows the registrar saw rejected stay rejected with the code she read. Only rows that
        proposed a write are re-judged, because those are the only ones whose verdict can
        have been invalidated by someone else's edit.
        """
        assertions: list[_Assertion] = []
        carried: list[ImportRow] = []
        for row in rows:
            if not row.is_written:
                carried.append(row)
                continue
            try:
                assertions.append(
                    _Assertion(
                        line=row.line_number,
                        number=StudentNumber(row.payload[_K_NUMBER]),
                        name_ar=str(row.payload.get(_K_NAME_AR, "")),
                        name_en=str(row.payload.get(_K_NAME_EN, "")),
                        year_code=AcademicYearCode(row.payload[_K_YEAR]),
                        class_code=ClassCode(row.payload[_K_CLASS]),
                        starts_on=date.fromisoformat(str(row.payload[_K_STARTS])),
                        # `.get`, unlike the required keys above: a batch previewed
                        # before this column existed is still a valid batch, and must
                        # commit rather than be reported as no longer rebuilding.
                        gender=Gender(str(row.payload.get(_K_GENDER, Gender.UNSPECIFIED))),
                    )
                )
            except (SisError, KeyError, TypeError, ValueError) as exc:
                # A payload that no longer rebuilds is, by definition, not the row that was
                # previewed. Recorded rather than raised so the rest of the batch still
                # lands (invariant 4).
                carried.append(
                    ImportRow(
                        line_number=row.line_number,
                        payload=row.payload,
                        outcome=StoredOutcome.REJECTED,
                        code=RowCode.CHANGED_SINCE_PREVIEW.value,
                        message=f"this row can no longer be read as it was previewed ({exc})",
                    )
                )
        return assertions, carried

    # -- shared judgement ---------------------------------------------------

    def _reconcile(self, plan: _Plan, approved: Mapping[int, StoredOutcome]) -> _Plan:
        """Refuse a row whose *identity* moved between the preview and this commit.

        A row the registrar approved as `created` that now resolves to a child already on
        file is not the row she read. Applying it anyway would attach this file's names and
        placement to an existing student nobody reviewed — and it is exactly what happens
        when the same roster is uploaded twice in two tabs, so refusing here is also what
        stops a second batch from re-applying the first one's work.

        Every other divergence has already produced its own specific code inside
        `_evaluate`, which is the point of re-running it: `changed_since_preview` is the
        residue, not the general answer.
        """
        if plan.superseded and approved.get(plan.line) is StoredOutcome.CREATED:
            return replace(
                plan,
                code=RowCode.CHANGED_SINCE_PREVIEW,
                outcome=StoredOutcome.REJECTED,
                message=(
                    f"student {plan.payload.get(_K_NUMBER)} was created by someone else "
                    "since this file was previewed; upload it again to see the current "
                    "outcome"
                ),
                field=_K_NUMBER,
                student=None,
                enrolment=None,
                close_open_on=None,
            )
        return plan

    def _load(self, uow: UnitOfWork, assertions: Sequence[_Assertion]) -> _Snapshot:
        """Three bulk reads for the whole file, whichever path is running.

        Each row's academic year comes from the row itself, so a commit is never re-scoped
        to a year the registrar did not review.
        """
        keys = {(str(a.year_code), str(a.class_code)) for a in assertions}
        numbers = [a.number for a in assertions]
        return _Snapshot(
            sections=uow.class_sections.get_many(keys),
            students=uow.students.get_many(numbers),
            enrolments=uow.enrolments.list_for_students(numbers),
        )

    def _evaluate(
        self, stated: _Assertion, snapshot: _Snapshot, seen: dict[str, int]
    ) -> _Plan:
        """The whole rule set for one row. Pure: same inputs, same verdict, no clock, no I/O.

        Order is deliberate — a duplicate line is reported as a duplicate even when it also
        names an unknown class, because the registrar deletes it either way and two codes
        for one line make the tally lie.
        """
        number = str(stated.number)
        payload = stated.payload

        first_seen = seen.get(number)
        if first_seen is not None:
            return _Plan(
                line=stated.line,
                payload=payload,
                code=RowCode.DUPLICATE_IN_FILE,
                outcome=StoredOutcome.REJECTED,
                message=f"student {number} is already on line {first_seen} of this file",
                field=_K_NUMBER,
            )
        seen[number] = stated.line

        key = (str(stated.year_code), str(stated.class_code))
        if key not in snapshot.sections:
            return _Plan(
                line=stated.line,
                payload=payload,
                code=RowCode.UNKNOWN_CLASS,
                outcome=StoredOutcome.REJECTED,
                message=f"class {stated.class_code} does not exist in {stated.year_code}",
                field=_K_CLASS,
            )

        existing = snapshot.students.get(number)
        if existing is not None and _names_collide(stated, existing):
            return _Plan(
                line=stated.line,
                payload=payload,
                code=RowCode.DUPLICATE_EXISTING,
                outcome=StoredOutcome.REJECTED,
                message=(
                    f"student number {number} is already on file for "
                    f"{existing.full_name_en or existing.full_name_ar}"
                ),
                field=_K_NUMBER,
            )

        student = self._merged_student(stated, existing)
        placements = list(snapshot.enrolments.get(number, ()))
        candidate = ClassEnrolment(
            student_number=stated.number,
            academic_year_code=stated.year_code,
            class_code=stated.class_code,
            starts_on=stated.starts_on,
        )

        # Already open in this very class: assert nothing. Re-opening the placement with
        # today's start would rewrite when she joined, and closing-then-reopening the same
        # class would invent a transfer that never happened.
        open_now = next((p for p in placements if p.is_open), None)
        if open_now is not None and (
            str(open_now.academic_year_code),
            str(open_now.class_code),
        ) == key:
            return _Plan(
                line=stated.line,
                payload=payload,
                code=RowCode.OK,
                outcome=(
                    StoredOutcome.UPDATED if student != existing else StoredOutcome.UNCHANGED
                ),
                student=student if student != existing else None,
                superseded=existing is not None,
            )

        # The same placement already recorded and since closed — a second commit of the
        # same file. Writing it again would be harmless (the repository is keyed on the
        # start date) but reporting it as created would not be.
        if any(
            (str(p.academic_year_code), str(p.class_code), p.starts_on)
            == (key[0], key[1], stated.starts_on)
            for p in placements
        ):
            return _Plan(
                line=stated.line,
                payload=payload,
                code=RowCode.OK,
                outcome=StoredOutcome.UNCHANGED,
                superseded=existing is not None,
            )

        close_on: date | None = None
        remaining = placements
        if open_now is not None:
            # Invariant 2: the old placement is closed, never edited to point at the new
            # class. Both rows stay true, and "which class in Term 1" keeps its answer.
            close_on = stated.starts_on - _ONE_DAY
            if close_on < open_now.starts_on:
                return _Plan(
                    line=stated.line,
                    payload=payload,
                    code=RowCode.ALREADY_ENROLLED_ELSEWHERE,
                    outcome=StoredOutcome.REJECTED,
                    message=(
                        f"student {number} is in {open_now.class_code} from "
                        f"{open_now.starts_on}, which is after this placement begins; "
                        "correct the start date or end that placement first"
                    ),
                    field=_K_STARTS,
                )
            remaining = [
                p if p is not open_now else ClassEnrolment(
                    student_number=open_now.student_number,
                    academic_year_code=open_now.academic_year_code,
                    class_code=open_now.class_code,
                    starts_on=open_now.starts_on,
                    ends_on=close_on,
                )
                for p in placements
            ]

        clash = next((p for p in remaining if p.conflicts_with(candidate)), None)
        if clash is not None:
            return _Plan(
                line=stated.line,
                payload=payload,
                code=RowCode.ALREADY_ENROLLED_ELSEWHERE,
                outcome=StoredOutcome.REJECTED,
                message=(
                    f"student {number} was already in {clash.class_code} from "
                    f"{clash.starts_on} to {clash.ends_on}, overlapping this placement"
                ),
                field=_K_STARTS,
            )

        return _Plan(
            line=stated.line,
            payload=payload,
            code=RowCode.OK,
            outcome=StoredOutcome.CREATED if existing is None else StoredOutcome.UPDATED,
            student=student,
            enrolment=candidate,
            close_open_on=close_on,
            superseded=existing is not None,
        )

    def _merged_student(self, stated: _Assertion, existing: Student | None) -> Student:
        """Fill in a detail the record lacks; never replace one it has.

        A file with an empty English column must not blank the English name a registrar
        typed by hand, and `_names_collide` has already refused the case where the two
        disagree — so what reaches here is only ever an addition.

        Gender follows the same rule for the same reason, and the rule matters more here
        because the blank is so much commoner: a sheet with no gender column at all would
        otherwise reset every child it re-imports back to `unspecified`, quietly undoing
        an earlier upload that did carry it. A registrar correcting a recorded sex states
        it, which lands as an `UNSPECIFIED` existing value only if it was never set.
        """
        if existing is None:
            return Student(
                student_number=stated.number,
                full_name_ar=stated.name_ar,
                full_name_en=stated.name_en,
                gender=stated.gender,
            )
        return Student(
            student_number=existing.student_number,
            full_name_ar=existing.full_name_ar or stated.name_ar,
            full_name_en=existing.full_name_en or stated.name_en,
            is_active=existing.is_active,
            gender=(
                stated.gender
                if stated.gender is not Gender.UNSPECIFIED
                else existing.gender
            ),
        )

    def _apply(self, uow: UnitOfWork, plans: Sequence[_Plan]) -> None:
        """Write the accepted plans in bulk, students before the placements that name them."""
        students = [p.student for p in plans if p.student is not None]
        if students:
            uow.students.upsert_many(students)

        # Closing is per-student because a transfer needs the child's own open row; there
        # are as many calls as there are transfers, not as there are rows.
        for plan in plans:
            if plan.close_open_on is not None and plan.enrolment is not None:
                uow.enrolments.close_open_enrolment(
                    StudentNumber(str(plan.enrolment.student_number)),
                    ends_on=plan.close_open_on,
                )

        enrolments = [p.enrolment for p in plans if p.enrolment is not None]
        if enrolments:
            uow.enrolments.upsert_many(enrolments)
