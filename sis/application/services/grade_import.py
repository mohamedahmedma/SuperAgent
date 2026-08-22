"""Importing stated marks: preview every row, then commit the ones a human approved.

The same two-step shape as the roster import, for student x subject x term — and for the
same reason. A marks file that resolves the wrong child, or the wrong class, does not
crash: it writes a well-formed grade onto somebody else's report card, and every layer
below the registrar reading the screen agrees it is fine. The only place that mistake is
catchable is a preview, before the write.

Two rules dominate this module.

**A blank cell is `None`, and `None` is never `0.0`** (invariant 1). It survives here in
two forms: a blank is stored as an ungraded row rather than as a zero, and a blank in an
uploaded file does not erase a mark already on file (see `_assess_row`). Both are the same
refusal — this service never invents a figure nobody stated.

**A mark is filed under the class the child was in as at the term** (invariant 2), which
is resolved through `resolve_sections_for_term` in `queries.py` rather than reimplemented
here. The import and the report card must agree about which class a Term 1 mark belongs
to, and the only way to guarantee that is one implementation of the rule.

Ports only. No sqlalchemy, no fastapi, no `sis.config`: the clock, the id generator, the
TTL and the size limit are all injected, so every rule below is testable with fake
repositories, a literal `b"..."` and a fixed `datetime`.
"""
import hashlib
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from sis.application.dto import (
    GradeCommitCommand,
    GradePreviewCommand,
    ImportCommitResult,
    ImportPreviewResult,
    ParsedGradeRow,
    RowCode,
)
# `RowOutcome` names two different things: a per-row diagnostic DTO and the domain enum of
# fates. Both are needed here, so the DTO is aliased. Importing either unaliased is how a
# row's `code` ends up holding "rejected" and its outcome holding a `RowCode`.
from sis.application.dto.common import RowOutcome as RowDiagnostic
from sis.application.ports.parsers import GradeFileParser
from sis.application.ports.repositories import GradeKey
from sis.application.ports.unit_of_work import UnitOfWork
from sis.application.services.queries import resolve_sections_for_term
from sis.domain.errors import (
    ImportBatchNotFound,
    InvalidPercentage,
    SisError,
    TermClosed,
    UploadTooLarge,
)
from sis.domain.grades import SubjectGrade
from sis.domain.imports import (
    ImportBatch,
    ImportKind,
    ImportRow,
    RowOutcome,
    tally,
)
from sis.domain.people import Student
from sis.domain.structure import ClassSection, Subject, Term
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    Percentage,
    StudentNumber,
    SubjectCode,
    TermCode,
)

__all__ = ["GradeImportService"]


def _utc_now() -> datetime:
    """Timezone-aware by construction — `ImportBatch` rejects a naive one, and rightly."""
    return datetime.now(UTC)


def _new_batch_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class _Request:
    """One row to validate: the parsed figures plus the class the caller expects.

    `expected_class` is what makes a commit able to notice that the world moved. At
    preview it holds the class the upload was targeted at, if any; at commit it holds the
    class the preview resolved, so a child who transferred in between resolves to a
    different section and the row is reported as changed instead of being written under a
    class the registrar never saw.
    """

    row: ParsedGradeRow
    expected_class: ClassCode | None = None


@dataclass(frozen=True, slots=True)
class _Assessment:
    """The verdict on one row, before anything is written.

    Carries the `SubjectGrade` it would write rather than a flag, so that commit writes
    the object validation already accepted. Rebuilding it after the check would be a
    second construction path, and the one that skips a check is always the second one.
    """

    line_number: int
    payload: dict[str, object]
    code: RowCode
    outcome: RowOutcome
    message: str = ""
    field: str | None = None
    grade: SubjectGrade | None = None

    @property
    def is_ok(self) -> bool:
        return self.code is RowCode.OK

    def as_import_row(self) -> ImportRow:
        if self.is_ok:
            return ImportRow.accepted(
                self.line_number, self.payload, self.outcome, message=self.message or None
            )
        # Built directly rather than through `ImportRow.rejected`, which takes a
        # `SisError`: several rejections here are decisions this service makes with no
        # error object behind them (a duplicate line, a class that no longer matches).
        # The code is still a `RowCode` member, never a hand-typed string.
        return ImportRow(
            line_number=self.line_number,
            payload=self.payload,
            outcome=RowOutcome.REJECTED,
            code=self.code.value,
            message=self.message,
            field=self.field,
        )

    def as_diagnostic(self) -> RowDiagnostic:
        return RowDiagnostic(
            line=self.line_number,
            code=self.code,
            message=self.message,
            payload=self.payload,
            field=self.field,
        )


class GradeImportService:
    """Preview and commit a marks upload, one stated figure per student, subject and term.

    Nothing here averages, weights or drops a lowest mark (invariant 5). A grade this
    service writes is one a teacher can find in the sheet they uploaded.
    """

    DEFAULT_PREVIEW_TTL: ClassVar[timedelta] = timedelta(minutes=30)

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        parser: GradeFileParser,
        *,
        clock: Callable[[], datetime] = _utc_now,
        new_batch_id: Callable[[], str] = _new_batch_id,
        preview_ttl: timedelta | None = None,
        max_upload_bytes: int | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._parser = parser
        self._clock = clock
        self._new_batch_id = new_batch_id
        self._preview_ttl = preview_ttl or self.DEFAULT_PREVIEW_TTL
        self._max_upload_bytes = max_upload_bytes

    # -- Step one: preview -------------------------------------------------

    def preview(self, command: GradePreviewCommand) -> ImportPreviewResult:
        """Validate every row against stored structure and write only an `ImportBatch`.

        Not one grade is touched here. The registrar sees per-row outcomes first, and the
        batch is the thing commit replays — so what is written later is what was reviewed,
        rather than whatever the client happens to upload the second time.
        """
        self._check_size(command.content)
        # Raises `UnsupportedFileType` / `UnreadableImportFile` only: those kill the batch
        # because there were never any rows. A row it could not read is a diagnostic.
        parsed = self._parser.parse(command.content, command.filename)
        now = self._clock()

        requests: list[_Request] = []
        assessments: list[_Assessment] = []
        for row in parsed.rows:
            mismatch = self._subject_guard(row, command.subject_code)
            if mismatch is not None:
                assessments.append(mismatch)
                continue
            # The file's term fills in for every row that does not name one, which is the
            # common single-term upload. Applied here so the batch payload records the
            # term explicitly and commit never has to re-derive it from a command it no
            # longer has.
            resolved = row if row.term_code is not None else replace(
                row, term_code=command.term_code
            )
            requests.append(_Request(row=resolved, expected_class=command.class_code))

        with self._uow_factory() as uow:
            assessments.extend(self._assess(uow, requests))
            rows = self._ordered_rows(parsed.diagnostics, assessments)
            batch = ImportBatch(
                batch_id=self._new_batch_id(),
                kind=ImportKind.GRADES,
                content_hash=self._digest(command.content),
                actor=command.actor,
                created_at=now,
                expires_at=now + self._preview_ttl,
                counts=tally(rows),
            )
            uow.imports.add(batch, rows)
            uow.commit()

        return ImportPreviewResult.from_rows(
            batch.batch_id,
            self._ordered_diagnostics(parsed.diagnostics, assessments),
            expires_at=batch.expires_at,
        )

    # -- Step two: commit --------------------------------------------------

    def commit(self, command: GradeCommitCommand) -> ImportCommitResult:
        """Write the previewed rows, re-validating each one against current data.

        Idempotent in two independent ways, because one is not enough. `assert_committable`
        refuses a second commit outright, which is what a double-clicked button gets. And
        the writes themselves are upserts on `(student, subject, term)`, so a batch that is
        somehow replayed restates the same figures rather than giving a child a second
        maths mark for Term 1.

        Every row is validated again rather than trusted from the preview. Between the two
        requests a child can transfer, a subject can be retired, a term can be closed — and
        a commit that replayed the preview's verdicts would report those rows as written
        when they were not.
        """
        batch_id = str(command.batch_id)
        now = self._clock()
        with self._uow_factory() as uow:
            batch = uow.imports.get(batch_id)
            if batch is None:
                raise ImportBatchNotFound(f"no import batch {batch_id}", field="batch_id")
            # The command carries no file, so the hash the batch already holds is what is
            # checked against: the status and expiry gates are the live ones here. Drift
            # between preview and commit is caught by re-validating the rows below, not by
            # the hash, which only ever guards a re-uploaded file.
            batch.assert_committable(now, batch.content_hash)

            stored = uow.imports.list_rows(batch_id)
            requests: list[_Request] = []
            previewed: list[ImportRow] = []
            carried: list[_Assessment] = []
            for stored_row in stored:
                if stored_row.is_rejected:
                    # A row rejected at preview stays rejected, with its original reason.
                    carried.append(self._carried(stored_row))
                    continue
                requests.append(self._request_from_payload(stored_row))
                previewed.append(stored_row)

            assessments = [
                self._as_changed_if_failed(before, after)
                for before, after in zip(
                    previewed, self._assess(uow, requests), strict=True
                )
            ]
            written = self._write(uow, assessments)
            final = carried + written
            rows = self._ordered_rows((), final)
            uow.imports.replace_rows(batch_id, rows)
            uow.imports.save(batch.committed(now))
            uow.commit()

        return ImportCommitResult.from_rows(
            batch_id, self._ordered_diagnostics((), final), committed_at=now
        )

    # -- Validation core ---------------------------------------------------

    def _assess(
        self, uow: UnitOfWork, requests: Sequence[_Request]
    ) -> list[_Assessment]:
        """Validate every row in bulk: three lookups for the file, not three per row.

        A per-row `get` here is what turns a 600-mark upload into two thousand queries and
        a registrar watching a spinner, so everything a row needs is loaded first and the
        loop below touches nothing but dictionaries.
        """
        if not requests:
            return []
        terms = uow.terms.get_many(
            {r.row.term_code for r in requests if r.row.term_code is not None}
        )
        # Batch-level, not per row: `RowCode` has no member for a closed term, and a
        # closed term is a decision about the whole upload anyway. Refusing loudly is the
        # point — a late write would silently change what last term's report card said.
        self._assert_terms_open(terms.values())
        students = uow.students.get_many({r.row.student_number for r in requests})
        # Keyed by `(academic year, subject code)`, because a subject belongs to a year and
        # a sheet may name its own term per row — so one upload can legitimately span two
        # years, and `MATH` means a different row in each. One query per distinct year
        # rather than one per row: a marks upload names one term in practice, occasionally
        # two, never forty.
        subjects = self._subjects_by_year(uow, requests, terms)
        sections = self._sections_by_term(uow, requests, terms, students)
        section_ids = uow.class_sections.ids_for({s.identity for s in sections.values()})
        existing = uow.grades.get_many(
            [self._key(r.row) for r in requests if r.row.term_code is not None]
        )

        seen: dict[GradeKey, int] = {}
        assessments: list[_Assessment] = []
        for request in requests:
            assessments.append(
                self._assess_row(
                    request,
                    terms=terms,
                    students=students,
                    subjects=subjects,
                    sections=sections,
                    section_ids=section_ids,
                    existing=existing,
                    seen=seen,
                )
            )
        return assessments

    def _assess_row(
        self,
        request: _Request,
        *,
        terms: Mapping[str, Term],
        students: Mapping[str, Student],
        subjects: Mapping[tuple[str, str], Subject],
        sections: Mapping[tuple[str, str], ClassSection],
        section_ids: Mapping[tuple[str, str], int],
        existing: Mapping[GradeKey, SubjectGrade],
        seen: dict[GradeKey, int],
    ) -> _Assessment:
        """One row's verdict. Every failure is a value; nothing raised leaves this method.

        The order of the checks is the order a registrar can act on them: the references
        first, because an unknown term or student explains every later failure, and the
        figure last, because a mark of 140 is only worth reporting once the row is known
        to be about a real child.
        """
        row = request.row
        payload = self._payload(row)
        line = row.line_number

        term = terms.get(str(row.term_code)) if row.term_code is not None else None
        if term is None:
            return _Assessment(
                line, payload, RowCode.UNKNOWN_TERM, RowOutcome.REJECTED,
                f"no term {row.term_code}", "term_code",
            )
        if str(row.student_number) not in students:
            return _Assessment(
                line, payload, RowCode.UNKNOWN_STUDENT, RowOutcome.REJECTED,
                f"no student {row.student_number} on file", "student_number",
            )
        # Under the term's year, not globally: a subject the school teaches this year is
        # not on file for last year's marks, and reporting it as present would write a
        # grade against a subject that year never taught.
        if (str(term.academic_year_code), str(row.subject_code)) not in subjects:
            return _Assessment(
                line, payload, RowCode.UNKNOWN_SUBJECT, RowOutcome.REJECTED,
                f"no subject {row.subject_code} in {term.academic_year_code}",
                "subject_code",
            )

        key = self._key(row)
        first = seen.get(key)
        if first is not None:
            # The first occurrence stands and later ones are rejected, rather than
            # last-one-wins: two different marks for one child and subject is a mistake in
            # the spreadsheet, and silently keeping whichever sorted last means the file
            # decides which is right instead of the registrar.
            return _Assessment(
                line, payload, RowCode.DUPLICATE_IN_FILE, RowOutcome.REJECTED,
                f"line {first} already states {row.subject_code} for "
                f"{row.student_number} in {row.term_code}",
                "student_number",
            )
        # Registered before the remaining checks, so a repeated pair is reported once as a
        # duplicate rather than three times as whatever else is wrong with it.
        seen[key] = line

        section = sections.get((str(row.term_code), str(row.student_number)))
        if section is None:
            return _Assessment(
                line, payload, RowCode.UNKNOWN_CLASS, RowOutcome.REJECTED,
                f"{row.student_number} had no class placement covering {row.term_code}; "
                "a mark cannot be filed under a guessed class",
                "class_code",
            )
        if request.expected_class is not None and str(section.code) != str(
            request.expected_class
        ):
            # The guard `GradePreviewCommand.class_code` exists for: a file uploaded
            # against the wrong class must be refused, not matched against the whole
            # school.
            return _Assessment(
                line, payload, RowCode.UNKNOWN_CLASS, RowOutcome.REJECTED,
                f"{row.student_number} was in {section.code} for {row.term_code}, not "
                f"{request.expected_class}",
                "class_code",
            )
        section_id = section_ids.get(section.identity)
        if section_id is None:
            return _Assessment(
                line, payload, RowCode.UNKNOWN_CLASS, RowOutcome.REJECTED,
                f"class {section.code} could not be resolved for {row.term_code}",
                "class_code",
            )
        payload["class_code"] = str(section.code)

        on_file = existing.get(key)
        if not row.is_graded:
            # Invariant 1, in the form that actually loses data. A blank cell means "not
            # graded yet" and must never become 0.0 -- and it must not erase a figure
            # already on file either: a school re-uploading one subject's sheet with the
            # other columns blank would otherwise wipe every other mark for the term, and
            # the wipe reads downstream as marks that were never entered. Clearing a mark
            # stays an explicit, single-grade act.
            if on_file is not None and on_file.is_graded:
                return _Assessment(
                    line, payload, RowCode.OK, RowOutcome.UNCHANGED,
                    "blank cell left the mark already on file untouched",
                )
        try:
            grade = self._build(row, section=section, section_id=section_id)
        except InvalidPercentage as error:
            # A stated figure outside 0..100 -- "17 out of 15", or a percentage column of
            # 140. Its own code because it is a conversation about the grading scale, not
            # a mistyped cell.
            return _Assessment(
                line, payload, RowCode.GRADE_OUT_OF_RANGE, RowOutcome.REJECTED,
                error.message, error.field,
            )
        except SisError as error:
            return _Assessment(
                line, payload, RowCode.INVALID_GRADE, RowOutcome.REJECTED,
                error.message, error.field,
            )

        if on_file is None:
            return _Assessment(line, payload, RowCode.OK, RowOutcome.CREATED, grade=grade)
        if self._same_figure(on_file, grade):
            return _Assessment(
                line, payload, RowCode.OK, RowOutcome.UNCHANGED,
                "already on file with the same figure", grade=grade,
            )
        # Restating a mark is the normal reason to upload a corrected sheet, so an existing
        # grade is an update and never `DUPLICATE_EXISTING`. That code belongs to imports
        # where a second row is a mistake; here it would reject exactly the corrections the
        # registrar came to make.
        return _Assessment(line, payload, RowCode.OK, RowOutcome.UPDATED, grade=grade)

    # -- Writing -----------------------------------------------------------

    def _write(
        self, uow: UnitOfWork, assessments: Sequence[_Assessment]
    ) -> list[_Assessment]:
        """Upsert every writable grade in one statement and report what each row became.

        The repository's created/updated flag wins over the preview's guess: another
        registrar may have entered the same mark in between, and reporting "created" for a
        row that updated is a small lie that makes an audit trail unusable.
        """
        writable = [
            a for a in assessments if a.grade is not None and a.outcome in _WRITES
        ]
        if not writable:
            return list(assessments)
        created = uow.grades.upsert_many([a.grade for a in writable])  # type: ignore[misc]
        actual = {
            a.line_number: (
                RowOutcome.CREATED
                if created.get(a.grade.identity, a.outcome is RowOutcome.CREATED)  # type: ignore[union-attr]
                else RowOutcome.UPDATED
            )
            for a in writable
        }
        return [
            replace(a, outcome=actual[a.line_number]) if a.line_number in actual else a
            for a in assessments
        ]

    # -- Helpers -----------------------------------------------------------

    def _build(
        self, row: ParsedGradeRow, *, section: ClassSection, section_id: int
    ) -> SubjectGrade:
        """Turn a parsed row into the mark to store, restating points only when needed.

        A stated percentage is stored verbatim and never recomputed from the points beside
        it (invariant 5): the two can disagree in a real sheet, and the figure the teacher
        wrote in the percentage column is the one the school published. Points are carried
        along so a later correction can be audited against what was actually on the sheet.
        """
        if row.percentage is None and row.points is None and row.max_points is None:
            return SubjectGrade(
                student_number=row.student_number,
                subject_code=row.subject_code,
                term_code=row.term_code,  # type: ignore[arg-type]  # checked by caller
                class_section_id=section_id,
                class_code=section.code,
                percentage=None,
            )
        if row.percentage is None and row.max_points is not None:
            return SubjectGrade.from_points(
                student_number=row.student_number,
                subject_code=row.subject_code,
                term_code=row.term_code,  # type: ignore[arg-type]
                class_section_id=section_id,
                class_code=section.code,
                points=row.points,
                max_points=row.max_points,
            )
        return SubjectGrade(
            student_number=row.student_number,
            subject_code=row.subject_code,
            term_code=row.term_code,  # type: ignore[arg-type]
            class_section_id=section_id,
            class_code=section.code,
            percentage=row.percentage,
            points=row.points,
            max_points=row.max_points,
        )

    @staticmethod
    def _subjects_by_year(
        uow: UnitOfWork,
        requests: Sequence[_Request],
        terms: Mapping[str, Term],
    ) -> dict[tuple[str, str], Subject]:
        """Resolve each named subject inside the year of the term the row is for.

        Keyed by `(academic year, subject code)` for the same reason `_sections_by_term` is
        keyed by term: the answer differs per year, and a cache keyed on the code alone
        would hand a row from last term's sheet whichever year's `MATH` happened to load.

        A row whose term is unknown contributes nothing — `_assess_row` rejects it on the
        term before it ever looks at the subject, so there is no year to ask under and
        nothing useful to fetch.
        """
        wanted: dict[str, set[SubjectCode]] = {}
        for request in requests:
            term = terms.get(str(request.row.term_code))
            if term is None or request.row.subject_code is None:
                continue
            wanted.setdefault(str(term.academic_year_code), set()).add(
                request.row.subject_code
            )

        resolved: dict[tuple[str, str], Subject] = {}
        for year_code, codes in wanted.items():
            for code, subject in uow.subjects.get_many(codes, AcademicYearCode(year_code)).items():
                resolved[(year_code, code)] = subject
        return resolved

    def _sections_by_term(
        self,
        uow: UnitOfWork,
        requests: Sequence[_Request],
        terms: Mapping[str, Term],
        students: Mapping[str, Student],
    ) -> dict[tuple[str, str], ClassSection]:
        """Resolve each child's class per term (invariant 2), one lookup per term.

        Keyed by term as well as student because the answer genuinely differs between
        terms for any child who transferred — that is the entire point of the invariant,
        and a cache keyed on the student alone would quietly give her Term 1 marks the
        class she moved to in March.
        """
        resolved: dict[tuple[str, str], ClassSection] = {}
        for code, term in terms.items():
            numbers = {
                r.row.student_number
                for r in requests
                if str(r.row.term_code) == code and str(r.row.student_number) in students
            }
            for number, section in resolve_sections_for_term(
                uow.enrolments, numbers, term
            ).items():
                resolved[(code, number)] = section
        return resolved

    def _subject_guard(
        self, row: ParsedGradeRow, expected: SubjectCode | None
    ) -> _Assessment | None:
        """Refuse a row naming a subject the upload was not for. See `class_code`'s guard.

        A single-subject upload that quietly accepted rows for other subjects would let a
        mis-picked dropdown write marks across the whole report card, and every row of it
        would look correct.
        """
        if expected is None or str(row.subject_code) == str(expected):
            return None
        return _Assessment(
            row.line_number,
            self._payload(row),
            RowCode.UNKNOWN_SUBJECT,
            RowOutcome.REJECTED,
            f"this upload is for {expected}, but the row states {row.subject_code}",
            "subject_code",
        )

    def _request_from_payload(self, stored: ImportRow) -> _Request:
        """Rebuild a row from what the preview recorded, so commit replays the review.

        The payload is the whole input: re-reading the uploaded file at commit is how a
        different file gets applied than the one a human approved.
        """
        payload = stored.payload
        percentage = payload.get("percentage")
        return _Request(
            row=ParsedGradeRow(
                line_number=stored.line_number,
                student_number=StudentNumber(str(payload["student_number"])),
                subject_code=SubjectCode(str(payload["subject_code"])),
                term_code=TermCode(str(payload["term_code"])),
                percentage=None if percentage is None else Percentage(float(percentage)),  # type: ignore[arg-type]
                points=None if payload.get("points") is None else float(payload["points"]),  # type: ignore[arg-type]
                max_points=(
                    None
                    if payload.get("max_points") is None
                    else float(payload["max_points"])  # type: ignore[arg-type]
                ),
            ),
            expected_class=(
                ClassCode(str(payload["class_code"]))
                if payload.get("class_code")
                else None
            ),
        )

    @staticmethod
    def _as_changed_if_failed(previewed: ImportRow, now: _Assessment) -> _Assessment:
        """Report a row that passed preview and fails now as `CHANGED_SINCE_PREVIEW`.

        The distinction the registrar needs: "your file is wrong" and "the world moved
        while you were reading" call for different actions, and merging them sends her to
        edit a spreadsheet that was correct when she uploaded it.
        """
        if now.is_ok or previewed.is_rejected:
            return now
        return replace(
            now,
            code=RowCode.CHANGED_SINCE_PREVIEW,
            message=f"changed since the preview: {now.message}",
        )

    @staticmethod
    def _carried(stored: ImportRow) -> _Assessment:
        """A preview rejection, replayed into the commit result with its original reason."""
        return _Assessment(
            line_number=stored.line_number,
            payload=dict(stored.payload),
            code=RowCode(stored.code),
            outcome=RowOutcome.REJECTED,
            message=stored.message or "",
            field=stored.field,
        )

    @staticmethod
    def _payload(row: ParsedGradeRow) -> dict[str, object]:
        """The row's identifying cells, as the UI and the commit step both read them.

        `percentage` is `None` for a blank and `is_graded` states that in its own right, so
        no reader of a stored batch has to decide what a missing key means — the decision
        that turns "not graded" into a zero.
        """
        return {
            "student_number": str(row.student_number),
            "subject_code": str(row.subject_code),
            "term_code": None if row.term_code is None else str(row.term_code),
            "class_code": None,
            "percentage": None if row.percentage is None else row.percentage.value,
            "points": row.points,
            "max_points": row.max_points,
            "is_graded": row.is_graded,
        }

    @staticmethod
    def _key(row: ParsedGradeRow) -> GradeKey:
        return (str(row.student_number), str(row.subject_code), str(row.term_code))

    @staticmethod
    def _same_figure(on_file: SubjectGrade, incoming: SubjectGrade) -> bool:
        """Whether a write would change anything a reader can see.

        The class is part of it: the same mark filed under a different section is a real
        change, because that is the heading it prints under.
        """
        return (
            on_file.percentage == incoming.percentage
            and on_file.points == incoming.points
            and on_file.max_points == incoming.max_points
            and on_file.class_section_id == incoming.class_section_id
        )

    @staticmethod
    def _assert_terms_open(terms: Iterable[Term]) -> None:
        for term in terms:
            if term.is_closed:
                raise TermClosed(
                    f"term {term.code} is closed; reopen it before importing marks",
                    field="term_code",
                )

    @staticmethod
    def _digest(content: bytes) -> str:
        """Identifies the previewed file, so a commit cannot apply a different one."""
        return hashlib.sha256(content).hexdigest()

    def _check_size(self, content: bytes) -> None:
        """Checked before parsing: a parser that discovers this has already spent the RAM."""
        if self._max_upload_bytes is not None and len(content) > self._max_upload_bytes:
            raise UploadTooLarge(
                f"upload is {len(content)} bytes, limit is {self._max_upload_bytes}",
                field="content",
            )

    @staticmethod
    def _ordered_rows(
        diagnostics: Sequence[RowDiagnostic], assessments: Sequence[_Assessment]
    ) -> list[ImportRow]:
        """Stored rows in file order, so the batch reads like the spreadsheet it came from."""
        rows = [
            ImportRow(
                line_number=d.line,
                payload=d.payload,
                outcome=RowOutcome.REJECTED,
                code=d.code.value,
                message=d.message,
                field=d.field,
            )
            for d in diagnostics
        ]
        rows.extend(a.as_import_row() for a in assessments)
        rows.sort(key=lambda row: row.line_number)
        return rows

    @staticmethod
    def _ordered_diagnostics(
        diagnostics: Sequence[RowDiagnostic], assessments: Sequence[_Assessment]
    ) -> list[RowDiagnostic]:
        merged = [*diagnostics, *(a.as_diagnostic() for a in assessments)]
        merged.sort(key=lambda row: row.line)
        return merged


# Outcomes that mean a write. `UNCHANGED` and `SKIPPED` deliberately are not here: a row
# that would store the same figure costs nothing to skip, and a blank cell over an existing
# mark must not reach the database at all.
_WRITES = (RowOutcome.CREATED, RowOutcome.UPDATED)
