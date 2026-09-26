"""Guardian import: preview which adults a file would attach to which children, then do it.

The same two-step shape as `roster_import`, and for the same reason. `preview()` judges
every row against what is on file, writes an `ImportBatch`, and **creates no guardian and
no link**. `commit()` loads that batch back, runs the *same* evaluator against state it
re-reads, and writes. The preview's verdict is a proposal, never an instruction: between
the two requests another registrar can create the child, register the phone, or change the
link, and a commit that trusted the preview would write against a world that no longer
exists while reporting success.

Two rules here are worth stating before the code, because both look stricter than they
need to be until the failure behind them is named.

**A phone already on file under a differently-named adult rejects the row.** It is the one
refusal a registrar might not expect, and it costs her a line when she meant to fix a
spelling. The trade is right: a rejected row is visible and fixed in one edit, whereas
accepting it silently attaches a family's records to whoever now answers a recycled
number — and once a parent can log in by receiving a code on that phone, it hands them the
records too. Numbers are reassigned constantly; names are how a human notices.

**The child must already exist.** This importer never creates a `Student`. A guardian file
naming an unknown student number is a typo or a roster that was never uploaded, and
inventing a child from a parent's row would produce a student with no class, no grades and
a plausible-looking number that quietly fails to match the real one.

Ports only. No sqlalchemy, no fastapi, no `sis.config`.
"""
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sis.application.dto import (
    GuardianCommitCommand,
    GuardianPreviewCommand,
    ImportCommitResult,
    ImportPreviewResult,
    RowCode,
    RowOutcome,
)
from sis.application.ports.parsers import GuardianFileParser
from sis.application.ports.unit_of_work import UnitOfWork
from sis.domain.errors import (
    ImportBatchNotFound,
    ImportContentMismatch,
    SisError,
    UploadTooLarge,
)
from sis.domain.guardians import Guardian, RelationshipType, StudentGuardian
from sis.domain.imports import (
    ImportBatch,
    ImportKind,
    ImportRow,
    # See `roster_import` for why the storage vocabulary is aliased: the domain's
    # `RowOutcome` is the write verb, the DTO's is the row report the API returns.
    RowOutcome as StoredOutcome,
    tally,
)
from sis.domain.people import Student
from sis.domain.value_objects import Phone, StudentNumber

__all__ = ["GuardianImportService"]


# Payload keys. Named once because `commit()` reads back what `preview()` wrote, and a
# typo in one of the two halves is a batch that previews perfectly and commits nothing.
_K_NUMBER = "student_number"
_K_PHONE = "phone"
_K_ALT_PHONE = "alt_phone"
_K_NAME_AR = "full_name_ar"
_K_NAME_EN = "full_name_en"
_K_RELATION = "relationship_type"
_K_LABEL = "relationship_label"
_K_PRIMARY = "is_primary_contact"
_K_CAN_VIEW = "can_view_records"
_K_NOTE = "restriction_note"


@dataclass(frozen=True, slots=True)
class _Assertion:
    """What one row claims, resolved but not yet judged.

    Built from a parsed row at preview and from a stored payload at commit, which is what
    lets one evaluator serve both paths.
    """

    line: int
    number: StudentNumber
    phone: Phone
    name_ar: str
    name_en: str
    relationship: RelationshipType
    label: str
    alt_phone: Phone | None = None
    is_primary_contact: bool = False
    can_view_records: bool = True
    restriction_note: str = ""

    @property
    def phones(self) -> tuple[Phone, ...]:
        """Her numbers, primary first, with an alternate that repeats it collapsed."""
        if self.alt_phone is None or str(self.alt_phone) == str(self.phone):
            return (self.phone,)
        return (self.phone, self.alt_phone)

    @property
    def payload(self) -> dict[str, object]:
        """The row as the registrar sees it, and as commit reads it back."""
        return {
            _K_NUMBER: str(self.number),
            _K_PHONE: str(self.phone),
            _K_ALT_PHONE: None if self.alt_phone is None else str(self.alt_phone),
            _K_NAME_AR: self.name_ar,
            _K_NAME_EN: self.name_en,
            _K_RELATION: self.relationship.value,
            _K_LABEL: self.label,
            _K_PRIMARY: self.is_primary_contact,
            _K_CAN_VIEW: self.can_view_records,
            _K_NOTE: self.restriction_note,
        }


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """Everything the evaluator is allowed to know about stored state.

    Three bulk reads for the whole file, never one per row. Holding it as a value also
    makes the evaluator a pure function of (assertion, snapshot), so a unit test of "the
    same mother on three rows becomes one guardian" builds two dicts and no database.

    `guardians` is keyed by the number that was *asked about*, not by each guardian's
    primary one, so a row quoting a mother's second line finds her under the number it
    quoted.
    """

    students: Mapping[str, Student]
    guardians: Mapping[str, Guardian]
    links: Mapping[str, Sequence[StudentGuardian]]


@dataclass(frozen=True, slots=True)
class _Plan:
    """One row's verdict *and* the writes it implies — decided once, applied later.

    Preview keeps the verdict and throws the writes away; commit keeps both. That is the
    whole of "nothing is created at preview": the two paths differ by whether `_apply` is
    called, not by which rules ran.
    """

    line: int
    payload: Mapping[str, object]
    code: RowCode
    outcome: StoredOutcome
    message: str = ""
    field: str | None = None
    guardian: Guardian | None = None
    link: StudentGuardian | None = None
    #: Set when this row resolved to a link that already existed.
    superseded: bool = False


def _norm_name(raw: str) -> str:
    """Collapse a name for comparison only. Never stored in this form."""
    return " ".join(raw.split()).casefold()


def _names_collide(stated: _Assertion, stored: Guardian) -> bool:
    """Does this number already belong to a *different* adult?

    Compared only where both sides state a name in the same script, so an Arabic-first
    sheet with no English column does not read as a conflict against a record holding
    both. See the module docstring for why this refuses rather than overwrites.
    """
    pairs = (
        (stated.name_ar, stored.full_name_ar),
        (stated.name_en, stored.full_name_en),
    )
    return any(a and b and _norm_name(a) != _norm_name(b) for a, b in pairs)


def _stored_row(plan: _Plan) -> ImportRow:
    """Persist the fine-grained `RowCode` in `code` and the write verb in `outcome`.

    Two vocabularies, as in `roster_import`: `outcome` answers "was anything written" and
    `code` answers "why", and only the latter can distinguish an unknown student from a
    phone that belongs to somebody else.
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


class GuardianImportService:
    """Preview and commit guardian uploads. Depends on ports only."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        parser: GuardianFileParser,
        *,
        preview_ttl: timedelta,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_upload_bytes: int | None = None,
        batch_ids: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        """A *factory* of units of work, because preview and commit are two transactions.

        The clock and the TTL arrive here rather than being read from `sis.config`, which
        no service may import: a use case that reads the environment cannot be tested
        without arranging it, and expiry cannot be tested at all without sleeping.
        """
        self._uow_factory = uow_factory
        self._parser = parser
        self._preview_ttl = preview_ttl
        self._clock = clock
        self._max_upload_bytes = max_upload_bytes
        self._batch_ids = batch_ids

    # -- preview ------------------------------------------------------------

    def preview(self, command: GuardianPreviewCommand) -> ImportPreviewResult:
        """Judge every row and write a batch. Not one guardian is created here.

        Raises `UploadTooLarge`, `UnsupportedFileType` or `UnreadableImportFile` when
        there was never a file worth reporting on. Unlike the roster importer there is no
        whole-batch reference to resolve first, because the upload names no target.
        """
        if (
            self._max_upload_bytes is not None
            and len(command.content) > self._max_upload_bytes
        ):
            raise UploadTooLarge(
                f"{command.filename} is larger than the {self._max_upload_bytes} byte limit",
                field="content",
            )
        parsed = self._parser.parse(command.content, command.filename)
        now = self._clock()

        assertions = [
            _Assertion(
                line=row.line_number,
                number=row.student_number,
                phone=row.phone,
                name_ar=row.full_name_ar,
                name_en=row.full_name_en,
                relationship=row.relationship_type,
                label=row.relationship_label,
                alt_phone=row.alt_phone,
                is_primary_contact=row.is_primary_contact,
                can_view_records=row.can_view_records,
                restriction_note=row.restriction_note,
            )
            for row in parsed.rows
        ]

        with self._uow_factory() as uow:
            snapshot = self._load(uow, assertions)

            seen: dict[tuple[str, str], int] = {}
            plans = [self._evaluate(a, snapshot, seen) for a in assertions]

            reports = [*parsed.diagnostics]
            reports.extend(_reported(plan) for plan in plans)
            stored = [
                *(_diagnostic_row(report) for report in parsed.diagnostics),
                *(_stored_row(plan) for plan in plans),
            ]
            stored.sort(key=lambda row: row.line_number)
            reports.sort(key=lambda report: report.line)

            batch = ImportBatch(
                batch_id=self._batch_ids(),
                kind=ImportKind.GUARDIANS,
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

    # -- commit -------------------------------------------------------------

    def commit(self, command: GuardianCommitCommand) -> ImportCommitResult:
        """Re-judge the previewed rows against current state, then write them.

        Raises `ImportBatchNotFound`, `ImportBatchExpired` or
        `ImportBatchAlreadyCommitted` — three distinct codes, because the registrar's next
        action differs for each. A second commit of a batch is refused rather than
        re-applied, which is what makes a double-clicked button safe.

        Every write lands in one transaction. A commit-per-row loop failing on row 140
        would leave 139 links written and 61 missing, which no screen reports as broken
        because every row it did write looks correct.
        """
        now = self._clock()
        batch_id = str(command.batch_id)

        with self._uow_factory() as uow:
            batch = uow.imports.get(batch_id)
            if batch is None:
                raise ImportBatchNotFound(f"no import batch {batch_id}", field="batch_id")
            if batch.kind is not ImportKind.GUARDIANS:
                raise ImportContentMismatch(
                    f"batch {batch_id} is a {batch.kind.value} import, not guardians",
                    field="batch_id",
                )
            batch.assert_committable(now, batch.content_hash)

            previewed = uow.imports.list_rows(batch_id)
            assertions, replayed = self._replay(previewed)
            approved = {
                row.line_number: row.outcome for row in previewed if row.is_written
            }
            snapshot = self._load(uow, assertions)

            seen: dict[tuple[str, str], int] = {}
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

        Rows the registrar saw rejected stay rejected with the code she read. Only rows
        that proposed a write are re-judged, because those are the only ones whose verdict
        someone else's edit can have invalidated.
        """
        assertions: list[_Assertion] = []
        carried: list[ImportRow] = []
        for row in rows:
            if not row.is_written:
                carried.append(row)
                continue
            try:
                alt = row.payload.get(_K_ALT_PHONE)
                assertions.append(
                    _Assertion(
                        line=row.line_number,
                        number=StudentNumber(row.payload[_K_NUMBER]),
                        phone=Phone(str(row.payload[_K_PHONE])),
                        name_ar=str(row.payload.get(_K_NAME_AR, "")),
                        name_en=str(row.payload.get(_K_NAME_EN, "")),
                        relationship=RelationshipType(
                            str(row.payload.get(_K_RELATION, RelationshipType.OTHER.value))
                        ),
                        label=str(row.payload.get(_K_LABEL, "")),
                        alt_phone=None if alt in (None, "") else Phone(str(alt)),
                        is_primary_contact=bool(row.payload.get(_K_PRIMARY, False)),
                        can_view_records=bool(row.payload.get(_K_CAN_VIEW, True)),
                        restriction_note=str(row.payload.get(_K_NOTE, "")),
                    )
                )
            except (SisError, KeyError, TypeError, ValueError) as exc:
                # A payload that no longer rebuilds is, by definition, not the row that
                # was previewed. Recorded rather than raised so the rest of the batch
                # still lands.
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
        """Refuse a row whose link stopped being new between the preview and this commit.

        A row the registrar approved as `created` that now finds a link already on file is
        not the row she read — someone else attached this adult to this child in the
        meantime, possibly with a different relationship or a different permission, and
        overwriting it silently is exactly what committing the same file in two tabs would
        do. Every other divergence already produced its own code inside `_evaluate`;
        `changed_since_preview` is the residue, not the general answer.
        """
        if plan.superseded and approved.get(plan.line) is StoredOutcome.CREATED:
            return replace(
                plan,
                code=RowCode.CHANGED_SINCE_PREVIEW,
                outcome=StoredOutcome.REJECTED,
                message=(
                    f"{plan.payload.get(_K_PHONE)} was linked to student "
                    f"{plan.payload.get(_K_NUMBER)} by someone else since this file was "
                    "previewed; upload it again to see the current outcome"
                ),
                field=_K_PHONE,
                guardian=None,
                link=None,
            )
        return plan

    def _load(self, uow: UnitOfWork, assertions: Sequence[_Assertion]) -> _Snapshot:
        """Three bulk reads for the whole file, whichever path is running."""
        numbers = [a.number for a in assertions]
        phones: list[Phone] = []
        for assertion in assertions:
            phones.extend(assertion.phones)
        return _Snapshot(
            students=uow.students.get_many(numbers),
            guardians=uow.guardians.get_many(phones),
            links=uow.student_guardians.list_for_students(numbers),
        )

    def _evaluate(
        self,
        stated: _Assertion,
        snapshot: _Snapshot,
        seen: dict[tuple[str, str], int],
    ) -> _Plan:
        """The whole rule set for one row. Pure: same inputs, same verdict, no clock, no I/O.

        Order is deliberate. A duplicated line is reported as a duplicate even when it also
        names an unknown student, because the registrar deletes it either way and two codes
        for one line make the tally lie.
        """
        number = str(stated.number)
        payload = stated.payload
        owner = snapshot.guardians.get(str(stated.phone))

        # Deduplicated on the *resolved adult* rather than on the raw number, so a mother
        # listed once under her mobile and once under the second line already on file is
        # recognised as the same pairing rather than written twice.
        identity = owner.identity if owner is not None else str(stated.phone)
        key = (number, identity)
        first_seen = seen.get(key)
        if first_seen is not None:
            return _Plan(
                line=stated.line,
                payload=payload,
                code=RowCode.DUPLICATE_IN_FILE,
                outcome=StoredOutcome.REJECTED,
                message=(
                    f"this guardian is already linked to student {number} on line "
                    f"{first_seen} of this file"
                ),
                field=_K_PHONE,
            )
        seen[key] = stated.line

        if number not in snapshot.students:
            return _Plan(
                line=stated.line,
                payload=payload,
                code=RowCode.UNKNOWN_STUDENT,
                outcome=StoredOutcome.REJECTED,
                message=(
                    f"student {number} is not on file; upload the roster before "
                    "attaching guardians to it"
                ),
                field=_K_NUMBER,
            )

        if owner is not None and _names_collide(stated, owner):
            return _Plan(
                line=stated.line,
                payload=payload,
                code=RowCode.DUPLICATE_EXISTING,
                outcome=StoredOutcome.REJECTED,
                message=(
                    f"{stated.phone} is already on file for "
                    f"{owner.full_name_en or owner.full_name_ar}"
                ),
                field=_K_PHONE,
            )

        # The alternate number must be free, or already this same adult's. Registering it
        # otherwise would move a number off the person it currently reaches — and the
        # database's uniqueness on a phone would refuse the write anyway, one layer too
        # late to tell the registrar which row was at fault.
        if stated.alt_phone is not None:
            alt_owner = snapshot.guardians.get(str(stated.alt_phone))
            if alt_owner is not None and (
                owner is None or alt_owner.identity != owner.identity
            ):
                return _Plan(
                    line=stated.line,
                    payload=payload,
                    code=RowCode.DUPLICATE_EXISTING,
                    outcome=StoredOutcome.REJECTED,
                    message=(
                        f"the second number {stated.alt_phone} is already on file for "
                        f"{alt_owner.full_name_en or alt_owner.full_name_ar}"
                    ),
                    field=_K_ALT_PHONE,
                )

        guardian = self._merged_guardian(stated, owner)
        candidate = StudentGuardian(
            student_number=stated.number,
            guardian_phone=guardian.primary_phone,
            relationship_type=stated.relationship,
            relationship_label=stated.label,
            is_primary_contact=stated.is_primary_contact,
            can_view_records=stated.can_view_records,
            restriction_note=stated.restriction_note,
        )

        existing_link = None
        if owner is not None:
            existing_link = next(
                (
                    link
                    for link in snapshot.links.get(number, ())
                    if owner.reachable_on(link.guardian_phone)
                ),
                None,
            )

        guardian_changed = guardian != owner
        link_changed = existing_link is None or candidate.differs_from(existing_link)

        if existing_link is None:
            outcome = StoredOutcome.CREATED
        elif link_changed or guardian_changed:
            outcome = StoredOutcome.UPDATED
        else:
            outcome = StoredOutcome.UNCHANGED

        return _Plan(
            line=stated.line,
            payload=payload,
            code=RowCode.OK,
            outcome=outcome,
            guardian=guardian if guardian_changed else None,
            link=candidate if link_changed else None,
            superseded=existing_link is not None,
        )

    def _merged_guardian(self, stated: _Assertion, existing: Guardian | None) -> Guardian:
        """Add what the row supplies; never replace what the record already states.

        A sheet with an empty English column must not blank the English name a registrar
        typed by hand, and `_names_collide` has already refused the case where the two
        disagree — so what reaches here is only ever an addition. Numbers accumulate for
        the same reason: an upload mentioning only her mobile must not drop the second
        line an earlier one recorded.
        """
        if existing is None:
            return Guardian(
                phones=stated.phones,
                full_name_ar=stated.name_ar,
                full_name_en=stated.name_en,
            )
        # The stored primary stays primary. Re-ordering it would change the identity every
        # link in the database already names.
        phones = list(existing.phones)
        known = {str(phone) for phone in phones}
        for phone in stated.phones:
            if str(phone) not in known:
                phones.append(phone)
                known.add(str(phone))
        return Guardian(
            phones=tuple(phones),
            full_name_ar=existing.full_name_ar or stated.name_ar,
            full_name_en=existing.full_name_en or stated.name_en,
            preferred_language=existing.preferred_language,
            is_active=existing.is_active,
        )

    def _apply(self, uow: UnitOfWork, plans: Sequence[_Plan]) -> None:
        """Write the accepted plans in bulk, guardians before the links that name them.

        The ordering is load-bearing, not stylistic: a link resolves its guardian by phone
        at write time, so a link written first names a row that does not exist yet.
        """
        guardians = [plan.guardian for plan in plans if plan.guardian is not None]
        if guardians:
            uow.guardians.upsert_many(guardians)

        links = [plan.link for plan in plans if plan.link is not None]
        if links:
            uow.student_guardians.upsert_many(links)
