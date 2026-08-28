"""`GuardianImportService` against fakes: the rules, with no database anywhere.

The database-backed half of this feature is `test_guardian_api`. What is asserted here is
the rule set — which row wins, which is refused, and what a second commit does — because
those are decisions the service makes and a test that needed an engine to reach them would
be slow enough that the twenty cases below would never all be written.

The most important assertion in the file is the first: **preview writes nothing**. Every
other guarantee is downstream of it, because the whole two-step flow exists so a registrar
can read what an upload would do before it does it.
"""
from datetime import UTC, datetime, timedelta

import pytest

from sis.application.dto import (
    GuardianCommitCommand,
    GuardianPreviewCommand,
    ParsedGuardianRow,
    ParseResult,
    RowCode,
    RowOutcome,
)
from sis.application.services.guardian_import import GuardianImportService
from sis.domain.errors import (
    ImportBatchAlreadyCommitted,
    ImportBatchExpired,
    ImportBatchNotFound,
    ImportContentMismatch,
    UploadTooLarge,
)
from sis.domain.guardians import Guardian, RelationshipType, StudentGuardian
from sis.domain.imports import ImportKind, RowOutcome as StoredOutcome
from sis.domain.people import Student
from sis.domain.value_objects import Phone, StudentNumber

from tests.sis.conftest import FakeUnitOfWork, StubParser

NOW = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
TTL = timedelta(minutes=30)

MOTHER = "+201001234567"
MOTHER_ALT = "+201119998888"
FATHER = "+201002223333"
BROTHER = "+201005554444"


def _row(
    line: int,
    number: str,
    phone: str,
    *,
    name_ar: str = "",
    name_en: str = "Fatma Ali",
    relationship: RelationshipType = RelationshipType.MOTHER,
    label: str = "",
    alt: str | None = None,
    can_view: bool = True,
    primary: bool = False,
) -> ParsedGuardianRow:
    return ParsedGuardianRow(
        line_number=line,
        student_number=StudentNumber(number),
        phone=Phone(phone),
        full_name_ar=name_ar,
        full_name_en=name_en,
        relationship_type=relationship,
        relationship_label=label,
        alt_phone=None if alt is None else Phone(alt),
        is_primary_contact=primary,
        can_view_records=can_view,
    )


def _service(uow: FakeUnitOfWork, rows, diagnostics=()) -> GuardianImportService:
    """A service whose parser returns exactly these rows, with a frozen clock."""
    parser = StubParser(
        ParseResult(
            rows=tuple(rows),
            diagnostics=tuple(diagnostics),
            total_lines=len(rows) + len(diagnostics),
            headers=("student_number", "phone"),
        )
    )
    return GuardianImportService(
        lambda: uow,
        parser,
        preview_ttl=TTL,
        clock=lambda: NOW,
        batch_ids=lambda: "batch-1",
    )


@pytest.fixture
def enrolled(fake_uow: FakeUnitOfWork) -> FakeUnitOfWork:
    """Two children on the roll and no guardians at all — guardians never create students."""
    fake_uow.students.upsert_many(
        [
            Student(
                student_number=StudentNumber("S-1"),
                full_name_ar="ليلى حسن",
                full_name_en="Layla Hassan",
            ),
            Student(
                student_number=StudentNumber("S-2"),
                full_name_ar="عمر خالد",
                full_name_en="Omar Khaled",
            ),
        ]
    )
    return fake_uow


def _preview(service: GuardianImportService):
    return service.preview(
        GuardianPreviewCommand(filename="g.csv", content=b"x", actor="registrar")
    )


def _commit(service: GuardianImportService, batch_id: str = "batch-1"):
    return service.commit(GuardianCommitCommand(batch_id=batch_id, actor="registrar"))


def test_preview_writes_no_guardian_and_no_link(enrolled: FakeUnitOfWork) -> None:
    """The guarantee the whole two-step flow rests on."""
    service = _service(enrolled, [_row(2, "S-1", MOTHER)])

    result = _preview(service)

    assert result.ok_count == 1
    assert enrolled.guardians.all() == ()
    assert enrolled.student_guardians.all() == ()


def test_commit_writes_the_guardian_and_the_link(enrolled: FakeUnitOfWork) -> None:
    service = _service(enrolled, [_row(2, "S-1", MOTHER, alt=MOTHER_ALT)])
    _preview(service)

    result = _commit(service)

    assert result.ok_count == 1
    (guardian,) = enrolled.guardians.all()
    assert [str(p) for p in guardian.phones] == [MOTHER, MOTHER_ALT]
    (link,) = enrolled.student_guardians.all()
    assert link.identity == ("S-1", MOTHER)
    assert link.relationship_type is RelationshipType.MOTHER


def test_one_bad_row_does_not_discard_the_good_ones(enrolled: FakeUnitOfWork) -> None:
    """Invariant 4. A parser diagnostic and a rejected row both leave the rest importable."""
    diagnostic = RowOutcome(
        line=3,
        code=RowCode.MISSING_PHONE,
        message="not a phone",
        payload={"student_number": "S-1"},
    )
    service = _service(
        enrolled,
        [_row(2, "S-1", MOTHER), _row(4, "S-404", FATHER), _row(5, "S-2", BROTHER)],
        diagnostics=[diagnostic],
    )

    result = _preview(service)
    codes = {row.line: row.code for row in result.rows}

    assert codes[2] is RowCode.OK
    assert codes[3] is RowCode.MISSING_PHONE
    assert codes[4] is RowCode.UNKNOWN_STUDENT
    assert codes[5] is RowCode.OK
    assert result.ok_count == 2


def test_a_guardian_upload_never_creates_a_student(enrolled: FakeUnitOfWork) -> None:
    """A parent's row is not evidence a child exists; inventing one is how a ghost
    student with no class and no grades appears on the roll."""
    service = _service(enrolled, [_row(2, "S-404", MOTHER)])
    _preview(service)
    _commit(service)

    assert enrolled.students.get(StudentNumber("S-404")) is None
    assert enrolled.student_guardians.all() == ()


def test_the_same_mother_on_two_children_is_one_guardian(
    enrolled: FakeUnitOfWork,
) -> None:
    """The deduplication every other guarantee depends on.

    If her two rows failed to resolve to one adult she would become two people, each
    holding one of her children, and no screen would report it.
    """
    service = _service(
        enrolled,
        [_row(2, "S-1", MOTHER, alt=MOTHER_ALT), _row(3, "S-2", MOTHER)],
    )
    _preview(service)
    _commit(service)

    assert len(enrolled.guardians.all()) == 1
    assert len(enrolled.student_guardians.all()) == 2
    # The alternate stated on only one of her rows survives the collapse.
    (guardian,) = enrolled.guardians.all()
    assert [str(p) for p in guardian.phones] == [MOTHER, MOTHER_ALT]


def test_a_row_quoting_her_second_number_finds_the_same_woman(
    enrolled: FakeUnitOfWork,
) -> None:
    """What makes the alternate number useful rather than decorative."""
    enrolled.guardians.upsert_many(
        [Guardian(phones=(Phone(MOTHER), Phone(MOTHER_ALT)), full_name_en="Fatma Ali")]
    )

    service = _service(enrolled, [_row(2, "S-2", MOTHER_ALT)])
    _preview(service)
    _commit(service)

    assert len(enrolled.guardians.all()) == 1


def test_an_exact_repeat_in_one_file_is_a_duplicate(enrolled: FakeUnitOfWork) -> None:
    service = _service(enrolled, [_row(2, "S-1", MOTHER), _row(3, "S-1", MOTHER)])

    result = _preview(service)
    codes = {row.line: row.code for row in result.rows}

    assert codes[2] is RowCode.OK
    assert codes[3] is RowCode.DUPLICATE_IN_FILE
    assert "line 2" in next(r.message for r in result.rows if r.line == 3)


def test_two_different_phones_for_one_child_are_not_duplicates(
    enrolled: FakeUnitOfWork,
) -> None:
    """The whole point of the many-to-many: a child has several adults, each with a number."""
    service = _service(
        enrolled,
        [
            _row(2, "S-1", MOTHER, relationship=RelationshipType.MOTHER),
            _row(3, "S-1", FATHER, name_en="Hassan Mahmoud", relationship=RelationshipType.FATHER),
            _row(4, "S-1", BROTHER, name_en="Karim Hassan", relationship=RelationshipType.SIBLING),
        ],
    )

    result = _preview(service)

    assert result.ok_count == 3
    assert result.rejected_count == 0


def test_one_phone_across_two_children_is_not_a_duplicate(
    enrolled: FakeUnitOfWork,
) -> None:
    """The other direction: one adult, several children."""
    service = _service(enrolled, [_row(2, "S-1", MOTHER), _row(3, "S-2", MOTHER)])

    assert _preview(service).ok_count == 2


def test_a_phone_held_by_a_differently_named_adult_is_refused(
    enrolled: FakeUnitOfWork,
) -> None:
    """A recycled number must not inherit the previous family's records."""
    enrolled.guardians.upsert_many(
        [Guardian(phones=(Phone(MOTHER),), full_name_en="Fatma Ali")]
    )

    service = _service(enrolled, [_row(2, "S-2", MOTHER, name_en="Someone Else")])
    result = _preview(service)

    (row,) = result.rows
    assert row.code is RowCode.DUPLICATE_EXISTING
    assert "Fatma Ali" in row.message


def test_a_blank_name_never_overwrites_a_stored_one(enrolled: FakeUnitOfWork) -> None:
    """A sheet with an empty column must not blank a name a registrar typed by hand."""
    enrolled.guardians.upsert_many(
        [
            Guardian(
                phones=(Phone(MOTHER),), full_name_ar="فاطمة علي", full_name_en="Fatma Ali"
            )
        ]
    )

    service = _service(enrolled, [_row(2, "S-1", MOTHER, name_en="", name_ar="فاطمة علي")])
    _preview(service)
    _commit(service)

    (guardian,) = enrolled.guardians.all()
    assert guardian.full_name_en == "Fatma Ali"


def test_the_sheet_can_withhold_records_access(enrolled: FakeUnitOfWork) -> None:
    """Granted by default because a registrar reviewed a preview; denied when she says so."""
    service = _service(
        enrolled,
        [
            _row(2, "S-1", MOTHER),
            _row(3, "S-1", BROTHER, name_en="Karim", can_view=False),
        ],
    )
    _preview(service)
    _commit(service)

    access = {
        str(link.guardian_phone): link.can_view_records
        for link in enrolled.student_guardians.all()
    }
    assert access == {MOTHER: True, BROTHER: False}


def test_relationship_and_label_are_both_stored(enrolled: FakeUnitOfWork) -> None:
    """Closing the vocabulary costs nothing a human typed."""
    service = _service(
        enrolled,
        [
            _row(
                2,
                "S-1",
                BROTHER,
                name_en="Karim",
                relationship=RelationshipType.SIBLING,
                label="big brother",
            )
        ],
    )
    _preview(service)
    _commit(service)

    (link,) = enrolled.student_guardians.all()
    assert link.relationship_type is RelationshipType.SIBLING
    assert link.relationship_label == "big brother"


def test_committing_the_same_batch_twice_is_refused(enrolled: FakeUnitOfWork) -> None:
    """What makes a double-clicked button safe: the first outcome stands."""
    service = _service(enrolled, [_row(2, "S-1", MOTHER)])
    _preview(service)
    _commit(service)

    with pytest.raises(ImportBatchAlreadyCommitted):
        _commit(service)

    assert len(enrolled.student_guardians.all()) == 1


def test_an_expired_preview_cannot_be_committed(enrolled: FakeUnitOfWork) -> None:
    """The TTL stops a preview taken against last week's roll landing today."""
    parser = StubParser(
        ParseResult(rows=(_row(2, "S-1", MOTHER),), diagnostics=(), total_lines=1, headers=())
    )
    clock = iter([NOW, NOW + TTL + timedelta(minutes=1)])
    service = GuardianImportService(
        lambda: enrolled,
        parser,
        preview_ttl=TTL,
        clock=lambda: next(clock),
        batch_ids=lambda: "batch-1",
    )
    _preview(service)

    with pytest.raises(ImportBatchExpired):
        _commit(service)


def test_an_unknown_batch_is_refused(enrolled: FakeUnitOfWork) -> None:
    service = _service(enrolled, [_row(2, "S-1", MOTHER)])

    with pytest.raises(ImportBatchNotFound):
        _commit(service, batch_id="no-such-batch")


def test_a_roster_batch_cannot_be_committed_as_guardians(
    enrolled: FakeUnitOfWork,
) -> None:
    """Kind is checked, never inferred: the wrong batch would write the wrong table."""
    from sis.domain.imports import ImportBatch

    enrolled.imports.add(
        ImportBatch(
            batch_id="roster-1",
            kind=ImportKind.ROSTER,
            content_hash="abc",
            actor="registrar",
            created_at=NOW,
            expires_at=NOW + TTL,
        ),
        [],
    )
    service = _service(enrolled, [])

    with pytest.raises(ImportContentMismatch):
        _commit(service, batch_id="roster-1")


def test_an_oversized_upload_is_refused_before_parsing(
    enrolled: FakeUnitOfWork,
) -> None:
    """Checked before the parser runs; discovering it afterwards has already spent the memory."""
    parser = StubParser(ParseResult(rows=(), diagnostics=(), total_lines=0, headers=()))
    service = GuardianImportService(
        lambda: enrolled, parser, preview_ttl=TTL, clock=lambda: NOW, max_upload_bytes=4
    )

    with pytest.raises(UploadTooLarge):
        service.preview(
            GuardianPreviewCommand(
                filename="g.csv", content=b"far too long", actor="registrar"
            )
        )
    assert parser.calls == []


def test_a_second_upload_of_the_same_file_reports_unchanged(
    enrolled: FakeUnitOfWork,
) -> None:
    """`UNCHANGED` rather than `UPDATED`, so a registrar can see the upload did nothing.

    Collapsing the two would make every re-import look like real work and hide the three
    rows that were the reason for it.
    """
    first = _service(enrolled, [_row(2, "S-1", MOTHER)])
    _preview(first)
    _commit(first)

    second = _service(enrolled, [_row(2, "S-1", MOTHER)])
    _preview(second)
    result = _commit(second)

    assert result.ok_count == 1
    assert len(enrolled.student_guardians.all()) == 1
    stored = enrolled.imports.list_rows("batch-1")
    assert [row.outcome for row in stored] == [StoredOutcome.UNCHANGED]


def test_a_changed_relationship_reports_updated(enrolled: FakeUnitOfWork) -> None:
    """The three rows that *are* the reason for a re-upload must stand out from the 200."""
    first = _service(enrolled, [_row(2, "S-1", MOTHER, relationship=RelationshipType.MOTHER)])
    _preview(first)
    _commit(first)

    second = _service(
        enrolled, [_row(2, "S-1", MOTHER, relationship=RelationshipType.GUARDIAN)]
    )
    _preview(second)
    _commit(second)

    stored = enrolled.imports.list_rows("batch-1")
    assert [row.outcome for row in stored] == [StoredOutcome.UPDATED]
    (link,) = enrolled.student_guardians.all()
    assert link.relationship_type is RelationshipType.GUARDIAN


def test_a_link_created_by_someone_else_since_the_preview_is_refused(
    enrolled: FakeUnitOfWork,
) -> None:
    """The row the registrar approved as `created` is not this row any more.

    This is also what stops the same file committed in two tabs from re-applying the
    first tab's work over the second's.
    """
    service = _service(enrolled, [_row(2, "S-1", MOTHER)])
    _preview(service)

    # Somebody else links the same pair, with a different relationship, in between.
    enrolled.guardians.upsert_many(
        [Guardian(phones=(Phone(MOTHER),), full_name_en="Fatma Ali")]
    )
    enrolled.student_guardians.upsert_many(
        [
            StudentGuardian(
                student_number=StudentNumber("S-1"),
                guardian_phone=Phone(MOTHER),
                relationship_type=RelationshipType.GUARDIAN,
                can_view_records=False,
            )
        ]
    )

    result = _commit(service)

    (row,) = result.rows
    assert row.code is RowCode.CHANGED_SINCE_PREVIEW
    # The other registrar's decision stands rather than being silently overwritten.
    (link,) = enrolled.student_guardians.all()
    assert link.relationship_type is RelationshipType.GUARDIAN
    assert link.can_view_records is False


def test_nothing_is_written_when_the_transaction_is_not_committed(
    enrolled: FakeUnitOfWork,
) -> None:
    """The fake rolls back for real, so "preview commits exactly once" is assertable."""
    service = _service(enrolled, [_row(2, "S-1", MOTHER)])
    _preview(service)

    assert enrolled.commits == 1
