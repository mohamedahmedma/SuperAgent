"""Who was told about which child, and when — recorded where the decision is made.

The audit moved here from `records/`. It sat there while the facade held the guardian
tables and made the call; it holds neither now, so an audit kept there would answer "what
did the relay pass on" rather than "who was told about this child".

Two properties carry the weight, and both are about refusals:

**A denial is recorded even though the caller is told nothing.** `no_children` and
`no_link` are the same `UnknownReference` on the wire — a caller who could tell them apart
could walk student numbers, or detect a custody restriction from outside the school. The
distinction survives here, which is the only place it is safe for it to exist.

**A denial's row survives the denial.** It is written in its own transaction and committed
before the refusal is raised. An audit rolled back alongside the request that failed
records only the accesses that succeeded, which is exactly backwards.
"""
from datetime import UTC, datetime

import pytest

from sis.application.services.queries import QueryService
from sis.domain.access import AccessAttempt, AccessReason
from sis.domain.errors import UnknownReference
from sis.domain.guardians import Guardian, RelationshipType, StudentGuardian
from sis.domain.people import Student
from sis.domain.value_objects import Phone, StudentNumber
from sis.tests.conftest import FakeUnitOfWork, reader_headers, registrar_headers

MOTHER = "+201001234567"
BARRED = "+201005554444"


@pytest.fixture()
def family() -> FakeUnitOfWork:
    """One mother who may read S001, and one adult linked to her but barred."""
    uow = FakeUnitOfWork()
    with uow:
        uow.students.upsert_many(
            [Student(student_number="S001", full_name_en="Layla", full_name_ar="ليلى")]
        )
        uow.guardians.upsert_many(
            [
                Guardian(full_name_ar="فاطمة", phones=(Phone(MOTHER),)),
                Guardian(full_name_ar="كريم", phones=(Phone(BARRED),)),
            ]
        )
        uow.student_guardians.upsert_many(
            [
                StudentGuardian(
                    student_number=StudentNumber("S001"),
                    guardian_phone=Phone(MOTHER),
                    relationship_type=RelationshipType.MOTHER,
                    can_view_records=True,
                ),
                StudentGuardian(
                    student_number=StudentNumber("S001"),
                    guardian_phone=Phone(BARRED),
                    relationship_type=RelationshipType.OTHER,
                    can_view_records=False,
                ),
            ]
        )
        uow.commit()
    return uow


@pytest.fixture()
def permitted(family: FakeUnitOfWork) -> str:
    """The mother's handle, asked of the repository rather than assumed.

    A guardian does not carry her own `public_id` — it is assigned by whatever stores her,
    which is the point of it being opaque. Deriving it here the way a caller would is what
    keeps this suite honest about the shape of the real lookup.
    """
    with family:
        return family.guardians.public_id_for(Phone(MOTHER))


@pytest.fixture()
def barred(family: FakeUnitOfWork) -> str:
    with family:
        return family.guardians.public_id_for(Phone(BARRED))


def _service(uow: FakeUnitOfWork) -> QueryService:
    """One unit of work handed back every time, so the audit written in its own
    transaction lands where the test can read it."""
    return QueryService(lambda: uow)


class TestWhatIsRecorded:
    def test_a_permitted_read_is_recorded(self, family, permitted):
        _service(family).require_guardian_may_see(permitted, StudentNumber("S001"))

        with family:
            rows = family.access_audit.recent()
        assert len(rows) == 1
        assert rows[0].allowed is True
        assert rows[0].reason is AccessReason.OK
        assert rows[0].guardian_public_id == permitted
        assert rows[0].student_number == "S001"

    def test_a_restricted_guardian_is_recorded_as_reaching_nobody(self, family, barred):
        """Her only link carries `can_view_records: false`, so she has no children to be
        told about — indistinguishable from an unknown handle to the caller, and recorded
        as `no_children` here."""
        with pytest.raises(UnknownReference):
            _service(family).require_guardian_may_see(barred, StudentNumber("S001"))

        with family:
            rows = family.access_audit.recent()
        assert rows[0].allowed is False
        assert rows[0].reason is AccessReason.NO_CHILDREN

    def test_a_real_parent_naming_another_child_is_recorded_as_no_link(self, family, permitted):
        """The signal worth alerting on: somebody walking student numbers against a
        handle that genuinely belongs to a parent of this school."""
        with pytest.raises(UnknownReference):
            _service(family).require_guardian_may_see(permitted, StudentNumber("S999"))

        with family:
            rows = family.access_audit.recent()
        assert rows[0].allowed is False
        assert rows[0].reason is AccessReason.NO_LINK
        assert rows[0].student_number == "S999"

    def test_an_unknown_handle_is_recorded_and_still_refused(self, family):
        with pytest.raises(UnknownReference):
            _service(family).require_guardian_may_see("G-nobody", StudentNumber("S001"))

        with family:
            rows = family.access_audit.recent()
        assert len(rows) == 1
        assert rows[0].reason is AccessReason.NO_CHILDREN

    def test_the_two_refusals_are_one_answer_outside_and_two_facts_inside(
        self, family, permitted, barred
    ):
        """The whole reason this table exists.

        A caller cannot tell a barred guardian from a stranger, or either from a child who
        does not exist. Somebody reading the audit can, and that difference is what turns
        a pile of 404s into "this handle is probing".
        """
        service = _service(family)
        refusals = []
        for guardian, student in ((barred, "S001"), (permitted, "S999")):
            with pytest.raises(UnknownReference) as refused:
                service.require_guardian_may_see(guardian, StudentNumber(student))
            refusals.append(str(refused.value))

        assert refusals[0] == refusals[1], "the two refusals differed on the wire"

        with family:
            reasons = {row.reason for row in family.access_audit.recent()}
        assert reasons == {AccessReason.NO_CHILDREN, AccessReason.NO_LINK}

    def test_the_caller_and_the_correlation_id_are_carried(self, family, permitted):
        _service(family).require_guardian_may_see(
            permitted, StudentNumber("S001"), actor="reader-abc12", request_id="turn-77"
        )

        with family:
            row = family.access_audit.recent()[0]
        assert row.actor == "reader-abc12"
        assert row.request_id == "turn-77"

    def test_a_failing_audit_does_not_fail_the_read(self, family, permitted, caplog):
        """Best effort, and loudly. A school whose audit table is briefly unwritable
        should still be able to tell a parent her daughter's marks."""
        def _explode(_attempt):
            raise RuntimeError("the audit table is on fire")

        family.access_audit.record = _explode
        _service(family).require_guardian_may_see(permitted, StudentNumber("S001"))

        assert "could not record an access attempt" in caplog.text


class TestTheDomainRefusesNonsense:
    def test_a_naive_timestamp_is_refused(self):
        """An audit that reads as local time to whoever queries it next answers "when"
        wrongly, and "when" is the whole point."""
        from sis.domain.errors import ValidationError

        with pytest.raises(ValidationError):
            AccessAttempt(
                guardian_public_id="G-1",
                student_number="S001",
                reason=AccessReason.OK,
                at=datetime(2026, 3, 1, 9, 0),
            )

    def test_an_attempt_naming_no_guardian_is_refused(self):
        from sis.domain.errors import ValidationError

        with pytest.raises(ValidationError):
            AccessAttempt(
                guardian_public_id="   ",
                student_number="S001",
                reason=AccessReason.OK,
                at=datetime.now(UTC),
            )

    def test_only_ok_counts_as_allowed(self):
        assert AccessReason.OK.is_allowed
        assert not AccessReason.NO_LINK.is_allowed
        assert not AccessReason.NO_CHILDREN.is_allowed


class TestOverHttp:
    """The route, and who may read it."""

    def test_a_reader_key_cannot_read_the_audit(self, client):
        """`records/` holds the reader key. It must be able to read a child's marks and
        must not be able to read the log of who else has been reading them."""
        response = client.get("/v1/admin/access-audit", headers=reader_headers())
        assert response.status_code == 403

    def test_an_anonymous_caller_cannot_read_the_audit(self, unauthenticated_client):
        assert unauthenticated_client.get("/v1/admin/access-audit").status_code == 401

    def test_a_registrar_reads_it(self, client):
        response = client.get("/v1/admin/access-audit", headers=registrar_headers())
        assert response.status_code == 200
        assert response.json() == []

    def test_there_is_no_way_to_write_or_delete_a_row(self, client):
        """Append-only by contract, and the contract is the absence of the route."""
        for method in (client.post, client.put, client.patch, client.delete):
            response = method("/v1/admin/access-audit", headers=registrar_headers())
            assert response.status_code == 405, (
                f"{method.__name__.upper()} /v1/admin/access-audit exists; the audit is "
                "supposed to be append-only with no edit path anywhere."
            )

    def test_a_refused_parent_read_lands_in_the_audit(self, client, registrar_headers_):
        """End to end: a guardian-scoped read that SIS refuses is recorded as refused."""
        refused = client.get(
            "/v1/guardians/by-id/G-nobody/students/S001/grades",
            params={"term": "2026-T1"},
            headers={**reader_headers(), "X-Request-Id": "turn-9"},
        )
        assert refused.status_code == 404

        rows = client.get(
            "/v1/admin/access-audit", params={"allowed": False}, headers=registrar_headers_
        ).json()
        assert rows, "a refused parent-facing read was not recorded"
        assert rows[0]["guardian_id"] == "G-nobody"
        assert rows[0]["allowed"] is False
        assert rows[0]["request_id"] == "turn-9"


@pytest.fixture()
def registrar_headers_() -> dict[str, str]:
    return registrar_headers()
