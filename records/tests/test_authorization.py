"""The authorisation rules, asserted.

These are the tests that justify the service existing. If any of them regress, a
parent can read another family's child's record — so they assert behaviour, not
implementation, and they cover the denial paths at least as heavily as the happy one.

Every parent-facing call now carries two credentials: the agent's API key and a
signed identity token naming the guardian. Tests for the second live in
`test_identity.py`; these assume it is valid and exercise the link table behind it.
"""
from records.models import AccessAudit
from records.tests.conftest import admin_headers, agent_headers


def test_no_key_is_rejected(client):
    response = client.get("/v1/guardians/G-1/students")
    assert response.status_code == 401


def test_invalid_key_is_rejected(client):
    response = client.get("/v1/guardians/G-1/students", headers={"X-API-Key": "nope-not-a-real-key"})
    assert response.status_code == 401


def test_admin_key_cannot_read_student_records(client):
    """Scopes do not nest.

    The admin key is the one shared with registrars and scripts. If it also read
    records, the school's most-copied credential would be its most dangerous one.
    """
    response = client.get("/v1/guardians/G-1/students/S-1001/grades", headers=admin_headers())
    assert response.status_code == 403


def test_agent_key_cannot_manage_links(client):
    """The reverse direction: a leaked agent key must not be able to grant itself access."""
    response = client.post(
        "/v1/admin/guardians/G-2/students",
        headers=agent_headers(),
        json={"student_id": "S-1001", "can_view_records": True},
    )
    assert response.status_code == 403


def test_permitted_guardian_sees_their_child(client):
    response = client.get("/v1/guardians/G-1/students", headers=agent_headers("G-1"))
    assert response.status_code == 200
    students = response.json()["students"]
    assert [s["student_id"] for s in students] == ["S-1001"]


def test_restricted_guardian_sees_no_children(client):
    """Linked, but `can_view_records` is False. The link must not imply access."""
    response = client.get("/v1/guardians/G-2/students", headers=agent_headers("G-2"))
    assert response.status_code == 200
    assert response.json()["students"] == []


def test_restricted_guardian_cannot_read_grades(client):
    response = client.get("/v1/guardians/G-2/students/S-1001/grades", headers=agent_headers("G-2"))
    assert response.status_code == 404


def test_unrelated_guardian_cannot_read_grades(client):
    response = client.get("/v1/guardians/G-3/students/S-1001/grades", headers=agent_headers("G-3"))
    assert response.status_code == 404


def test_guardian_cannot_read_another_familys_child(client):
    """The core leak this service prevents.

    A valid token, a valid key, the guardian's own id in the path — and still no,
    because the link table says this is not their child.
    """
    response = client.get("/v1/guardians/G-1/students/S-2002/grades", headers=agent_headers("G-1"))
    assert response.status_code == 404


def test_denial_reasons_are_indistinguishable_to_the_caller(client):
    """Restricted, unrelated and nonexistent must look identical from outside.

    A caller who can tell them apart can enumerate the student body and detect the
    existence of a custody restriction from error codes alone.
    """
    responses = [
        client.get("/v1/guardians/G-2/students/S-1001/grades", headers=agent_headers("G-2")),
        client.get("/v1/guardians/G-3/students/S-1001/grades", headers=agent_headers("G-3")),
        client.get("/v1/guardians/G-1/students/S-9999/grades", headers=agent_headers("G-1")),
    ]
    assert {r.status_code for r in responses} == {404}
    assert len({r.json()["detail"]["message"] for r in responses}) == 1


def test_denials_are_audited_with_the_real_reason(client, db):
    """The response hides why; the audit must not."""
    client.get("/v1/guardians/G-2/students/S-1001/grades", headers=agent_headers("G-2"))

    row = (
        db.query(AccessAudit)
        .filter(AccessAudit.guardian_external_id == "G-2")
        .order_by(AccessAudit.id.desc())
        .first()
    )
    assert row is not None
    assert row.allowed is False
    # `no_children`, not `records_restricted`. Guardian links moved to SIS, which filters
    # restricted ones out before answering, so a barred parent and an unknown handle now
    # arrive here identical. That indistinguishability is the point on the *response* side
    # and a genuine loss on the *audit* side: this service can no longer record which of
    # the two it was. The reason worth keeping is still recorded — a run of these against
    # one handle is somebody probing — and the full detail now lives in SIS's own audit.
    assert row.reason == "no_children"


def test_successful_reads_are_audited(client, db):
    client.get("/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1"))

    row = (
        db.query(AccessAudit)
        .filter(AccessAudit.guardian_external_id == "G-1", AccessAudit.allowed.is_(True))
        .order_by(AccessAudit.id.desc())
        .first()
    )
    assert row is not None
    assert row.student_external_id == "S-1001"
    assert row.reason == "ok"


def test_request_id_is_carried_into_the_audit(client, db):
    """Ties a records read back to the chat turn that caused it."""
    client.get(
        "/v1/guardians/G-1/students/S-1001/grades",
        headers={**agent_headers("G-1"), "X-Request-Id": "turn-abc-123"},
    )

    row = db.query(AccessAudit).filter(AccessAudit.request_id == "turn-abc-123").first()
    assert row is not None


def test_the_old_granting_route_is_gone_rather_than_quietly_useless(client):
    """It used to write this service's tables. Those tables are no longer read.

    Left alone it would accept the write, answer 201, and change nothing — so a registrar
    would believe a parent had been granted access to their child's records when nobody
    had. 410 says the capability moved and names where, which is the one answer that
    cannot be mistaken for success.
    """
    refused = client.post(
        "/v1/admin/guardians/G-3/students",
        headers=admin_headers(),
        json={"student_id": "S-2002", "can_view_records": True},
    )
    assert refused.status_code == 410
    assert refused.json()["detail"]["code"] == "moved"

    # And the grant genuinely did not happen.
    after = client.get("/v1/guardians/G-3/students", headers=agent_headers("G-3"))
    assert after.json()["students"] == []


def test_link_without_explicit_grant_reveals_nothing(client):
    """A half-finished import must not leak."""
    client.post(
        "/v1/admin/guardians/G-3/students",
        headers=admin_headers(),
        json={"student_id": "S-2002"},
    )

    response = client.get("/v1/guardians/G-3/students", headers=agent_headers("G-3"))
    assert response.json()["students"] == []
