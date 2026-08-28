"""The authorisation rules, asserted.

These are the tests that justify the service existing. If any of them regress, a
parent can read another family's child's record — so they assert behaviour, not
implementation, and they cover the denial paths at least as heavily as the happy one.

Every parent-facing call now carries two credentials: the agent's API key and a
signed identity token naming the guardian. Tests for the second live in
`test_identity.py`; these assume it is valid and exercise the link table behind it.
"""
from tests.records.conftest import agent_headers


def test_no_key_is_rejected(client):
    response = client.get("/v1/guardians/G-1/students")
    assert response.status_code == 401


def test_invalid_key_is_rejected(client):
    response = client.get("/v1/guardians/G-1/students", headers={"X-API-Key": "nope-not-a-real-key"})
    assert response.status_code == 401


def test_there_is_no_credential_that_can_grant_access(client):
    """This service used to have two scopes, and the pairing was the point: an admin key
    managed links and could not read records, an agent key read records and could not
    grant itself access.

    Both halves are gone, and the property is stronger for it. There is no admin scope
    because there is nothing to administer — guardians and their custody flags are the
    registrar's, in `sis/` — so the route that once granted access cannot be reached with
    any credential at all. **410, not 403.** A refusal invites somebody to find the right
    key; this says the capability does not exist here.
    """
    response = client.post(
        "/v1/admin/guardians/G-2/students",
        headers=agent_headers(),
        json={"student_id": "S-1001", "can_view_records": True},
    )
    assert response.status_code == 410
    assert response.json()["detail"]["code"] == "moved"


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


def test_a_denial_is_not_reported_twice(client, caplog):
    """An answer about a child belongs to `sis/`, and only to `sis/`.

    A refused read is recorded there with the reason that was actually true —
    `no_link` or `no_children` — in a table a school can query. This service must not
    also emit it: the response deliberately withholds the distinction, and a log line
    here would both double-count the event and put back the thing the 404 hides.

    What this service does report is everything that never reaches SIS at all — a bad
    key, an unverifiable token, a guardian mismatch. Those have nowhere else to go.
    """
    with caplog.at_level("WARNING"):
        response = client.get(
            "/v1/guardians/G-2/students/S-1001/grades", headers=agent_headers("G-2")
        )

    assert response.status_code == 404
    assert "records.access.refused" not in caplog.text


def test_a_bad_key_is_reported_because_sis_never_hears_about_it(client, caplog):
    """The other half of the split, asserted so the two stay distinguishable."""
    with caplog.at_level("WARNING"):
        client.get(
            "/v1/guardians/G-1/students/S-1001/grades",
            headers={"X-API-Key": "not-the-configured-secret"},
        )

    assert "records.access.refused" in caplog.text
    assert "not_authorized" in caplog.text


def test_the_old_granting_route_is_gone_rather_than_quietly_useless(client):
    """It used to write this service's tables. Those tables are no longer read.

    Left alone it would accept the write, answer 201, and change nothing — so a registrar
    would believe a parent had been granted access to their child's records when nobody
    had. 410 says the capability moved and names where, which is the one answer that
    cannot be mistaken for success.
    """
    refused = client.post(
        "/v1/admin/guardians/G-3/students",
        headers=agent_headers("G-1"),
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
        headers=agent_headers("G-1"),
        json={"student_id": "S-2002"},
    )

    response = client.get("/v1/guardians/G-3/students", headers=agent_headers("G-3"))
    assert response.json()["students"] == []


# ---------------------------------------------------------------------------
# The subject reaches the system of record
# ---------------------------------------------------------------------------
#
# This service decides whether a read may proceed, and it always did. What it did NOT do
# was tell the system of record which parent it was reading for — the SIS adapter called
# the registrar-scoped routes, so the answer to "whose child is this" was computed here
# and then discarded at the last hop.
#
# It is carried now, and SIS re-checks it. These assert the carrying, because that is the
# half that fails silently: if a route stops passing the handle, every one of these
# suites still passes and the second check quietly stops happening.


def test_the_guardian_reaches_the_backend_on_a_grades_read(client, fake_lms):
    client.get("/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1"))

    assert fake_lms.asked, "the backend was never asked at all"
    assert [guardian for _, _, guardian in fake_lms.asked] == ["G-1"]


def test_the_guardian_reaches_the_backend_on_an_attendance_read(client, fake_lms):
    client.get(
        "/v1/guardians/G-1/students/S-1001/attendance", headers=agent_headers("G-1")
    )

    assert fake_lms.asked, "the backend was never asked at all"
    assert all(guardian == "G-1" for _, _, guardian in fake_lms.asked)


def test_the_guardian_sent_is_the_signed_one_not_the_one_in_the_path(client, fake_lms):
    """The path is a URL; the claim is a signature.

    A mismatch is refused outright, so what this pins is that the value handed onward is
    the one that was *proved* — not the one a caller typed. If these ever diverged, a
    compromised caller could satisfy this service with a real token and still name
    somebody else downstream.
    """
    client.get("/v1/guardians/G-2/students/S-1001/grades", headers=agent_headers("G-1"))

    assert fake_lms.asked == [], "a mismatched request reached the system of record"
