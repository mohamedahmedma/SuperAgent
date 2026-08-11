"""End to end through the facade: adapter -> assembler -> contract.

The unit tests either side of this cover the adapter's transport and the assembler's
rules. What is asserted here is that the wiring between them survives — that a figure
the system of record computed reaches a parent unaltered, that an unpublished course
does not, and that an unreachable LMS produces a refusal rather than a guess.
"""
from records.lms import LmsUnavailable, SubjectGrade
from records.models import AccessAudit, CourseBinding
from records.tests.conftest import agent_headers


class TestGradesFlow:
    def test_both_percentages_reach_the_contract(self, client):
        """The measured real case: 65% official, 80% academic, same subject.

        A school that grades attendance produces two true answers and they are not
        interchangeable. Losing either one in the wiring is invisible unless the
        fixture makes them differ.
        """
        response = client.get(
            "/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1")
        )

        assert response.status_code == 200
        course = response.json()["courses"][0]
        assert course["computed_percentage"] == 65.0
        assert course["academic"]["percentage"] == 80.0

    def test_each_figure_carries_its_own_letter(self, client):
        course = client.get(
            "/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1")
        ).json()["courses"][0]

        assert course["letter_grade"] == "D"
        assert course["academic"]["letter_grade"] == "B"

    def test_the_subject_name_comes_from_the_school_not_the_lms(self, client):
        course = client.get(
            "/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1")
        ).json()["courses"][0]

        assert course["subject_name_en"] == "Mathematics"
        assert course["subject_name_ar"] == "الرياضيات"

    def test_counts_reach_the_contract(self, client):
        course = client.get(
            "/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1")
        ).json()["courses"][0]

        assert course["excused_count"] == 1
        assert course["pending_count"] == 1
        assert course["is_complete"] is False

    def test_an_unpublished_course_never_reaches_a_parent(self, client, db, fake_lms):
        """A teacher's sandbox, or a subject whose grades are not released yet.

        The LMS answers for every course the student is enrolled in; the school decides
        which a parent may see.
        """
        db.query(CourseBinding).update({"is_published": False})
        db.commit()

        response = client.get(
            "/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1")
        )

        assert response.status_code == 200
        assert response.json()["courses"] == []

    def test_a_subject_with_no_binding_at_all_is_dropped(self, client, fake_lms):
        """The LMS reporting a course the school has not bound is not an error."""
        fake_lms.grades[("S-1001", "2026-T1-")] = [
            SubjectGrade(course_ref="2026-T1-G7A-SANDBOX", subject_name="Sandbox",
                         percentage=99.0),
        ]

        response = client.get(
            "/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1")
        )

        assert response.json()["courses"] == []


class TestAttendanceFlow:
    def test_the_term_figure_reaches_the_contract(self, client):
        response = client.get(
            "/v1/guardians/G-1/students/S-1001/attendance", headers=agent_headers("G-1")
        )

        assert response.status_code == 200
        body = response.json()
        # 7 of 8 points across the term — the system of record's own weighting.
        assert body["attendance_rate"] == 87.5
        assert body["total_sessions"] == 4

    def test_status_counts_are_mapped_from_their_descriptions(self, client):
        body = client.get(
            "/v1/guardians/G-1/students/S-1001/attendance", headers=agent_headers("G-1")
        ).json()

        assert body["present_count"] == 3
        assert body["late_count"] == 1

    def test_day_detail_is_absent_rather_than_invented(self, client):
        """The plugin summarises server-side, which is what keeps one child's register
        from arriving with their classmates' attached."""
        body = client.get(
            "/v1/guardians/G-1/students/S-1001/attendance", headers=agent_headers("G-1")
        ).json()

        assert body["recent_days"] == []


class TestHonestFailure:
    def test_an_unreachable_lms_is_a_503_not_a_guess(self, client, fake_lms):
        """The whole point of the failure path.

        The agent turns this into "I cannot reach the school records right now". Any
        other outcome — an empty list, a remembered figure — is worse than useless to a
        parent.
        """
        fake_lms.unavailable = True

        response = client.get(
            "/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1")
        )

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "lms_unavailable"

    def test_attendance_fails_the_same_way(self, client, fake_lms):
        fake_lms.unavailable = True

        response = client.get(
            "/v1/guardians/G-1/students/S-1001/attendance", headers=agent_headers("G-1")
        )

        assert response.status_code == 503

    def test_an_outage_is_audited(self, client, db, fake_lms):
        """A spike of these against one student is how a sync problem gets noticed
        before a parent reports it."""
        fake_lms.unavailable = True
        client.get("/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1"))

        row = (
            db.query(AccessAudit)
            .filter(AccessAudit.reason == "lms_unavailable")
            .order_by(AccessAudit.id.desc())
            .first()
        )
        assert row is not None
        assert row.allowed is False

    def test_no_published_courses_means_the_lms_is_never_asked(self, client, db, fake_lms):
        """If nothing is published there is nothing a parent may see, so there is no
        reason to ask the system of record about this child at all — and an outage
        must not turn an empty answer into an error."""
        db.query(CourseBinding).update({"is_published": False})
        db.commit()
        fake_lms.unavailable = True

        response = client.get(
            "/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1")
        )

        assert response.status_code == 200
        assert response.json()["courses"] == []
