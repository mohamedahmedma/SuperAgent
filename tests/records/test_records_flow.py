"""End to end through the facade: adapter -> assembler -> contract.

The unit tests either side of this cover the adapter's transport and the assembler's
rules. What is asserted here is that the wiring between them survives — that a figure the
system of record computed reaches a parent unaltered, and that an unreachable system of
record produces a refusal rather than a guess.
"""
from dataclasses import replace

from records.domain.errors import LmsUnavailable
from records.domain.marks import SubjectGrade
from tests.records.conftest import agent_headers


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

    def test_a_subject_is_named_in_both_scripts_when_the_school_has_both(self, client, fake_lms):
        """The names come from the system of record now, which is the only place that has
        them: there is no binding table left to override a course title with.

        Falling back to the other script rather than to blank is the rule worth pinning —
        a subject rendered with no name at all is worse than one rendered in the wrong
        language, and this service holds no translation to invent the missing side with.
        """
        fake_lms.grades[("S-1001", "2026-T1")][0] = replace(
            fake_lms.grades[("S-1001", "2026-T1")][0], subject_name_ar="الرياضيات"
        )
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


class TestTimetableFlow:
    """The fifth read, and the only one whose subject is a room rather than the child.

    What the wiring has to preserve is the indirection: the caller names a student, the
    system of record resolves the class, and the class never appears in the request. Plus
    the three-way `status`, which is the one thing a consumer must not have to infer.
    """

    def test_the_week_reaches_the_contract_named_and_ordered(self, client):
        response = client.get(
            "/v1/guardians/G-1/students/S-1001/timetable", headers=agent_headers("G-1")
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["class_code"] == "3A"
        assert body["class_name_ar"] == "الثالث أ"
        assert body["days"][0] == "sunday"
        assert body["teaching_slots"] == 5
        assert body["lessons"][0]["subject_name_ar"] == "الرياضيات"
        # The child still travels on the payload, so the agent can say who it answered about.
        assert body["student"]["student_id"] == "S-1001"

    def test_the_break_and_the_free_period_survive_as_themselves(self, client):
        """The two rows most easily mistaken for a lesson.

        A break drawn as a lesson invents one the child does not have; a stated free period
        dropped from the list turns a finished week into one that looks unfinished.
        """
        body = client.get(
            "/v1/guardians/G-1/students/S-1001/timetable", headers=agent_headers("G-1")
        ).json()

        assert [p["is_teaching"] for p in body["periods"]] == [True, False]
        assert body["periods"][1]["name_ar"] == "فسحة"
        # An unfixed bell stays empty rather than acquiring a plausible hour.
        assert body["periods"][1]["starts_at"] == ""
        assert body["lessons"][1]["subject_code"] == ""

    def test_the_request_never_names_a_class(self, client, fake_timetables):
        """The property the whole design rests on.

        A parent has no class code and this service holds none. The port is asked for a
        student and a term, and the room is resolved on the other side of it — which is what
        keeps a code that changes mid-year from being cached anywhere up here.
        """
        client.get(
            "/v1/guardians/G-1/students/S-1001/timetable", headers=agent_headers("G-1")
        )

        assert fake_timetables.asked == [("S-1001", "2026-T1", "G-1")]

    def test_the_guardian_travels_to_the_system_of_record(self, client, fake_timetables):
        """So SIS re-checks the link itself, from the registrar's own data, on this read.

        Asserted here because nothing else would fail if a route stopped passing it: the
        week comes back identical and only the second refusal disappears.
        """
        client.get(
            "/v1/guardians/G-1/students/S-1001/timetable", headers=agent_headers("G-1")
        )

        (_, _, guardian_ref) = fake_timetables.asked[0]
        assert guardian_ref == "G-1"

    def test_a_child_who_is_not_this_guardian_s_is_never_asked_about(
        self, client, fake_timetables
    ):
        """Step 3 never runs before step 1 succeeds — the stated security property.

        The arguments are all known before the link check returns, so the two calls *could*
        be issued together. They are not, and this is what pins it: a restricted guardian
        must not cause the system of record to be asked about the child at all.
        """
        refused = client.get(
            "/v1/guardians/G-2/students/S-1001/timetable", headers=agent_headers("G-2")
        )

        assert refused.status_code == 404
        assert fake_timetables.asked == []

    def test_no_class_and_no_timetable_stay_different_on_the_wire(
        self, client, fake_timetables
    ):
        """Both are empty; only one of them means the school still has typing to do."""
        no_class = client.get(
            "/v1/guardians/G-1/students/S-1001/timetable?term=2026-T1",
            headers=agent_headers("G-1"),
        )
        # The fixture holds a week for this child; drop it and she has no placement.
        fake_timetables.weeks.clear()
        after = client.get(
            "/v1/guardians/G-1/students/S-1001/timetable?term=2026-T1",
            headers=agent_headers("G-1"),
        )

        assert no_class.json()["status"] == "ok"
        assert after.json()["status"] == "no_class"
        assert after.json()["class_code"] == ""
        assert after.json()["lessons"] == []


class TestClassroomFlow:
    """Three routes over one read, and the projections that must not lose anything.

    What the wiring has to preserve: the class NAME (not just its code), the rung's subject
    board in the school's order, every teacher including a co-teacher, and no staff contact
    details anywhere. Plus the property the design rests on — three URLs, one read, one
    room.
    """

    def _get(self, client, path, guardian="G-1", student="S-1001"):
        return client.get(
            f"/v1/guardians/{guardian}/students/{student}/{path}",
            headers=agent_headers(guardian),
        )

    def test_the_class_name_is_what_reaches_the_contract(self, client):
        """A parent asked what class their child is in; the answer is its NAME.

        The fixture's code and name differ on purpose, so reading the code by mistake fails
        rather than passing by coincidence.
        """
        response = self._get(client, "class")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["class_name_ar"] == "الثالث ١"
        assert body["class_name_en"] == "Primary 3 Class 1"
        # Carried for correlation, but it is not the answer.
        assert body["class_code"] == "3A"
        assert body["year_level_name_ar"] == "الصف الثالث"

    def test_the_subject_board_arrives_named_and_in_the_school_s_order(self, client):
        body = self._get(client, "subjects").json()

        assert body["status"] == "ok"
        assert [s["code"] for s in body["subjects"]] == ["MATH", "SCI"]
        assert [s["name_ar"] for s in body["subjects"]] == ["الرياضيات", "العلوم"]

    def test_every_teacher_reaches_the_parent_including_a_co_teacher(self, client):
        """The failure this guards: a subject with two teachers reported as having one.

        Legal in the system of record, guarded in only one of its two write paths, and
        invisible from any fixture that gives each subject a single teacher.
        """
        body = self._get(client, "teachers").json()

        assert body["status"] == "ok"
        pairs = {(t["subject_code"], t["full_name_ar"]) for t in body["teachers"]}
        assert pairs == {
            ("MATH", "أ. سامي"),
            ("SCI", "أ. هدى"),
            ("SCI", "أ. منى"),
        }

    def test_no_staff_contact_details_exist_on_the_contract_at_all(self, client):
        """The shape is the privacy boundary, so this asserts the shape.

        A parent needs to know who teaches their child, not how to reach a member of staff
        directly. There is deliberately nowhere on `ClassTeacherOut` to put an email, a
        phone number or a staff number, so a future adapter cannot leak one by filling a
        field in.
        """
        body = self._get(client, "teachers").json()

        forbidden = {"email", "phone", "staff_number", "username", "user_id"}
        for teacher in body["teachers"]:
            assert forbidden.isdisjoint(teacher.keys()), teacher

    def test_three_routes_ask_the_system_of_record_once_each(
        self, client, fake_classrooms
    ):
        """Three URLs, three reads — but each read asks about ONE room.

        The routes are separate because the questions are; the READ is not divisible, which
        is why one port method serves all three. This pins that each route asks once and
        names only a student and a term: no class code is ever sent, because this service
        holds none.
        """
        self._get(client, "class")
        self._get(client, "subjects")
        self._get(client, "teachers")

        assert fake_classrooms.asked == [
            ("S-1001", "2026-T1", "G-1"),
            ("S-1001", "2026-T1", "G-1"),
            ("S-1001", "2026-T1", "G-1"),
        ]

    def test_a_child_who_is_not_this_guardian_s_is_never_asked_about(
        self, client, fake_classrooms
    ):
        """Step 3 never runs before step 1 succeeds, on all three routes."""
        for path in ("class", "subjects", "teachers"):
            refused = self._get(client, path, guardian="G-2")
            assert refused.status_code == 404, path

        assert fake_classrooms.asked == []

    def test_no_class_this_term_is_a_status_and_not_an_empty_room(
        self, client, fake_classrooms
    ):
        """The one emptiness that is about the CHILD, kept apart from the other two.

        With no placement there is no room to ask about, so the lists are empty for a
        reason that has nothing to do with the school's admin — and a consumer must be able
        to tell which it is looking at.
        """
        fake_classrooms.rooms.clear()

        for path, key in (("subjects", "subjects"), ("teachers", "teachers")):
            body = self._get(client, path).json()
            assert body["status"] == "no_class", path
            assert body["class_code"] == "", path
            assert body[key] == [], path

        klass = self._get(client, "class").json()
        assert klass["status"] == "no_class"
        assert klass["class_name_ar"] == ""


class TestHonestFailure:
    def test_an_unreachable_classroom_is_a_503_not_a_child_with_no_teachers(
        self, client, fake_classrooms
    ):
        """An empty room would tell a parent her daughter has no class and no teachers.

        Same code as the marks path — `lms_unavailable` — deliberately: the agent is written
        against that one signal.
        """
        fake_classrooms.unavailable = True

        for path in ("class", "subjects", "teachers"):
            response = client.get(
                f"/v1/guardians/G-1/students/S-1001/{path}", headers=agent_headers("G-1")
            )
            assert response.status_code == 503, path
            assert response.json()["detail"]["code"] == "lms_unavailable", path

    def test_an_unreachable_timetable_is_a_503_not_an_empty_week(
        self, client, fake_timetables
    ):
        """An empty week would tell a parent her daughter has no lessons.

        Same code as the marks path — `lms_unavailable` — deliberately: the agent is written
        against that one signal, and a second failure vocabulary is one it would eventually
        handle wrong.
        """
        fake_timetables.unavailable = True

        response = client.get(
            "/v1/guardians/G-1/students/S-1001/timetable", headers=agent_headers("G-1")
        )

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "lms_unavailable"

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

    def test_an_outage_is_reported(self, client, caplog, fake_lms):
        """A spike of these against one student is how a sync problem gets noticed
        before a parent reports it."""
        fake_lms.unavailable = True
        with caplog.at_level("WARNING"):
            client.get(
                "/v1/guardians/G-1/students/S-1001/grades", headers=agent_headers("G-1")
            )

        assert "lms_unavailable" in caplog.text
        assert "S-1001" in caplog.text

