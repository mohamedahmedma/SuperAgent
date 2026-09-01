"""The weekly plan, over HTTP.

A timetable is a plan, not a record, and every test here is about a way a plan can be
wrong rather than about a fact being preserved. That is the opposite emphasis from the
grades and attendance suites, and it is deliberate: the value of this table is the clashes
it refuses.

Five properties, one per requirement:

**Every class has its own week.** Two classes in the same school, same term, same slot are
two different lessons and neither knows about the other. The test that matters is the
negative: placing a lesson in 3A must not appear in 3B's week.

**The week is the school's.** `School.working_days` has said since it was added that it
existed for "the future timetable". A school that opens Sunday to Wednesday has a four-day
grid, and a Thursday lesson is refused — not stored and hidden, refused.

**Lessons sit in periods the school runs.** Including the negative cases that are easy to
get wrong: period 9 of a seven-period day, and the break.

**One slot holds one lesson, and one teacher is in one room.** The first is exercised over
HTTP. The second is asserted against the database directly, because teacher management does
not exist yet and the point of the constraint is that it is already law when it does.

**A class may only be timetabled a subject its grade is assigned.** Stage 5's rule arriving
one table further on, and the same test proves the Arabic and Languages sections stay
apart — they are different rungs with different assignments, so neither can borrow the
other's subjects.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.conftest import registrar_headers

SCHOOL = "TT"
YEAR = "TT-2026"
TERM = f"{YEAR}-T1"
#: Sunday to Wednesday. Deliberately *not* the five-day default, so every assertion that
#: Thursday is refused is a statement about this school rather than about a constant.
WEEK = ["sunday", "monday", "tuesday", "wednesday"]


@pytest.fixture()
def registrar() -> dict[str, str]:
    return registrar_headers()


@pytest.fixture()
def school(client: TestClient, registrar: dict[str, str]) -> None:
    """A bilingual school, one year, one primary rung per track, two classes on the Arabic one.

    Two classes because "every class has its own timetable" cannot be tested with one, and
    two tracks because the separation has to be shown rather than asserted.
    """
    assert client.post(
        "/v1/schools",
        json={
            "code": SCHOOL,
            "name_en": "Timetable School",
            "name_ar": "مدرسة الجدول",
            "language_type": "both",
            "kg_grade_count": 0,
            "primary_grade_count": 1,
            "preparatory_grade_count": 0,
            "secondary_grade_count": 1,
            "term_count": 1,
            "working_days": WEEK,
        },
        headers=registrar,
    ).status_code == 201
    assert client.post(
        "/v1/academic-years",
        json={
            "code": YEAR,
            "school_code": SCHOOL,
            "name_en": "2026/2027",
            "name_ar": "٢٠٢٦",
            "starts_on": "2026-09-01",
            "ends_on": "2027-06-30",
            "is_current": True,
        },
        headers=registrar,
    ).status_code == 201

    for code, track, stage in (
        ("AR-P1", "AR", "primary"),
        ("AR-S1", "AR", "secondary"),
        ("LG-P1", "LANG", "primary"),
    ):
        assert client.post(
            "/v1/structure/levels",
            json={
                "code": code,
                "school_code": SCHOOL,
                "track_code": track,
                "name_en": code,
                "name_ar": code,
                "display_order": 1,
                "stage": stage,
            },
            headers=registrar,
        ).status_code == 201

    for code, level in (("P1A", "AR-P1"), ("P1B", "AR-P1"), ("LGA", "LG-P1")):
        assert client.post(
            "/v1/structure/classes",
            json={
                "code": code,
                "academic_year_code": YEAR,
                "year_level_code": level,
                "name_en": code,
                "name_ar": code,
            },
            headers=registrar,
        ).status_code == 201

    for code in ("MATH", "PHYS"):
        assert client.post(
            "/v1/subjects",
            json={
                "code": code,
                "academic_year_code": YEAR,
                "name_en": code.title(),
                "name_ar": code,
            },
            headers=registrar,
        ).status_code == 201

    # Maths is taught on the Arabic primary rung; Physics only on the secondary one. That
    # is the stage-5 fact every subject assertion below rests on.
    for subject, level in (("MATH", "AR-P1"), ("PHYS", "AR-S1")):
        assert client.put(
            "/v1/subject-assignments",
            json={
                "academic_year_code": YEAR,
                "subject_code": subject,
                "year_level_code": level,
                "assigned": True,
            },
            headers=registrar,
        ).status_code == 204


def _periods(client: TestClient, headers: dict[str, str], *, count: int = 3):
    """A day of `count` teaching periods, with a break in the middle of a longer one."""
    body = [
        {"period_number": n, "name_en": f"Period {n}", "name_ar": f"حصة {n}"}
        for n in range(1, count + 1)
    ]
    return client.put(
        f"/v1/schools/{SCHOOL}/timetable-periods",
        json={"periods": body},
        headers=headers,
    )


def _place(client: TestClient, headers: dict[str, str], entries: list[dict]):
    return client.put(
        "/v1/timetable",
        json={"academic_year_code": YEAR, "entries": entries},
        headers=headers,
    )


def _lesson(class_code: str, day: str, period: int, subject: str | None = "MATH") -> dict:
    return {
        "class_code": class_code,
        "term_code": TERM,
        "day_of_week": day,
        "period_number": period,
        "subject_code": subject,
    }


def _week(client: TestClient, headers: dict[str, str], class_code: str) -> dict:
    response = client.get(
        f"/v1/timetable/week?academic_year={YEAR}&class_code={class_code}&term={TERM}",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


# -- The school's day -------------------------------------------------------


def test_a_school_lays_out_its_day_and_periods_may_be_untimed(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """Times are optional for the reason term dates are: the grid is agreed first."""
    empty = client.get(f"/v1/schools/{SCHOOL}/timetable-periods", headers=registrar)
    assert empty.status_code == 200, empty.text
    # A school with no bell schedule yet is an empty list, not a 404. It exists; its day
    # has not been decided.
    assert empty.json() == []

    stored = _periods(client, registrar, count=3)
    assert stored.status_code == 200, stored.text
    assert [row["period_number"] for row in stored.json()] == [1, 2, 3]
    assert all(row["starts_at"] is None for row in stored.json())
    assert all(row["is_timed"] is False for row in stored.json())

    timed = client.put(
        f"/v1/schools/{SCHOOL}/timetable-periods",
        json={
            "periods": [
                {"period_number": 1, "starts_at": "08:00:00", "ends_at": "08:45:00"},
                {"period_number": 2, "starts_at": "08:45:00", "ends_at": "09:30:00"},
                {"period_number": 3, "name_en": "Break", "name_ar": "فسحة", "is_teaching": False},
            ]
        },
        headers=registrar,
    )
    assert timed.status_code == 200, timed.text
    rows = timed.json()
    assert rows[0]["is_timed"] is True
    assert rows[2]["is_teaching"] is False and rows[2]["is_timed"] is False

    # An inverted stated range is still refused: optional widened what may be absent, not
    # what may be wrong.
    bad = client.put(
        f"/v1/schools/{SCHOOL}/timetable-periods",
        json={"periods": [{"period_number": 1, "starts_at": "10:00:00", "ends_at": "09:00:00"}]},
        headers=registrar,
    )
    assert bad.status_code == 422, bad.text


def test_shortening_the_day_is_refused_while_lessons_sit_in_the_period(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """Otherwise the lesson survives in a row the grid no longer draws — a silent orphan."""
    assert _periods(client, registrar, count=3).status_code == 200
    assert _place(client, registrar, [_lesson("P1A", "sunday", 3)]).status_code == 200

    refused = _periods(client, registrar, count=2)
    assert refused.status_code == 409, refused.text
    assert "3" in refused.json()["detail"]["message"]

    # Clear the lesson and the same call succeeds. The rule is about stranding data, not
    # about forbidding a school from changing its mind.
    assert client.post(
        "/v1/timetable/clear",
        json={
            "academic_year_code": YEAR,
            "slots": [
                {
                    "class_code": "P1A",
                    "term_code": TERM,
                    "day_of_week": "sunday",
                    "period_number": 3,
                }
            ],
        },
        headers=registrar,
    ).status_code == 200
    assert _periods(client, registrar, count=2).status_code == 200


# -- Every class has its own week -------------------------------------------


def test_every_class_has_its_own_timetable(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """Requirement 1, stated as the negative that could actually fail."""
    assert _periods(client, registrar, count=3).status_code == 200
    assert _place(
        client,
        registrar,
        [_lesson("P1A", "sunday", 1), _lesson("P1A", "monday", 2)],
    ).status_code == 200

    mine = _week(client, registrar, "P1A")
    assert [(row["day_of_week"], row["period_number"]) for row in mine["entries"]] == [
        ("sunday", 1),
        ("monday", 2),
    ]

    # The class next door is untouched. Same school, same rung, same term, same slots.
    assert _week(client, registrar, "P1B")["entries"] == []

    # And the same slot in the other class is a different lesson, not a clash.
    assert _place(client, registrar, [_lesson("P1B", "sunday", 1)]).status_code == 200
    assert len(_week(client, registrar, "P1A")["entries"]) == 2
    assert len(_week(client, registrar, "P1B")["entries"]) == 1


def test_the_week_is_the_schools_own_days_in_the_schools_own_order(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """Requirement 2. Only selected working days appear, and Thursday is refused."""
    assert _periods(client, registrar, count=3).status_code == 200

    plan = _week(client, registrar, "P1A")
    assert plan["days"] == WEEK, "the grid must offer the school's week, not a default one"
    assert "thursday" not in plan["days"]

    # Thursday is a real weekday and a real `WorkingDay`; this school simply shuts on it.
    # 422, not 409: the request could not have meant anything valid.
    refused = _place(client, registrar, [_lesson("P1A", "thursday", 1)])
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["field"] == "day_of_week"

    # And nothing landed. A refusal that half-applied would be worse than the mistake.
    assert _week(client, registrar, "P1A")["entries"] == []


def test_lessons_only_sit_in_periods_the_school_runs(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """Requirement 3's boundary cases: past the end of the day, and into the break."""
    assert client.put(
        f"/v1/schools/{SCHOOL}/timetable-periods",
        json={
            "periods": [
                {"period_number": 1},
                {"period_number": 2, "name_en": "Break", "name_ar": "فسحة", "is_teaching": False},
                {"period_number": 3},
            ]
        },
        headers=registrar,
    ).status_code == 200

    past_the_end = _place(client, registrar, [_lesson("P1A", "sunday", 9)])
    assert past_the_end.status_code == 422, past_the_end.text
    assert past_the_end.json()["detail"]["field"] == "period_number"

    into_the_break = _place(client, registrar, [_lesson("P1A", "sunday", 2)])
    assert into_the_break.status_code == 409, into_the_break.text

    assert _place(client, registrar, [_lesson("P1A", "sunday", 3)]).status_code == 200


def test_a_lesson_cannot_be_timetabled_before_the_school_has_a_day(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """An empty grid refuses loudly rather than accepting a lesson nothing can draw."""
    refused = _place(client, registrar, [_lesson("P1A", "sunday", 1)])
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["field"] == "period_number"


# -- Conflicts --------------------------------------------------------------


def test_one_slot_holds_one_lesson(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """Requirement 5. Re-placing is a replace, and two-in-one-batch is a refusal."""
    assert _periods(client, registrar, count=3).status_code == 200
    assert _place(client, registrar, [_lesson("P1A", "sunday", 1, "MATH")]).status_code == 200

    # The slot is the identity, so this replaces rather than adding a second lesson at the
    # same moment — which is what makes laying out a grid safe to click twice.
    again = _place(client, registrar, [_lesson("P1A", "sunday", 1, None)])
    assert again.status_code == 200, again.text
    entries = _week(client, registrar, "P1A")["entries"]
    assert len(entries) == 1
    assert entries[0]["subject_code"] is None

    # Two lessons for one slot inside a single request is a conflict, not a silent
    # last-one-wins. Last-one-wins is how a registrar loses a lesson without being told.
    clash = _place(
        client,
        registrar,
        [_lesson("P1A", "tuesday", 1, "MATH"), _lesson("P1A", "tuesday", 1, None)],
    )
    assert clash.status_code == 409, clash.text
    # Nothing from the refused batch landed.
    assert len(_week(client, registrar, "P1A")["entries"]) == 1


def test_one_teacher_cannot_be_in_two_rooms_at_once(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """The rule for a stage that has not happened yet, asserted where it lives.

    Teacher management is not in this stage, so there is no route that sets `teacher_id`
    and this cannot be exercised over HTTP. The constraint is written now because it costs
    nothing today and a migration later — and because the whole point of "prepare the model
    to connect with a teacher" is that the connection is already safe when it is made.

    It works because `teacher_id` is nullable and SQL treats NULLs as distinct: every
    unassigned lesson may share a slot, and one named teacher may not.
    """
    assert _periods(client, registrar, count=3).status_code == 200
    assert _place(
        client,
        registrar,
        [_lesson("P1A", "sunday", 1), _lesson("P1B", "sunday", 1)],
    ).status_code == 200

    with SqlAlchemyUnitOfWork() as unit:
        session = unit._session
        school_id = session.execute(
            text("SELECT id FROM schools WHERE code = :code"), {"code": SCHOOL}
        ).scalar_one()
        session.execute(
            text(
                "INSERT INTO teachers (staff_number, school_id, full_name_en, "
                "full_name_ar, email, phone, is_active, created_at) "
                "VALUES ('T-1', :school, 'A Teacher', 'مدرس', '', '', 1, CURRENT_TIMESTAMP)"
            ),
            {"school": school_id},
        )
        teacher_id = session.execute(
            text("SELECT id FROM teachers WHERE staff_number = 'T-1'")
        ).scalar_one()

        # Both lessons are in the same slot in two different rooms, which is legal. Giving
        # them the same teacher is not.
        session.execute(
            text(
                "UPDATE timetable_entries SET teacher_id = :teacher WHERE id = "
                "(SELECT MIN(id) FROM timetable_entries)"
            ),
            {"teacher": teacher_id},
        )
        unit.commit()

    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        with SqlAlchemyUnitOfWork() as unit:
            unit._session.execute(
                text(
                    "UPDATE timetable_entries SET teacher_id = :teacher WHERE id = "
                    "(SELECT MAX(id) FROM timetable_entries)"
                ),
                {"teacher": teacher_id},
            )
            unit.commit()


# -- Subjects, and the two tracks -------------------------------------------


def test_a_class_is_only_timetabled_a_subject_its_grade_teaches(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """Stage 5's rule, one table further on.

    Physics is assigned to the secondary rung and to nothing else, so a primary class
    cannot be timetabled it. Without this check the assignment work would hold on the marks
    screens and leak straight back in through the timetable.
    """
    assert _periods(client, registrar, count=3).status_code == 200

    assert _place(client, registrar, [_lesson("P1A", "sunday", 1, "MATH")]).status_code == 200

    refused = _place(client, registrar, [_lesson("P1A", "monday", 1, "PHYS")])
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["field"] == "subject_code"

    unknown = _place(client, registrar, [_lesson("P1A", "monday", 1, "NOPE")])
    assert unknown.status_code == 409, unknown.text


def test_the_two_academic_tracks_keep_separate_timetables(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """No code mentions a track, and the sections are separate anyway.

    That is the design rather than an accident: a class belongs to a rung and a rung to one
    track, so the Languages section's class cannot be given the Arabic section's subject —
    it is not assigned to *its* rung — and its week is its own.
    """
    assert _periods(client, registrar, count=3).status_code == 200
    assert _place(client, registrar, [_lesson("P1A", "sunday", 1, "MATH")]).status_code == 200

    # MATH is assigned to AR-P1. The Languages primary rung has no assignment at all, so
    # the same subject is refused there rather than inherited.
    refused = _place(client, registrar, [_lesson("LGA", "sunday", 1, "MATH")])
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["field"] == "subject_code"

    # A free period is still placeable — the grid exists for both sections.
    assert _place(client, registrar, [_lesson("LGA", "sunday", 1, None)]).status_code == 200
    assert _week(client, registrar, "LGA")["entries"][0]["subject_code"] is None
    # And the Arabic section is untouched by any of it.
    assert _week(client, registrar, "P1A")["entries"][0]["subject_code"] == "MATH"


# -- Clearing ---------------------------------------------------------------


def test_a_free_period_and_an_unplanned_slot_are_different_things(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """The distinction a registrar needs and a naive model loses.

    A row with no subject says "this class has this period off". No row says "nobody has
    planned this far". Rendering them identically would leave no way to tell a finished
    timetable from an abandoned one.
    """
    assert _periods(client, registrar, count=3).status_code == 200
    assert _place(client, registrar, [_lesson("P1A", "sunday", 1, None)]).status_code == 200

    stated_free = _week(client, registrar, "P1A")["entries"]
    assert len(stated_free) == 1 and stated_free[0]["subject_code"] is None

    cleared = client.post(
        "/v1/timetable/clear",
        json={
            "academic_year_code": YEAR,
            "slots": [
                {
                    "class_code": "P1A",
                    "term_code": TERM,
                    "day_of_week": "sunday",
                    "period_number": 1,
                }
            ],
        },
        headers=registrar,
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["removed"] == 1
    assert _week(client, registrar, "P1A")["entries"] == []

    # Clearing an empty slot removes nothing and is not an error: the caller's intent is
    # already true.
    again = client.post(
        "/v1/timetable/clear",
        json={
            "academic_year_code": YEAR,
            "slots": [
                {
                    "class_code": "P1A",
                    "term_code": TERM,
                    "day_of_week": "sunday",
                    "period_number": 1,
                }
            ],
        },
        headers=registrar,
    )
    assert again.status_code == 200 and again.json()["removed"] == 0


def test_the_week_read_carries_the_grid_and_counts_its_own_slots(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """One request draws the screen: the days, the rows, the lessons and the denominator."""
    assert client.put(
        f"/v1/schools/{SCHOOL}/timetable-periods",
        json={
            "periods": [
                {"period_number": 1},
                {"period_number": 2, "name_en": "Break", "name_ar": "فسحة", "is_teaching": False},
                {"period_number": 3},
            ]
        },
        headers=registrar,
    ).status_code == 200

    plan = _week(client, registrar, "P1A")
    assert plan["days"] == WEEK
    assert [row["period_number"] for row in plan["periods"]] == [1, 2, 3]
    # Four open days x two teaching periods. The break is a row on the grid and not a slot
    # anything can be scheduled into.
    assert plan["teaching_slots"] == 8


def test_unknown_codes_are_refused_rather_than_answered_empty(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """A typo and an empty timetable read identically once the answer is `[]`."""
    assert _periods(client, registrar, count=3).status_code == 200

    assert client.get(
        f"/v1/timetable/week?academic_year={YEAR}&class_code=NOPE&term={TERM}",
        headers=registrar,
    ).status_code == 404
    assert client.get(
        f"/v1/timetable/week?academic_year=NOPE&class_code=P1A&term={TERM}",
        headers=registrar,
    ).status_code == 404
    assert client.get(
        f"/v1/timetable/week?academic_year={YEAR}&class_code=P1A&term=NOPE",
        headers=registrar,
    ).status_code == 404
    assert client.get(
        f"/v1/schools/NOPE/timetable-periods", headers=registrar
    ).status_code == 404
