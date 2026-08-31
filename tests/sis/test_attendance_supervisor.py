"""Stage 13: finding the registers you may take, and closing one.

The scope boundary itself — a supervisor of P1A refused P1B — is already proved in
`test_rbac_api.py`, and is not restated here. What this file covers is the two things
Stage 13 added, both of which are about a register being *taken* rather than about who
may touch it:

**Step one of the workflow had no route.** An attendance supervisor holds classrooms and
nothing above them, so every listing that narrows a grade or a year refuses them: they
could read a register only by already knowing its class code. `GET /v1/attendance/classes`
answers from their own grants instead, which is what makes "pick a grade, pick a class"
possible for the role the workflow was written for.

**"Everyone not marked present is absent" is a statement, not a default.** The service
reads an unnamed child as *not reached yet*, deliberately — that is what keeps a partial
register honest. `absent_unlisted` is the caller saying the pass is finished. The tests
below pin both halves: that it fills every blank, and that it overwrites nothing.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from sis.domain.rbac import RoleCode, ScopeType
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.test_rbac_api import _grant, _make_user, _sign_in, ids  # noqa: F401
from tests.sis.test_timetable_api import SCHOOL, YEAR, registrar, school  # noqa: F401

DAY = "2025-11-20"


@pytest.fixture()
def roll(client: TestClient, registrar: dict[str, str], school: None) -> list[str]:
    """Four children in P1A and one in P1B, placed from the first day of the year.

    P1B exists so that "the classes I may take" has something to leave out that is not
    merely absent from the school, and the single child there makes its register a real
    one rather than an empty list that would pass any assertion.
    """
    numbers = []
    for index, (number, klass) in enumerate(
        [("S-1", "P1A"), ("S-2", "P1A"), ("S-3", "P1A"), ("S-4", "P1A"), ("S-9", "P1B")]
    ):
        assert client.post(
            "/v1/students",
            headers=registrar,
            json={"student_number": number, "full_name_en": f"Child {index}",
                  "full_name_ar": f"طفل {index}"},
        ).status_code in (200, 201)
        assert client.post(
            f"/v1/students/{number}/placements",
            headers=registrar,
            json={"academic_year_code": YEAR, "class_code": klass, "starts_on": "2025-09-01"},
        ).status_code in (200, 201)
        if klass == "P1A":
            numbers.append(number)
    return numbers


@pytest.fixture()
def supervisor(client: TestClient, ids: dict[str, int]) -> dict[str, str]:
    """Granted P1A and nothing else — one room out of the school's three."""
    with SqlAlchemyUnitOfWork() as uow:
        user_id = _make_user(uow._session, "register.taker", school_id=ids["school"])
        _grant(
            uow._session, user_id, RoleCode.ATTENDANCE_SUPERVISOR,
            ScopeType.CLASS_SECTION, ids["class_P1A"],
        )
        uow.commit()
    return _sign_in(client, "register.taker")


# -- Step one: which registers may I take ------------------------------------


def test_a_class_scoped_supervisor_is_told_which_classes_they_hold(
    client: TestClient, supervisor: dict[str, str], roll: list[str]
) -> None:
    """The route that makes the workflow's first step possible at all.

    Every other way to reach a class narrows a grade or a year, and this caller holds
    neither — so the same question asked through `structure/classes` is refused, which is
    the assertion at the bottom and the reason this route exists.
    """
    response = client.get(
        "/v1/attendance/classes", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
    )
    assert response.status_code == 200, response.text
    listed = response.json()["classes"]
    assert [row["class_code"] for row in listed] == ["P1A"]

    only = listed[0]
    # The grade rides along so a client groups by it without a second call.
    assert only["year_level_code"] == "AR-P1"
    assert only["may_record"] is True
    assert (only["size"], only["marked"], only["is_complete"]) == (4, 0, False)

    assert client.get(
        "/v1/structure/classes", headers=supervisor,
        params={"academic_year": YEAR, "year_level": "AR-P1"},
    ).status_code == 403


def test_the_same_route_answers_a_school_wide_caller_with_the_school(
    client: TestClient, registrar: dict[str, str], roll: list[str]
) -> None:
    """One route for every scope. The answer is the union over the caller's grants, so a
    registrar is not a special case with a listing of its own."""
    response = client.get(
        "/v1/attendance/classes", headers=registrar,
        params={"academic_year": YEAR, "on": DAY},
    )
    assert response.status_code == 200, response.text
    assert {row["class_code"] for row in response.json()["classes"]} == {"P1A", "P1B", "LGA"}


def test_the_listing_reports_the_day_a_register_has_reached(
    client: TestClient, supervisor: dict[str, str], roll: list[str]
) -> None:
    """Progress per class, which is what stops a day being recorded twice by somebody who
    could not otherwise tell it had been recorded once."""
    assert client.put(
        "/v1/classes/P1A/attendance", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
        json={"entries": [{"student_number": roll[0], "state": "present"}]},
    ).status_code == 200

    partial = client.get(
        "/v1/attendance/classes", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
    ).json()["classes"][0]
    assert (partial["marked"], partial["is_complete"]) == (1, False)

    # A different day is a different register, and is still untouched.
    other = client.get(
        "/v1/attendance/classes", headers=supervisor,
        params={"academic_year": YEAR, "on": "2025-11-21"},
    ).json()["classes"][0]
    assert (other["marked"], other["is_complete"]) == (0, False)


# -- Steps three and four: mark the present, the rest are absent -------------


def test_closing_the_register_records_every_unmarked_child_absent(
    client: TestClient, supervisor: dict[str, str], roll: list[str]
) -> None:
    """The workflow in one request: name the children in the room, and the rest are away."""
    response = client.put(
        "/v1/classes/P1A/attendance", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
        json={
            "entries": [{"student_number": roll[0], "state": "present"}],
            "absent_unlisted": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_complete"] is True
    assert body["unmarked"] == 0
    assert body["counts"]["present"] == 1
    assert body["counts"]["absent"] == 3


def test_closing_fills_blanks_and_overwrites_nothing(
    client: TestClient, supervisor: dict[str, str], roll: list[str]
) -> None:
    """The rule that keeps the flag from destroying an earlier pass.

    A child excused at nine o'clock is still excused when the register is closed at noon.
    Without this, the fastest button on the screen would quietly overwrite the one state
    that carries a typed reason, and a register cannot show afterwards that it did.
    """
    assert client.put(
        "/v1/classes/P1A/attendance", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
        json={"entries": [
            {"student_number": roll[0], "state": "excused", "note": "medical"},
            {"student_number": roll[1], "state": "late"},
        ]},
    ).status_code == 200

    closed = client.put(
        "/v1/classes/P1A/attendance", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
        json={"entries": [], "absent_unlisted": True},
    )
    assert closed.status_code == 200, closed.text
    by_number = {row["student_number"]: row for row in closed.json()["students"]}
    assert by_number[roll[0]]["state"] == "excused"
    assert by_number[roll[0]]["note"] == "medical"
    assert by_number[roll[1]]["state"] == "late"
    assert by_number[roll[2]]["state"] == "absent"
    assert closed.json()["is_complete"] is True


def test_an_empty_register_is_still_refused_when_it_is_not_being_closed(
    client: TestClient, supervisor: dict[str, str], roll: list[str]
) -> None:
    """Saying nothing and saying "everybody else is absent" are different requests.

    The first has nothing to record and is refused, as it was before this flag existed.
    """
    assert client.put(
        "/v1/classes/P1A/attendance", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
        json={"entries": []},
    ).status_code == 422


# -- Editing, and not recording the same day twice ---------------------------


def test_recording_the_same_day_twice_leaves_one_row_per_child(
    client: TestClient, supervisor: dict[str, str], roll: list[str]
) -> None:
    """The duplicate guarantee, asserted against the table rather than the response.

    `(student, day)` is unique in the database, so a double-submitted register cannot
    become two statements about one morning however the client behaves.
    """
    for _ in range(3):
        assert client.put(
            "/v1/classes/P1A/attendance", headers=supervisor,
            params={"academic_year": YEAR, "on": DAY},
            json={
                "entries": [{"student_number": roll[0], "state": "present"}],
                "absent_unlisted": True,
            },
        ).status_code == 200

    with SqlAlchemyUnitOfWork() as uow:
        rows = uow._session.execute(
            text("SELECT COUNT(*) FROM attendance WHERE on_date = :day"), {"day": DAY}
        ).scalar()
        duplicates = uow._session.execute(
            text(
                "SELECT COUNT(*) FROM (SELECT student_id FROM attendance "
                "WHERE on_date = :day GROUP BY student_id, on_date HAVING COUNT(*) > 1)"
            ),
            {"day": DAY},
        ).scalar()
    assert rows == 4
    assert duplicates == 0


def test_a_closed_register_is_still_editable_by_whoever_may_write_it(
    client: TestClient, supervisor: dict[str, str], roll: list[str]
) -> None:
    """"Support editing according to the existing permission rules" — the permission rules
    being the ones already there: the write is scoped, and a closed day is not frozen."""
    assert client.put(
        "/v1/classes/P1A/attendance", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
        json={"entries": [], "absent_unlisted": True},
    ).status_code == 200

    corrected = client.put(
        "/v1/classes/P1A/attendance", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
        json={"entries": [{"student_number": roll[0], "state": "present"}]},
    )
    assert corrected.status_code == 200, corrected.text
    assert corrected.json()["counts"]["present"] == 1
    assert corrected.json()["counts"]["absent"] == 3


def test_closing_a_register_is_refused_on_a_class_the_supervisor_was_not_given(
    client: TestClient, supervisor: dict[str, str], roll: list[str]
) -> None:
    """The new flag is not a way around the scope. It is the same route and the same check."""
    refused = client.put(
        "/v1/classes/P1B/attendance", headers=supervisor,
        params={"academic_year": YEAR, "on": DAY},
        json={"entries": [], "absent_unlisted": True},
    )
    assert refused.status_code == 403
    assert "attendance.write" in refused.json()["detail"]["message"]
