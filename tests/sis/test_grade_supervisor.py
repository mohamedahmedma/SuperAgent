"""Stage 12: grade-scoped reads and eligible teacher-to-class assignment."""
import pytest
from fastapi.testclient import TestClient

from sis.domain.rbac import RoleCode, ScopeType
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.test_rbac_api import _grant, _make_user, _sign_in, ids  # noqa: F401
from tests.sis.test_timetable_api import (  # noqa: F401
    SCHOOL,
    TERM,
    YEAR,
    _lesson,
    _periods,
    _place,
    registrar,
    school,
)


@pytest.fixture()
def grade_supervisor(client: TestClient, ids: dict[str, int]) -> dict[str, str]:
    with SqlAlchemyUnitOfWork() as uow:
        user_id = _make_user(uow._session, "grade.supervisor.12", school_id=ids["school"])
        _grant(
            uow._session, user_id, RoleCode.YEAR_SUPERVISOR,
            ScopeType.YEAR_LEVEL, ids["level_p1"],
        )
        uow.commit()
    return _sign_in(client, "grade.supervisor.12")


@pytest.fixture()
def eligible_teacher(client: TestClient, registrar: dict[str, str], school: None) -> None:
    response = client.put(
        f"/v1/schools/{SCHOOL}/teachers/T-12",
        headers=registrar,
        json={
            "full_name_en": "Eligible Mathematics Teacher",
            "assignments": [{
                "academic_year_code": YEAR,
                "subject_code": "MATH",
                "year_level_code": "AR-P1",
                "class_codes": ["P1A"],
            }],
        },
    )
    assert response.status_code == 200, response.text


def test_supervisor_reads_only_the_assigned_grade_classes(
    client: TestClient, grade_supervisor: dict[str, str]
) -> None:
    allowed = client.get(
        "/v1/structure/classes", headers=grade_supervisor,
        params={"academic_year": YEAR, "year_level": "AR-P1"},
    )
    assert allowed.status_code == 200, allowed.text
    assert {row["code"] for row in allowed.json()} == {"P1A", "P1B"}

    assert client.get(
        "/v1/structure/classes", headers=grade_supervisor,
        params={"academic_year": YEAR, "year_level": "AR-S1"},
    ).status_code == 403
    assert client.get(
        "/v1/structure/classes", headers=grade_supervisor,
        params={"academic_year": YEAR},
    ).status_code == 403


def test_assignment_flow_lists_only_eligible_teachers_and_updates_selected_classes(
    client: TestClient, grade_supervisor: dict[str, str], eligible_teacher: None
) -> None:
    path = f"/v1/schools/{SCHOOL}/grades/AR-P1/teacher-assignment-options"
    options = client.get(
        path, headers=grade_supervisor,
        params={"academic_year": YEAR, "subject": "MATH"},
    )
    assert options.status_code == 200, options.text
    assert {row["code"] for row in options.json()["classes"]} == {"P1A", "P1B"}
    assert [row["staff_number"] for row in options.json()["eligible_teachers"]] == ["T-12"]

    assigned = client.put(
        f"/v1/schools/{SCHOOL}/grades/AR-P1/teacher-class-assignments",
        headers=grade_supervisor,
        json={
            "academic_year_code": YEAR, "subject_code": "MATH",
            "staff_number": "T-12", "class_codes": ["P1A", "P1B"],
        },
    )
    assert assigned.status_code == 200, assigned.text
    assert assigned.json()["eligible_teachers"][0]["assigned_class_codes"] == ["P1A", "P1B"]


def test_supervisor_cannot_assign_or_list_an_unrelated_grade(
    client: TestClient, grade_supervisor: dict[str, str], eligible_teacher: None
) -> None:
    refused = client.get(
        f"/v1/schools/{SCHOOL}/grades/AR-S1/teacher-assignment-options",
        headers=grade_supervisor, params={"academic_year": YEAR, "subject": "MATH"},
    )
    assert refused.status_code == 403
    # The broad teacher directory is school-wide and therefore unavailable to a
    # grade-scoped supervisor; eligible teachers come only from the scoped flow.
    assert client.get(
        f"/v1/schools/{SCHOOL}/teachers", headers=grade_supervisor
    ).status_code == 403


def test_grade_supervisor_only_writes_the_timetable_in_their_scope(
    client: TestClient, grade_supervisor: dict[str, str]
) -> None:
    permissions = set(client.get("/v1/auth/me", headers=grade_supervisor).json()["profile"]["permissions"])
    assert "teachers.assign_classes" in permissions
    assert {permission for permission in permissions if permission.endswith(".write")} == {
        "timetable.write"
    }
    assert "system.manage" not in permissions


# -- Reading the grade's teaching staff -------------------------------------
#
# The role has carried `teachers.read` at `year_level` scope since it was defined, and
# until these tests there was no route that would honour it: the only teacher directory
# was school-wide, so the permission was granted and unusable. A grant nothing can
# satisfy is worse than a missing one — it reads as working.


@pytest.fixture()
def teacher_of_two_grades(client: TestClient, registrar: dict[str, str], school: None) -> None:
    """One teacher who works on the supervised grade *and* on an unrelated one.

    The whole point of the fixture. A teacher who only ever taught Primary 1 cannot show
    that the directory is filtered, because every honest answer looks the same.
    """
    assert client.put(
        f"/v1/schools/{SCHOOL}/teachers/T-BOTH",
        headers=registrar,
        json={
            "full_name_en": "Teaches Two Grades",
            "assignments": [
                {"academic_year_code": YEAR, "subject_code": "MATH",
                 "year_level_code": "AR-P1", "class_codes": ["P1A"]},
                {"academic_year_code": YEAR, "subject_code": "PHYS",
                 "year_level_code": "AR-S1", "class_codes": []},
            ],
        },
    ).status_code == 200


def test_the_grade_directory_narrows_the_record_as_well_as_the_list(
    client: TestClient, grade_supervisor: dict[str, str], teacher_of_two_grades: None
) -> None:
    """A teacher of two grades arrives holding one of them.

    Filtering only *which* teachers come back would still hand a Primary 1 supervisor
    the secondary timetable of everybody who works on both rungs — the unrelated grade
    walking in inside a related teacher.
    """
    response = client.get(
        f"/v1/schools/{SCHOOL}/teachers",
        headers=grade_supervisor,
        params={"year_level": "AR-P1"},
    )
    assert response.status_code == 200, response.text
    assert [row["staff_number"] for row in response.json()] == ["T-BOTH"]
    assignments = response.json()[0]["assignments"]
    assert [row["year_level_code"] for row in assignments] == ["AR-P1"]
    assert [row["subject_code"] for row in assignments] == ["MATH"]

    one = client.get(
        f"/v1/schools/{SCHOOL}/teachers/T-BOTH",
        headers=grade_supervisor,
        params={"year_level": "AR-P1"},
    )
    assert one.status_code == 200, one.text
    assert [row["year_level_code"] for row in one.json()["assignments"]] == ["AR-P1"]


def test_the_directory_is_refused_school_wide_and_on_an_unrelated_grade(
    client: TestClient, grade_supervisor: dict[str, str], teacher_of_two_grades: None
) -> None:
    """Naming no grade is a school-wide read, and a school-wide read is not held."""
    assert client.get(
        f"/v1/schools/{SCHOOL}/teachers", headers=grade_supervisor
    ).status_code == 403
    assert client.get(
        f"/v1/schools/{SCHOOL}/teachers",
        headers=grade_supervisor,
        params={"year_level": "AR-S1"},
    ).status_code == 403
    assert client.get(
        f"/v1/schools/{SCHOOL}/teachers/T-BOTH", headers=grade_supervisor
    ).status_code == 403


def test_a_teacher_outside_the_grade_is_reported_as_no_such_teacher(
    client: TestClient, grade_supervisor: dict[str, str], registrar: dict[str, str],
    school: None,
) -> None:
    """404, not 403 — and deliberately the same answer as a staff number that never was.

    A supervisor who could tell "exists, but not yours" from "does not exist" could walk
    the staff numbers and learn who works at the school.
    """
    assert client.put(
        f"/v1/schools/{SCHOOL}/teachers/T-SEC",
        headers=registrar,
        json={
            "full_name_en": "Secondary Only",
            "assignments": [{"academic_year_code": YEAR, "subject_code": "PHYS",
                             "year_level_code": "AR-S1", "class_codes": []}],
        },
    ).status_code == 200

    outside = client.get(
        f"/v1/schools/{SCHOOL}/teachers/T-SEC",
        headers=grade_supervisor,
        params={"year_level": "AR-P1"},
    )
    invented = client.get(
        f"/v1/schools/{SCHOOL}/teachers/T-NOBODY",
        headers=grade_supervisor,
        params={"year_level": "AR-P1"},
    )
    assert outside.status_code == invented.status_code == 404
    assert outside.json()["detail"]["code"] == invented.json()["detail"]["code"]


def test_the_registrar_still_reads_the_whole_school_directory(
    client: TestClient, registrar: dict[str, str], teacher_of_two_grades: None
) -> None:
    """The narrowing is opt-in. A school-wide caller naming no grade still gets all of it."""
    response = client.get(f"/v1/schools/{SCHOOL}/teachers", headers=registrar)
    assert response.status_code == 200, response.text
    assert [row["staff_number"] for row in response.json()] == ["T-BOTH"]
    assert {row["year_level_code"] for row in response.json()[0]["assignments"]} == {
        "AR-P1", "AR-S1"
    }


# -- Reading the grade's academic data --------------------------------------


def test_the_whole_school_timetable_is_refused_and_one_grade_is_not(
    client: TestClient, grade_supervisor: dict[str, str], registrar: dict[str, str],
    school: None,
) -> None:
    """The year-wide lesson list is a school-wide read, and it never checked that it was.

    It was gated on holding `timetable.read` *somewhere* rather than on where, so a
    supervisor of one rung was handed every lesson in the school. The grade has to be
    named for the grant to match, and naming it also filters what comes back.
    """
    assert _periods(client, registrar).status_code == 200
    assert _place(
        client, registrar,
        [_lesson("P1A", "sunday", 1), _lesson("LGA", "sunday", 1, subject=None)],
    ).status_code == 200

    assert client.get(
        "/v1/timetable", headers=grade_supervisor, params={"academic_year": YEAR}
    ).status_code == 403
    assert client.get(
        "/v1/timetable", headers=grade_supervisor,
        params={"academic_year": YEAR, "year_level": "LG-P1"},
    ).status_code == 403

    mine = client.get(
        "/v1/timetable", headers=grade_supervisor,
        params={"academic_year": YEAR, "year_level": "AR-P1"},
    )
    assert mine.status_code == 200, mine.text
    assert [row["class_code"] for row in mine.json()] == ["P1A"]

    # And the school-wide caller is untouched: both lessons, no grade named.
    whole = client.get("/v1/timetable", headers=registrar, params={"academic_year": YEAR})
    assert whole.status_code == 200, whole.text
    assert {row["class_code"] for row in whole.json()} == {"P1A", "LGA"}
