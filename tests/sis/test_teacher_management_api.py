"""Stage 8: a teacher identity plus valid subject, grade, track, and class scope."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.test_timetable_api import SCHOOL, YEAR, registrar, school


def _save(client: TestClient, headers: dict[str, str], body: dict, staff: str = "T-100"):
    return client.put(f"/v1/schools/{SCHOOL}/teachers/{staff}", json=body, headers=headers)


def test_teacher_may_hold_multiple_grades_tracks_and_classes(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    # Configure Maths for the Languages grade as well. This remains an explicit
    # subject/grade decision; merely belonging to the same school is not enough.
    assert client.put("/v1/subject-assignments", json={
        "academic_year_code": YEAR, "subject_code": "MATH",
        "year_level_code": "LG-P1", "assigned": True,
    }, headers=registrar).status_code == 204

    response = _save(client, registrar, {
        "full_name_en": "Mona Teacher", "full_name_ar": "منى",
        "username": "mona.teacher", "password": "safe-password",
        "assignments": [
            {"academic_year_code": YEAR, "subject_code": "MATH",
             "year_level_code": "AR-P1", "class_codes": ["P1A", "P1B"]},
            {"academic_year_code": YEAR, "subject_code": "MATH",
             "year_level_code": "LG-P1", "class_codes": ["LGA"]},
        ],
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["username"] == "mona.teacher"
    assert {(a["year_level_code"], a["track_code"]) for a in body["assignments"]} == {
        ("AR-P1", "AR"), ("LG-P1", "LANG")
    }
    assert body["assignments"][0]["class_codes"] == ["P1A", "P1B"]

    # Creating an account here does not silently promote it into any role.
    with SqlAlchemyUnitOfWork() as uow:
        grants = uow._session.execute(text(
            "SELECT COUNT(*) FROM user_roles ur JOIN users u ON u.id=ur.user_id "
            "WHERE u.username='mona.teacher'"
        )).scalar_one()
    assert grants == 0


def test_subject_must_be_configured_for_the_selected_grade(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    response = _save(client, registrar, {
        "full_name_en": "Invalid Assignment",
        "assignments": [{"academic_year_code": YEAR, "subject_code": "PHYS",
                         "year_level_code": "AR-P1", "class_codes": ["P1A"]}],
    })
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["field"] == "subject_code"

    # The transaction is atomic: the rejected identity did not remain behind.
    assert client.get(
        f"/v1/schools/{SCHOOL}/teachers/T-100", headers=registrar
    ).status_code == 404


def test_classes_must_belong_to_the_assignment_grade_and_year(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    response = _save(client, registrar, {
        "full_name_en": "Wrong Room",
        "assignments": [{"academic_year_code": YEAR, "subject_code": "MATH",
                         "year_level_code": "AR-P1", "class_codes": ["LGA"]}],
    })
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["field"] == "class_codes"
