"""Stage 15: manager eligibility and supervisor class assignment stay separate."""
from fastapi.testclient import TestClient

from tests.sis.test_grade_supervisor import grade_supervisor  # noqa: F401
from tests.sis.test_rbac_api import ids, principal  # noqa: F401
from tests.sis.test_timetable_api import SCHOOL, YEAR, registrar, school  # noqa: F401


def test_principal_can_define_subject_grade_and_derived_track(
    client: TestClient, principal: dict[str, str], school: None
) -> None:
    response = client.put(
        f"/v1/schools/{SCHOOL}/teachers/T-15",
        headers=principal,
        json={
            "full_name_en": "Arabic Teacher",
            "assignments": [{
                "academic_year_code": YEAR,
                "subject_code": "MATH",
                "year_level_code": "AR-P1",
                "class_codes": [],
            }],
        },
    )
    assert response.status_code == 200, response.text
    assignment = response.json()["assignments"][0]
    assert assignment["track_code"] == "AR"
    assert assignment["class_codes"] == []


def test_invalid_subject_is_rejected_before_teacher_selection(
    client: TestClient, grade_supervisor: dict[str, str], school: None
) -> None:
    response = client.get(
        f"/v1/schools/{SCHOOL}/grades/AR-P1/teacher-assignment-options",
        headers=grade_supervisor,
        params={"academic_year": YEAR, "subject": "PHYS"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["field"] == "subject_code"


def test_inactive_teacher_is_neither_listed_nor_assignable(
    client: TestClient,
    registrar: dict[str, str],
    grade_supervisor: dict[str, str],
    school: None,
) -> None:
    saved = client.put(
        f"/v1/schools/{SCHOOL}/teachers/T-INACTIVE-15",
        headers=registrar,
        json={
            "full_name_en": "Inactive Teacher",
            "is_active": False,
            "assignments": [{
                "academic_year_code": YEAR,
                "subject_code": "MATH",
                "year_level_code": "AR-P1",
                "class_codes": [],
            }],
        },
    )
    assert saved.status_code == 200, saved.text
    path = f"/v1/schools/{SCHOOL}/grades/AR-P1/teacher-assignment-options"
    options = client.get(
        path,
        headers=grade_supervisor,
        params={"academic_year": YEAR, "subject": "MATH"},
    )
    assert options.status_code == 200
    assert "T-INACTIVE-15" not in {
        row["staff_number"] for row in options.json()["eligible_teachers"]
    }
    assigned = client.put(
        f"/v1/schools/{SCHOOL}/grades/AR-P1/teacher-class-assignments",
        headers=grade_supervisor,
        json={
            "academic_year_code": YEAR,
            "subject_code": "MATH",
            "staff_number": "T-INACTIVE-15",
            "class_codes": ["P1A"],
        },
    )
    assert assigned.status_code == 409
    assert assigned.json()["detail"]["field"] == "staff_number"


def test_grade_scope_cannot_be_reused_against_another_school(
    client: TestClient,
    registrar: dict[str, str],
    grade_supervisor: dict[str, str],
    school: None,
) -> None:
    assert client.post("/v1/schools", headers=registrar, json={
        "code": "OTHER15", "name_en": "Other School", "name_ar": "Other",
        "language_type": "arabic", "kg_grade_count": 0,
        "primary_grade_count": 1, "preparatory_grade_count": 0,
        "secondary_grade_count": 0, "term_count": 1,
        "working_days": ["sunday"],
    }).status_code == 201
    assert client.post("/v1/academic-years", headers=registrar, json={
        "code": "OTHER15-2025", "school_code": "OTHER15",
        "name_en": "2025", "name_ar": "2025",
        "starts_on": "2025-09-01", "ends_on": "2026-06-30",
        "is_current": True,
    }).status_code == 201
    assert client.post("/v1/structure/levels", headers=registrar, json={
        "code": "AR-P1", "school_code": "OTHER15", "track_code": "AR",
        "name_en": "Primary 1", "name_ar": "Primary 1",
        "display_order": 1, "stage": "primary",
    }).status_code == 201
    response = client.get(
        "/v1/schools/OTHER15/grades/AR-P1/teacher-assignment-options",
        headers=grade_supervisor,
        params={"academic_year": "OTHER15-2025"},
    )
    assert response.status_code == 403
