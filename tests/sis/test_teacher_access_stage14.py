"""Stage 14: a teacher's boundary is the exact class-subject assignment."""
from fastapi.testclient import TestClient
from sqlalchemy import select

from sis.demo.seeder import sync_roles
from sis.domain.rbac import RoleCode, ScopeType
from sis.infrastructure.db import models as m
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.test_rbac_api import _grant, _make_user, _sign_in
from tests.sis.test_timetable_api import YEAR, registrar, school


def _teacher(client: TestClient) -> dict[str, str]:
    with SqlAlchemyUnitOfWork() as uow:
        session = uow._session
        sync_roles(session)
        school_id = session.scalar(select(m.School.id))
        user_id = _make_user(session, "stage14.teacher", school_id=school_id)
        teacher = m.Teacher(
            staff_number="T-14",
            school_id=school_id,
            user_id=user_id,
            full_name_en="Stage 14 Teacher",
            full_name_ar="Stage 14 Teacher",
        )
        session.add(teacher)
        session.flush()
        math = session.scalar(select(m.Subject).where(m.Subject.code == "MATH"))
        p1a = session.scalar(select(m.ClassSection).where(m.ClassSection.code == "P1A"))
        p1b = session.scalar(select(m.ClassSection).where(m.ClassSection.code == "P1B"))
        assert math and p1a and p1b
        for section in (p1a, p1b):
            _grant(session, user_id, RoleCode.TEACHER, ScopeType.CLASS_SECTION, section.id)
            session.add(m.TeacherClassSection(
                teacher_id=teacher.id,
                class_section_id=section.id,
                subject_id=math.id,
                assigned_by="test",
            ))
        uow.commit()
    return _sign_in(client, "stage14.teacher")


def test_teacher_discovers_all_assigned_classes_and_only_those(
    client: TestClient, school: None
) -> None:
    headers = _teacher(client)
    response = client.get(
        "/v1/teaching/assignments",
        headers=headers,
        params={"academic_year": YEAR},
    )
    assert response.status_code == 200, response.text
    assert {
        (row["class_code"], row["subject_code"])
        for row in response.json()["assignments"]
    } == {("P1A", "MATH"), ("P1B", "MATH")}


def test_teacher_cannot_read_or_write_an_unrelated_class(
    client: TestClient, school: None
) -> None:
    headers = _teacher(client)
    query = {"academic_year": YEAR, "term": f"{YEAR}-T1", "subject": "MATH"}
    assert client.get("/v1/classes/LGA/grades", headers=headers, params=query).status_code == 403
    assert client.put(
        "/v1/classes/LGA/grades",
        headers=headers,
        params={"academic_year": YEAR},
        json={
            "term_code": f"{YEAR}-T1",
            "subject_code": "MATH",
            "marks": [{"student_number": "100", "percentage": 90}],
        },
    ).status_code == 403


def test_teacher_cannot_read_or_write_a_colleagues_subject_in_their_class(
    client: TestClient, school: None, registrar: dict[str, str]
) -> None:
    assert client.post(
        "/v1/subjects",
        headers=registrar,
        json={
            "code": "ARAB",
            "academic_year_code": YEAR,
            "name_en": "Arabic",
            "name_ar": "Arabic",
        },
    ).status_code == 201
    assert client.put(
        "/v1/subject-assignments",
        headers=registrar,
        json={
            "academic_year_code": YEAR,
            "subject_code": "ARAB",
            "year_level_code": "AR-P1",
            "assigned": True,
        },
    ).status_code == 204
    headers = _teacher(client)
    query = {"academic_year": YEAR, "term": f"{YEAR}-T1", "subject": "ARAB"}
    denied_read = client.get("/v1/classes/P1A/grades", headers=headers, params=query)
    assert denied_read.status_code == 403
    denied_write = client.put(
        "/v1/classes/P1A/grades",
        headers=headers,
        params={"academic_year": YEAR},
        json={
            "term_code": f"{YEAR}-T1",
            "subject_code": "ARAB",
            "marks": [{"student_number": "100", "percentage": 90}],
        },
    )
    assert denied_write.status_code == 403
    assert denied_write.json()["detail"]["code"] == "not_authorized"
