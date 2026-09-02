"""Stage 11 boundaries for School Owner and School Manager / Principal."""
from datetime import date

from fastapi.testclient import TestClient
from sqlalchemy import select

from sis.demo.seeder import sync_roles
from sis.domain.rbac import RoleCode, ScopeType
from sis.infrastructure.crypto import hash_password
from sis.infrastructure.db import models as m
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.test_rbac_api import PASSWORD, _grant, _make_user, _sign_in, ids, principal  # noqa: F401
from tests.sis.test_timetable_api import SCHOOL, registrar, school  # noqa: F401


def _role_headers(client: TestClient, role: RoleCode, school_id: int | None) -> dict[str, str]:
    username = f"stage11.{role.value}"
    with SqlAlchemyUnitOfWork() as uow:
        sync_roles(uow._session)
        user_id = _make_user(uow._session, username, school_id=school_id)
        _grant(
            uow._session, user_id, role,
            ScopeType.GLOBAL if role is RoleCode.SYSTEM_ADMIN else ScopeType.SCHOOL,
            None if role is RoleCode.SYSTEM_ADMIN else school_id,
        )
        uow.commit()
    return _sign_in(client, username)


def test_owner_can_read_the_school_but_cannot_change_normal_data(
    client: TestClient, ids: dict[str, int]
) -> None:
    owner = _role_headers(client, RoleCode.SCHOOL_OWNER, ids["school"])
    assert client.get("/v1/schools", headers=owner).status_code == 200
    refused = client.post(
        "/v1/schools", headers=owner,
        json={"code": "NOPE", "name_en": "Must not be created"},
    )
    assert refused.status_code == 403


def test_principal_is_general_read_only_and_never_a_system_admin(
    client: TestClient, ids: dict[str, int], principal: dict[str, str]
) -> None:
    profile = client.get("/v1/auth/me", headers=principal).json()["profile"]
    assert profile["is_system_admin"] is False
    assert "roles.assign" in profile["permissions"]
    assert "teacher_attendance.read" in profile["permissions"]
    # A principal admits and corrects children in their own school: the console's Edit button
    # is theirs, and refusing it would leave the head of the school unable to fix a misspelt
    # name. Reading marks is the same story. What stays out of reach is everything that
    # rewrites the school itself or the system it runs on.
    assert "students.create" in profile["permissions"]
    assert "students.write" in profile["permissions"]
    assert "grades.read" in profile["permissions"]
    forbidden = {
        "schools.write", "structure.write", "grades.write",
        "guardians.write", "imports.run", "system.manage", "system.status.write",
    }
    assert forbidden.isdisjoint(profile["permissions"])


def test_principal_can_view_teacher_attendance_but_not_record_it(
    client: TestClient, ids: dict[str, int], principal: dict[str, str]
) -> None:
    with SqlAlchemyUnitOfWork() as uow:
        teacher = m.Teacher(
            staff_number="T-STAGE11", school_id=ids["school"],
            full_name_en="Stage Eleven Teacher", full_name_ar="Teacher",
        )
        uow._session.add(teacher)
        uow._session.flush()
        uow._session.add(m.TeacherAttendance(
            teacher_id=teacher.id, school_id=ids["school"], on_date=date(2026, 8, 31),
            state="present", recorded_by="system",
        ))
        uow.commit()

    rows = client.get(f"/v1/schools/{SCHOOL}/teachers/attendance", headers=principal)
    assert rows.status_code == 200, rows.text
    assert rows.json()[0]["staff_number"] == "T-STAGE11"
    refused = client.put(
        f"/v1/schools/{SCHOOL}/teachers/T-STAGE11/attendance/2026-09-01",
        headers=principal, json={"state": "absent"},
    )
    assert refused.status_code == 403


def test_principal_adds_supervisor_roles_without_replacing_teacher(
    client: TestClient, ids: dict[str, int], principal: dict[str, str]
) -> None:
    with SqlAlchemyUnitOfWork() as uow:
        user_id = _make_user(uow._session, "stage11.teacher", school_id=ids["school"])
        _grant(uow._session, user_id, RoleCode.TEACHER, ScopeType.SCHOOL, ids["school"])
        uow.commit()

    for code in ("grade_supervisor", "attendance_supervisor"):
        response = client.post(
            f"/v1/rbac/users/{user_id}/roles", headers=principal,
            json={"role_code": code, "scope_type": "school", "scope_id": ids["school"]},
        )
        assert response.status_code == 200, response.text
    held = {row["role_code"] for row in response.json()}
    assert held == {"teacher", "year_supervisor", "attendance_supervisor"}


def test_principal_cannot_delegate_owner_principal_or_system_admin(
    client: TestClient, ids: dict[str, int], principal: dict[str, str]
) -> None:
    with SqlAlchemyUnitOfWork() as uow:
        user_id = _make_user(uow._session, "stage11.no-escalation", school_id=ids["school"])
        uow.commit()
    for code in ("school_owner", "principal", "system_admin"):
        response = client.post(
            f"/v1/rbac/users/{user_id}/roles", headers=principal,
            json={"role_code": code, "scope_type": "school", "scope_id": ids["school"]},
        )
        assert response.status_code == 403, (code, response.text)
