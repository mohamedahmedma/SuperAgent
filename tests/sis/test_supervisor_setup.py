"""Supervisor creation is atomic and grants only the selected grade role."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from sis.infrastructure.db import models as m
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.test_rbac_api import PASSWORD, _sign_in, ids, principal  # noqa: F401
from tests.sis.test_timetable_api import registrar, school  # noqa: F401


@pytest.mark.parametrize("role", ["floor_supervisor", "attendance_supervisor"])
def test_create_remove_and_replace_supervisor(client: TestClient, ids, principal, role):
    body = {
        "username": f"new.{role}", "password": PASSWORD,
        "full_name_en": "New supervisor", "full_name_ar": "مشرف جديد",
        "role_code": role, "year_level_id": ids["level_p1"],
    }
    response = client.post("/v1/rbac/supervisors", headers=principal, json=body)
    assert response.status_code == 201, response.text
    user = response.json()
    assert user["school_id"] == ids["school"]
    assert len(user["roles"]) == 1
    grant = {"role_code": role, "scope_type": "year_level", "scope_id": ids["level_p1"]}
    assert all(user["roles"][0][key] == value for key, value in grant.items())
    headers = _sign_in(client, body["username"])
    before = client.get("/v1/auth/me", headers=headers).json()["profile"]
    assert "attendance.read" in before["permissions"]
    assert "roles.assign" not in before["permissions"]
    # Removing this grade must preserve any supervision of another grade.
    other = {**grant, "scope_id": ids["level_s1"]}
    assert client.post(f'/v1/rbac/users/{user["id"]}/roles', headers=principal, json=other).status_code == 200
    removed = client.request("DELETE", f'/v1/rbac/users/{user["id"]}/roles', headers=principal, json=grant)
    assert removed.status_code == 200, removed.text
    assert len(removed.json()) == 1
    assert removed.json()[0]["scope_id"] == ids["level_s1"]
    assert client.request("DELETE", f'/v1/rbac/users/{user["id"]}/roles', headers=principal, json=other).status_code == 200
    after = client.get("/v1/auth/me", headers=headers).json()["profile"]
    assert "attendance.read" not in after["permissions"]
    replacement = client.post("/v1/rbac/supervisors", headers=principal, json={**body, "username": f"replacement.{role}"})
    assert replacement.status_code == 201, replacement.text


@pytest.mark.parametrize("changes,status", [
    ({"role_code": "system_admin"}, 422),
    ({"year_level_id": 999999}, 404),
    ({"username": "   "}, 422),
    ({"school_id": 999999}, 403),
])
def test_invalid_supervisor_leaves_no_account(client: TestClient, ids, principal, changes, status):
    body = {"username": "invalid.supervisor", "password": PASSWORD,
            "role_code": "floor_supervisor", "year_level_id": ids["level_p1"], **changes}
    response = client.post("/v1/rbac/supervisors", headers=principal, json=body)
    assert response.status_code == status, response.text
    with SqlAlchemyUnitOfWork() as uow:
        assert uow._session.scalar(select(m.User.id).where(m.User.username == body["username"].strip())) is None


def test_supervisor_cannot_create_accounts(client: TestClient, ids, principal):
    body = {"username": "restricted.supervisor", "password": PASSWORD,
            "role_code": "attendance_supervisor", "year_level_id": ids["level_p1"]}
    assert client.post("/v1/rbac/supervisors", headers=principal, json=body).status_code == 201
    headers = _sign_in(client, body["username"])
    response = client.post("/v1/rbac/supervisors", headers=headers, json={**body, "username": "unauthorized"})
    assert response.status_code == 403
