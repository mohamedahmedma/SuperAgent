"""Stage 10: global administrator authority and the reversible maintenance gate."""
from fastapi.testclient import TestClient
from sqlalchemy import select

from sis.demo.seeder import sync_roles
from sis.domain.rbac import RoleCode, ScopeType
from sis.infrastructure.crypto import hash_password
from sis.infrastructure.db import models as m
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.test_timetable_api import registrar, school  # noqa: F401

PASSWORD = "system-admin-test-password"


def _account(client: TestClient, role: RoleCode, username: str) -> dict[str, str]:
    with SqlAlchemyUnitOfWork() as uow:
        sync_roles(uow._session)
        school_id = uow._session.scalar(select(m.School.id).limit(1))
        user = m.User(
            username=username,
            password_hash=hash_password(PASSWORD),
            school_id=None if role is RoleCode.SYSTEM_ADMIN else school_id,
        )
        uow._session.add(user)
        uow._session.flush()
        role_id = uow._session.scalar(select(m.Role.id).where(m.Role.code == role.value))
        uow._session.add(
            m.UserRole(
                user_id=user.id,
                role_id=role_id,
                scope_type=(
                    ScopeType.GLOBAL.value
                    if role is RoleCode.SYSTEM_ADMIN
                    else ScopeType.SCHOOL.value
                ),
                scope_id=None if role is RoleCode.SYSTEM_ADMIN else school_id,
                granted_by="test",
            )
        )
        uow.commit()
    response = client.post(
        "/v1/auth/login", json={"username": username, "password": PASSWORD}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_paused_system_blocks_normal_users_but_not_system_administrator(
    client: TestClient, school: None,
) -> None:
    admin = _account(client, RoleCode.SYSTEM_ADMIN, "sysadmin.stage10")
    principal = _account(client, RoleCode.PRINCIPAL, "principal.stage10")

    paused = client.put(
        "/v1/admin/system/status",
        headers=admin,
        json={"status": "paused", "note": "Database upgrade"},
    )
    assert paused.status_code == 200, paused.text
    assert paused.json()["updated_by"] == "sysadmin.stage10"

    refused = client.get("/v1/rbac/roles", headers=principal)
    assert refused.status_code == 503
    assert refused.json()["detail"]["system_status"] == "paused"
    assert refused.headers["retry-after"] == "300"

    assert client.get("/v1/rbac/roles", headers=admin).status_code == 200
    restored = client.put(
        "/v1/admin/system/status",
        headers=admin,
        json={"status": "active", "note": ""},
    )
    assert restored.status_code == 200


def test_maintenance_is_read_only_for_normal_users(client: TestClient, school: None) -> None:
    admin = _account(client, RoleCode.SYSTEM_ADMIN, "sysadmin.maintenance")
    principal = _account(client, RoleCode.PRINCIPAL, "principal.maintenance")
    assert client.put(
        "/v1/admin/system/status",
        headers=admin,
        json={"status": "maintenance", "note": "Applying updates"},
    ).status_code == 200

    assert client.get("/v1/rbac/roles", headers=principal).status_code == 200
    blocked = client.post(
        "/v1/rbac/users",
        headers=principal,
        json={"username": "blocked", "password": "long-enough-password"},
    )
    assert blocked.status_code == 503
    assert client.put(
        "/v1/admin/system/status",
        headers=admin,
        json={"status": "active"},
    ).status_code == 200


def test_system_administrator_can_manage_accounts(client: TestClient, school: None) -> None:
    admin = _account(client, RoleCode.SYSTEM_ADMIN, "sysadmin.users")
    created = client.post(
        "/v1/rbac/users",
        headers=admin,
        json={
            "username": "new.operator",
            "password": "long-enough-password",
            "full_name_en": "New Operator",
            "preferred_language": "en",
        },
    )
    assert created.status_code == 201, created.text
    user_id = created.json()["id"]
    updated = client.patch(
        f"/v1/rbac/users/{user_id}", headers=admin, json={"is_active": False}
    )
    assert updated.status_code == 200
    assert updated.json()["is_active"] is False
