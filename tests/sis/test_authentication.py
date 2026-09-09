"""The API-key integration door is authenticated and scope constrained."""
from datetime import UTC, datetime, timedelta

import pytest

from sis.api.deps import hash_api_key, key_prefix
from sis.domain.auth import ApiKey, Scope
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.conftest import reader_headers


WRITE_ROUTE = "/v1/admin/api-keys"
READ_ROUTE = "/v1/schools"


def _store(raw: str, scope: Scope, **overrides) -> None:
    fields = {
        "prefix": key_prefix(raw), "key_hash": hash_api_key(raw),
        "label": "test key", "scope": scope, "is_active": True,
        "expires_at": None, "created_at": datetime.now(UTC),
    }
    fields.update(overrides)
    with SqlAlchemyUnitOfWork() as uow:
        uow.api_keys.add(ApiKey(**fields))
        uow.commit()


class TestCredentialsAreRequired:
    def test_read_and_write_routes_refuse_anonymous_callers(self, unauthenticated_client):
        assert unauthenticated_client.get(READ_ROUTE).status_code == 401
        assert unauthenticated_client.post(
            WRITE_ROUTE, json={"label": "mine", "scope": "reader"}
        ).status_code == 401

    def test_guardian_data_route_refuses_anonymous_callers(self, unauthenticated_client):
        response = unauthenticated_client.post(
            "/v1/guardians/resolve", json={"phone": "+201001234567"}
        )
        assert response.status_code == 401


class TestInvalidKeysAreRefused:
    @pytest.mark.parametrize("key", ["nothing-like-a-real-key-at-all", ""])
    def test_unknown_key_is_refused(self, unauthenticated_client, key):
        assert unauthenticated_client.get(READ_ROUTE, headers={"X-API-Key": key}).status_code == 401

    def test_revoked_and_expired_keys_are_refused(self, client):
        _store("revoked-key-000000000000000000", Scope.READER, is_active=False)
        _store(
            "expired-key-000000000000000000", Scope.READER,
            expires_at=datetime.now(UTC) - timedelta(days=1),
        )
        assert client.get(READ_ROUTE, headers={"X-API-Key": "revoked-key-000000000000000000"}).status_code == 401
        assert client.get(READ_ROUTE, headers={"X-API-Key": "expired-key-000000000000000000"}).status_code == 401


class TestScopesAreEnforced:
    def test_registrar_key_is_accepted_for_a_write(self, client):
        assert client.post(
            WRITE_ROUTE, json={"label": "valid", "scope": "reader"}
        ).status_code == 201

    def test_reader_key_cannot_mint_key_or_change_guardian_access(self, client):
        assert client.post(
            WRITE_ROUTE, json={"label": "mine", "scope": "registrar"}, headers=reader_headers()
        ).status_code == 403
        # This guardian-write route has an additional caller guard; either refusal is
        # safe, while the preceding API-key mint assertion proves scope enforcement.
        assert client.patch(
            "/v1/students/S-1/guardians/+201001234567",
            json={"can_view_records": True, "restriction_note": ""},
            headers=reader_headers(),
        ).status_code in {401, 403}
