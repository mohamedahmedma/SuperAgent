"""Who may call this service, and how each way of not being allowed is refused.

`188fd45` removed API-key authentication and the tests that asserted it, leaving the note
that it "needs restoring before the port is reachable by anything else, and certainly
before parent login makes guardian data a login credential." Both of those have since
happened: `identity/` resolves a parent's phone against these guardian routes, and
`records/` reads a child's marks through them.

So this file is the restored claim, plus the two things the original could not assert
because they did not exist yet: keys live in a school's own database, and the parent-facing
guardian routes are reachable by a `reader` credential that cannot write.

Every refusal here is checked for *how* it refuses as well as whether. An unknown prefix, a
wrong secret, a revoked key and an expired one all have to answer identically — a caller
who can tell them apart can enumerate which handles are real and learn that a leaked key
was noticed, from error text alone.
"""
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from sis.api.deps import hash_api_key, key_prefix
from sis.domain.auth import ApiKey, Scope
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from sis.tests.conftest import (
    READER_KEY,
    REGISTRAR_KEY,
    reader_headers,
    registrar_headers,
)

# A route that only a registrar may reach, and one any reader may. Both are deliberately
# boring: what is under test is the credential, not the handler.
WRITE_ROUTE = "/v1/admin/api-keys"
READ_ROUTE = "/v1/schools"


def _store(raw: str, scope: Scope, **overrides) -> None:
    """Put a key in the school's database, hashed the way the API authenticates it."""
    fields = {
        "prefix": key_prefix(raw),
        "key_hash": hash_api_key(raw),
        "label": "test key",
        "scope": scope,
        "is_active": True,
        "expires_at": None,
        "created_at": datetime.now(UTC),
    }
    fields.update(overrides)
    with SqlAlchemyUnitOfWork() as uow:
        uow.api_keys.add(ApiKey(**fields))
        uow.commit()


class TestAnonymousCallers:
    def test_a_read_route_refuses_a_caller_with_no_key(self, unauthenticated_client):
        response = unauthenticated_client.get(READ_ROUTE)
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "not_authorized"

    def test_a_write_route_refuses_a_caller_with_no_key(self, unauthenticated_client):
        response = unauthenticated_client.post(
            WRITE_ROUTE, json={"label": "mine", "scope": "reader"}
        )
        assert response.status_code == 401

    def test_the_refusal_names_the_header_to_present(self, unauthenticated_client):
        """A 401 with no `WWW-Authenticate` leaves an integrator guessing."""
        response = unauthenticated_client.get(READ_ROUTE)
        assert response.headers.get("www-authenticate") == "X-API-Key"

    def test_a_guardian_route_is_not_quietly_open(self, unauthenticated_client):
        """The route `identity/` resolves a parent's phone through.

        It answers whether a number belongs to a parent of this school, which is exactly
        the question an unauthenticated caller must not be able to ask in a loop.
        """
        response = unauthenticated_client.post(
            "/v1/guardians/resolve", json={"phone": "+201001234567"}
        )
        assert response.status_code == 401


class TestBadCredentials:
    """Every wrong key is wrong in the same way, as far as the caller can tell."""

    def _refusal(self, client: TestClient, key: str) -> tuple[int, str]:
        response = client.get(READ_ROUTE, headers={"X-API-Key": key})
        return response.status_code, response.json()["detail"]["message"]

    def test_an_unknown_prefix_is_refused(self, client):
        status, _ = self._refusal(client, "nothing-like-a-real-key-at-all")
        assert status == 401

    def test_a_real_prefix_with_a_wrong_secret_is_refused(self, client):
        forged = REGISTRAR_KEY[:12] + "-but-the-rest-is-invented"
        status, _ = self._refusal(client, forged)
        assert status == 401

    def test_a_revoked_key_is_refused(self, client):
        _store("revoked-key-000000000000000000", Scope.READER, is_active=False)
        status, _ = self._refusal(client, "revoked-key-000000000000000000")
        assert status == 401

    def test_an_expired_key_is_refused(self, client):
        _store(
            "expired-key-000000000000000000",
            Scope.READER,
            expires_at=datetime.now(UTC) - timedelta(days=1),
        )
        status, _ = self._refusal(client, "expired-key-000000000000000000")
        assert status == 401

    def test_a_key_expiring_later_still_works(self, client):
        _store(
            "future-key-0000000000000000000",
            Scope.READER,
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        response = client.get(
            READ_ROUTE, headers={"X-API-Key": "future-key-0000000000000000000"}
        )
        assert response.status_code == 200

    def test_every_kind_of_wrong_key_is_indistinguishable(self, client):
        """The property, stated once.

        Unknown, forged, revoked and expired must produce one answer. If this test fails,
        the error text has become an oracle: it tells a caller which handles exist, and it
        tells whoever leaked a key that somebody noticed.
        """
        _store("revoked-two-00000000000000000", Scope.READER, is_active=False)
        _store(
            "expired-two-00000000000000000",
            Scope.READER,
            expires_at=datetime.now(UTC) - timedelta(days=1),
        )
        answers = {
            self._refusal(client, key)
            for key in (
                "nothing-like-a-real-key-at-all",
                REGISTRAR_KEY[:12] + "-but-the-rest-is-invented",
                "revoked-two-00000000000000000",
                "expired-two-00000000000000000",
            )
        }
        assert len(answers) == 1, f"these four told a caller different things: {answers}"

    def test_a_non_ascii_key_is_refused_rather_than_crashing(self, client, bootstrap_key):
        """The header is entirely caller-controlled, and it is not guaranteed ASCII.

        Sent as raw bytes because `httpx` refuses to encode a non-ASCII header *string* —
        but nothing stops a hostile client putting those bytes on the wire, and starlette
        decodes them latin-1 into a `str` that is not ASCII. `secrets.compare_digest`
        raises `TypeError` on exactly that, which would turn one crafted request into a
        500 instead of a refusal. The bootstrap key is enabled here because the comparison
        that would raise is the one guarding it.
        """
        response = client.get(
            READ_ROUTE, headers={"X-API-Key": "clé-invalide".encode("latin-1")}
        )
        assert response.status_code == 401


class TestScopesDoNotNest:
    """`registrar` does not imply `reader` and `reader` never implies a write."""

    def test_a_reader_key_cannot_mint_a_key(self, client):
        response = client.post(
            WRITE_ROUTE, json={"label": "mine", "scope": "registrar"}, headers=reader_headers()
        )
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "not_authorized"

    def test_a_reader_key_cannot_change_a_guardians_access(self, client):
        """The custody route. A read-only integration must never reach it.

        This is the one `188fd45` removed by name, and it is the most consequential: the
        route grants and revokes a parent's access to a child's records.
        """
        response = client.patch(
            "/v1/students/S-1/guardians/+201001234567",
            json={"can_view_records": True, "restriction_note": ""},
            headers=reader_headers(),
        )
        assert response.status_code == 403

    def test_a_reader_key_cannot_upload_a_roster(self, client):
        response = client.post(
            "/v1/imports/roster/preview",
            files={"file": ("roster.xlsx", b"not really a workbook", "application/vnd.ms-excel")},
            headers=reader_headers(),
        )
        assert response.status_code == 403

    def test_a_registrar_key_may_still_read(self, client):
        """Reads a registrar legitimately performs are not locked behind `reader`.

        The registrar console prints report cards. If `require_read_access` narrowed to
        `reader`, the console would need a second key and somebody would give it a
        registrar one for everything instead.
        """
        assert client.get(READ_ROUTE, headers=registrar_headers()).status_code == 200

    def test_a_reader_key_may_read(self, client):
        assert client.get(READ_ROUTE, headers=reader_headers()).status_code == 200

    def test_a_reader_key_reaches_the_parent_facing_guardian_route(self, client):
        """What `records/` holds, and the whole reason `reader` exists.

        A 404 is the pass condition: the credential was accepted and the handle simply
        names nobody. A 401 or 403 would mean the adapter cannot do its job with a
        read-only key and would end up holding the school's write credential.
        """
        response = client.get(
            "/v1/guardians/by-id/no-such-handle/students", headers=reader_headers()
        )
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "unknown_reference"


class TestBootstrapKey:
    def test_it_authenticates_as_a_registrar(self, client, bootstrap_key):
        response = client.post(
            WRITE_ROUTE,
            json={"label": "first real key", "scope": "reader"},
            headers={"X-API-Key": bootstrap_key},
        )
        assert response.status_code == 201

    def test_it_does_nothing_when_unset(self, client):
        """Unsetting the variable revokes it, with no migration and no stored row."""
        response = client.get(
            READ_ROUTE, headers={"X-API-Key": "bootstrap-fixture-key-000000000"}
        )
        assert response.status_code == 401


class TestUseIsRecorded:
    def test_a_successful_call_stamps_last_used(self, client):
        """So an operator can see which keys are dead weight before revoking one."""
        client.get(READ_ROUTE, headers=reader_headers())
        with SqlAlchemyUnitOfWork() as uow:
            stored = uow.api_keys.get_by_prefix(key_prefix(READER_KEY))
        assert stored is not None
        assert stored.last_used_at is not None

    def test_a_refused_call_does_not(self, client):
        """A wrong secret against a real prefix must not touch the key it guessed at."""
        client.get(
            READ_ROUTE, headers={"X-API-Key": REGISTRAR_KEY[:12] + "-wrong-secret-here"}
        )
        with SqlAlchemyUnitOfWork() as uow:
            stored = uow.api_keys.get_by_prefix(key_prefix(REGISTRAR_KEY))
        assert stored is not None
        assert stored.last_used_at is None


class TestKeysBelongToOneSchool:
    """Physical separation reaching authentication rather than stopping short of it."""

    def test_a_key_minted_for_one_school_is_refused_at_another(self, split_school_client):
        client, keys = split_school_client
        nc_key = keys["NC"]

        assert client.get(
            READ_ROUTE, headers={"X-API-Key": nc_key, "X-School-Code": "NC"}
        ).status_code == 200

        crossed = client.get(
            READ_ROUTE, headers={"X-API-Key": nc_key, "X-School-Code": "MD"}
        )
        assert crossed.status_code == 401, (
            "a key minted at one branch authenticated at another. Physical separation "
            "stops at the door if the credential is estate-wide."
        )


@pytest.fixture()
def split_school_client(two_databases, bootstrap_key):
    """Two schools, each with its own stored registrar key.

    Reuses `test_physical_separation`'s fixture rather than building a third split
    estate — the fixture is the one place that knows how the registry is pointed at two
    files, and a second copy would drift from it.
    """
    from sis.app import app

    minted: dict[str, str] = {}
    for code in ("NC", "MD"):
        raw = f"{code.lower()}-school-key-00000000000000"
        with SqlAlchemyUnitOfWork(school_code=code) as uow:
            uow.api_keys.add(
                ApiKey(
                    prefix=key_prefix(raw),
                    key_hash=hash_api_key(raw),
                    label=f"{code} registrar",
                    scope=Scope.REGISTRAR,
                    is_active=True,
                    expires_at=None,
                    created_at=datetime.now(UTC),
                )
            )
            uow.commit()
        minted[code] = raw

    with TestClient(app) as test_client:
        yield test_client, minted
