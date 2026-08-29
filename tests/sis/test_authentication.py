"""This service authenticates nobody, and this file is the proof that it is on purpose.

API-key checking was removed from `sis/api/deps.py`: `X-API-Key` is not read, the
`api_keys` table is not consulted on a request, and no route refuses a caller for want
of a credential. Sign-in with a username and password is meant to replace it.

The tests below are written the way the refusal tests were — one claim each, stated out
loud — because a service that admits everyone should say so somewhere that runs. If a
future change puts authentication back, every test in this file fails at once, which is
the point: the day the door closes should be a deliberate edit here rather than a
surprise for whoever is holding the console.

**Reading these as reassurance would be the wrong reading.** Everything asserted here is
also true of anyone else who can reach the port. Until sign-in exists, the only thing
between this service and the internet is the network it is on.
"""
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from sis.api.deps import hash_api_key, key_prefix
from sis.domain.auth import ApiKey, Scope
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.conftest import reader_headers

# A route that used to require `registrar`, and one that used to require `reader`. Both
# are deliberately boring: what is under test is who may call them, which is everyone.
WRITE_ROUTE = "/v1/admin/api-keys"
READ_ROUTE = "/v1/schools"


def _store(raw: str, scope: Scope, **overrides) -> None:
    """Put a key in the school's database, hashed the way the API used to check it."""
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


class TestNoCredentialIsNeeded:
    def test_a_read_route_answers_a_caller_with_no_key(self, unauthenticated_client):
        assert unauthenticated_client.get(READ_ROUTE).status_code == 200

    def test_a_write_route_answers_a_caller_with_no_key(self, unauthenticated_client):
        """The consequential half. Reads being open is a disclosure; writes being open
        means anyone who can reach the port can change the school's records."""
        response = unauthenticated_client.post(
            WRITE_ROUTE, json={"label": "mine", "scope": "reader"}
        )
        assert response.status_code == 201

    def test_a_guardian_route_answers_a_caller_with_no_key(self, unauthenticated_client):
        """The route `identity/` resolves a parent's phone through.

        It answers whether a number belongs to a parent of this school, and it now
        answers that to anybody — including in a loop. Recorded here rather than left
        implicit, because this is the route that turns "open service" into "a stranger
        can enumerate which families attend".
        """
        response = unauthenticated_client.post(
            "/v1/guardians/resolve", json={"phone": "+201001234567"}
        )
        assert response.status_code != 401


class TestAKeyIsNotEvenLookedAt:
    """A key that would once have been refused now makes no difference at all."""

    @pytest.mark.parametrize(
        "key",
        [
            "nothing-like-a-real-key-at-all",
            "",
            "dev-sis-registrar",  # what the shipped console sends when nobody typed one
        ],
    )
    def test_an_unknown_key_is_ignored(self, unauthenticated_client, key):
        response = unauthenticated_client.get(READ_ROUTE, headers={"X-API-Key": key})
        assert response.status_code == 200

    def test_a_revoked_key_is_ignored(self, client):
        _store("revoked-key-000000000000000000", Scope.READER, is_active=False)
        response = client.get(
            READ_ROUTE, headers={"X-API-Key": "revoked-key-000000000000000000"}
        )
        assert response.status_code == 200, (
            "revoking a key changed nothing, which is what removing authentication "
            "means: there is no revocation left to perform."
        )

    def test_an_expired_key_is_ignored(self, client):
        _store(
            "expired-key-000000000000000000",
            Scope.READER,
            expires_at=datetime.now(UTC) - timedelta(days=1),
        )
        response = client.get(
            READ_ROUTE, headers={"X-API-Key": "expired-key-000000000000000000"}
        )
        assert response.status_code == 200

    def test_a_non_ascii_key_does_not_crash_the_request(self, client):
        """The header is caller-controlled and is not guaranteed ASCII.

        Sent as raw bytes because `httpx` refuses to encode a non-ASCII header *string* —
        but nothing stops a hostile client putting those bytes on the wire, and starlette
        decodes them latin-1 into a `str` that is not ASCII. Nothing reads the header any
        more, so the crafted value has to be inert rather than a 500.
        """
        response = client.get(
            READ_ROUTE, headers={"X-API-Key": "cle-invalide\xe9".encode("latin-1")}
        )
        assert response.status_code == 200


class TestScopesNoLongerSeparateAnything:
    """`reader` and `registrar` still name what a route is for. Neither gates it."""

    def test_a_reader_key_can_mint_a_key(self, client):
        response = client.post(
            WRITE_ROUTE,
            json={"label": "mine", "scope": "registrar"},
            headers=reader_headers(),
        )
        assert response.status_code == 201

    def test_a_reader_key_can_change_a_guardians_access(self, client):
        """The custody route: it grants and revokes a parent's access to a child's
        records, and it is now open to whoever asks.

        Anything but a 403 passes — the request was authorised, and what happens next is
        about the student handle rather than about the caller.
        """
        response = client.patch(
            "/v1/students/S-1/guardians/+201001234567",
            json={"can_view_records": True, "restriction_note": ""},
            headers=reader_headers(),
        )
        assert response.status_code != 403

    def test_a_reader_key_can_upload_a_roster(self, client):
        """A 4xx about the file is fine; a 403 about the credential is what must be gone."""
        response = client.post(
            "/v1/imports/roster/preview",
            files={
                "file": ("roster.xlsx", b"not really a workbook", "application/vnd.ms-excel")
            },
            headers=reader_headers(),
        )
        assert response.status_code != 403


class TestSchoolsAreStillSeparated:
    """Removing authentication did not touch which database answers a request.

    `X-School-Code` chooses the connection, and always did that job rather than deciding
    who may ask. It is asserted here so the two are not confused later: this service no
    longer knows who is calling, and it still knows exactly which school it is answering
    about.
    """

    def test_each_school_is_answered_from_its_own_database(self, split_school_client):
        for code in ("NC", "MD"):
            response = split_school_client.get(READ_ROUTE, headers={"X-School-Code": code})
            assert response.status_code == 200

    def test_a_request_naming_no_school_is_still_refused(self, split_school_client):
        """The one refusal left on this path, and it is not about credentials."""
        response = split_school_client.get(READ_ROUTE)
        assert response.status_code == 422


@pytest.fixture()
def split_school_client(two_databases):
    """Two schools, two databases, no credential anywhere.

    Reuses `test_physical_separation`'s fixture rather than building a third split
    estate — that fixture is the one place that knows how the registry is pointed at two
    files, and a second copy would drift from it.
    """
    from sis.app import app

    with TestClient(app) as test_client:
        yield test_client
