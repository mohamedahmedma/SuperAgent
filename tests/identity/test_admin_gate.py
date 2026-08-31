"""Who may reach `/v1/admin/*`, now that there is no shared key.

The routes used to take an `X-Admin-Key` header holding one secret shared by every script
and every operator. Two things were wrong with it, and the second is the one that actually
bit: it was unset on the deployed system, so every admin route answered 503 and nobody could
create an account or bind a guardian at all; and a shared secret has no identity, so the
binding route — audited separately precisely because "who decided this parent is that
guardian" is the first question after a records leak — could not answer that question.

They now take an administrator's own access token. This file pins what that means at the
edge: which callers get in, which are turned away, and with which status.

The status codes are load-bearing and are asserted individually:

  401  the credential is missing, malformed, expired or forged. Re-presenting it is futile,
       but obtaining a good one is not.
  403  the credential is genuine and belongs to somebody who is not an administrator.

Collapsing those two would tell a parent holding a valid token that their token is invalid,
which sends them to re-authenticate in a loop that cannot succeed.
"""
import pytest

from tests.identity.conftest import BOOTSTRAP_ADMIN_PASSWORD, BOOTSTRAP_ADMIN_USER

#: One representative of each shape of admin route: a body POST, a body PUT on a path
#: parameter, and a bodyless DELETE. Every one of them must be gated identically — a route
#: added later that forgets the dependency is exactly the failure this parametrisation is
#: here to catch.
ADMIN_CALLS = (
    ("post", "/v1/admin/accounts", {"username": "x", "password": "y"}),
    ("put", "/v1/admin/accounts/someone/guardian-binding", {"guardian_external_id": "G-1"}),
    ("delete", "/v1/admin/accounts/someone/guardian-binding", None),
)


def call(client, method, path, body, headers=None):
    kwargs = {"headers": headers or {}}
    if body is not None:
        kwargs["json"] = body
    return getattr(client, method)(path, **kwargs)


class TestTheAdministratorGetsIn:
    """The happy path, end to end: seeded account -> login -> admin route."""

    def test_the_seeded_administrator_can_create_an_account(self, client, admin_headers):
        response = client.post(
            "/v1/admin/accounts",
            headers=admin_headers,
            json={"username": "0501112222", "password": "correct-horse-battery"},
        )

        assert response.status_code == 201, response.text
        assert response.json()["username"] == "0501112222"

    def test_the_seeded_administrator_can_bind_a_guardian(self, client, admin_headers):
        """The most sensitive write in the system, reachable without any shared secret."""
        client.post(
            "/v1/admin/accounts",
            headers=admin_headers,
            json={"username": "0503334444", "password": "correct-horse-battery"},
        )

        response = client.put(
            "/v1/admin/accounts/0503334444/guardian-binding",
            headers=admin_headers,
            json={"guardian_external_id": "G-7"},
        )

        assert response.status_code == 200, response.text
        assert response.json()["guardian_id"] == "G-7"

    def test_the_token_comes_from_an_ordinary_login(self, client):
        """No special ceremony: the admin signs in exactly as anyone else does.

        Which also means the account obeys the lockout policy, and its password can be
        changed through the API rather than by editing `.env` and redeploying.
        """
        response = client.post(
            "/v1/auth/login",
            json={"username": BOOTSTRAP_ADMIN_USER, "password": BOOTSTRAP_ADMIN_PASSWORD},
        )

        assert response.status_code == 200
        assert response.json()["role"] == "admin"


class TestEverybodyElseIsTurnedAway:
    @pytest.mark.parametrize("method,path,body", ADMIN_CALLS)
    def test_no_credential_at_all_is_401(self, client, method, path, body):
        assert call(client, method, path, body).status_code == 401

    @pytest.mark.parametrize("method,path,body", ADMIN_CALLS)
    def test_a_garbage_token_is_401(self, client, method, path, body):
        headers = {"Authorization": "Bearer not.a.real.token"}
        assert call(client, method, path, body, headers).status_code == 401

    @pytest.mark.parametrize("method,path,body", ADMIN_CALLS)
    def test_the_old_shared_key_no_longer_opens_anything(self, client, method, path, body):
        """The regression that proves the key is genuinely gone.

        An old script still sending `X-Admin-Key` must be refused rather than quietly
        served — a half-removed credential that still works somewhere is worse than either
        keeping it or removing it.
        """
        headers = {"X-Admin-Key": "test-admin-key"}
        assert call(client, method, path, body, headers).status_code == 401

    @pytest.mark.parametrize("method,path,body", ADMIN_CALLS)
    def test_a_non_admin_token_is_403_not_401(self, client, admin_headers, method, path, body):
        """A genuine credential belonging to somebody who is not an administrator.

        403, because 401 would tell a signed-in parent their session is broken and send
        them round a re-authentication loop that cannot ever succeed.
        """
        client.post(
            "/v1/admin/accounts",
            headers=admin_headers,
            json={
                "username": "0505556666",
                "password": "correct-horse-battery",
                "role": "parent",
            },
        )
        token = client.post(
            "/v1/auth/login",
            json={"username": "0505556666", "password": "correct-horse-battery"},
        ).json()["access_token"]

        response = call(
            client, method, path, body, {"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 403, response.text

    def test_a_parent_token_cannot_bind_a_guardian(self, client, admin_headers):
        """Stated on its own because it is the attack that matters.

        A parent who could reach the binding route could name themselves any guardian and
        read that family's marks. The old design made this unreachable by credential TYPE;
        the new one rejects it on the role claim, and that rejection is the whole safety
        argument, so it gets its own test rather than only a parametrised one.
        """
        client.post(
            "/v1/admin/accounts",
            headers=admin_headers,
            json={
                "username": "0507778888",
                "password": "correct-horse-battery",
                "role": "parent",
            },
        )
        token = client.post(
            "/v1/auth/login",
            json={"username": "0507778888", "password": "correct-horse-battery"},
        ).json()["access_token"]

        response = client.put(
            "/v1/admin/accounts/0507778888/guardian-binding",
            headers={"Authorization": f"Bearer {token}"},
            json={"guardian_external_id": "G-1"},
        )

        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "not_authorized"


class TestSwaggerCanActuallyBeUsed:
    """The workflow these routes exist for: an operator managing accounts through /docs.

    That workflow was impossible even with a valid token, because identity published no
    security scheme at all — every credential was read from a raw `Header` parameter, so
    Swagger rendered no Authorize button and there was nowhere to put one. The routes were
    reachable by curl and by nothing a person would use.
    """

    def test_a_bearer_scheme_is_advertised(self, client):
        schemes = client.app.openapi()["components"]["securitySchemes"]

        assert len(schemes) == 1, schemes
        scheme = next(iter(schemes.values()))
        assert scheme["type"] == "http"
        assert scheme["scheme"] == "bearer"

    def test_the_admin_routes_require_it(self, client):
        """Without this the Authorize button exists and those routes ignore it."""
        paths = client.app.openapi()["paths"]

        for path, method in (
            ("/v1/admin/accounts", "post"),
            ("/v1/admin/accounts/{username}/guardian-binding", "put"),
            ("/v1/admin/accounts/{username}/guardian-binding", "delete"),
        ):
            security = paths[path][method].get("security")
            assert security, f"{method.upper()} {path} advertises no credential"

    def test_no_password_grant_is_advertised(self, client):
        """This service issues tokens, but never through an OAuth2 flow.

        Declaring one would make Swagger render a username/password form and POST it
        form-encoded, while `/v1/auth/login` reads a JSON body — the same mistake the chat
        backend had, where the Authorize button was a 422 waiting to happen.
        """
        rendered = repr(client.app.openapi())

        assert "oauth2" not in rendered
        assert "tokenUrl" not in rendered


class TestTheCredentialIsRevocable:
    """What a named account buys that a shared key could not."""

    def test_deactivating_the_administrator_closes_the_door_on_the_next_login(
        self, client, admin_headers
    ):
        """A shared key could only be rotated by editing config and redeploying everything.

        An account is deactivated in the database, and `SessionService.login` refuses it on
        the next attempt — see identity/application/services/sessions.py, which checks
        `is_active` before the password.
        """
        from identity.infrastructure.db.models import Account
        from identity.infrastructure.db.session import new_session

        db = new_session()
        try:
            account = (
                db.query(Account).filter(Account.username == BOOTSTRAP_ADMIN_USER).first()
            )
            account.is_active = False
            db.commit()
        finally:
            db.close()

        response = client.post(
            "/v1/auth/login",
            json={"username": BOOTSTRAP_ADMIN_USER, "password": BOOTSTRAP_ADMIN_PASSWORD},
        )

        assert response.status_code == 401
