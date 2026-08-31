"""What this API tells a client about how to authenticate, and how it refuses one.

The backend verifies bearer tokens and issues none. It said otherwise for a while: its
security scheme was `OAuth2PasswordBearer`, which advertises the OAuth2 password grant and
carries a `tokenUrl`. Three things followed, and all three are asserted here.

  * Swagger rendered a sign-in form — username, password, client_id, client_secret —
    against a service that has no sign-in route.
  * That form could not have worked at any URL. The grant is form-encoded by
    specification; identity's `/v1/auth/login` reads a JSON body. The Authorize button was
    a 422 waiting to happen, and on a deployment whose `IDENTITY_BASE_URL` named an
    in-cluster host it did not get that far — the browser could not resolve it at all.
  * `tokenUrl` is copied verbatim into the published `openapi.json`, so that in-cluster
    hostname was printed in a public document.

The runtime half is asserted too, because the fix must not have moved it: the refusals a
caller sees for a missing, malformed or invalid credential are the ones it saw before.
"""
import unittest

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import backend.infra.auth as backend_auth


def _protected_app() -> FastAPI:
    """The smallest app that mounts the real dependency.

    Composed here rather than via `backend.app.create_app` on purpose: this file is about
    the security scheme, and building the whole application would drag in a profile, a
    database and a middleware stack that have nothing to say about it.
    """
    app = FastAPI()

    @app.get("/protected")
    def protected(token: str = Depends(backend_auth.bearer_token)) -> dict:
        # Echoes the token back so the happy path can prove the *parsed* value arrives,
        # not merely that the request was let through.
        return {"token": token}

    return app


class SchemeShapeTests(unittest.TestCase):
    """What `openapi.json` advertises."""

    def setUp(self):
        self.schema = _protected_app().openapi()
        self.schemes = self.schema.get("components", {}).get("securitySchemes", {})

    def test_the_scheme_is_http_bearer(self):
        """One box that takes a token — not a grant this service cannot perform."""
        self.assertEqual(1, len(self.schemes), self.schemes)
        scheme = next(iter(self.schemes.values()))
        self.assertEqual("http", scheme.get("type"))
        self.assertEqual("bearer", scheme.get("scheme"))

    def test_no_password_grant_is_advertised(self):
        """The specific regression: `type: oauth2` with a `password` flow.

        Asserted over the whole document rather than the one scheme, so re-introducing
        the grant anywhere — a second dependency, a router with its own security — fails
        here too.
        """
        rendered = repr(self.schema)
        self.assertNotIn("oauth2", rendered)
        self.assertNotIn("password", rendered)
        self.assertNotIn("client_secret", rendered)

    def test_no_token_url_and_so_no_service_address_is_published(self):
        """`tokenUrl` was the only route by which another service's address got in here.

        `identity:8200` is the value the deployment actually published; `localhost:8200`
        was the module default that would have been published everywhere else.
        """
        rendered = repr(self.schema)
        self.assertNotIn("tokenUrl", rendered)
        self.assertNotIn("identity:8200", rendered)
        self.assertNotIn("localhost:8200", rendered)

    def test_the_document_no_longer_depends_on_identity_base_url(self):
        """Setting it must change nothing, which is what makes it safe to stop setting."""
        from unittest.mock import patch

        with patch.dict(
            "os.environ", {"IDENTITY_BASE_URL": "http://should-not-appear:9999"}, clear=False
        ):
            import importlib

            importlib.reload(backend_auth)
            try:
                rendered = repr(_protected_app().openapi())
                self.assertNotIn("should-not-appear", rendered)
            finally:
                # Restore the module every other test in the process shares.
                importlib.reload(backend_auth)


class RefusalTests(unittest.TestCase):
    """How a caller is turned away. Unchanged from the OAuth2 scheme, deliberately.

    Both schemes build the identical refusal — 401, detail "Not authenticated", and
    `WWW-Authenticate: Bearer` — so these are a guard against the swap having quietly
    moved a status code, not a description of new behaviour.
    """

    def setUp(self):
        self.client = TestClient(_protected_app())

    def test_a_well_formed_token_reaches_the_route(self):
        """The happy path: the dependency yields the token, unwrapped and unmodified."""
        response = self.client.get(
            "/protected", headers={"Authorization": "Bearer a.b.c"}
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual("a.b.c", response.json()["token"])

    def test_a_lowercase_scheme_is_accepted(self):
        """RFC 7235 makes the scheme case-insensitive, and the old scheme accepted it."""
        response = self.client.get(
            "/protected", headers={"Authorization": "bearer a.b.c"}
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual("a.b.c", response.json()["token"])

    def test_a_missing_header_is_401_and_names_the_challenge(self):
        response = self.client.get("/protected")
        self.assertEqual(401, response.status_code)
        self.assertEqual("Bearer", response.headers.get("www-authenticate"))
        self.assertEqual("Not authenticated", response.json()["detail"])

    def test_another_scheme_is_401_not_500(self):
        """A browser with a stale Basic credential must be refused, not crash the route."""
        response = self.client.get(
            "/protected", headers={"Authorization": "Basic dXNlcjpwYXNz"}
        )
        self.assertEqual(401, response.status_code)
        self.assertEqual("Bearer", response.headers.get("www-authenticate"))

    def test_a_bearer_with_no_token_is_401(self):
        response = self.client.get("/protected", headers={"Authorization": "Bearer "})
        self.assertEqual(401, response.status_code)

    def test_a_garbage_header_is_401(self):
        response = self.client.get("/protected", headers={"Authorization": "a.b.c"})
        self.assertEqual(401, response.status_code)


class ContractTests(unittest.TestCase):
    """The seam the rest of the backend depends on, held still."""

    def test_get_current_user_still_takes_a_token_string(self):
        """Every route and every test calls it as `get_current_user(token=..., db=...)`.

        The scheme changed; this signature deliberately did not, which is why no call site
        moved. `bearer_token` absorbs the difference.
        """
        import inspect

        parameters = inspect.signature(backend_auth.get_current_user).parameters
        self.assertIn("token", parameters)
        self.assertIs(str, parameters["token"].annotation)

    def test_the_historical_export_still_names_the_token_dependency(self):
        """`oauth2_scheme` is named in the module docstring as part of the contract."""
        self.assertIs(backend_auth.bearer_token, backend_auth.oauth2_scheme)


if __name__ == "__main__":
    unittest.main()
