"""The HTTP surface a non-bundled UI actually meets: CORS, and the static mount.

Both settings under test here were silently wrong, and neither could fail visibly
in the repo's own setup — the bundled Vue app is served from the backend's own
origin, so it never issues a cross-origin preflight and never depends on a 404
being JSON. The bugs only surface once someone points a separately hosted UI at
this API, which is exactly when there is no test to catch them.

So these assertions are written from the client's side of the wire: what an
`Access-Control-*` header set has to look like for a browser to honour it, not
what the middleware was configured with.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

# Note the absence of a `with TestClient(...)` anywhere below. Entering the context
# manager runs the app's startup hook, which initialises the database; none of these
# tests touch a route that needs one, and requiring Postgres to assert on a CORS
# header would be a poor trade.
#
# Imported as a module rather than `from backend import app`, which would bind the
# name to the module and collide with the FastAPI instance of the same name — the
# shape ImportShapeTests in test_request_context_di.py exists to prevent.
import backend.app as app_module


def build_client(**env) -> TestClient:
    """A client over a freshly composed app, with `env` applied during composition.

    `create_app` reads the environment as it builds the middleware stack, so the
    patch has to be in force for the call, not merely for the request.
    """
    with patch.dict("os.environ", env, clear=False):
        return TestClient(app_module.create_app())


class CorsTests(unittest.TestCase):
    """A bearer-token API, so credentialed mode is neither needed nor compatible."""

    def test_wildcard_default_never_claims_credentials(self):
        """The original bug, stated as the browser sees it.

        `Access-Control-Allow-Origin: *` together with
        `Access-Control-Allow-Credentials: true` is not a permissive combination —
        it is a rejected one. The fetch spec forbids the wildcard in credentialed
        mode, so a browser discards the header and the request fails. Shipping both
        is strictly worse than shipping either.
        """
        client = build_client(CORS_ALLOW_ORIGINS="")
        response = client.options(
            "/auth/login",
            headers={
                "Origin": "https://some-ui.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(response.headers.get("access-control-allow-origin"), "*")
        self.assertNotEqual(
            response.headers.get("access-control-allow-credentials"), "true",
            "the wildcard and credentialed mode cannot both be advertised",
        )

    def test_allowlist_echoes_permitted_origin_and_omits_others(self):
        client = build_client(
            CORS_ALLOW_ORIGINS="https://app.example.com, http://localhost:5173"
        )

        for origin in ("https://app.example.com", "http://localhost:5173"):
            with self.subTest(origin=origin):
                response = client.options(
                    "/auth/login",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                    },
                )
                self.assertEqual(
                    response.headers.get("access-control-allow-origin"), origin
                )

        denied = client.options(
            "/auth/login",
            headers={
                "Origin": "https://not-our-ui.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertIsNone(denied.headers.get("access-control-allow-origin"))

    def test_trailing_slash_in_configured_origin_still_matches(self):
        """An Origin header carries no path, so "https://app.example.com/" would
        otherwise match nothing at all — a misconfiguration with no error message."""
        client = build_client(CORS_ALLOW_ORIGINS="https://app.example.com/")
        response = client.options(
            "/auth/login",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "https://app.example.com",
        )

    def test_etag_is_readable_cross_origin(self):
        """Asset delivery revalidates on ETag, and the UI fetches assets from JS to
        attach its bearer token. An unexposed ETag means If-None-Match is never sent
        and the 304 branch in the asset route is dead code for every remote UI.

        Asserted on a real request rather than a preflight: Access-Control-Expose-
        Headers is only meaningful on the actual response, and the middleware sends
        it nowhere else.
        """
        client = build_client(CORS_ALLOW_ORIGINS="https://app.example.com")
        response = client.get(
            "/documents", headers={"Origin": "https://app.example.com"}
        )

        exposed = response.headers.get("access-control-expose-headers", "")
        self.assertIn("ETag", [item.strip() for item in exposed.split(",")])


class StaticMountTests(unittest.TestCase):
    """Whether the bundled UI is served is configuration, not a filesystem accident."""

    def test_backend_only_does_not_answer_root_from_disk(self):
        """The health-check trap. With the mount, "/" is a 200 read off disk and says
        nothing about whether the backend works; without it, "/" is an ordinary 404
        and a check has to name a real endpoint to pass."""
        client = build_client(SERVE_FRONTEND="false")
        response = client.get("/")

        self.assertEqual(response.status_code, 404)
        self.assertIn("application/json", response.headers.get("content-type", ""))

    def test_bundled_ui_still_served_when_enabled(self):
        """The opt-out must not become an opt-in: an unchanged deployment that
        relies on the backend serving frontend/dist keeps working."""
        with TemporaryDirectory() as tmp:
            dist = Path(tmp)
            (dist / "index.html").write_text("<!doctype html><title>UI</title>")

            with patch.object(app_module, "FRONTEND_DIR", dist):
                client = build_client(SERVE_FRONTEND="true")
                response = client.get("/")

            self.assertEqual(response.status_code, 200)
            self.assertIn("<title>UI</title>", response.text)

    def test_api_routes_win_over_the_static_mount(self):
        """Ordering regression guard: the router is included before the mount, so a
        real endpoint must never be shadowed by the static handler."""
        with TemporaryDirectory() as tmp:
            dist = Path(tmp)
            (dist / "index.html").write_text("<!doctype html><title>UI</title>")

            with patch.object(app_module, "FRONTEND_DIR", dist):
                client = build_client(SERVE_FRONTEND="true")
                # Unauthenticated, so it is rejected by the dependency rather than
                # answered — but rejected as an API route, which is the point.
                response = client.get("/documents")

            self.assertEqual(response.status_code, 401)
            self.assertNotIn("<title>UI</title>", response.text)


if __name__ == "__main__":
    unittest.main()
