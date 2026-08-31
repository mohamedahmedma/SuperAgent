"""SIS's cross-origin policy is its own, and must stay that way.

Three services in this estate configure CORS. `backend/` and `identity/` share
`CORS_ALLOW_ORIGINS`, deliberately: the parents' Vue app calls both, and two variables
would let them disagree about which origin the UI is. `sis/` reads `SIS_CORS_ORIGINS`
instead, and the difference in name looks like an inconsistency somebody forgot to tidy.

It is not, and tidying it is a security regression. `CORS_ALLOW_ORIGINS` names the PARENTS'
app. That app never calls sis — sis is reached from the registrar console it serves itself
at `/ui`, same-origin, needing no CORS at all. Folding the names together would tell a
service that answers with named children's marks to start accepting requests from the
parents' front end, and sis authenticates nobody at present, so an allowed origin is most
of what stands between a page and a term's marks.

So this file pins the separation rather than the convenience: unset means CLOSED, and
`CORS_ALLOW_ORIGINS` must not open it.
"""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import sis.app as sis_app


def build_client(**env) -> TestClient:
    """A client over a freshly composed app, with `env` in force during composition.

    The middleware stack is built inside `create_app`, so the patch has to cover the call
    and not merely the request — the same reason `tests/general/test_http_surface.py`
    builds its clients this way.
    """
    with patch.dict(os.environ, env, clear=False):
        return TestClient(sis_app.create_app())


def preflight(client: TestClient, origin: str):
    return client.options(
        "/health",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


class ClosedByDefaultTests(unittest.TestCase):
    def test_no_origin_is_allowed_when_nothing_is_configured(self):
        """The default, and it must stay closed.

        A permissive default would mean any page the registrar happens to have open can
        read children's marks the moment a browser session exists — and nobody would
        notice, because the service works exactly as well either way.
        """
        client = build_client(SIS_CORS_ORIGINS="")
        response = preflight(client, "https://anything.example.com")
        self.assertIsNone(response.headers.get("access-control-allow-origin"))


class NoInheritanceTests(unittest.TestCase):
    """The regression this file exists for."""

    def test_the_parents_app_origin_does_not_open_sis(self):
        """`CORS_ALLOW_ORIGINS` is the parents' Vue app. It must not reach sis.

        If this fails, somebody unified the two variable names — which reads like a
        cleanup and hands the parent-facing front end cross-origin access to the
        registrar's API.
        """
        client = build_client(
            CORS_ALLOW_ORIGINS="https://superagent.aurexis.cc",
            SIS_CORS_ORIGINS="",
        )
        response = preflight(client, "https://superagent.aurexis.cc")
        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_sis_stays_closed_even_when_the_shared_list_is_a_wildcard(self):
        client = build_client(CORS_ALLOW_ORIGINS="*", SIS_CORS_ORIGINS="")
        response = preflight(client, "https://anything.example.com")
        self.assertIsNone(response.headers.get("access-control-allow-origin"))


class ItsOwnListTests(unittest.TestCase):
    """What SIS_CORS_ORIGINS does when it IS set — a separately hosted console."""

    def test_a_named_origin_is_echoed(self):
        client = build_client(SIS_CORS_ORIGINS="https://registrar.example.com")
        response = preflight(client, "https://registrar.example.com")
        self.assertEqual(
            "https://registrar.example.com",
            response.headers.get("access-control-allow-origin"),
        )

    def test_an_origin_not_on_the_list_is_refused(self):
        client = build_client(SIS_CORS_ORIGINS="https://registrar.example.com")
        response = preflight(client, "https://superagent.aurexis.cc")
        self.assertNotEqual(
            "https://superagent.aurexis.cc",
            response.headers.get("access-control-allow-origin"),
        )

    def test_a_wildcard_never_claims_credentials(self):
        """`*` with `Allow-Credentials: true` is a combination browsers REJECT.

        The fetch spec forbids the wildcard in credentialed mode, so advertising both is
        strictly worse than advertising either — the browser discards the header and every
        call fails. `sis/app.py` forces credentials off against a wildcard; this holds it.
        """
        client = build_client(
            SIS_CORS_ORIGINS="*", SIS_CORS_ALLOW_CREDENTIALS="true"
        )
        response = preflight(client, "https://anything.example.com")
        self.assertEqual("*", response.headers.get("access-control-allow-origin"))
        self.assertNotEqual(
            "true", response.headers.get("access-control-allow-credentials")
        )


if __name__ == "__main__":
    unittest.main()
