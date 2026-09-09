"""End-to-end: the real application, on a real socket, over real HTTP.

A live uvicorn server running `backend.app.create_app()` — the same object that serves
production — with real Postgres, real Redis, real Milvus, the real embedder and, where
marked, the real LLM. Requests go over TCP through the real middleware stack. Nothing
is patched.

This is the only layer that can catch the class of bug the other 1100 tests structurally
cannot: middleware ordering, CORS headers as a browser sees them, auth wired to the
wrong dependency, a route shadowed by the static mount, serialisation that works on a
dict and not on the wire, and anything that only appears when several requests are in
flight at once.

The server starts once for the module. Booting it loads bge-m3, which is the slowest
thing here by a wide margin, and doing that per test class would trade minutes for
nothing.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from tests.general.integration_support import (
    TEST_PREFIX,
    postgres_available,
    requires_llm,
    temporary_user,
    wait_until,
)

_SERVER = None
_BASE = None
_THREAD = None

# The identity service, booted alongside. Authentication left the chat backend, so an
# end-to-end test of it now genuinely spans two processes — which is the point: a token
# minted by one and verified by the other over a real socket is the only way to catch
# an issuer, audience or key mismatch between them.
_IDENTITY_SERVER = None
_IDENTITY_BASE = None
_IDENTITY_THREAD = None
_SAVED_IDENTITY_KEY = None
_ADMIN_TOKEN = None

#: The administrator identity seeds on startup, and this suite manages accounts as.
E2E_ADMIN_USER = "e2e-administrator"
E2E_ADMIN_PASSWORD = "e2e-administrator-password"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def setUpModule():
    """Boot the real app once. Skips the whole module if it cannot serve."""
    global _SERVER, _BASE, _THREAD

    if not postgres_available():
        raise unittest.SkipTest("no reachable Postgres; the app cannot start")

    import uvicorn

    from backend.app import create_app

    _start_identity()

    # A separate origin, so the CORS assertions below exercise a real cross-origin
    # request rather than a same-origin one that never triggers the middleware.
    os.environ["CORS_ALLOW_ORIGINS"] = "https://ui.example.com"
    os.environ["SERVE_FRONTEND"] = "false"

    port = _free_port()
    _BASE = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port,
                            log_level="error", access_log=False)
    _SERVER = uvicorn.Server(config)
    _THREAD = threading.Thread(target=_SERVER.run, daemon=True)
    _THREAD.start()

    def up():
        return requests.get(f"{_BASE}/health", timeout=2).status_code == 200

    if not wait_until(up, timeout=300, interval=0.25):
        raise unittest.SkipTest("the app did not become live within 300s")


def _start_identity():
    """Boot the identity service and point the backend at its signing key.

    Two details make this work regardless of what else the suite has imported:

    `IDENTITY_PUBLIC_KEY_PEM` is read by `backend.infra.identity` on every call, not at
    import, so setting it here takes effect even though the backend package was
    imported long ago.

    Issuer and audience used to be module constants identity read at import time, so
    which value it captured depended on whether `records/tests/conftest.py` had been
    collected first. They are now resolved lazily from `identity.config`, so setting
    the variables here — before identity's lifespan runs — is enough, and the ordering
    dependency is gone rather than merely worked around.
    """
    global _IDENTITY_SERVER, _IDENTITY_BASE, _IDENTITY_THREAD, _SAVED_IDENTITY_KEY

    import uvicorn

    import backend.infra.identity as backend_identity
    from identity.config import settings as identity_settings
    from identity.infrastructure.crypto.keys import signing_key_from
    from identity.infrastructure.db.session import reset_engine

    workdir = tempfile.mkdtemp(prefix="e2e-identity-")
    os.environ["IDENTITY_DATABASE_URL"] = f"sqlite:///{workdir}/identity.db"
    os.environ["IDENTITY_DEV_KEY_FILE"] = f"{workdir}/dev-key.pem"
    # There is no shared admin key any more. `/v1/admin/*` takes an administrator's own
    # access token, so this suite needs a real account — seeded by identity's lifespan
    # when the server below starts.
    # Assigned, not `setdefault`. The project's `.env` ships both of these declared and
    # EMPTY — that is what leaves a real deployment with no administrator until an operator
    # chooses a password — and `setdefault` treats a declared-but-empty variable as already
    # set, so it would quietly seed nothing and every account this suite creates would 401.
    os.environ["IDENTITY_BOOTSTRAP_ADMIN_USER"] = E2E_ADMIN_USER
    os.environ["IDENTITY_BOOTSTRAP_ADMIN_PASSWORD"] = E2E_ADMIN_PASSWORD

    # Set as environment variables rather than poked onto a module. Identity now
    # resolves its issuer and audience lazily from settings, so the values only have
    # to be in the environment before its lifespan runs — which removes the ordering
    # dependency this block used to work around rather than merely re-pointing it.
    os.environ["IDENTITY_ISSUER"] = backend_identity.ISSUER
    os.environ["IDENTITY_AUDIENCE"] = backend_identity.AUDIENCE
    # Drops the cached settings *and* any engine another suite built from them, so the
    # database URL set above is the one this server actually opens.
    reset_engine()
    _SAVED_IDENTITY_KEY = os.environ.get("IDENTITY_PUBLIC_KEY_PEM")
    os.environ["IDENTITY_PUBLIC_KEY_PEM"] = signing_key_from(
        identity_settings()
    ).public_pem

    from identity.app import app as identity_app

    port = _free_port()
    _IDENTITY_BASE = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(identity_app, host="127.0.0.1", port=port,
                            log_level="error", access_log=False)
    _IDENTITY_SERVER = uvicorn.Server(config)
    _IDENTITY_THREAD = threading.Thread(target=_IDENTITY_SERVER.run, daemon=True)
    _IDENTITY_THREAD.start()

    def up():
        return requests.get(f"{_IDENTITY_BASE}/health", timeout=2).status_code == 200

    if not wait_until(up, timeout=60, interval=0.25):
        raise unittest.SkipTest("the identity service did not become live within 60s")


def tearDownModule():
    for server, thread in ((_SERVER, _THREAD), (_IDENTITY_SERVER, _IDENTITY_THREAD)):
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=15)

    # Hand the process back the verification key it had. This module points the whole
    # interpreter at a live identity service's key, and other suites pin their own.
    if _SAVED_IDENTITY_KEY is None:
        os.environ.pop("IDENTITY_PUBLIC_KEY_PEM", None)
    else:
        os.environ["IDENTITY_PUBLIC_KEY_PEM"] = _SAVED_IDENTITY_KEY


def url(path: str) -> str:
    return f"{_BASE}{path}"


def identity_url(path: str) -> str:
    return f"{_IDENTITY_BASE}{path}"


def admin_token() -> str:
    """Sign in as the seeded administrator, once per module run.

    Cached because every account this suite creates needs it and a PBKDF2 login is not
    free. Not cached across modules: the token is bound to this run's identity server.
    """
    global _ADMIN_TOKEN
    if _ADMIN_TOKEN is None:
        response = requests.post(
            identity_url("/v1/auth/login"),
            json={"username": E2E_ADMIN_USER, "password": E2E_ADMIN_PASSWORD},
            timeout=30,
        )
        assert response.status_code == 200, (
            f"the seeded administrator could not log in: {response.text[:300]}"
        )
        _ADMIN_TOKEN = response.json()["access_token"]
    return _ADMIN_TOKEN


def register(username: str, password: str, **kwargs):
    """Create an ordinary account on the identity service.

    Named `register` still, because that is what it does from this suite's point of view,
    but it goes through `/v1/admin/accounts` now — `/v1/auth/register` is gone. Self-service
    account creation was closed when the shared admin key was removed: an administrator is
    now exactly an account with role=admin, so a public route that could mint one would be a
    public route into every family's records.

    `role="user"` is explicit because the admin route defaults to "parent", and an ordinary
    account is what every caller here wants.
    """
    return requests.post(
        identity_url("/v1/admin/accounts"),
        json={"username": username, "password": password, "role": "user"},
        headers={"Authorization": f"Bearer {admin_token()}"},
        timeout=30,
        **kwargs,
    )


def login(username: str, password: str):
    return requests.post(
        identity_url("/v1/auth/login"),
        json={"username": username, "password": password},
        timeout=30,
    )


def auth_headers(username: str, password: str) -> dict:
    """Register, sign in, and return a header the CHAT BACKEND will accept.

    The round trip is the assertion: the token is minted by one service and verified by
    another, over a socket.
    """
    register(username, password)
    token = login(username, password).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class ProbeTests(unittest.TestCase):
    """Liveness and readiness, as an orchestrator would call them."""

    def test_health_is_reachable_without_credentials(self):
        response = requests.get(url("/health"), timeout=10)
        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["status"])

    def test_health_reports_uptime(self):
        body = requests.get(url("/health"), timeout=10).json()
        self.assertIn("uptime_seconds", body)
        self.assertGreaterEqual(body["uptime_seconds"], 0)

    def test_ready_is_reachable_without_credentials(self):
        response = requests.get(url("/ready"), timeout=30)
        self.assertIn(response.status_code, (200, 503))

    def test_ready_becomes_true_once_the_embedder_is_warm(self):
        """Startup calls warm_up, so a live server should reach ready."""
        def ready():
            return requests.get(url("/ready"), timeout=10).status_code == 200

        self.assertTrue(wait_until(ready, timeout=300, interval=0.5),
                        "the server never became ready")

    def test_ready_names_each_dependency(self):
        body = requests.get(url("/ready"), timeout=30).json()
        self.assertIn("embedder", body["checks"])
        self.assertIn("database", body["checks"])

    def test_probes_respond_quickly(self):
        """A probe that blocks behind a turn causes the orchestrator to kill a healthy
        container. This is the event-loop regression, phrased as an SLO."""
        start = time.time()
        requests.get(url("/health"), timeout=10)
        self.assertLess(time.time() - start, 2.0)


class RoutingTests(unittest.TestCase):
    def test_an_unknown_path_is_a_json_404(self):
        response = requests.get(url("/no-such-route"), timeout=10)
        self.assertEqual(404, response.status_code)
        self.assertIn("application/json", response.headers.get("content-type", ""))

    def test_the_root_is_not_served_from_disk_in_backend_only_mode(self):
        self.assertEqual(404, requests.get(url("/"), timeout=10).status_code)

    def test_openapi_is_served(self):
        response = requests.get(url("/openapi.json"), timeout=15)
        self.assertEqual(200, response.status_code)
        self.assertIn("paths", response.json())

    def test_the_documented_routes_include_the_api(self):
        paths = requests.get(url("/openapi.json"), timeout=15).json()["paths"]
        for expected in ("/health", "/ready", "/chat", "/chat/stream", "/sessions"):
            self.assertIn(expected, paths)

    def test_the_backend_no_longer_serves_authentication(self):
        """Login and registration moved to the identity service.

        Asserted here rather than only in a unit test because a stale route left
        mounted would still answer over HTTP, and that is the shape of the bug that
        keeps two auth systems alive after a migration is declared finished.
        """
        paths = requests.get(url("/openapi.json"), timeout=15).json()["paths"]
        for removed in ("/auth/login", "/auth/register", "/auth/me"):
            self.assertNotIn(removed, paths)

        self.assertEqual(404, requests.post(
            url("/auth/login"), json={"username": "x", "password": "y"}, timeout=15
        ).status_code)


class CorsTests(unittest.TestCase):
    """As a browser sees them. A dict-level assertion cannot catch a discarded header."""

    def test_a_permitted_origin_is_echoed_on_a_preflight(self):
        response = requests.options(
            url("/chat"),
            headers={"Origin": "https://ui.example.com",
                     "Access-Control-Request-Method": "POST"},
            timeout=10,
        )
        self.assertEqual("https://ui.example.com",
                         response.headers.get("access-control-allow-origin"))

    def test_an_unlisted_origin_is_not_echoed(self):
        response = requests.options(
            url("/chat"),
            headers={"Origin": "https://evil.example.com",
                     "Access-Control-Request-Method": "POST"},
            timeout=10,
        )
        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_credentials_are_never_advertised_with_a_wildcard(self):
        """The original bug: browsers discard `*` in credentialed mode, so advertising
        both is strictly worse than either."""
        response = requests.options(
            url("/chat"),
            headers={"Origin": "https://ui.example.com",
                     "Access-Control-Request-Method": "POST"},
            timeout=10,
        )
        if response.headers.get("access-control-allow-origin") == "*":
            self.assertNotEqual("true",
                                response.headers.get("access-control-allow-credentials"))

    def test_etag_is_exposed_so_a_remote_ui_can_revalidate(self):
        response = requests.get(
            url("/documents"), headers={"Origin": "https://ui.example.com"}, timeout=15
        )
        exposed = response.headers.get("access-control-expose-headers", "")
        self.assertIn("ETag", [item.strip() for item in exposed.split(",")])


class AuthTests(unittest.TestCase):
    """Real registration, real password hashing, real RS256, real 401s — across two
    services. The token is minted by identity and verified by the backend."""

    def post_json(self, path, payload, **kwargs):
        return requests.post(url(path), json=payload, timeout=30, **kwargs)

    def test_a_protected_route_refuses_an_anonymous_caller(self):
        self.assertEqual(401, requests.get(url("/documents"), timeout=15).status_code)

    def test_a_protected_route_refuses_a_forged_token(self):
        response = requests.get(
            url("/documents"),
            headers={"Authorization": "Bearer not.a.real.token"},
            timeout=15,
        )
        self.assertEqual(401, response.status_code)

    def test_a_protected_route_refuses_a_malformed_header(self):
        for header in ("Bearer", "Basic abc", "", "Bearer  "):
            with self.subTest(header=header):
                response = requests.get(
                    url("/documents"), headers={"Authorization": header}, timeout=15
                )
                self.assertEqual(401, response.status_code)

    def test_register_then_login_then_use_the_token(self):
        """The cross-service round trip, which is now the real integration risk.

        A mismatched issuer, audience or signing key between the two services shows up
        here and nowhere else — each side's own tests pass happily while the pair is
        unusable.
        """
        with temporary_user() as (username, password):
            registered = register(username, password)
            self.assertIn(registered.status_code, (200, 201),
                          f"register failed: {registered.text[:300]}")

            signed_in = login(username, password)
            self.assertEqual(200, signed_in.status_code, signed_in.text[:300])
            token = signed_in.json().get("access_token")
            self.assertTrue(token)

            response = requests.get(
                url("/sessions"), headers={"Authorization": f"Bearer {token}"}, timeout=30
            )
            self.assertEqual(200, response.status_code, response.text[:300])

    def test_a_fresh_account_carries_no_guardian_binding(self):
        """A newly created account must never carry a guardian binding.

        Creation and binding are two separate admin calls precisely so that an
        interrupted bulk import leaves accounts that can log in and read nothing.
        """
        with temporary_user() as (username, password):
            register(username, password)
            body = login(username, password).json()
            self.assertIsNone(body["guardian_id"])
            self.assertEqual("user", body["role"])

    def test_a_wrong_password_is_rejected(self):
        with temporary_user() as (username, password):
            register(username, password)
            self.assertIn(login(username, "definitely-not-it").status_code, (400, 401))

    def test_an_unknown_user_cannot_log_in(self):
        self.assertIn(login(f"{TEST_PREFIX}-ghost", "whatever").status_code, (400, 401))

    def test_registering_the_same_username_twice_is_refused(self):
        with temporary_user() as (username, password):
            self.assertIn(register(username, password).status_code, (200, 201))
            self.assertGreaterEqual(register(username, password).status_code, 400)


class ValidationTests(unittest.TestCase):
    """Malformed input must produce a 4xx, never a 500 and never a stack trace."""

    def test_a_missing_body_is_a_422_not_a_500(self):
        response = requests.post(identity_url("/v1/auth/login"), json={}, timeout=15)
        self.assertLess(response.status_code, 500)

    def test_malformed_json_is_rejected_cleanly(self):
        response = requests.post(
            identity_url("/v1/auth/login"),
            data="{not json",
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        self.assertIn(response.status_code, (400, 422))

    def test_a_wrong_method_is_a_405(self):
        self.assertEqual(405, requests.delete(url("/health"), timeout=10).status_code)

    def test_hostile_strings_do_not_produce_a_500(self):
        hostile = [
            "'; DROP TABLE users; --",
            "<script>alert(1)</script>",
            "../../../../etc/passwd",
            "\x00\x01\x02",
            "{{7*7}}",
            "A" * 5000,
        ]
        for value in hostile:
            with self.subTest(value=value[:30]):
                response = requests.post(
                    identity_url("/v1/auth/login"),
                    json={"username": value, "password": value},
                    timeout=30,
                )
                self.assertLess(response.status_code, 500,
                                f"a 5xx on hostile input: {response.text[:200]}")

    def test_the_accounts_table_survived_that(self):
        self.assertEqual(200, requests.get(identity_url("/health"), timeout=10).status_code)
        self.assertIn(login("x", "y").status_code, (400, 401))


class ConcurrencyTests(unittest.TestCase):
    """The property the async fixes exist for: one user's request must not stall
    everyone else's."""

    def test_many_parallel_probes_all_succeed(self):
        def probe(_):
            return requests.get(url("/health"), timeout=30).status_code

        with ThreadPoolExecutor(max_workers=32) as pool:
            codes = list(pool.map(probe, range(64)))
        self.assertEqual({200}, set(codes))

    def test_probes_stay_responsive_while_readiness_checks_run(self):
        """/ready touches the database; /health must not queue behind it."""
        stop = threading.Event()
        latencies = []

        def hammer_ready():
            while not stop.is_set():
                try:
                    requests.get(url("/ready"), timeout=30)
                except Exception:
                    pass

        def measure_health():
            for _ in range(20):
                start = time.time()
                requests.get(url("/health"), timeout=30)
                latencies.append(time.time() - start)

        loaders = [threading.Thread(target=hammer_ready, daemon=True) for _ in range(6)]
        for loader in loaders:
            loader.start()
        try:
            measure_health()
        finally:
            stop.set()
            for loader in loaders:
                loader.join(timeout=10)

        worst = max(latencies)
        self.assertLess(worst, 5.0, f"/health blocked for {worst:.2f}s under load")

    def test_parallel_authenticated_reads_all_succeed(self):
        with temporary_user() as (username, password):
            headers = auth_headers(username, password)

            def read(_):
                return requests.get(url("/sessions"), headers=headers, timeout=60).status_code

            with ThreadPoolExecutor(max_workers=16) as pool:
                codes = list(pool.map(read, range(32)))
            self.assertEqual({200}, set(codes))


class IdentityBackendIntegrationTests(unittest.TestCase):
    """The two public services talking to each other, wired the way production wires them.

    Every other test in this module hands the backend identity's public key directly, in
    `IDENTITY_PUBLIC_KEY_PEM`. That is a legitimate shortcut for testing everything else,
    and it silently skips the one thing that actually connects these two deployments: the
    backend FETCHING `IDENTITY_JWKS_URL` over HTTP from identity, matching the token's `kid`
    against what comes back, and caching it.

    In production that fetch is the whole seam. `auth.aurexis.cc` publishes the key,
    `api.aurexis.cc` goes and gets it, and when that fails every authenticated request in
    the estate answers 503 "Authentication is temporarily unavailable" — with nothing in the
    message naming a URL. So this class removes the pinned PEM entirely. If the JWKS fetch
    does not work, nothing below can pass.

    The journey is the one an operator actually performs: the seeded administrator signs in
    to identity, creates an account there, and that account then uses the CHAT BACKEND — a
    token minted by one service, verified by another, over a socket, with the key travelling
    between them over HTTP.
    """

    @classmethod
    def setUpClass(cls):
        import backend.infra.identity as backend_identity
        import schoolauth

        cls._saved_pem = os.environ.get("IDENTITY_PUBLIC_KEY_PEM")
        cls._saved_jwks = backend_identity.JWKS_URL

        # No pinned key. The JWKS fetch is now the only way a signature can be checked.
        os.environ.pop("IDENTITY_PUBLIC_KEY_PEM", None)
        backend_identity.JWKS_URL = identity_url("/.well-known/jwks.json")
        schoolauth.reset_key_cache()

    @classmethod
    def tearDownClass(cls):
        import backend.infra.identity as backend_identity
        import schoolauth

        backend_identity.JWKS_URL = cls._saved_jwks
        if cls._saved_pem is None:
            os.environ.pop("IDENTITY_PUBLIC_KEY_PEM", None)
        else:
            os.environ["IDENTITY_PUBLIC_KEY_PEM"] = cls._saved_pem
        schoolauth.reset_key_cache()

    # -- the key travelling between the services ----------------------------

    def test_identity_publishes_a_jwks_the_backend_can_use(self):
        """The document itself, as the backend's verifier requires it to be."""
        response = requests.get(identity_url("/.well-known/jwks.json"), timeout=15)

        self.assertEqual(200, response.status_code, response.text[:300])
        keys = response.json().get("keys")
        self.assertTrue(keys, "no keys published")
        self.assertTrue(keys[0].get("kid"), "a key with no kid cannot be matched to a token")
        self.assertNotIn("d", keys[0], "the PRIVATE exponent was published")

    # -- the happy path -----------------------------------------------------

    def test_an_admin_created_account_can_use_the_backend(self):
        """The whole point, end to end, with no shared secret anywhere in it.

        admin signs in -> admin creates an account -> that account signs in -> the backend
        accepts its token after fetching identity's key over HTTP.
        """
        with temporary_user() as (username, password):
            created = register(username, password)
            self.assertEqual(201, created.status_code, created.text[:300])

            signed_in = login(username, password)
            self.assertEqual(200, signed_in.status_code, signed_in.text[:300])
            token = signed_in.json()["access_token"]

            response = requests.get(
                url("/sessions"), headers={"Authorization": f"Bearer {token}"}, timeout=30
            )

        self.assertEqual(
            200, response.status_code,
            f"the backend rejected a token identity had just minted: {response.text[:300]}",
        )

    def test_the_seeded_administrator_reaches_the_backends_admin_routes(self):
        """The role claim crossing the wire.

        `backend/infra/auth.py` require_admin reads `role` off the signed token, so an
        administrator minted by identity is an administrator to the backend with nothing
        configured on the backend side at all.
        """
        response = requests.get(
            url("/documents"),
            headers={"Authorization": f"Bearer {admin_token()}"},
            timeout=60,
        )

        self.assertNotIn(
            response.status_code, (401, 403),
            f"the admin role did not survive the trip between services: {response.text[:300]}",
        )

    # -- the paths that must fail -------------------------------------------

    def test_an_ordinary_account_is_refused_by_the_backends_admin_routes(self):
        """403, not 401. The credential is genuine; the role is not."""
        with temporary_user() as (username, password):
            register(username, password)
            token = login(username, password).json()["access_token"]

            response = requests.get(
                url("/documents"),
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            )

        self.assertEqual(403, response.status_code, response.text[:300])

    def test_no_credential_is_refused(self):
        response = requests.get(url("/sessions"), timeout=30)
        self.assertEqual(401, response.status_code)

    def test_a_garbage_token_is_refused(self):
        response = requests.get(
            url("/sessions"),
            headers={"Authorization": "Bearer not.a.real.token"},
            timeout=30,
        )
        self.assertEqual(401, response.status_code)

    def test_a_token_signed_by_a_foreign_key_is_refused(self):
        """The attack the JWKS fetch exists to stop.

        Correct issuer, correct audience, correct claims — and a key identity never
        published. If this passes, the backend is reading claims without checking who
        signed them, and anyone can mint themselves an administrator.
        """
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from jose import jwt

        import backend.infra.identity as backend_identity

        forged_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = forged_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        now = int(time.time())
        forged = jwt.encode(
            {
                "iss": backend_identity.ISSUER,
                "aud": backend_identity.AUDIENCE,
                "sub": "not-a-real-person",
                "role": "admin",
                "iat": now,
                "exp": now + 1800,
            },
            pem,
            algorithm="RS256",
        )

        response = requests.get(
            url("/documents"),
            headers={"Authorization": f"Bearer {forged}"},
            timeout=30,
        )

        self.assertEqual(401, response.status_code, response.text[:300])

    def test_a_token_for_another_audience_is_refused(self):
        """Without the audience check a token minted for one service works against another.

        Same key, same issuer, different intended reader — so this is not a formality.
        """
        from jose import jwt

        import backend.infra.identity as backend_identity
        from identity.config import settings as identity_settings
        from identity.infrastructure.crypto.keys import signing_key_from

        key = signing_key_from(identity_settings())
        now = int(time.time())
        wrong_audience = jwt.encode(
            {
                "iss": backend_identity.ISSUER,
                "aud": "somebody-elses-service",
                "sub": "someone",
                "role": "user",
                "iat": now,
                "exp": now + 1800,
            },
            key.private_pem,
            algorithm=key.algorithm,
            headers={"kid": key.kid},
        )

        response = requests.get(
            url("/sessions"),
            headers={"Authorization": f"Bearer {wrong_audience}"},
            timeout=30,
        )

        self.assertEqual(401, response.status_code, response.text[:300])

    def test_an_expired_token_is_refused(self):
        """Signed by the real key, and past its expiry."""
        from jose import jwt

        import backend.infra.identity as backend_identity
        from identity.config import settings as identity_settings
        from identity.infrastructure.crypto.keys import signing_key_from

        key = signing_key_from(identity_settings())
        now = int(time.time())
        expired = jwt.encode(
            {
                "iss": backend_identity.ISSUER,
                "aud": backend_identity.AUDIENCE,
                "sub": "someone",
                "role": "user",
                "iat": now - 7200,
                "exp": now - 300,
            },
            key.private_pem,
            algorithm=key.algorithm,
            headers={"kid": key.kid},
        )

        response = requests.get(
            url("/sessions"),
            headers={"Authorization": f"Bearer {expired}"},
            timeout=30,
        )

        self.assertEqual(401, response.status_code, response.text[:300])

    # -- what happens to a session after the account changes ----------------

    def test_a_deleted_account_cannot_get_a_new_access_token(self):
        """Deletion on identity has to end the session on the backend too.

        It does so by revoking the refresh token: the access token already in a browser
        stays valid until it expires — the price of verifying offline — but it cannot be
        renewed, so the session is bounded rather than indefinite.
        """
        with temporary_user() as (username, password):
            register(username, password)
            tokens = login(username, password).json()

            deleted = requests.delete(
                identity_url(f"/v1/admin/accounts/{username}"),
                headers={"Authorization": f"Bearer {admin_token()}"},
                timeout=30,
            )
            self.assertEqual(204, deleted.status_code, deleted.text[:300])

            refreshed = requests.post(
                identity_url("/v1/auth/refresh"),
                json={"refresh_token": tokens["refresh_token"]},
                timeout=30,
            )

        self.assertEqual(401, refreshed.status_code, refreshed.text[:300])

    def test_a_deactivated_account_cannot_log_in_or_refresh(self):
        with temporary_user() as (username, password):
            register(username, password)
            tokens = login(username, password).json()

            requests.patch(
                identity_url(f"/v1/admin/accounts/{username}"),
                headers={"Authorization": f"Bearer {admin_token()}"},
                json={"is_active": False},
                timeout=30,
            )

            again = login(username, password)
            refreshed = requests.post(
                identity_url("/v1/auth/refresh"),
                json={"refresh_token": tokens["refresh_token"]},
                timeout=30,
            )

        self.assertEqual(401, again.status_code, again.text[:300])
        self.assertEqual(401, refreshed.status_code, refreshed.text[:300])

    # -- what happens when identity is not there ----------------------------

    def test_a_cached_key_survives_an_identity_outage(self):
        """A stale key is still a valid key. Identity restarting must not stop chat.

        Simulated the way an outage actually presents: the URL is UNCHANGED and the service
        behind it stops answering. That distinction is the whole test. Pointing the backend
        at a different, dead URL is not an outage — the cache is keyed by URL
        (`schoolauth/verification.py` says so, "so a process verifying against two identity
        services cannot serve one's keys for the other's tokens"), so a new URL is a COLD
        cache and correctly fails closed. Writing it that way asserts the opposite of what
        it appears to.

        So: warm the cache through a real request, age the entry past its TTL to force a
        refresh, and make the network fail. `_fetch_and_store` then falls back to the
        cached document, which is what keeps every service verifying through an identity
        restart.
        """
        from unittest.mock import patch

        import schoolauth.verification as verification

        import backend.infra.identity as backend_identity

        with temporary_user() as (username, password):
            register(username, password)
            token = login(username, password).json()["access_token"]

            warm = requests.get(
                url("/sessions"), headers={"Authorization": f"Bearer {token}"}, timeout=30
            )
            self.assertEqual(200, warm.status_code, warm.text[:300])

            jwks_url = backend_identity.JWKS_URL
            document, _fetched_at = verification._cached_jwks[jwks_url]
            # Aged past any TTL, so the next verification MUST attempt a refresh rather
            # than sailing through on a fresh entry and proving nothing.
            verification._cached_jwks[jwks_url] = (document, 0.0)

            with patch.object(
                verification.urllib.request,
                "urlopen",
                side_effect=OSError("identity is unreachable"),
            ):
                response = requests.get(
                    url("/sessions"),
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                )

        self.assertEqual(
            200, response.status_code,
            f"an identity outage took the backend down with it: {response.text[:300]}",
        )

    def test_it_fails_closed_when_it_has_never_reached_identity(self):
        """No key, no cache, no verification — and therefore no access.

        The refusal must not be a 200. Whether it presents as 401 or 503 is a judgement the
        verifier makes; letting the request through is the only wrong answer.
        """
        import backend.infra.identity as backend_identity
        import schoolauth

        with temporary_user() as (username, password):
            register(username, password)
            token = login(username, password).json()["access_token"]

            saved = backend_identity.JWKS_URL
            backend_identity.JWKS_URL = f"http://127.0.0.1:{_free_port()}/nope.json"
            schoolauth.reset_key_cache()
            try:
                response = requests.get(
                    url("/sessions"),
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                )
            finally:
                backend_identity.JWKS_URL = saved
                schoolauth.reset_key_cache()

        self.assertIn(
            response.status_code, (401, 503),
            f"unverifiable identity was not refused: {response.text[:300]}",
        )


class LiveChatTests(unittest.TestCase):
    """The whole stack including a real LLM call. Skipped when the provider is not
    configured or RUN_LLM_TESTS is off."""

    @classmethod
    def setUpClass(cls):
        cls._exit = None
        cls.username = None

    def authenticated(self):
        context = temporary_user()
        username, password = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return auth_headers(username, password)

    @requires_llm
    def test_a_question_gets_an_answer(self):
        headers = self.authenticated()
        response = requests.post(
            url("/chat"),
            json={"message": "what are the school fees?",
                  "session_id": f"{TEST_PREFIX}-chat-1"},
            headers=headers,
            timeout=180,
        )
        self.assertEqual(200, response.status_code, response.text[:400])
        body = response.json()
        self.assertIn("response", body)
        self.assertTrue(str(body["response"]).strip(), "the assistant returned nothing")

    @requires_llm
    def test_an_arabic_question_gets_an_answer(self):
        headers = self.authenticated()
        response = requests.post(
            url("/chat"),
            json={"message": "ما هي الرسوم الدراسية؟",
                  "session_id": f"{TEST_PREFIX}-chat-ar"},
            headers=headers,
            timeout=180,
        )
        self.assertEqual(200, response.status_code, response.text[:400])
        self.assertTrue(str(response.json().get("response", "")).strip())

    @requires_llm
    def test_the_streaming_endpoint_emits_server_sent_events(self):
        headers = self.authenticated()
        response = requests.post(
            url("/chat/stream"),
            json={"message": "when does the second term start?",
                  "session_id": f"{TEST_PREFIX}-stream-1"},
            headers=headers,
            timeout=180,
            stream=True,
        )
        self.assertEqual(200, response.status_code)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

        events = []
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
            if len(events) >= 3:
                break
        response.close()
        self.assertTrue(events, "the stream produced no events")
        self.assertTrue(all("type" in event for event in events))

    @requires_llm
    def test_streaming_never_sends_an_unauthenticated_stream(self):
        response = requests.post(
            url("/chat/stream"),
            json={"message": "hello", "session_id": f"{TEST_PREFIX}-stream-anon"},
            timeout=30,
        )
        self.assertEqual(401, response.status_code)

    @requires_llm
    def test_two_users_are_served_concurrently(self):
        """The point of the event-loop work, end to end: two turns at once must not
        serialise into twice one turn's latency."""
        headers = [self.authenticated() for _ in range(2)]

        def ask(index):
            start = time.time()
            response = requests.post(
                url("/chat"),
                json={"message": "what are the admission requirements?",
                      "session_id": f"{TEST_PREFIX}-conc-{index}"},
                headers=headers[index],
                timeout=300,
            )
            return response.status_code, time.time() - start

        start = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(ask, range(2)))
        wall = time.time() - start

        codes = [code for code, _ in outcomes]

        # A 429 is the provider's rate limit, not this application serialising, and it
        # is the honest answer to "how many users can we serve": once the event loop is
        # no longer the bottleneck, the upstream quota is. Skipping rather than failing
        # keeps the test measuring our code and not somebody else's billing tier.
        if 429 in codes:
            self.skipTest(
                "the LLM provider rate-limited concurrent turns (429) — raise the quota "
                "to measure application-side concurrency here"
            )

        for code in codes:
            self.assertEqual(200, code)
        slowest = max(duration for _, duration in outcomes)
        self.assertLess(
            wall, slowest * 1.8,
            f"two turns took {wall:.1f}s but the slowest alone was {slowest:.1f}s — "
            "they appear to have serialised",
        )


if __name__ == "__main__":
    unittest.main()
