"""Test fixtures for the identity service.

The dev signing key is written into the temp directory so a test run never touches
the developer's real `identity-dev-key.pem`, and never leaves one behind.

## Installing fakes

The service used to hold its gateway and its directory in module globals, which a test
replaced by calling `whatsapp.set_gateway()` before the app started. They now live on
`app.state.channels`, built once by the lifespan — so a test installs its fakes by
replacing that object *after* the client has started, which is what `install_channels`
below does.

That is a better arrangement to test against, not merely a different one: the swap is
scoped to the app object the test is holding, so a suite that forgets to undo it cannot
change what a later suite sees. The old globals could, and the ordering failures that
produced are what `identity/infrastructure/db/session.py` documents at length.
"""
import os
import tempfile

_TMPDIR = tempfile.mkdtemp(prefix="identity-tests-")
os.environ["IDENTITY_DATABASE_URL"] = f"sqlite:///{_TMPDIR}/test.db"
os.environ["IDENTITY_DEV_KEY_FILE"] = f"{_TMPDIR}/dev-key.pem"
# The administrator this suite manages accounts as. There is no shared admin key any more:
# `/v1/admin/*` is reached with a real account's access token, so the suite needs a real
# account. Seeded by the app's own lifespan on every `TestClient(app)` entry, which is also
# the mechanism under test in `test_bootstrap_admin.py`.
BOOTSTRAP_ADMIN_USER = "suite-administrator"
BOOTSTRAP_ADMIN_PASSWORD = "suite-administrator-password"
os.environ["IDENTITY_BOOTSTRAP_ADMIN_USER"] = BOOTSTRAP_ADMIN_USER
os.environ["IDENTITY_BOOTSTRAP_ADMIN_PASSWORD"] = BOOTSTRAP_ADMIN_PASSWORD
os.environ["IDENTITY_ISSUER"] = "test-identity"
os.environ["IDENTITY_AUDIENCE"] = "test-services"
# Low enough to trigger in a test without a loop of thirty requests.
os.environ["IDENTITY_MAX_FAILED_ATTEMPTS"] = "3"
# PBKDF2 at 310,000 rounds is ~100ms per hash, and this suite hashes on nearly every
# request it makes. The rounds are a property of the deployment, not of the rules being
# asserted, so the suite runs at a number that keeps it fast.
os.environ["IDENTITY_PBKDF2_ROUNDS"] = "2000"

# `identity/app.py` loads the project's `.env`, which is right for a deployment and wrong
# for a suite: `IDENTITY_SIS_BASE_URL` there makes the lifespan install a real
# `SisGuardianDirectory` over the fake these tests register, so every guardian lookup
# leaves the process and fails against a school that is not running.
#
# Blanked rather than pointed somewhere harmless, because an unset base URL is exactly
# what makes the in-memory fake the default — and a fake is what these tests assert
# against. The matching key is blanked too, since a base URL without one is now a startup
# failure by design (SIS authenticates its callers).
#
# BOTH spellings are blanked. `identity/config.py` falls back to the shared `SIS_BASE_URL`
# when the identity-specific name is unset, which is the point of that fallback — but it
# means blanking only the specific one would let the deployment's real SIS address through
# and put a live `SisGuardianDirectory` under a suite that expects the fake.
for _name in (
    "IDENTITY_SIS_BASE_URL",
    "IDENTITY_SIS_API_KEY",
    "SIS_BASE_URL",
    "SIS_API_KEY",
):
    os.environ[_name] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from identity.app import app  # noqa: E402
from identity.config import reset_settings  # noqa: E402
from identity.infrastructure.db.base import Base  # noqa: E402
from identity.infrastructure.db.session import (  # noqa: E402
    get_engine,
    new_session,
    reset_engine,
)
from identity.infrastructure.directory.fake import FakeGuardianDirectory  # noqa: E402
from identity.infrastructure.whatsapp.channels import WhatsAppChannels  # noqa: E402
from identity.infrastructure.whatsapp.gateways import (  # noqa: E402
    RecordingWhatsAppGateway,
)
from identity.domain.schools import SchoolRegistry  # noqa: E402



def use_setting(monkeypatch, name: str, value: str) -> None:
    """Set one environment variable and drop the settings cache, for this test only.

    Configuration is no longer a module global a test can `setattr`. It is a frozen
    `Settings` resolved lazily and cached, and `api/deps.py` reads it per request — so
    setting the variable and clearing the cache is what makes the next request see the
    new value. `monkeypatch` restores the variable afterwards, and `_fresh_settings`
    below drops the cache again so the restored value is what the next test resolves.
    """
    monkeypatch.setenv(name, value)
    reset_settings()



@pytest.fixture(autouse=True)
def _fresh_settings():
    """Every test starts and ends with an unresolved settings cache, and this suite's admin.

    The cache half is cheap — resolving is a few dozen `os.getenv` calls — and it removes a
    whole class of ordering failure: a test that changes a variable can no longer leave a
    frozen `Settings` behind for the next one to resolve against.

    The administrator half exists because these names are set at import, and that is not
    enough. `tests/general/test_e2e_api.py` boots its own identity server and legitimately
    assigns IDENTITY_BOOTSTRAP_ADMIN_USER for itself; running after it, every test here
    seeded one account and tried to log in as another, and the whole suite failed on
    `admin_headers` with a 401 that named nothing.

    Autouse, so it is in force before `client` starts the app — which is when the lifespan
    actually seeds the account.
    """
    previous = {
        name: os.environ.get(name)
        for name in ("IDENTITY_BOOTSTRAP_ADMIN_USER", "IDENTITY_BOOTSTRAP_ADMIN_PASSWORD")
    }
    os.environ["IDENTITY_BOOTSTRAP_ADMIN_USER"] = BOOTSTRAP_ADMIN_USER
    os.environ["IDENTITY_BOOTSTRAP_ADMIN_PASSWORD"] = BOOTSTRAP_ADMIN_PASSWORD
    reset_settings()

    yield

    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    reset_settings()


def install_channels(
    test_client,
    *,
    gateway=None,
    directory=None,
    business_number: str = "",
    verify_token: str = "",
    app_secret: str = "",
    registry: SchoolRegistry | None = None,
    by_school: dict | None = None,
) -> WhatsAppChannels:
    """Replace the running app's channels with ones built from fakes.

    Returns the installed object so a test can read `gateway.sent` off it. Scoped to the
    app the client is holding, so nothing here leaks into another suite.
    """
    channels = WhatsAppChannels(
        registry=registry or SchoolRegistry(),
        directory=directory or FakeGuardianDirectory(),
        verify_token=verify_token,
        app_secret=app_secret,
        business_number=business_number,
        default_gateway=gateway or RecordingWhatsAppGateway(),
        by_school=dict(by_school or {}),
    )
    test_client.app.state.channels = channels
    return channels


def _claim_database() -> None:
    """Point IDENTITY_DATABASE_URL back at this suite's database, and drop any engine built from another.

    Set at import above, and re-asserted here because the variable is process-global and
    this is not the only suite that wants one. pytest imports every collected module
    before running anything, so in a session covering several suites the last import
    silently owns it — and the loser fails a long way from the cause, with `no such
    table` from a server pointed at somebody else's file.

    Now that the engine and the settings are both built lazily, re-asserting actually
    works: before, the engine was captured at import and no later environment change
    could move it.
    """
    os.environ["IDENTITY_DATABASE_URL"] = f"sqlite:///{_TMPDIR}/test.db"
    os.environ["IDENTITY_DEV_KEY_FILE"] = f"{_TMPDIR}/dev-key.pem"
    reset_settings()
    reset_engine()


@pytest.fixture()
def db():
    _claim_database()
    Base.metadata.drop_all(bind=get_engine())
    Base.metadata.create_all(bind=get_engine())
    session = new_session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def admin_headers(client):
    """Authorization header for the seeded administrator.

    This replaced a module-level `{"X-Admin-Key": ...}` constant, and the change is the
    point rather than an inconvenience: it can no longer be a constant, because the
    credential is now a real account's access token that has to be obtained by logging in.
    A test that reaches an admin route therefore proves the whole chain — the account was
    seeded at startup, the password verifies, the token carries role=admin — instead of
    proving that a string matched a string.
    """
    response = client.post(
        "/v1/auth/login",
        json={"username": BOOTSTRAP_ADMIN_USER, "password": BOOTSTRAP_ADMIN_PASSWORD},
    )
    assert response.status_code == 200, (
        "the seeded administrator could not log in; "
        f"the admin routes are unreachable for this test ({response.text})"
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture()
def parent(client, admin_headers):
    """A parent account, created and bound — the normal end state of onboarding."""
    client.post(
        "/v1/admin/accounts",
        headers=admin_headers,
        json={"username": "0501234567", "password": "correct-horse-battery", "display_name": "Umm Layla"},
    )
    client.put(
        "/v1/admin/accounts/0501234567/guardian-binding",
        headers=admin_headers,
        json={"guardian_external_id": "G-1"},
    )
    return {"username": "0501234567", "password": "correct-horse-battery"}


@pytest.fixture()
def unbound_parent(client, admin_headers):
    """Created but not yet bound — the safe half-finished state of a bulk import."""
    client.post(
        "/v1/admin/accounts",
        headers=admin_headers,
        json={"username": "0509999999", "password": "correct-horse-battery"},
    )
    return {"username": "0509999999", "password": "correct-horse-battery"}
