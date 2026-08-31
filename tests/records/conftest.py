"""Test fixtures.

Environment is set before any `records` module is imported, because issuer and audience
are read at import time.

**There is no database to set up.** The service holds none: every fact it serves is asked
for at request time, so the fixtures below register fakes for the three things it asks —
the guardian directory, the school calendar, and the marks adapter — and that is the whole
of the arrangement. What used to be here, a temporary SQLite file and a schema created per
test, went with the tables.

The identity keypair is generated here rather than imported from the `identity` package on
purpose. The records facade must need nothing but a public key, and a test suite that
reached into the identity service to mint a token would hide the day that stopped being
true.
"""
import os

os.environ["RECORDS_LMS"] = "fake"
# The service key this suite presents. Read per request by `records.auth`, so setting it
# here is enough and no fixture has to install it.
os.environ["RECORDS_API_KEY"] = "agentkey-fixture-0000000000000000"
os.environ["IDENTITY_ISSUER"] = "test-identity"
os.environ["IDENTITY_AUDIENCE"] = "test-services"

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PRIVATE_PEM = _PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode("utf-8")
PUBLIC_PEM = (
    _PRIVATE_KEY.public_key()
    .public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode("utf-8")
)
os.environ["IDENTITY_PUBLIC_KEY_PEM"] = PUBLIC_PEM

from datetime import datetime, timedelta, timezone  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from jose import jwt  # noqa: E402

from records.adapters.fake.calendar import FakeSchoolCalendar  # noqa: E402
from records.adapters.fake.directory import FakeGuardianDirectory  # noqa: E402
from records.adapters.fake.lms import FakeLms  # noqa: E402
from records.app import app  # noqa: E402
from records.config import reset_settings  # noqa: E402
from records.domain.marks import SubjectAttendance, SubjectGrade  # noqa: E402
from records.domain.people import PermittedStudent  # noqa: E402
from records.domain.terms import SchoolTerm  # noqa: E402

def _isolate_from_the_environment() -> None:
    """Blank the settings that decide who this service talks to.

    `records/app.py` loads the project's `.env`, which is right for a deployment and wrong
    for a suite: `SIS_BASE_URL` there makes the lifespan install a real
    `SisGuardianDirectory` over the fake these tests register, so every guardian lookup
    leaves the process and fails against a school that is not running.

    Blanked rather than pointed somewhere harmless: an unset value is what makes the
    in-memory fake the default, and a fake is what these tests assert against.

    There is no database URL to re-claim any more. That line existed because the variable
    was process-global and several suites wanted it; the service holds no rows now, so
    there is nothing for another suite to take.
    """
    for name in ("SIS_BASE_URL", "SIS_API_KEY", "IDENTITY_JWKS_URL"):
        os.environ[name] = ""


#: The one credential this service has. There is no admin scope any more: the routes that
#: needed one minted keys and read an audit, and both moved out.
AGENT_KEY = "agentkey-fixture-0000000000000000"


def mint_token(
    guardian_id: str | None,
    *,
    issuer: str = "test-identity",
    audience: str = "test-services",
    expired: bool = False,
    key_pem: str | None = None,
) -> str:
    """Mint an identity token the way the identity service would.

    The keyword arguments exist so tests can produce the *wrong* token deliberately —
    wrong audience, wrong signer, expired — which is the only way to prove the
    verifier actually checks those things rather than just the signature.
    """
    now = datetime.now(timezone.utc)
    exp = now - timedelta(minutes=5) if expired else now + timedelta(minutes=30)
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": "parent-user",
        "role": "parent",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    if guardian_id:
        claims["guardian_id"] = guardian_id
    return jwt.encode(claims, key_pem or PRIVATE_PEM, algorithm="RS256")


def agent_headers(guardian_id: str | None = None, token: str | None = None) -> dict:
    """Both credentials: the system key and the parent's signed identity."""
    headers = {"X-API-Key": AGENT_KEY}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    elif guardian_id is not None:
        headers["Authorization"] = f"Bearer {mint_token(guardian_id)}"
    return headers


#: The environment this suite is written against, in one place.
#:
#: The same names are set at import, and that used to be enough. It is not, because another
#: suite in the same run legitimately changes them — `tests/general/test_parent_journey.py`
#: boots a real sis, records and identity and points the process at all three — so what a
#: records test actually runs against depended on collection order.
_SUITE_ENV = {
    # Fixtures, not a real school. With `sis` here instead, the app's lifespan raises
    # "RECORDS_LMS=sis requires SIS_BASE_URL" and every test errors in setup.
    "RECORDS_LMS": "fake",
    # Read per request by `records.auth`, and the credential every request here presents.
    "RECORDS_API_KEY": "agentkey-fixture-0000000000000000",
    # What this suite's tokens are minted with, and so what it must verify against.
    "IDENTITY_ISSUER": "test-identity",
    "IDENTITY_AUDIENCE": "test-services",
    # Blanked, not overridden. An unset base URL is what selects the in-memory fakes these
    # tests assert against; a real one installs adapters that leave the process.
    "SIS_BASE_URL": "",
    "SIS_API_KEY": "",
    # A JWKS URL would send verification to the network. The pinned PEM wins over it inside
    # `schoolauth`, but only while the PEM is set, so this is blanked rather than trusted
    # to be shadowed.
    "IDENTITY_JWKS_URL": "",
}


@pytest.fixture(autouse=True)
def _pin_verification():
    """Re-pin everything this suite verifies with, for every test.

    The key was already re-pinned here, for a reason the docstring gave: `test_e2e_api.py`
    boots a real identity service and legitimately points the whole process at *its* key.
    The same argument applies to every other name in `_SUITE_ENV`, and to the cached
    settings object that holds them — and until it was extended to those, this suite failed
    as twenty-five tests whenever it ran after `tests/general/`.

    The mechanism, because the symptom names none of it: `records.config.settings()` is
    `lru_cache`d. These values are set at import, so the first resolution is correct. Then
    another suite changes the environment and drops that cache — and the next resolution
    picks up `IDENTITY_ISSUER` from the project's `.env`, which is unset, so it defaults to
    `school-identity`. The tokens here still say `test-identity`. Every request comes back
    401 with `guardian_id: ""` and reason `not_authorized`, which reads exactly like a
    broken signature and mentions no issuer at all.

    So the cache is dropped on the way IN as well as on the way out: leaving it holding a
    previous suite's resolution is the whole failure.
    """
    previous = {name: os.environ.get(name) for name in _SUITE_ENV}
    previous["IDENTITY_PUBLIC_KEY_PEM"] = os.environ.get("IDENTITY_PUBLIC_KEY_PEM")

    os.environ.update(_SUITE_ENV)
    os.environ["IDENTITY_PUBLIC_KEY_PEM"] = PUBLIC_PEM
    reset_settings()

    yield

    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    reset_settings()


def install_adapters(test_client, *, calendar=None, directory=None, lms=None):
    """Point the running app at these fakes.

    The three seams used to be module-level slots a fixture could assign before the app
    started. They are fields on `app.state` now, built by the lifespan, so a test installs
    its own **after** the client has started — which is strictly better to test against:
    the swap is scoped to the app object this test holds, so a suite that forgets to undo
    it cannot change what a later suite sees.
    """
    state = test_client.app.state
    if calendar is not None:
        state.calendar = calendar
    if directory is not None:
        state.directory = directory
    if lms is not None:
        state.lms = lms


@pytest.fixture()
def seeded():
    """The three things this service asks for, answered by fakes.

    Three guardians, and they are the whole point: one permitted, one linked but
    restricted, one linked to nobody. Every authorisation test is a comparison between
    them.

    Nothing is written anywhere, because there is nowhere to write. This service holds no
    tables: the children come back from the guardian directory it asks, the term from the
    calendar it asks, and the marks from the adapter. The `db` fixture that used to create
    a schema for each test went with them.
    """
    _isolate_from_the_environment()
    now = datetime.now(timezone.utc)

    term = SchoolTerm(
        code="2026-T1",
        name_en="Term 1",
        name_ar="الفصل الأول",
        academic_year="2025-2026",
        starts_on=now - timedelta(days=30),
        ends_on=now + timedelta(days=60),
    )
    calendar = FakeSchoolCalendar([term])

    #   G-1  permitted   — may be told about Layla
    #   G-2  restricted  — a real guardian of Layla's, barred by a court order. SIS filters
    #                      restricted links out, so she arrives here holding no children,
    #                      which is what makes "restricted" and "not a parent" look alike
    #                      from outside.
    #   G-3  unrelated   — a parent of somebody else entirely
    directory = FakeGuardianDirectory(
        {
            "G-1": [
                PermittedStudent(
                    student_id="S-1001",
                    full_name_en="Layla Hassan",
                    full_name_ar="ليلى حسن",
                    grade_level="G7",
                    section="A",
                )
            ],
            "G-2": [],
            "G-3": [],
        }
    )
    return {"term": term, "calendar": calendar, "directory": directory}


@pytest.fixture()
def fake_lms():
    """A subject where the official total and the academic figure DIFFER.

    Keyed by (student reference, term prefix) — the shape the adapter protocol now
    takes. Using the school's student number rather than an LMS user id is the point:
    nothing outside the system of record should have to know a Moodle id.

    65% against 80% is the measured real case: a graded attendance item drags the
    official course total below the mark the child earned on assessments. A fixture
    where the two coincided would let a bug that returns one for the other pass
    unnoticed.
    """
    adapter = FakeLms(
        grades={
            ("S-1001", "2026-T1"): [
                SubjectGrade(
                    course_ref="2026-T1-G7A-MATH",
                    subject_name="Mathematics",
                    percentage=65.0,
                    academic_percentage=80.0,
                    graded_count=3,
                    excluded_count=1,
                    pending_count=1,
                    is_complete=False,
                )
            ]
        },
        attendance={
            ("S-1001", "2026-T1"): [
                SubjectAttendance(
                    course_ref="2026-T1-G7A-MATH",
                    subject_name="Mathematics",
                    percentage=87.5,
                    taken_sessions=4,
                    by_status=(
                        {"acronym": "P", "description": "Present", "count": 3},
                        {"acronym": "L", "description": "Late", "count": 1},
                    ),
                    points=7.0,
                    max_points=8.0,
                )
            ]
        },
    )
    return adapter


@pytest.fixture()
def client(seeded, fake_lms):
    with TestClient(app) as test_client:
        # After the lifespan, which built its own bare fakes from the (blank) environment.
        install_adapters(
            test_client,
            calendar=seeded["calendar"],
            directory=seeded["directory"],
            lms=fake_lms,
        )
        yield test_client
