"""Test fixtures.

Environment is set before any `records` module is imported, because engines and
issuer/audience constants are read at import time. A file-backed temporary SQLite
rather than `sqlite://` — an in-memory database gives each pooled connection its own
empty schema, which shows up as tests that pass alone and fail together.

The identity keypair is generated here rather than imported from the `identity`
package on purpose. The records facade must need nothing but a public key, and a test
suite that reached into the identity service to mint a token would hide the day that
stopped being true.
"""
import os
import tempfile

_TMPDIR = tempfile.mkdtemp(prefix="records-tests-")
os.environ["RECORDS_DATABASE_URL"] = f"sqlite:///{_TMPDIR}/test.db"
os.environ["RECORDS_LMS"] = "fake"
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

from records import auth, lms  # noqa: E402
from records.app import app  # noqa: E402
from records.db import Base, get_engine, new_session, reset_engine  # noqa: E402
from records import calendar as school_calendar  # noqa: E402
from records import guardian_directory  # noqa: E402
from records.calendar import FakeSchoolCalendar, SchoolTerm  # noqa: E402
from records.guardian_directory import (  # noqa: E402
    FakeGuardianDirectory,
    PermittedStudent,
)
from records.models import ApiKey, CourseBinding  # noqa: E402

def _claim_database() -> None:
    """Point RECORDS_DATABASE_URL back at this suite's database, and drop any engine built from another.

    Set at import above, and re-asserted here because the variable is process-global and
    this is not the only suite that wants one. pytest imports every collected module
    before running anything, so in a session covering several suites the last import
    silently owns it — and the loser fails a long way from the cause, with `no such
    table` from a server pointed at somebody else's file.

    Now that the engine is built lazily, re-asserting actually works: before, the engine
    was captured at import and no later environment change could move it.
    """
    os.environ["RECORDS_DATABASE_URL"] = f"sqlite:///{_TMPDIR}/test.db"
    reset_engine()

    # And the settings that decide WHO this service talks to, not just where its rows
    # live. `records/app.py` now loads the project's `.env`, which is right for a
    # deployment and wrong for a suite: `SIS_BASE_URL` there makes the lifespan install a
    # real `SisGuardianDirectory` over the fake these tests just registered, so every
    # guardian lookup leaves the process and fails against a school that is not running.
    #
    # Blanked rather than pointed somewhere harmless: an unset value is what makes the
    # in-memory fake the default, and a fake is what these tests are asserting against.
    for name in ("SIS_BASE_URL", "SIS_API_KEY", "IDENTITY_JWKS_URL"):
        os.environ[name] = ""


AGENT_KEY = "agentkey-fixture-0000000000000000"
ADMIN_KEY = "adminkey-fixture-0000000000000000"


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


def admin_headers() -> dict:
    return {"X-API-Key": ADMIN_KEY}


@pytest.fixture(autouse=True)
def _pin_verification_key():
    """Re-pin this suite's public key for every test.

    It is set at import as well, but `tests/test_e2e_api.py` boots a real identity
    service and legitimately points the whole process at *its* key. Whichever module
    runs second would otherwise verify tokens against the other one's key and fail with
    a signature error that looks nothing like a test-ordering problem.
    """
    previous = os.environ.get("IDENTITY_PUBLIC_KEY_PEM")
    os.environ["IDENTITY_PUBLIC_KEY_PEM"] = PUBLIC_PEM
    yield
    if previous is None:
        os.environ.pop("IDENTITY_PUBLIC_KEY_PEM", None)
    else:
        os.environ["IDENTITY_PUBLIC_KEY_PEM"] = previous


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
def seeded(db):
    """One term, one course binding, two API keys, and three guardians in the directory.

    The three guardians are the whole point: one permitted, one linked but restricted, one
    linked to nobody. Every authorisation test is a comparison between them.

    Note what is NOT seeded any more: no students, no guardians, no links, no terms as
    rows. This service holds none of that. The children come back from the guardian
    directory it asks, the term from the calendar it asks, and the only rows written here
    are the two things it genuinely owns — a key that says which system is calling, and a
    binding that names a course.
    """
    now = datetime.now(timezone.utc)

    # The calendar is the school's, asked over HTTP in a deployment and faked here. There
    # is no local `terms` table left to seed it from: that row survived only because
    # report cards keyed on its id, and both are gone.
    term = SchoolTerm(
        code="2026-T1",
        name_en="Term 1",
        name_ar="الفصل الأول",
        academic_year="2025-2026",
        starts_on=now - timedelta(days=30),
        ends_on=now + timedelta(days=60),
    )
    school_calendar.set_calendar(FakeSchoolCalendar([term]))

    # Guardian links no longer live in this service. It asks SIS on every request, so the
    # three parents below are seeded into the directory it asks rather than into a table
    # it reads — see records/guardian_directory.py for why the data moved.
    #
    #   G-1  permitted   — may be told about Layla
    #   G-2  restricted  — a real guardian of Layla's, barred by a court order. SIS filters
    #                      restricted links out, so she arrives here holding no children,
    #                      which is what makes "restricted" and "not a parent" look alike
    #                      from outside.
    #   G-3  unrelated   — a parent of somebody else entirely
    guardian_directory.set_directory(
        FakeGuardianDirectory(
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
    )

    db.add(
        CourseBinding(
            lms_course_id=9001,
            lms_idnumber="2026-T1-G7A-MATH",
            term_code=term.code,
            subject_code="MATH",
            subject_name_en="Mathematics",
            subject_name_ar="الرياضيات",
            grade_level="G7",
            section="A",
            is_published=True,
        )
    )

    for raw, scope in ((AGENT_KEY, "agent"), (ADMIN_KEY, "admin")):
        db.add(
            ApiKey(
                prefix=raw[: auth.KEY_PREFIX_LENGTH],
                key_hash=auth._hash_key(raw),
                label=f"test {scope}",
                scope=scope,
            )
        )

    db.commit()
    return {"term": term}


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
    adapter = lms.FakeLms(
        grades={
            ("S-1001", "2026-T1-"): [
                lms.SubjectGrade(
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
            ("S-1001", "2026-T1-"): [
                lms.SubjectAttendance(
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
    lms.set_adapter(adapter)
    return adapter


@pytest.fixture()
def client(seeded, fake_lms):
    with TestClient(app) as test_client:
        # The app's lifespan resets the adapter to a bare FakeLms; put the fixture
        # data back so the client and the adapter agree.
        lms.set_adapter(fake_lms)
        yield test_client
