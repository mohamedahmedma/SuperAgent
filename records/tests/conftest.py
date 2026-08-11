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
from records.db import Base, SessionLocal, engine  # noqa: E402
from records.models import ApiKey, CourseBinding, Guardian, GuardianStudent, Student, Term  # noqa: E402

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
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def seeded(db):
    """One term, one course, two students, three guardians with different rights.

    The three guardians are the whole point: one permitted, one linked but
    restricted, one linked to nobody. Every authorisation test is a comparison
    between them.
    """
    now = datetime.now(timezone.utc)
    term = Term(
        code="2026-T1",
        name_en="Term 1",
        name_ar="الفصل الأول",
        academic_year="2025-2026",
        starts_on=now - timedelta(days=30),
        ends_on=now + timedelta(days=60),
    )
    db.add(term)
    db.flush()

    student = Student(
        external_id="S-1001",
        lms_user_id=501,
        full_name_en="Layla Hassan",
        full_name_ar="ليلى حسن",
        grade_level="G7",
        section="A",
    )
    other_student = Student(
        external_id="S-2002",
        lms_user_id=502,
        full_name_en="Omar Khaled",
        full_name_ar="عمر خالد",
        grade_level="G7",
        section="A",
    )
    db.add_all([student, other_student])
    db.flush()

    permitted = Guardian(external_id="G-1", full_name_en="Permitted Parent")
    restricted = Guardian(external_id="G-2", full_name_en="Restricted Parent")
    unrelated = Guardian(external_id="G-3", full_name_en="Unrelated Parent")
    db.add_all([permitted, restricted, unrelated])
    db.flush()

    db.add_all(
        [
            GuardianStudent(guardian_id=permitted.id, student_id=student.id, can_view_records=True),
            # Linked — a real guardian on file — but barred from academic records.
            GuardianStudent(
                guardian_id=restricted.id,
                student_id=student.id,
                can_view_records=False,
                restriction_note="Court order 2025/114",
            ),
        ]
    )

    db.add(
        CourseBinding(
            lms_course_id=9001,
            lms_idnumber="2026-T1-G7A-MATH",
            term_id=term.id,
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
    return {"term": term, "student": student, "other_student": other_student}


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
