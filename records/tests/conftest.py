"""Test fixtures.

The database URL is set before any `records` module is imported, because the engine
is built at import time. A file-backed temporary SQLite rather than `sqlite://` — an
in-memory database gives each pooled connection its own empty schema, which shows up
as tests that pass alone and fail together.
"""
import os
import tempfile

_TMPDIR = tempfile.mkdtemp(prefix="records-tests-")
os.environ["RECORDS_DATABASE_URL"] = f"sqlite:///{_TMPDIR}/test.db"
os.environ["RECORDS_LMS"] = "fake"

from datetime import datetime, timedelta, timezone  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from records import auth, lms  # noqa: E402
from records.app import app  # noqa: E402
from records.db import Base, SessionLocal, engine  # noqa: E402
from records.models import ApiKey, CourseBinding, Guardian, GuardianStudent, Student, Term  # noqa: E402
from records.schemas import AssignmentGrade, SubmissionStatus  # noqa: E402

AGENT_KEY = "agentkey-fixture-0000000000000000"
ADMIN_KEY = "adminkey-fixture-0000000000000000"


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
    """A course whose assignments include one excused item — the case that matters."""
    now = datetime.now(timezone.utc)
    grades = {
        (501, 9001): [
            AssignmentGrade(
                assignment_id="a1",
                title="Homework 1",
                status=SubmissionStatus.GRADED,
                score=90.0,
                max_score=100.0,
                percentage=90.0,
                category="homework",
                due_date=now - timedelta(days=10),
            ),
            AssignmentGrade(
                assignment_id="a2",
                title="Homework 2",
                status=SubmissionStatus.EXCUSED,
                category="homework",
                due_date=now - timedelta(days=5),
            ),
        ]
    }
    adapter = lms.FakeLms(grades=grades)
    lms.set_adapter(adapter)
    return adapter


@pytest.fixture()
def client(seeded, fake_lms):
    with TestClient(app) as test_client:
        # The app's lifespan resets the adapter to a bare FakeLms; put the fixture
        # data back so the client and the adapter agree.
        lms.set_adapter(fake_lms)
        yield test_client


def agent_headers() -> dict:
    return {"X-API-Key": AGENT_KEY}


def admin_headers() -> dict:
    return {"X-API-Key": ADMIN_KEY}
