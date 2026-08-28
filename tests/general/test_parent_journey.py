"""The parent's whole journey, across three live services, with nothing mocked between them.

`sis/` and `records/` are served by real uvicorn workers on loopback ports; `identity/` is
driven through its own `TestClient`. The only fake is Meta, which records a message instead
of sending it — exactly what a school without Cloud API credentials gets.

**This is the only file in the repo that imports more than one service, and it is a test.**
The services are independent projects with their own databases and no imports between them;
that isolation is the architecture, and the one thing it cannot check is whether they agree
on the wire. Field names, error codes, the E.164 spelling of a number, whether a term is
named `2026-T1` or `2026-T1-` — each is a place where four separately-correct services are
collectively wrong, and each has already broken here at least once.

Every failure this file has caught looked like success: a `200` carrying an empty list,
which a parent reads as "the school has recorded nothing about my child" and which no
status code anywhere reports as wrong. That is why the assertions are about *content* and
not about status codes.

Run it on its own — it sets service environment variables at import, as each service's own
conftest does:

    pytest tests/test_parent_journey.py -q
"""
import hashlib
import hmac
import itertools
import json
import os
import socket
import tempfile
import threading
import time
from datetime import date

import pytest

# --------------------------------------------------------------------------
# What this file needs the three services configured with — DECLARED here and
# APPLIED in a fixture, never at import.
#
# It used to be applied here, because each service read its database URL once at
# import into a module-global engine, so a fixture would have been too late. That is
# no longer true: all three build their engine on first use.
#
# Applying it at import was actively harmful. pytest imports every collected module
# before running anything, so in a session that also collects `records/tests` or
# `identity/tests`, these values reached those suites too — and configured them.
# `records/` then started against a real SIS over HTTP instead of its fake and failed
# with `not_configured` where it expected `lms_unavailable`; three separate variables
# caused three separate rounds of that before the pattern was worth naming.
#
# A test file may configure the process while its own tests run. It may not configure
# it for everybody else's.
# --------------------------------------------------------------------------

_TMP = tempfile.mkdtemp(prefix="parent-journey-")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


SIS_PORT = _free_port()
RECORDS_PORT = _free_port()

APP_SECRET = "journey-app-secret"
VERIFY_TOKEN = "journey-verify-token"
SCHOOL_NUMBER = "+201288339613"

#: The registrar's own SIS credential — uploads, commits, custody changes.
#:
#: Separate from `SIS_API_KEY` below, and that separation is now load-bearing rather than
#: tidy: `Scope.permits` is exact equality, so the reader key `records/` holds is *refused*
#: by every route this one reaches. The journey exercises both, which is the only way to
#: notice if that stopped being true.
SIS_REGISTRAR_KEY = "journey-registrar"

ENVIRONMENT = {
    "SIS_DATABASE_URL": f"sqlite:///{_TMP}/sis.db",
    "SIS_DEFAULT_COUNTRY_CODE": "+20",
    # No RECORDS_DATABASE_URL: the facade holds no database at all. Its one credential is
    # this secret, read per request from the environment by both sides — the backend sends
    # it, records compares against it.
    "RECORDS_API_KEY": "journey-records-agent",
    "RECORDS_LMS": "sis",
    "SIS_BASE_URL": f"http://127.0.0.1:{SIS_PORT}",
    "SIS_API_KEY": "journey-reader",
    "IDENTITY_DATABASE_URL": f"sqlite:///{_TMP}/identity.db",
    "IDENTITY_DEV_KEY_FILE": f"{_TMP}/dev-key.pem",
    "IDENTITY_ISSUER": "school-identity",
    "IDENTITY_AUDIENCE": "school-services",
    # Blanked, not pointed at the SIS below. The `gateway` fixture installs its own
    # directory *after* identity's lifespan has run, and a base URL here would make the
    # lifespan build a second one first — and, since a base URL without a key is now a
    # startup failure by design, take the whole suite down before any of that.
    "IDENTITY_SIS_BASE_URL": "",
    "IDENTITY_SIS_API_KEY": "",
}

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config as AlembicConfig  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from identity.config import reset_settings as reset_identity_settings  # noqa: E402
from identity.config import settings as identity_settings  # noqa: E402
from identity.infrastructure.crypto.keys import signing_key_from  # noqa: E402
from identity.infrastructure.directory.fake import (  # noqa: E402
    FakeGuardianDirectory,
)
from identity.infrastructure.directory.sis import SisGuardianDirectory  # noqa: E402
from identity.domain.schools import SchoolRegistry  # noqa: E402
from identity.infrastructure.whatsapp.channels import (  # noqa: E402
    WhatsAppChannels,
)
from identity.infrastructure.whatsapp.gateways import (  # noqa: E402
    RecordingWhatsAppGateway,
)

# --- the cast --------------------------------------------------------------

MOTHER = "+201001234567"          # Fatma — two children, two numbers
MOTHER_ALT = "+201119998888"
MOTHER_WA = "201001234567"
MOTHER_ALT_WA = "201119998888"

FATHER = "+201002223333"          # Hassan — one child
FATHER_WA = "201002223333"

BROTHER = "+201005554444"         # Karim — on file, barred by a court order
BROTHER_WA = "201005554444"

OTHER_PARENT = "+201007778888"    # Mona — a different family entirely
OTHER_PARENT_WA = "201007778888"

STRANGER_WA = "201110000000"      # nobody the school has ever heard of

TERM = "2026-T1"


# --------------------------------------------------------------------------
# The school
# --------------------------------------------------------------------------


def _seed_sis() -> None:
    """One school, two classes, three children, four adults, marks and a register.

    Shaped so the interesting cases exist rather than having to be constructed per test:
    a mother on two children and two numbers, a father on one, a sibling whose access the
    registrar restricted, an unrelated parent, a real zero beside an unmarked subject, and
    a fortnight of register days including one excused absence.
    """
    from sis.domain.attendance import AttendanceMark, AttendanceState
    from sis.domain.grades import SubjectGrade
    from sis.domain.guardians import Guardian, RelationshipType, StudentGuardian
    from sis.domain.people import ClassEnrolment, Student
    from sis.domain.structure import (
        AcademicYear,
        ClassSection,
        School,
        Subject,
        Term,
        YearLevel,
    )
    from sis.domain.value_objects import Percentage, Phone, StudentNumber
    from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

    sections = [
        ClassSection(code=code, academic_year_code="2025-2026", year_level_code="3",
                     name_en=f"Year 3 {code[-1]}", name_ar=f"الثالث {code[-1]}")
        for code in ("3A", "3B")
    ]

    with SqlAlchemyUnitOfWork() as uow:
        uow.schools.upsert_many([School(code="MAIN", name_en="Main", name_ar="الرئيسي")])
        uow.academic_years.upsert_many([
            AcademicYear(code="2025-2026", school_code="MAIN", name_en="2025/2026",
                         name_ar="٢٠٢٥/٢٠٢٦", starts_on=date(2025, 9, 1),
                         ends_on=date(2026, 6, 30), is_current=True)])
        uow.year_levels.upsert_many([
            YearLevel(code="3", school_code="MAIN", name_en="Year 3", name_ar="الثالث",
                      display_order=3)])
        uow.class_sections.upsert_many(sections)
        uow.terms.upsert_many([
            Term(code=TERM, academic_year_code="2025-2026", name_en="Term 1",
                 name_ar="الفصل الأول", starts_on=date(2025, 9, 1),
                 ends_on=date(2025, 12, 15), sequence=1)])
        uow.subjects.upsert_many([
            Subject(code="MATH", academic_year_code="2025-2026", name_en="Mathematics",
                    name_ar="الرياضيات", display_order=1),
            Subject(code="ARB", academic_year_code="2025-2026", name_en="Arabic",
                    name_ar="اللغة العربية", display_order=2),
            Subject(code="SCI", academic_year_code="2025-2026", name_en="Science",
                    name_ar="العلوم", display_order=3)])
        uow.students.upsert_many([
            Student(student_number=StudentNumber("S001"), full_name_ar="ليلى أحمد",
                    full_name_en="Layla Ahmed"),
            Student(student_number=StudentNumber("S002"), full_name_ar="عمر خالد",
                    full_name_en="Omar Khaled"),
            Student(student_number=StudentNumber("S003"), full_name_ar="نادية سمير",
                    full_name_en="Nadia Samir")])
        uow.enrolments.upsert_many([
            ClassEnrolment(student_number=StudentNumber(n), academic_year_code="2025-2026",
                           class_code=c, starts_on=date(2025, 9, 1))
            for n, c in (("S001", "3A"), ("S002", "3A"), ("S003", "3B"))])
        uow.guardians.upsert_many([
            Guardian(phones=(Phone(MOTHER), Phone(MOTHER_ALT)), full_name_ar="فاطمة علي",
                     full_name_en="Fatma Ali"),
            Guardian(phones=(Phone(FATHER),), full_name_en="Hassan Mahmoud"),
            Guardian(phones=(Phone(BROTHER),), full_name_en="Karim Hassan"),
            Guardian(phones=(Phone(OTHER_PARENT),), full_name_en="Mona Said")])
        uow.student_guardians.upsert_many([
            StudentGuardian(student_number=StudentNumber("S001"), guardian_phone=Phone(MOTHER),
                            relationship_type=RelationshipType.MOTHER,
                            is_primary_contact=True, can_view_records=True),
            StudentGuardian(student_number=StudentNumber("S002"), guardian_phone=Phone(MOTHER),
                            relationship_type=RelationshipType.MOTHER, can_view_records=True),
            StudentGuardian(student_number=StudentNumber("S001"), guardian_phone=Phone(FATHER),
                            relationship_type=RelationshipType.FATHER, can_view_records=True),
            # On file as a contact, barred from the records by a court order.
            StudentGuardian(student_number=StudentNumber("S001"), guardian_phone=Phone(BROTHER),
                            relationship_type=RelationshipType.SIBLING,
                            relationship_label="big brother", can_view_records=False,
                            restriction_note="court order 2026/114"),
            StudentGuardian(student_number=StudentNumber("S003"), guardian_phone=Phone(OTHER_PARENT),
                            relationship_type=RelationshipType.MOTHER, can_view_records=True)])
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        ids = uow.class_sections.ids_for([s.identity for s in sections])
        a_id = ids[("2025-2026", "3A")]
        uow.grades.upsert_many([
            SubjectGrade(student_number=StudentNumber("S001"), subject_code="MATH",
                         term_code=TERM, class_section_id=a_id, class_code="3A",
                         percentage=Percentage(88.5)),
            # Unmarked. Must reach a parent as "not marked yet", never as 0.
            SubjectGrade(student_number=StudentNumber("S001"), subject_code="ARB",
                         term_code=TERM, class_section_id=a_id, class_code="3A",
                         percentage=None),
            # A real zero, beside the blank above. The pair is what proves the two stay
            # distinguishable all the way to the parent.
            SubjectGrade(student_number=StudentNumber("S001"), subject_code="SCI",
                         term_code=TERM, class_section_id=a_id, class_code="3A",
                         percentage=Percentage(0.0)),
            SubjectGrade(student_number=StudentNumber("S002"), subject_code="MATH",
                         term_code=TERM, class_section_id=a_id, class_code="3A",
                         percentage=Percentage(61.0))])
        marks = []
        for index in range(10):
            state = (
                AttendanceState.ABSENT if index == 3
                else AttendanceState.EXCUSED if index == 5
                else AttendanceState.LATE if index == 7
                else AttendanceState.PRESENT
            )
            marks.append(AttendanceMark(
                student_number=StudentNumber("S001"), on_date=date(2025, 9, 1 + index),
                state=state, class_section_id=a_id, class_code="3A",
                note="doctor's note" if state is AttendanceState.EXCUSED else ""))
        uow.attendance.upsert_many(marks, recorded_by="journey")
        uow.commit()

    _seed_sis_api_keys()


def _seed_sis_api_keys() -> None:
    """The two credentials this estate presents to `sis/`, stored so they verify.

    `SIS_API_KEY` used to be a value nothing checked. `sis/` authenticates every caller
    again, so the journey only proves something if the keys it sends are keys SIS accepts —
    unstored ones would 401 on every hop and this suite would be asserting that a broken
    estate fails politely.

    **Two keys, on purpose.** `records/` holds a `reader`; the registrar's uploads and
    custody changes hold a `registrar`. `Scope.permits` is exact equality, so the reader is
    refused by every write route — which means the process answering parents cannot rewrite
    a term's marks even if it is fully compromised. Giving both jobs one key would pass
    this suite and quietly delete that property.
    """
    from datetime import UTC, datetime

    from sis.api.deps import hash_api_key, key_prefix
    from sis.domain.auth import ApiKey, Scope
    from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

    minted = (
        (ENVIRONMENT["SIS_API_KEY"], Scope.READER, "records adapter (journey)"),
        (SIS_REGISTRAR_KEY, Scope.REGISTRAR, "registrar office (journey)"),
    )
    with SqlAlchemyUnitOfWork() as uow:
        for raw, scope, label in minted:
            uow.api_keys.add(
                ApiKey(
                    prefix=key_prefix(raw),
                    key_hash=hash_api_key(raw),
                    label=label,
                    scope=scope,
                    is_active=True,
                    expires_at=None,
                    created_at=datetime.now(UTC),
                )
            )
        uow.commit()


def _serve(app, port: str) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(400):
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError(f"a service did not start on port {port}")


@pytest.fixture(scope="session", autouse=True)
def _own_the_environment():
    """Configure the three services for this file, and put it back afterwards.

    **Session-scoped, and `estate` depends on it**, because ordering is the point. As a
    per-test fixture this ran after the session-scoped `estate`, which had already
    migrated and served whichever database it inherited: the servers came up on another
    suite's file and every guardian lookup failed with `no such table`.

    Restored on the way out, so a suite that runs after this one in the same session
    gets the environment it expected rather than this file's. `records/` in particular
    reads `RECORDS_LMS` and `SIS_BASE_URL` when its app starts up, so a leaked value
    silently swaps its fake adapter for a real HTTP client.

    The engines are dropped as well as the variables set. `sis/` and `identity/` each
    memoise an engine built from their own variable, so resetting one leaves the other
    serving whichever database its own suite bound it to. `records/` needs no reset: it
    holds no database at all.
    """
    import identity.infrastructure.db.session as identity_session
    from sis.config import reset_settings_cache
    from sis.infrastructure.db.session import reset_engine

    def _drop_engines() -> None:
        reset_settings_cache()
        reset_engine()
        # Identity's engine and its settings are dropped together: the engine is
        # rebuilt from the settings, so clearing one without the other would rebuild
        # against the database it was just pointed away from.
        reset_identity_settings()
        identity_session.reset_engine()

    previous = {key: os.environ.get(key) for key in ENVIRONMENT}
    os.environ.update(ENVIRONMENT)
    _drop_engines()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _drop_engines()


@pytest.fixture(scope="session", autouse=True)
def estate(_own_the_environment):
    """sis/ and records/ running for real, for the whole session.

    Session-scoped because starting two ASGI servers per test would dominate the runtime
    and none of these tests mutate the schema. The ones that *do* change data — a
    revoked custody, a fresh guardian upload — either add rows or undo themselves.
    """
    from sis.config import reset_settings_cache

    reset_settings_cache()
    command.upgrade(AlembicConfig("sis/alembic.ini"), "head")

    from sis.infrastructure.db.session import reset_engine

    reset_engine()
    _seed_sis()

    from sis.app import app as sis_app

    sis_server = _serve(sis_app, SIS_PORT)

    # records verifies identity's tokens offline. Handing it the public key directly is
    # the documented alternative to a JWKS URL and needs no third server.
    # The key the identity app will build for itself, built here so records can be
    # handed the public half before either process starts.
    os.environ["IDENTITY_PUBLIC_KEY_PEM"] = signing_key_from(
        identity_settings()
    ).public_pem

    from records.app import app as records_app

    records_server = _serve(records_app, RECORDS_PORT)

    yield

    sis_server.should_exit = True
    records_server.should_exit = True
    time.sleep(0.3)


@pytest.fixture(autouse=True)
def _fresh_challenges():
    """Clear spent verification challenges between tests.

    Identity rate-limits a number to a handful of challenges per quarter of an hour, which
    is right: a parent tapping the link twice is normal, a script walking nonces is not.
    This file signs the same mother in a dozen times in under a second, so from the
    limiter's point of view it *is* the attack — and the symptom is a login that quietly
    sends no code at all, which surfaces as an IndexError reading the recorder rather than
    as anything resembling its cause.

    Clearing the table is preferred over widening the limit: the limit is production
    behaviour and should not be softened to suit a test, and it keeps its own coverage in
    identity's own suite where a fake clock can exercise it honestly.
    """
    from identity.infrastructure.db.schema import init_db
    from identity.infrastructure.db.session import new_session

    # Identity builds its own schema in its lifespan, which has not run yet the first time
    # this fixture fires. `init_db` is `create_all` and idempotent, so calling it here
    # costs nothing and removes the ordering dependency.
    init_db()

    from identity.infrastructure.db.models import VerificationChallenge

    session = new_session()
    try:
        session.query(VerificationChallenge).delete()
        session.commit()
    finally:
        session.close()
    yield


def _channels(gateway, *, directory=None) -> WhatsAppChannels:
    """What the identity app would have built, pointed at this test's SIS and gateway.

    The real `SisGuardianDirectory` by default, because this suite is the one place the
    two services are exercised against each other over a real socket — a fake here would
    make the journey prove nothing about the seam it exists to prove.
    """
    return WhatsAppChannels(
        registry=SchoolRegistry(),
        directory=directory
        or SisGuardianDirectory(
            base_url=f"http://127.0.0.1:{SIS_PORT}", api_key=ENVIRONMENT["SIS_API_KEY"]
        ),
        verify_token=VERIFY_TOKEN,
        app_secret=APP_SECRET,
        business_number=SCHOOL_NUMBER,
        default_gateway=gateway,
    )


@pytest.fixture()
def gateway():
    """Every WhatsApp message the school would have sent, kept instead of sent."""
    return RecordingWhatsAppGateway()


@pytest.fixture()
def identity(gateway):
    from identity.app import app as identity_app

    with TestClient(identity_app) as client:
        # After the lifespan, so this beats whatever app.py built from the (unset)
        # environment. One object holds the gateway, the directory and the webhook
        # secrets, so they are installed together and cannot disagree.
        client.app.state.channels = _channels(gateway)
        yield client


@pytest.fixture(scope="session")
def agent_key() -> str:
    """The credential the chat backend presents to `records/`.

    Configuration rather than a minted row: `records/` holds no database, so there is no
    key table to insert into and no admin route to insert through. Both sides read the
    same variable — the backend to send it, `records/` to compare against it.
    """
    return ENVIRONMENT["RECORDS_API_KEY"]


@pytest.fixture()
def sis_client():
    """The registrar's own client: uploads, commits, and custody changes.

    Carries the `registrar` key, not the `reader` one `records/` uses — these are write
    routes and the reader is refused by them, which is the separation working.
    """
    with httpx.Client(
        base_url=f"http://127.0.0.1:{SIS_PORT}",
        timeout=10,
        headers={"X-API-Key": SIS_REGISTRAR_KEY},
    ) as client:
        yield client


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


#: A fresh WhatsApp message id per delivery.
#:
#: The webhook deduplicates on this — Meta replays an unacknowledged delivery for up to
#: seven days, and one parent tap must not send several codes. The identity database
#: outlives a single test, so a fixed id here makes the *second* login in the file look
#: like a retry of the first: no code is sent, and the test reads a stale one out of the
#: recorder and fails somewhere far from the cause.
_message_ids = itertools.count(1)


def _next_message_id() -> str:
    return f"wamid.J{next(_message_ids)}"


def _deliver(client, wa_id: str, text: str, *, message_id: str = ""):
    """One inbound WhatsApp message, signed the way Meta signs it."""
    message_id = message_id or _next_message_id()
    payload = {
        "object": "whatsapp_business_account",
        "entry": [{"id": "waba", "changes": [{"field": "messages", "value": {
            "messaging_product": "whatsapp",
            "metadata": {"display_phone_number": "201288339613", "phone_number_id": "pn"},
            "contacts": [{"profile": {"name": "فاطمة علي"}, "wa_id": wa_id}],
            "messages": [{"from": wa_id, "id": message_id, "timestamp": "1",
                          "type": "text", "text": {"body": text}}]}}]}],
    }
    # ensure_ascii=False so Arabic travels as UTF-8, which is what Meta sends and what the
    # signature has to be computed over.
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    signature = hmac.new(APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return client.post("/v1/auth/whatsapp/webhook", content=raw, headers={
        "X-Hub-Signature-256": f"sha256={signature}", "Content-Type": "application/json"})


def _sign_in(identity, gateway, wa_id: str, *, message_id: str = "") -> dict:
    """The whole login, as a parent performs it. Returns the token payload."""
    started = identity.post("/v1/auth/whatsapp/start").json()
    delivered = _deliver(
        identity, wa_id, started["message"], message_id=message_id or _next_message_id()
    )
    assert delivered.status_code == 200
    code = "".join(c for c in gateway.sent[-1][1] if c.isdigit())
    verified = identity.post("/v1/auth/whatsapp/verify",
                             json={"poll_secret": started["poll_secret"], "code": code})
    assert verified.status_code == 200, verified.text
    return verified.json()


def _parent_client(token: str, agent_key: str) -> httpx.Client:
    """records/, addressed the way the chat backend addresses it.

    Two credentials, and both are required: the key says which *system* is calling, the
    bearer token says which *parent* it is calling for. Neither substitutes for the other.
    """
    return httpx.Client(
        base_url=f"http://127.0.0.1:{RECORDS_PORT}",
        headers={"X-API-Key": agent_key, "Authorization": f"Bearer {token}"},
        timeout=10,
    )


# --------------------------------------------------------------------------


class TestSigningIn:
    """WhatsApp proves a number; the school's records decide whether it is a parent's."""

    def test_a_parent_signs_in_and_the_token_names_her(self, identity, gateway):
        session = _sign_in(identity, gateway, MOTHER_WA)

        assert session["role"] == "parent"
        assert session["guardian_id"]
        assert session["access_token"] and session["refresh_token"]

    def test_the_link_points_at_the_school_s_own_number(self, identity, gateway):
        started = identity.post("/v1/auth/whatsapp/start").json()

        # Digits only and no leading zero. The national spelling produces a link to a
        # number that does not exist, and it fails completely silently.
        assert started["link"].startswith("https://wa.me/201288339613?text=")
        assert started["business_number"] == SCHOOL_NUMBER

    def test_either_of_her_numbers_reaches_the_same_account(self, identity, gateway):
        """A parent who verifies her WhatsApp line is the woman who verified her mobile."""
        first = _sign_in(identity, gateway, MOTHER_WA, message_id="wamid.A1")
        second = _sign_in(identity, gateway, MOTHER_ALT_WA, message_id="wamid.A2")

        assert first["guardian_id"] == second["guardian_id"]
        assert first["username"] == second["username"]

    def test_two_parents_of_one_child_are_two_different_people(self, identity, gateway):
        mother = _sign_in(identity, gateway, MOTHER_WA, message_id="wamid.B1")
        father = _sign_in(identity, gateway, FATHER_WA, message_id="wamid.B2")

        assert mother["guardian_id"] != father["guardian_id"]

    def test_a_number_the_school_does_not_hold_is_refused(self, identity, gateway):
        started = identity.post("/v1/auth/whatsapp/start").json()

        assert _deliver(identity, STRANGER_WA, started["message"]).status_code == 200

        reply = gateway.sent[-1][1]
        assert "not registered" in reply
        # No code, and nothing that would tell them which numbers *are* registered.
        assert not any(character.isdigit() for character in reply.replace(" ", ""))

    def test_the_start_of_a_login_asks_for_no_phone_number(self, identity, gateway):
        """There is nothing here to probe.

        The only way to ask "is this number a parent" is to send a message from it, and
        the answer goes to that number over WhatsApp rather than into this response.
        """
        started = identity.post("/v1/auth/whatsapp/start").json()

        assert "phone" not in started
        assert "guardian_id" not in started

    def test_metas_retries_do_not_send_a_second_code(self, identity, gateway):
        """Meta replays an unacknowledged delivery for up to seven days."""
        started = identity.post("/v1/auth/whatsapp/start").json()
        before = len(gateway.sent)

        for _ in range(3):
            _deliver(identity, MOTHER_WA, started["message"], message_id="wamid.SAME")

        assert len(gateway.sent) - before == 1

    def test_an_unsigned_webhook_is_refused(self, identity, gateway):
        """The webhook is public. Without the signature anyone could claim any number."""
        started = identity.post("/v1/auth/whatsapp/start").json()
        before = len(gateway.sent)

        unsigned = identity.post(
            "/v1/auth/whatsapp/webhook",
            content=json.dumps({"entry": []}).encode(),
            headers={"Content-Type": "application/json"},
        )

        assert unsigned.status_code == 403
        assert len(gateway.sent) == before


class TestWhatAParentCanRead:
    """The figures, and the two distinctions that must survive every hop."""

    def test_she_sees_both_of_her_children_and_no_others(self, identity, gateway, agent_key):
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            body = parent.get(f"/v1/guardians/{session['guardian_id']}/students").json()

        assert {row["student_id"] for row in body["students"]} == {"S001", "S002"}

    def test_a_father_sees_only_his_own_child(self, identity, gateway, agent_key):
        session = _sign_in(identity, gateway, FATHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            body = parent.get(f"/v1/guardians/{session['guardian_id']}/students").json()

        assert [row["student_id"] for row in body["students"]] == ["S001"]

    def test_an_unmarked_subject_is_null_and_a_zero_is_a_zero(
        self, identity, gateway, agent_key
    ):
        """The invariant the whole estate is built around, checked at the far end.

        Zero is a mark a child earned. "Not marked yet" is not a mark at all. Anywhere the
        two converge, a school tells a family their daughter scored 0% in a subject nobody
        has graded — and it is byte-identical to a real zero, so nothing downstream can
        tell it went wrong.
        """
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            body = parent.get(
                f"/v1/guardians/{session['guardian_id']}/students/S001/grades",
                params={"term": TERM},
            ).json()

        by_code = {course["subject_code"]: course for course in body["courses"]}
        assert by_code["MATH"]["computed_percentage"] == 88.5
        assert by_code["SCI"]["computed_percentage"] == 0.0
        assert by_code["ARB"]["computed_percentage"] is None

    def test_every_subject_reaches_her_named_in_arabic(self, identity, gateway, agent_key):
        """A silently-dropped subject is the failure this file exists to catch.

        It arrives as a `200` with a shorter list, which reads as "that is all the school
        has recorded" and is indistinguishable from the truth without knowing the answer.
        """
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            body = parent.get(
                f"/v1/guardians/{session['guardian_id']}/students/S001/grades",
                params={"term": TERM},
            ).json()

        assert len(body["courses"]) == 3
        assert {c["subject_name_ar"] for c in body["courses"]} == {
            "الرياضيات", "اللغة العربية", "العلوم"
        }

    def test_each_child_has_her_own_marks(self, identity, gateway, agent_key):
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            base = f"/v1/guardians/{session['guardian_id']}/students"
            layla = parent.get(f"{base}/S001/grades", params={"term": TERM}).json()
            omar = parent.get(f"{base}/S002/grades", params={"term": TERM}).json()

        assert layla["student"]["student_id"] == "S001"
        assert omar["student"]["student_id"] == "S002"
        assert {c["subject_code"]: c["computed_percentage"] for c in omar["courses"]} == {
            "MATH": 61.0
        }

    def test_attendance_counts_an_excused_day_as_attended(
        self, identity, gateway, agent_key
    ):
        """Where the two services genuinely disagree, and this contract's answer wins.

        SIS's `in_the_room` is present-plus-late. This contract counts excused as
        attended and the template says so to the parent, so publishing SIS's narrower
        figure would show a child with a doctor's note as having missed school.
        """
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            body = parent.get(
                f"/v1/guardians/{session['guardian_id']}/students/S001/attendance",
                params={"term": TERM},
            ).json()

        assert body["total_sessions"] == 10
        assert body["present_count"] == 7
        assert body["absent_count"] == 1
        assert body["excused_count"] == 1
        # 9 of 10: everything except the one unexcused absence.
        assert body["attendance_rate"] == 90.0

    def test_one_subject_can_be_asked_about_on_its_own(self, identity, gateway, agent_key):
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            body = parent.get(
                f"/v1/guardians/{session['guardian_id']}/students/S001/grades/MATH",
                params={"term": TERM},
            ).json()

        assert body["course"]["subject_name_ar"] == "الرياضيات"
        assert body["course"]["computed_percentage"] == 88.5

    def test_the_term_is_named_from_the_school_s_own_calendar(
        self, identity, gateway, agent_key
    ):
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            terms = parent.get("/v1/terms").json()

        assert [t["term_id"] for t in terms] == [TERM]
        assert terms[0]["name_ar"] == "الفصل الأول"


class TestWhatAParentCannotRead:
    """Every refusal, and the ones that must look identical from outside."""

    def test_another_family_s_child_is_refused(self, identity, gateway, agent_key):
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            refused = parent.get(
                f"/v1/guardians/{session['guardian_id']}/students/S003/grades",
                params={"term": TERM},
            )

        assert refused.status_code == 404

    def test_a_child_who_does_not_exist_looks_the_same(self, identity, gateway, agent_key):
        """Otherwise a caller walks student numbers and learns which ones are real."""
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            base = f"/v1/guardians/{session['guardian_id']}/students"
            not_hers = parent.get(f"{base}/S003/grades", params={"term": TERM})
            no_such = parent.get(f"{base}/S999/grades", params={"term": TERM})

        assert not_hers.status_code == no_such.status_code == 404
        assert not_hers.json()["detail"] == no_such.json()["detail"]

    def test_a_guardian_barred_by_a_court_order_reads_nothing(
        self, identity, gateway, agent_key
    ):
        """He is a real guardian, on file, and linked to her. He may not read her records.

        Restricted and unknown look identical from here, deliberately: a caller able to
        tell them apart could detect a custody restriction from outside the school.
        """
        session = _sign_in(identity, gateway, BROTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            children = parent.get(f"/v1/guardians/{session['guardian_id']}/students").json()
            marks = parent.get(
                f"/v1/guardians/{session['guardian_id']}/students/S001/grades",
                params={"term": TERM},
            )

        assert children["students"] == []
        assert marks.status_code == 404

    def test_a_forged_token_reads_nothing(self, agent_key):
        with _parent_client("not-a-real-token", agent_key) as impostor:
            refused = impostor.get("/v1/guardians/anything/students")

        assert refused.status_code == 401

    def test_one_parent_s_token_cannot_ask_about_another_parent(
        self, identity, gateway, agent_key
    ):
        """The signed claim has to match the guardian named in the path.

        This is what stops a compromised chat backend from reading a family it holds no
        token for: it can relay a parent's own identity and nothing else.
        """
        mother = _sign_in(identity, gateway, MOTHER_WA, message_id="wamid.X1")
        other = _sign_in(identity, gateway, OTHER_PARENT_WA, message_id="wamid.X2")

        with _parent_client(mother["access_token"], agent_key) as parent:
            refused = parent.get(f"/v1/guardians/{other['guardian_id']}/students")

        assert refused.status_code == 403

    def test_an_agent_key_alone_proves_nothing(self, agent_key):
        """A leaked key must be worth nothing on its own — it says which system, not who."""
        with httpx.Client(base_url=f"http://127.0.0.1:{RECORDS_PORT}", timeout=10) as caller:
            refused = caller.get(
                "/v1/guardians/anything/students", headers={"X-API-Key": agent_key}
            )

        assert refused.status_code == 401

    def test_the_route_that_granted_access_cannot_be_reached_at_all(
        self, identity, gateway, agent_key
    ):
        """It used to be refused by scope. It is now gone, which is stronger.

        Granting a guardian access to a child is the registrar's act and lives in `sis/`.
        A 403 would invite somebody to look for the right key; 410 says there is no key,
        and names where the capability went.
        """
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            refused = parent.post(
                "/v1/admin/guardians/G-1/students",
                json={"student_id": "S001", "can_view_records": True},
            )

        assert refused.status_code == 410
        assert refused.json()["detail"]["code"] == "moved"


class TestWhenSomethingIsDown:
    """A service that cannot answer must never be rendered as an answer."""

    def test_an_unreachable_school_does_not_become_no_children(
        self, identity, gateway, agent_key
    ):
        """"Not registered" and "we cannot reach the records" are different sentences.

        Telling a real parent they are unknown because another service blinked is worse
        than telling them to try again.
        """
        from records import guardian_directory as records_directory

        session = _sign_in(identity, gateway, MOTHER_WA)
        healthy = records_directory.get_directory()
        records_directory.set_directory(
            records_directory.FakeGuardianDirectory(unavailable=True)
        )
        try:
            with _parent_client(session["access_token"], agent_key) as parent:
                refused = parent.get(
                    f"/v1/guardians/{session['guardian_id']}/students/S001/grades",
                    params={"term": TERM},
                )
        finally:
            records_directory.set_directory(healthy)

        assert refused.status_code == 503
        assert refused.json()["detail"]["code"] == "not_configured"

    def test_an_unreachable_school_leaves_a_login_retryable(self, identity, gateway):
        """Our problem, not the parent's — so the challenge survives for another attempt."""
        healthy = identity.app.state.channels
        identity.app.state.channels = _channels(
            gateway, directory=FakeGuardianDirectory(unavailable=True)
        )
        try:
            started = identity.post("/v1/auth/whatsapp/start").json()
            _deliver(identity, MOTHER_WA, started["message"], message_id="wamid.DOWN")
            status = identity.post("/v1/auth/whatsapp/status",
                                   json={"poll_secret": started["poll_secret"]}).json()
        finally:
            identity.app.state.channels = healthy

        assert "try again" in gateway.sent[-1][1]
        assert status["status"] == "pending"

    def test_a_term_nobody_has_marked_reports_nothing_rather_than_zeros(
        self, identity, gateway, agent_key
    ):
        """An unknown term is a 404, not a page of zeroes."""
        session = _sign_in(identity, gateway, MOTHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            missing = parent.get(
                f"/v1/guardians/{session['guardian_id']}/students/S001/grades",
                params={"term": "2099-T9"},
            )

        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "unknown_term"


class TestTheRegistrarChangesSomething:
    """The school's decisions have to reach a parent's next question, not the next sync."""

    def test_revoking_access_takes_effect_immediately(
        self, identity, gateway, agent_key, sis_client
    ):
        """A court order arrives and the office acts. Nothing here is cached.

        The father is used rather than the mother so the revocation can be undone without
        leaving the rest of the file depending on ordering.
        """
        session = _sign_in(identity, gateway, FATHER_WA)

        with _parent_client(session["access_token"], agent_key) as parent:
            before = parent.get(f"/v1/guardians/{session['guardian_id']}/students").json()
            assert [row["student_id"] for row in before["students"]] == ["S001"]

            revoked = sis_client.patch(
                f"/v1/students/S001/guardians/{FATHER}",
                json={"can_view_records": False, "restriction_note": "journey test"},
            )
            assert revoked.status_code == 200, revoked.text
            try:
                after = parent.get(
                    f"/v1/guardians/{session['guardian_id']}/students"
                ).json()
                marks = parent.get(
                    f"/v1/guardians/{session['guardian_id']}/students/S001/grades",
                    params={"term": TERM},
                )
            finally:
                sis_client.patch(
                    f"/v1/students/S001/guardians/{FATHER}",
                    json={"can_view_records": True, "restriction_note": ""},
                )

        assert after["students"] == []
        assert marks.status_code == 404

    def test_a_guardian_uploaded_today_can_sign_in_today(
        self, identity, gateway, agent_key, sis_client
    ):
        """The registrar's spreadsheet, straight through to a parent holding a token.

        The one path that proves the whole product: a name and a number typed into a sheet
        this morning, and that parent reading her child's marks this afternoon without
        anybody touching a database.
        """
        sheet = (
            "student_number,guardian name (arabic),phone,relationship\n"
            "S003,سعاد إبراهيم,01234567890,mother\n"
        ).encode("utf-8")

        preview = sis_client.post(
            "/v1/imports/guardians/preview",
            files={"file": ("guardians.csv", sheet, "text/csv")},
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["ok_count"] == 1

        committed = sis_client.post(
            f"/v1/imports/guardians/{preview.json()['batch_id']}/commit"
        )
        assert committed.status_code == 200, committed.text

        session = _sign_in(identity, gateway, "201234567890", message_id="wamid.NEW")

        with _parent_client(session["access_token"], agent_key) as parent:
            children = parent.get(f"/v1/guardians/{session['guardian_id']}/students").json()

        assert [row["student_id"] for row in children["students"]] == ["S003"]
