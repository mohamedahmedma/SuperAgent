"""HTTP tests: the whole stack, over a real migrated database, through `TestClient`.

Everything below the API is unit-tested with fakes. What only shows up here is the
wiring — that a route is mounted at the path the contract promises, that the scope
dependency is attached to the write routes and not merely defined, that a refusal reaches
the client in one envelope shape, and that a registrar's afternoon (generate, create,
upload, commit, read a report card) actually works end to end.

Two of these are guarding against a *missing* line rather than a wrong one. A router that
forgets `caller: Registrar` serves every write to anybody and passes every unit test its
service has, because the service never sees a caller; a serialiser that renders an
unmarked subject as `0` is a correct number in a correct-looking field. Neither has a
symptom short of the response body, so the response body is what is asserted.

The database is built by `alembic upgrade head` (invariant 8), and the app's own startup
check runs against it unmocked — a suite that skipped that check would not notice the day
the migration and the code stopped agreeing.
"""
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from sis.app import create_app
from sis.config import reset_settings_cache
from sis.domain.structure import AcademicYear, ClassSection, YearLevel
from sis.infrastructure.db.session import reset_engine
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

_ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"

# Long enough that its 12-character prefix is a handle rather than most of the secret.
BOOTSTRAP_KEY = "bootstrap-registrar-key-0123456789abcdef"

# Small enough that a test can exceed it with a few kilobytes, large enough that every
# legitimate upload in this file passes comfortably.
MAX_UPLOAD_BYTES = 4096

YEAR_CODE = "2025-2026"
TERM_CODE = "2026-T1"

ROSTER_CSV = (
    "Student Number,Arabic Name,English Name,Class\n"
    "S001,ليلى أحمد,Layla Ahmed,3A\n"
    "S002,عمر خالد,Omar Khaled,3A\n"
).encode("utf-8")

# Three deliberate shapes: a stated mark, a blank cell, and an earned zero. The last two
# must not be able to render as the same thing.
GRADES_CSV = (
    "Student Number,Subject,Percentage\n"
    "S001,MATH,87.5\n"
    "S001,SCI,\n"
    "S002,MATH,0\n"
).encode("utf-8")


@pytest.fixture
def sis_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """An empty, migrated database, with the service pointed at it."""
    url = f"sqlite:///{(tmp_path / 'sis.db').as_posix()}"
    monkeypatch.setenv("SIS_DATABASE_URL", url)
    monkeypatch.setenv("SIS_BOOTSTRAP_REGISTRAR_KEY", BOOTSTRAP_KEY)
    monkeypatch.setenv("SIS_MAX_UPLOAD_BYTES", str(MAX_UPLOAD_BYTES))
    reset_settings_cache()
    reset_engine()

    command.upgrade(Config(str(_ALEMBIC_INI)), "head")
    yield

    reset_engine()
    reset_settings_cache()


@pytest.fixture
def client(sis_database: None) -> Iterator[TestClient]:
    """A client over a freshly built app, entered so the lifespan's schema check runs."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def registrar() -> dict[str, str]:
    """The bootstrap key: an empty database plus one variable is a working registrar."""
    return {"X-API-Key": BOOTSTRAP_KEY}


@pytest.fixture
def reader(client: TestClient, registrar: dict[str, str]) -> dict[str, str]:
    """A minted `reader` key — the credential every write route below must refuse."""
    response = client.post(
        "/v1/admin/api-keys",
        json={"label": "reporting dashboard", "scope": "reader"},
        headers=registrar,
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["scope"] == "reader"
    return {"X-API-Key": body["api_key"]}


def _seed_academic_year() -> None:
    """The one thing the HTTP surface cannot create, and everything else hangs from it."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.academic_years.upsert_many(
            [
                AcademicYear(
                    code=YEAR_CODE,
                    name_en="2025-2026",
                    name_ar="٢٠٢٥-٢٠٢٦",
                    starts_on=date(2025, 9, 1),
                    ends_on=date(2026, 6, 30),
                    is_current=True,
                )
            ]
        )
        uow.commit()


def _seed_generated_structure() -> None:
    """The rung and sections `/v1/structure/generate` would produce, written directly.

    Written directly rather than over HTTP to keep this fixture independent of the
    generate route, which has its own test above. The codes and labels are exactly what
    `classes_by_year={"3": 2}` produces through the default templates, so the two stay
    in step.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.year_levels.upsert_many(
            [YearLevel(code="3", name_en="Year 3", name_ar="السنة 3", display_order=3)]
        )
        uow.class_sections.upsert_many(
            [
                ClassSection(
                    code=f"3{suffix}",
                    academic_year_code=YEAR_CODE,
                    year_level_code="3",
                    name_en=f"Year 3 {suffix}",
                    name_ar=f"السنة 3 {suffix}",
                )
                for suffix in ("A", "B")
            ]
        )
        uow.commit()


def assert_error_envelope(response: Any, *, status: int, code: str | None = None) -> None:
    """Every refusal is `{"detail": {"code", "message"}}`, whatever raised it.

    Asserted on shape rather than on wording because `message` is prose and this school
    will translate it, while `code` is the contract a client branches on.
    """
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, dict), detail
    assert detail["code"] and isinstance(detail["code"], str)
    assert detail["message"] and isinstance(detail["message"], str)
    if code is not None:
        assert detail["code"] == code


def _csv_upload(name: str = "roster.csv", content: bytes = ROSTER_CSV) -> dict[str, Any]:
    return {"file": (name, content, "text/csv")}


# Every route that writes, or reads a write's audit trail. Each carries a well-formed
# payload on purpose: if the body were invalid, a 422 could mask a missing scope check
# and the test would pass against an unguarded route.
WRITE_ROUTES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "POST",
        "/v1/structure/generate",
        {"json": {"academic_year_code": YEAR_CODE, "year_count": 1, "classes_per_year": 1}},
    ),
    (
        "POST",
        "/v1/terms",
        {
            "json": {
                "code": TERM_CODE,
                "academic_year_code": YEAR_CODE,
                "name_en": "Term 1",
                "name_ar": "الفصل الأول",
                "starts_on": "2025-09-01",
                "ends_on": "2025-12-15",
            }
        },
    ),
    ("POST", "/v1/subjects", {"json": {"code": "MATH", "name_en": "Mathematics"}}),
    (
        "POST",
        "/v1/imports/roster/preview",
        {"files": _csv_upload(), "data": {"academic_year_code": YEAR_CODE}},
    ),
    ("POST", "/v1/imports/roster/any-batch/commit", {}),
    (
        "POST",
        "/v1/imports/grades/preview",
        {
            "files": _csv_upload("grades.csv", GRADES_CSV),
            "data": {"term_code": TERM_CODE},
        },
    ),
    ("POST", "/v1/imports/grades/any-batch/commit", {}),
    ("GET", "/v1/imports/any-batch", {}),
    ("POST", "/v1/admin/api-keys", {"json": {"label": "another", "scope": "reader"}}),
)

_ROUTE_IDS = [f"{method} {path}" for method, path, _ in WRITE_ROUTES]


@pytest.mark.parametrize(("method", "path", "payload"), WRITE_ROUTES, ids=_ROUTE_IDS)
def test_write_route_refuses_a_request_carrying_no_key(
    client: TestClient, method: str, path: str, payload: dict[str, Any]
) -> None:
    response = client.request(method, path, **payload)

    assert_error_envelope(response, status=401, code="not_authorized")
    # The refusal must come from the dependency, before the handler: a 404 here would
    # mean an unauthenticated caller had already been told whether a batch id exists.
    assert response.headers.get("WWW-Authenticate") == "X-API-Key"


@pytest.mark.parametrize(("method", "path", "payload"), WRITE_ROUTES, ids=_ROUTE_IDS)
def test_write_route_refuses_a_reader_key(
    client: TestClient,
    reader: dict[str, str],
    method: str,
    path: str,
    payload: dict[str, Any],
) -> None:
    """`Scope.permits` is exact equality, and this is where that has to hold.

    A reader key reaching a write route is how a reporting dashboard handed the wrong
    credential silently gains the ability to rewrite a term's grades — and nothing in a
    log distinguishes that from a registrar doing her job.
    """
    response = client.request(method, path, headers=reader, **payload)

    assert_error_envelope(response, status=403, code="not_authorized")


def test_a_registrar_key_is_refused_by_reader_only_checks_nowhere_it_reads(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Reads a registrar legitimately performs must not be locked behind `reader`.

    The mirror of the test above, and the reason `require_read_access` names both scopes
    out loud: exact-equality scopes would otherwise refuse the registrar the very
    dropdowns she generates the structure from.
    """
    _seed_academic_year()

    assert client.get("/v1/structure/years", headers=registrar).status_code == 200
    assert client.get("/v1/subjects", headers=registrar).status_code == 200


def test_an_oversized_upload_is_refused_before_it_is_buffered(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """413, not 400: the file was merely too big, and "bad request" sends the registrar
    looking for a corruption that is not there."""
    bulky = b"Student Number,Arabic Name,English Name,Class\n" + (
        b"S001,\xd9\x84\xd9\x8a\xd9\x84\xd9\x89,Layla,3A\n" * 800
    )
    assert len(bulky) > MAX_UPLOAD_BYTES

    response = client.post(
        "/v1/imports/roster/preview",
        files=_csv_upload("big.csv", bulky),
        data={"academic_year_code": YEAR_CODE},
        headers=registrar,
    )

    assert_error_envelope(response, status=413, code="upload_too_large")


def test_a_file_this_service_cannot_read_is_refused_by_name(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """A PDF is refused at the door rather than parsed by guessing where its columns are.

    The refusal is the design: a column boundary inferred one glyph wide attaches the
    wrong child to the wrong class, and every row of that import still looks well formed.
    """
    response = client.post(
        "/v1/imports/roster/preview",
        files=_csv_upload("roster.pdf", b"%PDF-1.7 not a spreadsheet"),
        data={"academic_year_code": YEAR_CODE},
        headers=registrar,
    )

    assert 400 <= response.status_code < 500
    assert_error_envelope(
        response, status=response.status_code, code="unsupported_file_type"
    )
    assert response.json()["detail"]["field"] == "file"


def test_structure_generate_creates_a_rung_and_its_sections_idempotently(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Invariant 3 over HTTP: a second click reports, and does not double the school."""
    _seed_academic_year()

    generated = client.post(
        "/v1/structure/generate",
        json={"academic_year_code": YEAR_CODE, "classes_by_year": {"3": 2}},
        headers=registrar,
    )
    assert generated.status_code == 200, generated.text
    body = generated.json()
    assert body["created_count"] == 3  # the rung, plus 3A and 3B
    assert [item["code"] for item in body["items"]] == ["3", "3A", "3B"]

    again = client.post(
        "/v1/structure/generate",
        json={"academic_year_code": YEAR_CODE, "classes_by_year": {"3": 2}},
        headers=registrar,
    )
    # `created: false` is a success, not a skip — that is what makes the button safe to
    # click twice, and 409 would be the wrong answer.
    assert again.status_code == 200
    assert again.json() == {
        **body,
        "created_count": 0,
        "existing_count": 3,
        "items": [{**item, "created": False} for item in body["items"]],
    }


def test_registrar_imports_a_roster_then_marks_and_reads_back_the_report_card(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """The registrar's afternoon, ending at what a parent is actually shown.

    Written as one test rather than six because the steps are not independent: a batch id
    comes from the preview before it, and a mark cannot be filed at all until the child
    has a placement covering the term. Split apart, each step would need the previous
    ones rebuilt as fixtures, and those fixtures would drift from the routes they imitate.

    Structure is seeded directly so this test fails for its own reasons rather than
    for generation's. Everything from the term onwards is exercised over HTTP.
    """
    _seed_academic_year()
    _seed_generated_structure()

    # -- term and subjects --------------------------------------------------
    term = client.post(
        "/v1/terms",
        json={
            "code": TERM_CODE,
            "academic_year_code": YEAR_CODE,
            "name_en": "Term 1",
            "name_ar": "الفصل الأول",
            "starts_on": "2025-09-01",
            "ends_on": "2025-12-15",
            "sequence": 1,
        },
        headers=registrar,
    )
    assert term.status_code == 201, term.text

    for code, name_en, name_ar in (
        ("MATH", "Mathematics", "الرياضيات"),
        ("SCI", "Science", "العلوم"),
    ):
        created = client.post(
            "/v1/subjects",
            json={"code": code, "name_en": name_en, "name_ar": name_ar},
            headers=registrar,
        )
        assert created.status_code == 201, created.text

    # -- roster: preview writes nothing, commit writes everything ----------
    preview = client.post(
        "/v1/imports/roster/preview",
        files=_csv_upload(),
        data={"academic_year_code": YEAR_CODE},
        headers=registrar,
    )
    assert preview.status_code == 200, preview.text
    previewed = preview.json()
    assert previewed["ok_count"] == 2
    assert previewed["rejected_count"] == 0

    # Nothing is written until the batch is committed, which is the whole of invariant 4.
    empty = client.get(
        "/v1/classes/3A/students",
        params={"academic_year": YEAR_CODE, "on": "2025-11-15"},
        headers=registrar,
    )
    assert empty.status_code == 200
    assert empty.json()["count"] == 0

    committed = client.post(
        f"/v1/imports/roster/{previewed['batch_id']}/commit", headers=registrar
    )
    assert committed.status_code == 200, committed.text
    assert committed.json()["ok_count"] == 2

    register = client.get(
        "/v1/classes/3A/students",
        params={"academic_year": YEAR_CODE, "on": "2025-11-15"},
        headers=registrar,
    )
    assert register.status_code == 200
    roll = register.json()
    assert roll["count"] == 2
    assert {entry["student_number"] for entry in roll["students"]} == {"S001", "S002"}

    # -- grades -------------------------------------------------------------
    marks_preview = client.post(
        "/v1/imports/grades/preview",
        files=_csv_upload("grades.csv", GRADES_CSV),
        data={"term_code": TERM_CODE},
        headers=registrar,
    )
    assert marks_preview.status_code == 200, marks_preview.text
    marks = marks_preview.json()
    assert marks["ok_count"] == 3, marks["rows"]
    assert marks["rejected_count"] == 0

    marks_committed = client.post(
        f"/v1/imports/grades/{marks['batch_id']}/commit", headers=registrar
    )
    assert marks_committed.status_code == 200, marks_committed.text
    assert marks_committed.json()["ok_count"] == 3

    # A second commit is refused rather than re-applied — what makes a double-clicked
    # button safe.
    replayed = client.post(
        f"/v1/imports/grades/{marks['batch_id']}/commit", headers=registrar
    )
    assert_error_envelope(replayed, status=409)

    # -- the report card ----------------------------------------------------
    report = client.get(
        "/v1/students/S001/grades", params={"term": TERM_CODE}, headers=registrar
    )
    assert report.status_code == 200, report.text
    card = report.json()

    assert card["student_number"] == "S001"
    assert card["full_name_en"] == "Layla Ahmed"
    assert card["full_name_ar"] == "ليلى أحمد"
    assert card["term_code"] == TERM_CODE
    # Resolved for the term through her placement, not read off a column on the student.
    assert card["class_code"] == "3A"
    assert card["subject_count"] == 2
    assert card["graded_count"] == 1

    lines = {line["subject_code"]: line for line in card["grades"]}
    assert lines["MATH"]["percentage"] == 87.5
    assert lines["MATH"]["is_graded"] is True
    assert lines["MATH"]["class_code"] == "3A"

    # Invariant 1 on the wire: the unmarked subject is `null`, and it is still a line on
    # the card rather than a subject quietly dropped.
    assert lines["SCI"]["percentage"] is None
    assert lines["SCI"]["is_graded"] is False
    assert lines["SCI"]["subject_name_en"] == "Science"

    # Nothing is aggregated (invariant 5): no average, total, GPA or rank is invented.
    assert not {"average", "total", "gpa", "rank"} & card.keys()

    # The other half of invariant 1: an earned zero is a mark, and must not read as a
    # blank the way `if grade.percentage:` would render it.
    zero = client.get(
        "/v1/students/S002/grades", params={"term": TERM_CODE}, headers=registrar
    ).json()
    maths = next(line for line in zero["grades"] if line["subject_code"] == "MATH")
    assert maths["percentage"] == 0.0
    assert maths["is_graded"] is True
    assert zero["graded_count"] == 1


def test_an_unknown_student_is_a_404_in_the_same_envelope(
    client: TestClient, registrar: dict[str, str]
) -> None:
    _seed_academic_year()
    client.post(
        "/v1/terms",
        json={
            "code": TERM_CODE,
            "academic_year_code": YEAR_CODE,
            "name_en": "Term 1",
            "name_ar": "الفصل الأول",
            "starts_on": "2025-09-01",
            "ends_on": "2025-12-15",
        },
        headers=registrar,
    )

    response = client.get(
        "/v1/students/NOBODY/grades", params={"term": TERM_CODE}, headers=registrar
    )

    assert_error_envelope(response, status=404, code="unknown_reference")
