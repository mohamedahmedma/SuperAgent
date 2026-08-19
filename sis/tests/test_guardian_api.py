"""Guardians over HTTP: the whole stack, a real migrated database, through `TestClient`.

What only shows up here is the wiring. Three things in particular have no symptom short
of a response body:

* **The router being mounted at all.** `all_routers()` is an explicit tuple, so a new
  module that nobody adds to it is unreachable in production while every unit test its
  service has still passes. `test_the_router_is_mounted` is the guard.
* **The scope dependency being attached** rather than merely defined — a write route that
  forgets `caller: Registrar` serves every reader key and passes every service test,
  because the service never sees a caller.
* **The unique constraint on a phone.** The fakes model it, but only a real database
  proves the schema agrees, and it is the constraint a future parent login depends on.
"""
import io
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from openpyxl import Workbook

from sis.app import create_app
from sis.config import reset_settings_cache
from sis.domain.structure import AcademicYear, ClassSection, YearLevel
from sis.infrastructure.db.session import reset_engine
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

_ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"

BOOTSTRAP_KEY = "bootstrap-registrar-key-0123456789abcdef"
YEAR_CODE = "2025-2026"
CLASS_CODE = "3A"

# The family the whole feature exists for: a mother with two numbers who is on *both*
# children, a father, and a big brother the school has restricted. Line 6 names a child
# who is not on the roll and line 7 a number that cannot be dialled — one of each kind of
# rejection, so "one bad row does not discard the good ones" is actually exercised.
GUARDIANS_CSV = (
    "student_number,guardian name (arabic),phone,alt phone,relationship,can view records\n"
    "S001,فاطمة علي,01001234567,01119998888,mother,\n"
    "S001,حسن محمود,01002223333,,أب,\n"
    "S001,كريم حسن,0100 555 4444,,big brother,no\n"
    "S002,فاطمة علي,01001234567,,mother,\n"
    "S999,مجهول,01007776666,,mother,\n"
    "S001,سيء,not a phone,,mother,\n"
).encode("utf-8")


@pytest.fixture
def sis_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """An empty, migrated database, with the service pointed at it."""
    url = f"sqlite:///{(tmp_path / 'sis.db').as_posix()}"
    monkeypatch.setenv("SIS_DATABASE_URL", url)
    monkeypatch.setenv("SIS_BOOTSTRAP_REGISTRAR_KEY", BOOTSTRAP_KEY)
    # Pinned rather than left to the environment: these tests assert exact E.164 output,
    # and a deployment default leaking in would make them pass or fail by locale.
    monkeypatch.setenv("SIS_DEFAULT_COUNTRY_CODE", "+20")
    reset_settings_cache()
    reset_engine()

    command.upgrade(Config(str(_ALEMBIC_INI)), "head")
    yield

    reset_engine()
    reset_settings_cache()


@pytest.fixture
def client(sis_database: None) -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def registrar() -> dict[str, str]:
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
    return {"X-API-Key": response.json()["api_key"]}


@pytest.fixture
def roll(client: TestClient, registrar: dict[str, str]) -> None:
    """Two children on the roll. Guardians attach to students; they never create them."""
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
        uow.year_levels.upsert_many(
            [YearLevel(code="3", name_en="Year 3", name_ar="السنة 3", display_order=3)]
        )
        uow.class_sections.upsert_many(
            [
                ClassSection(
                    code=CLASS_CODE,
                    academic_year_code=YEAR_CODE,
                    year_level_code="3",
                    name_en="Year 3 A",
                    name_ar="الثالث أ",
                )
            ]
        )
        uow.commit()

    roster = (
        "Student Number,Arabic Name,English Name\n"
        "S001,ليلى أحمد,Layla Ahmed\n"
        "S002,عمر خالد,Omar Khaled\n"
    ).encode("utf-8")
    preview = client.post(
        "/v1/imports/roster/preview",
        files={"file": ("roster.csv", roster, "text/csv")},
        data={"academic_year_code": YEAR_CODE, "class_code": CLASS_CODE},
        headers=registrar,
    )
    assert preview.status_code == 200, preview.text
    committed = client.post(
        f"/v1/imports/roster/{preview.json()['batch_id']}/commit", headers=registrar
    )
    assert committed.status_code == 200, committed.text


def _upload(
    client: TestClient,
    headers: dict[str, str],
    content: bytes = GUARDIANS_CSV,
    filename: str = "guardians.csv",
) -> dict:
    """Preview and commit one guardians file, asserting both halves succeeded.

    `filename` is a parameter because the reader is chosen by extension: handing xlsx
    bytes to a name ending `.csv` is refused, correctly and confusingly.
    """
    preview = client.post(
        "/v1/imports/guardians/preview",
        files={"file": (filename, content, "text/csv")},
        headers=headers,
    )
    assert preview.status_code == 200, preview.text
    commit = client.post(
        f"/v1/imports/guardians/{preview.json()['batch_id']}/commit", headers=headers
    )
    assert commit.status_code == 200, commit.text
    return {"preview": preview.json(), "commit": commit.json()}


def test_the_router_is_mounted(client: TestClient) -> None:
    """`all_routers()` is a hand-written tuple; a module left out of it 404s in production.

    Asserted against the app's own route table rather than by calling the endpoint,
    because a 404 from an unmounted router and a 404 from an unknown student are the same
    status and this must fail for only one of those reasons.
    """
    paths = {route.path for route in client.app.routes}
    assert "/v1/students/{student_number}/guardians" in paths
    assert "/v1/guardians/{phone}/students" in paths
    assert "/v1/imports/guardians/preview" in paths
    assert "/v1/imports/guardians/{batch_id}/commit" in paths


def test_preview_writes_nothing(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """The promise the two-step flow rests on: a preview is a report, not a write."""
    preview = client.post(
        "/v1/imports/guardians/preview",
        files={"file": ("guardians.csv", GUARDIANS_CSV, "text/csv")},
        headers=registrar,
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["ok_count"] == 4

    after = client.get("/v1/students/S001/guardians", headers=registrar)
    assert after.status_code == 200
    assert after.json()["count"] == 0


def test_one_bad_row_does_not_discard_the_good_ones(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """Four rows land, two are refused, and each refusal names its own reason."""
    result = _upload(client, registrar)
    codes = {row["line"]: row["code"] for row in result["commit"]["rows"]}

    assert result["commit"]["ok_count"] == 4
    assert codes[6] == "unknown_student"  # names a child who is not on the roll
    assert codes[7] == "missing_phone"  # a number that cannot be dialled


def test_a_child_has_every_guardian_the_file_named(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """Mother, father and big brother on one child — the shape a phone column cannot hold."""
    _upload(client, registrar)
    body = client.get("/v1/students/S001/guardians", headers=registrar).json()

    assert body["count"] == 3
    by_phone = {row["phone"]: row for row in body["guardians"]}
    assert set(by_phone) == {"+201001234567", "+201002223333", "+201005554444"}

    mother = by_phone["+201001234567"]
    assert mother["relationship_type"] == "mother"
    # Both her numbers, primary first -- the whole reason phones are their own table.
    assert mother["phones"] == ["+201001234567", "+201119998888"]

    # "أب" is bucketed by the bilingual synonym table, not stored as typed.
    assert by_phone["+201002223333"]["relationship_type"] == "father"

    brother = by_phone["+201005554444"]
    assert brother["relationship_type"] == "sibling"
    # Closing the vocabulary costs nothing a human typed.
    assert brother["relationship_label"] == "big brother"
    # `can view records` said "no", so the grant the sheet otherwise implies is withheld.
    assert brother["can_view_records"] is False


def test_one_mother_across_two_children_is_one_guardian(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """The deduplication the whole normalisation exists for.

    She appears on two rows of the file. If the two spellings of her number failed to
    collide she would become two people, each holding one of her children — and the
    unique constraint on a phone would have refused the second write outright.
    """
    _upload(client, registrar)
    seen = client.get("/v1/guardians/+201001234567/students", headers=registrar).json()

    assert seen["count"] == 2
    assert {row["student_number"] for row in seen["students"]} == {"S001", "S002"}


def test_a_guardian_is_found_by_her_second_number(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """What makes the alternate number useful rather than decorative.

    A parent who one day verifies the WhatsApp line she gave the school must reach the
    same children as one who verifies the mobile.
    """
    _upload(client, registrar)
    by_alt = client.get("/v1/guardians/+201119998888/students", headers=registrar).json()

    assert by_alt["count"] == 2
    assert by_alt["full_name_ar"] == "فاطمة علي"


def test_a_restricted_guardian_reads_nothing(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """On the contact list, and barred from the records. Both facts, kept separately."""
    _upload(client, registrar)

    visible = client.get("/v1/guardians/+201005554444/students", headers=registrar)
    assert visible.json()["count"] == 0

    # Still on file: a restriction removes the reading, never the contact.
    everything = client.get(
        "/v1/guardians/+201005554444/students",
        params={"include_restricted": True},
        headers=registrar,
    )
    assert everything.json()["count"] == 1


def test_re_uploading_the_same_file_creates_no_duplicates(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """Idempotence. A registrar who uploads twice has not doubled every contact list."""
    _upload(client, registrar)
    second = _upload(client, registrar)

    assert client.get("/v1/students/S001/guardians", headers=registrar).json()["count"] == 3
    assert client.get("/v1/students/S002/guardians", headers=registrar).json()["count"] == 1
    # Reported as unchanged rather than created -- the count a registrar reads to decide
    # whether the second upload did anything.
    assert second["commit"]["totals"].get("ok") == 4


def test_a_phone_belonging_to_someone_else_is_refused(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """A recycled number must not silently inherit the previous family's records.

    This is the one refusal a registrar might not expect, so it earns a test: after OTP
    login exists, accepting it hands one family's grades to whoever now answers the phone.
    """
    _upload(client, registrar)
    # The name is in Arabic because the stored guardian's is: names are compared only
    # where both sides state one in the *same* script, so an English name here would be
    # incomparable against an Arabic-only record and would merge rather than conflict.
    # That is the documented rule and the deliberate cost of accepting Arabic-only sheets
    # (see `_names_collide`); this test exercises the rule, not the gap.
    stolen = (
        "student_number,guardian name (arabic),phone,relationship\n"
        "S002,شخص آخر تماما,01001234567,mother\n"
    ).encode("utf-8")

    preview = client.post(
        "/v1/imports/guardians/preview",
        files={"file": ("stolen.csv", stolen, "text/csv")},
        headers=registrar,
    )
    rows = preview.json()["rows"]
    assert [row["code"] for row in rows] == ["duplicate_existing"]
    assert "فاطمة علي" in rows[0]["message"]


def test_access_can_be_revoked_without_an_upload(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """The urgent custody path. A court order arrives and the office has to act now."""
    _upload(client, registrar)
    assert client.get("/v1/guardians/+201002223333/students", headers=registrar).json()["count"] == 1

    revoked = client.patch(
        "/v1/students/S001/guardians/+201002223333",
        json={"can_view_records": False, "restriction_note": "court order 2026/114"},
        headers=registrar,
    )
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["can_view_records"] is False

    assert client.get("/v1/guardians/+201002223333/students", headers=registrar).json()["count"] == 0
    # The link survives; only the reading was removed.
    still_listed = client.get("/v1/students/S001/guardians", headers=registrar).json()
    father = next(g for g in still_listed["guardians"] if g["phone"] == "+201002223333")
    assert father["restriction_note"] == "court order 2026/114"


def test_an_unknown_student_is_a_404_not_an_empty_list(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """"No such child" and "no guardians recorded yet" must stay distinguishable.

    The second is the normal state of every child between the two uploads, so answering
    both with an empty list would send a registrar looking for a typo that is not there.
    """
    assert client.get("/v1/students/S404/guardians", headers=registrar).status_code == 404

    known = client.get("/v1/students/S001/guardians", headers=registrar)
    assert known.status_code == 200
    assert known.json()["count"] == 0


def test_a_reader_key_cannot_import_or_change_access(
    client: TestClient, reader: dict[str, str], roll: None
) -> None:
    """Scope is compared by exact equality; a route that forgot its dependency serves all."""
    refused = client.post(
        "/v1/imports/guardians/preview",
        files={"file": ("guardians.csv", GUARDIANS_CSV, "text/csv")},
        headers=reader,
    )
    assert refused.status_code == 403

    patched = client.patch(
        "/v1/students/S001/guardians/+201001234567",
        json={"can_view_records": False},
        headers=reader,
    )
    assert patched.status_code == 403

    # Reads stay open to both scopes.
    assert client.get("/v1/students/S001/guardians", headers=reader).status_code == 200


def test_a_batch_cannot_be_committed_twice(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """What makes a double-clicked button safe: the first outcome stands."""
    preview = client.post(
        "/v1/imports/guardians/preview",
        files={"file": ("guardians.csv", GUARDIANS_CSV, "text/csv")},
        headers=registrar,
    )
    batch_id = preview.json()["batch_id"]
    assert client.post(f"/v1/imports/guardians/{batch_id}/commit", headers=registrar).status_code == 200

    again = client.post(f"/v1/imports/guardians/{batch_id}/commit", headers=registrar)
    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "batch_already_committed"


def test_a_roster_batch_cannot_be_committed_as_guardians(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """Kind is checked, never inferred. Committing the wrong batch writes the wrong table."""
    roster = "Student Number,English Name\nS003,Nadia Samir\n".encode("utf-8")
    preview = client.post(
        "/v1/imports/roster/preview",
        files={"file": ("roster.csv", roster, "text/csv")},
        data={"academic_year_code": YEAR_CODE, "class_code": CLASS_CODE},
        headers=registrar,
    )
    batch_id = preview.json()["batch_id"]

    wrong = client.post(f"/v1/imports/guardians/{batch_id}/commit", headers=registrar)
    assert wrong.status_code == 409
    assert wrong.json()["detail"]["code"] == "content_mismatch"


def test_an_excel_mangled_phone_still_reaches_the_parent(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """Excel types a phone column as a number and eats the leading zero.

    `01001234567` arrives as `1001234567.0`. Left alone it is a number that reaches
    nobody, and nothing downstream can tell it was ever wrong — so the recovery is
    asserted through a real .xlsx rather than trusted to the unit test of `Phone`.
    """
    book = Workbook()
    sheet = book.active
    sheet.append(["student_number", "guardian name (english)", "phone", "relationship"])
    sheet.append(["S001", "Fatma Ali", 1001234567, "mother"])
    buffer = io.BytesIO()
    book.save(buffer)

    _upload(client, registrar, buffer.getvalue(), filename="guardians.xlsx")
    body = client.get("/v1/students/S001/guardians", headers=registrar).json()
    assert [row["phone"] for row in body["guardians"]] == ["+201001234567"]
