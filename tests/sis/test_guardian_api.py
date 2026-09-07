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
from sis.domain.people import ClassEnrolment
from sis.domain.structure import AcademicYear, ClassSection, School, Term, YearLevel
from sis.domain.value_objects import StudentNumber
from sis.infrastructure.db.session import reset_engine
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.conftest import Clock

_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "sis" / "alembic.ini"

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
def client(sis_database: None, clock: Clock) -> Iterator[TestClient]:
    # Its own app, so the clock has to be pinned on this one rather than on the module
    # singleton the shared fixture pins.
    app = create_app()
    clock.install(app)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def registrar() -> dict[str, str]:
    return {"X-API-Key": BOOTSTRAP_KEY}


@pytest.fixture
def roll(client: TestClient, registrar: dict[str, str]) -> None:
    """Two children on the roll. Guardians attach to students; they never create them."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.schools.upsert_many([School(code="MAIN", name_en="Main School", name_ar="المدرسة")])
        uow.academic_years.upsert_many(
            [
                AcademicYear(
                    code=YEAR_CODE,
                    school_code="MAIN",
                    name_en="2025-2026",
                    name_ar="٢٠٢٥-٢٠٢٦",
                    starts_on=date(2025, 9, 1),
                    ends_on=date(2026, 6, 30),
                    is_current=True,
                )
            ]
        )
        uow.year_levels.upsert_many(
            [YearLevel(code="3", school_code="MAIN", name_en="Year 3", name_ar="السنة 3", display_order=3)]
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


def test_a_number_resolves_to_a_stable_handle(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """What an authentication service calls once it has proved somebody holds a number.

    It gets back a handle and no phone number, which is the whole point: the caller stores
    something opaque and permanent instead of PII it would then have to protect.
    """
    _upload(client, registrar)

    resolved = client.post(
        "/v1/guardians/resolve", json={"phone": "+201001234567"}, headers=registrar
    )
    assert resolved.status_code == 200, resolved.text
    body = resolved.json()
    assert body["full_name_ar"] == "فاطمة علي"
    assert body["public_id"]
    # The response must not hand the number back — a caller that stored this whole body
    # would be storing the PII the handle exists to avoid.
    assert "phone" not in body


def test_the_handle_is_the_same_through_either_of_her_numbers(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """A parent who verifies her WhatsApp line is the same person as one who verifies her
    mobile, and must resolve to one account rather than two."""
    _upload(client, registrar)

    first = client.post(
        "/v1/guardians/resolve", json={"phone": "+201001234567"}, headers=registrar
    ).json()
    second = client.post(
        "/v1/guardians/resolve", json={"phone": "+201119998888"}, headers=registrar
    ).json()

    assert first["public_id"] == second["public_id"]


def test_the_handle_survives_a_re_upload(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """It is stored, not derived. A handle that changed on re-import would silently
    detach every account bound to it."""
    _upload(client, registrar)
    before = client.post(
        "/v1/guardians/resolve", json={"phone": "+201001234567"}, headers=registrar
    ).json()["public_id"]

    _upload(client, registrar)
    after = client.post(
        "/v1/guardians/resolve", json={"phone": "+201001234567"}, headers=registrar
    ).json()["public_id"]

    assert before == after


def test_a_national_format_number_resolves(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """The caller holds whatever a parent typed and should not have to know the rules."""
    _upload(client, registrar)

    resolved = client.post(
        "/v1/guardians/resolve", json={"phone": "0100 123 4567"}, headers=registrar
    )
    assert resolved.status_code == 200, resolved.text


def test_an_unknown_number_is_a_404(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """An ordinary answer, not an error: most numbers in the world are not this school's.

    It has to be reachable without an exception, because the authentication service asks
    this question about every number that messages it.
    """
    _upload(client, registrar)

    missing = client.post(
        "/v1/guardians/resolve", json={"phone": "+201119990000"}, headers=registrar
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "unknown_reference"


def test_an_unusable_number_is_refused_rather_than_resolved(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """422, not 404. "That is not a phone number" and "that number is not a parent here"
    are different answers and the caller acts differently on each."""
    refused = client.post(
        "/v1/guardians/resolve", json={"phone": "not a phone"}, headers=registrar
    )
    assert refused.status_code == 422


def test_a_handle_lists_her_children_without_naming_her_number(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """The question a parent-facing service actually asks.

    It is handed a handle when the parent signs in, and never the number — so the phone
    stays out of a process that runs a language model over untrusted input, out of its
    logs, and out of its memory.
    """
    _upload(client, registrar)
    handle = client.post(
        "/v1/guardians/resolve", json={"phone": "+201001234567"}, headers=registrar
    ).json()["public_id"]

    seen = client.get(f"/v1/guardians/by-id/{handle}/students", headers=registrar)
    assert seen.status_code == 200, seen.text
    body = seen.json()

    assert {row["student_number"] for row in body["students"]} == {"S001", "S002"}
    assert body["full_name_ar"] == "فاطمة علي"
    # The number is not handed back to a caller that only ever knew the handle.
    assert body["phone"] == ""


def test_the_handle_answer_matches_the_phone_answer(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """Two ways of asking one question must not drift apart."""
    _upload(client, registrar)
    handle = client.post(
        "/v1/guardians/resolve", json={"phone": "+201001234567"}, headers=registrar
    ).json()["public_id"]

    by_phone = client.get("/v1/guardians/+201001234567/students", headers=registrar).json()
    by_handle = client.get(f"/v1/guardians/by-id/{handle}/students", headers=registrar).json()

    assert by_phone["count"] == by_handle["count"]
    assert [s["student_number"] for s in by_phone["students"]] == [
        s["student_number"] for s in by_handle["students"]
    ]


def test_a_restricted_child_is_absent_from_the_handle_answer(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """The custody restriction has to hold on the route the chatbot actually calls.

    A rule enforced on one of two equivalent routes is a rule that will be bypassed by
    whichever caller happens to use the other.
    """
    _upload(client, registrar)
    handle = client.post(
        "/v1/guardians/resolve", json={"phone": "+201005554444"}, headers=registrar
    ).json()["public_id"]

    visible = client.get(f"/v1/guardians/by-id/{handle}/students", headers=registrar).json()
    assert visible["count"] == 0

    everything = client.get(
        f"/v1/guardians/by-id/{handle}/students",
        params={"include_restricted": True},
        headers=registrar,
    ).json()
    assert everything["count"] == 1


def test_an_unknown_handle_is_a_404(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """"Not a guardian" must stay distinguishable from "a guardian who may see nobody" —
    the second is what a custody restriction looks like, and it must not read as a broken
    token."""
    missing = client.get("/v1/guardians/by-id/not-a-real-handle/students", headers=registrar)
    assert missing.status_code == 404


def _handle_for(client: TestClient, headers: dict[str, str], phone: str) -> str:
    return client.post(
        "/v1/guardians/resolve", json={"phone": phone}, headers=headers
    ).json()["public_id"]


def test_a_guardian_may_read_her_own_child_s_marks(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """The read the chatbot actually performs once it knows which child."""
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    marks = client.get(
        f"/v1/guardians/by-id/{handle}/students/S001/grades",
        params={"term": "2026-T1"},
        headers=registrar,
    )
    # 404 only because this fixture seeds no term; what matters is that the guardian check
    # passed rather than refusing her outright.
    assert marks.status_code in (200, 404)
    if marks.status_code == 404:
        assert marks.json()["detail"]["field"] != "student_number"


def test_a_guardian_cannot_read_a_child_who_is_not_hers(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """The check that makes this route worth having.

    The chat service filters to a parent's own children before asking — but it is a
    process running a language model over text a stranger can write, so its filtering is a
    convenience and not a boundary. A prompt that talks the model into naming another
    child has to meet a server that says no.
    """
    _upload(client, registrar)
    # The big brother, whose access the sheet restricted, is a guardian of S001 only.
    brother = _handle_for(client, registrar, "+201005554444")

    refused = client.get(
        f"/v1/guardians/by-id/{brother}/students/S002/grades",
        params={"term": "2026-T1"},
        headers=registrar,
    )
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "student_number"


def test_a_restricted_guardian_is_refused_her_own_linked_child(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """A custody restriction has to hold on the grades route, not only on the list.

    The brother IS linked to S001 — the sheet said `can view records: no`. A rule enforced
    only where children are listed is a rule bypassed by asking for the marks directly.
    """
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    refused = client.get(
        f"/v1/guardians/by-id/{brother}/students/S001/grades",
        params={"term": "2026-T1"},
        headers=registrar,
    )
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "student_number"


def test_an_unknown_child_and_someone_else_s_child_look_identical(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """Otherwise a caller could walk student numbers and learn which ones exist."""
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    not_hers = client.get(
        f"/v1/guardians/by-id/{brother}/students/S002/grades",
        params={"term": "2026-T1"}, headers=registrar,
    ).json()["detail"]
    no_such = client.get(
        f"/v1/guardians/by-id/{brother}/students/S999/grades",
        params={"term": "2026-T1"}, headers=registrar,
    ).json()["detail"]

    assert not_hers["code"] == no_such["code"]
    assert not_hers["message"] == no_such["message"]


# ---------------------------------------------------------------------------
# The same guard, on attendance
# ---------------------------------------------------------------------------
#
# Grades had a guardian-scoped route and attendance did not, so anything asking a parent
# "how many days has she missed" had to reach the registrar route and be trusted to have
# filtered first. These assert that the second route enforces exactly what the first does —
# written as near-copies on purpose, because the failure worth catching is the two drifting.


def test_a_guardian_may_read_her_own_child_s_attendance(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    record = client.get(
        f"/v1/guardians/by-id/{handle}/students/S001/attendance", headers=registrar
    )
    assert record.status_code == 200, record.text
    assert record.json()["student_number"] == "S001"


def test_a_guardian_cannot_read_the_attendance_of_a_child_who_is_not_hers(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    refused = client.get(
        f"/v1/guardians/by-id/{brother}/students/S002/attendance", headers=registrar
    )
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "student_number"


def test_a_restricted_guardian_is_refused_her_own_linked_child_s_attendance(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    """The custody restriction has to hold here too.

    The brother IS linked to S001; the sheet said `can view records: no`. Whether a child
    was in school on Tuesday is exactly the kind of thing a court order bars an adult from
    being told, and a rule enforced only on grades is a rule with a door beside it.
    """
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    refused = client.get(
        f"/v1/guardians/by-id/{brother}/students/S001/attendance", headers=registrar
    )
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "student_number"


def test_an_unknown_child_and_someone_else_s_child_look_identical_on_attendance(
    client: TestClient, registrar: dict[str, str], roll: None
) -> None:
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    not_hers = client.get(
        f"/v1/guardians/by-id/{brother}/students/S002/attendance", headers=registrar
    ).json()["detail"]
    no_such = client.get(
        f"/v1/guardians/by-id/{brother}/students/S999/attendance", headers=registrar
    ).json()["detail"]

    assert not_hers["code"] == no_such["code"]
    assert not_hers["message"] == no_such["message"]


# ---------------------------------------------------------------------------
# The same guard, on the timetable
# ---------------------------------------------------------------------------
#
# Third near-copy of the block above, for the same reason the second one exists: the rule
# is only a rule where it is enforced, and a parent-facing route added without it is the
# door beside the lock. Written as copies deliberately — the failure worth catching is the
# three drifting apart.
#
# What is *not* a copy is the class. Grades and attendance are facts about a child and are
# keyed on her number; a timetable is a fact about a ROOM, and a child reaches one only
# through the placement she holds for the term. So these also assert the resolution itself
# — that the room is the one she sat in for the term asked about, and that having no room
# is an answer rather than a missing record.

FIRST_TERM = "2026-T1"
SECOND_TERM = "2026-T2"


@pytest.fixture
def week(client: TestClient, registrar: dict[str, str], roll: None) -> None:
    """Two dated terms, a second class, a bell schedule, and one lesson in 3A's Sunday.

    Dated terms are the point of the fixture. `resolve_section_for_term` asks about a
    term's last day and falls back to its first, so a term with no dates would resolve
    against the whole year and make the transfer assertion below vacuous.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.terms.upsert_many(
            [
                Term(
                    code=FIRST_TERM,
                    academic_year_code=YEAR_CODE,
                    name_en="Term 1",
                    name_ar="الفصل الأول",
                    starts_on=date(2025, 9, 1),
                    ends_on=date(2025, 12, 15),
                    sequence=1,
                ),
                Term(
                    code=SECOND_TERM,
                    academic_year_code=YEAR_CODE,
                    name_en="Term 2",
                    name_ar="الفصل الثاني",
                    starts_on=date(2026, 1, 5),
                    ends_on=date(2026, 6, 30),
                    sequence=2,
                ),
            ]
        )
        uow.class_sections.upsert_many(
            [
                ClassSection(
                    code="3B",
                    academic_year_code=YEAR_CODE,
                    year_level_code="3",
                    name_en="Year 3 B",
                    name_ar="الثالث ب",
                )
            ]
        )
        uow.commit()

    assert client.put(
        "/v1/schools/MAIN/timetable-periods",
        json={
            "periods": [
                {"period_number": 1, "name_en": "Period 1", "name_ar": "حصة ١"},
                {"period_number": 2, "name_en": "Break", "name_ar": "فسحة", "is_teaching": False},
                {"period_number": 3, "name_en": "Period 3", "name_ar": "حصة ٣"},
            ]
        },
        headers=registrar,
    ).status_code == 200

    assert client.post(
        "/v1/subjects",
        json={
            "code": "MATH",
            "academic_year_code": YEAR_CODE,
            "name_en": "Mathematics",
            "name_ar": "الرياضيات",
        },
        headers=registrar,
    ).status_code == 201
    assert client.put(
        "/v1/subject-assignments",
        json={
            "academic_year_code": YEAR_CODE,
            "subject_code": "MATH",
            "year_level_code": "3",
            "assigned": True,
        },
        headers=registrar,
    ).status_code == 204

    # 3A sits maths on Sunday; 3B sits it on Monday. Two rooms with different weeks, so
    # "which room did we resolve" is answerable from the lessons alone.
    assert client.put(
        "/v1/timetable",
        json={
            "academic_year_code": YEAR_CODE,
            "entries": [
                {
                    "class_code": "3A",
                    "term_code": FIRST_TERM,
                    "day_of_week": "sunday",
                    "period_number": 1,
                    "subject_code": "MATH",
                },
                {
                    "class_code": "3A",
                    "term_code": FIRST_TERM,
                    "day_of_week": "monday",
                    "period_number": 3,
                    "subject_code": None,
                },
                {
                    "class_code": "3B",
                    "term_code": SECOND_TERM,
                    "day_of_week": "monday",
                    "period_number": 1,
                    "subject_code": "MATH",
                },
            ],
        },
        headers=registrar,
    ).status_code == 200


def _timetable(
    client: TestClient, headers: dict[str, str], handle: str, student: str, term: str
):
    return client.get(
        f"/v1/guardians/by-id/{handle}/students/{student}/timetable",
        params={"term": term},
        headers=headers,
    )


def test_a_guardian_may_read_her_own_child_s_timetable(
    client: TestClient, registrar: dict[str, str], week: None
) -> None:
    """The read the chatbot performs, and the one thing it must not have to supply: a class.

    A parent has no class code and the chat service has no business holding one. It asks
    about a child; the room is resolved here.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    answer = _timetable(client, registrar, handle, "S001", FIRST_TERM)
    assert answer.status_code == 200, answer.text
    body = answer.json()

    assert body["student_number"] == "S001"
    assert body["class_code"] == "3A"
    assert body["class_name_ar"] == "الثالث أ"
    # The school's own week, in the school's own order, and never sorted alphabetically.
    assert body["days"] == ["sunday", "monday", "tuesday", "wednesday", "thursday"]
    # The break travels with the grid: a client cannot draw the day without it.
    assert [p["period_number"] for p in body["periods"]] == [1, 2, 3]
    assert body["periods"][1]["is_teaching"] is False
    # Two teaching periods across five open days.
    assert body["teaching_slots"] == 10

    lessons = body["lessons"]
    assert [(l["day_of_week"], l["period_number"]) for l in lessons] == [
        ("sunday", 1),
        ("monday", 3),
    ]
    # The names a parent reads, not the key the school files under.
    assert lessons[0]["subject_name_ar"] == "الرياضيات"
    assert lessons[0]["subject_code"] == "MATH"
    # A stated free period stays distinguishable from a slot nobody planned: it has a row.
    assert lessons[1]["subject_code"] is None
    assert lessons[1]["subject_name_ar"] == ""


def test_the_timetable_is_the_class_she_sat_in_for_that_term(
    client: TestClient, registrar: dict[str, str], week: None
) -> None:
    """Invariant 2, reached through the parent-facing route.

    S001 moves 3A -> 3B over the winter. Term 1 must keep answering 3A's week — that is
    the week she actually sat — while Term 2 answers 3B's. A route that read her *current*
    placement would reprint January's room over an autumn a parent is asking about.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    with SqlAlchemyUnitOfWork() as uow:
        uow.enrolments.close_open_enrolment(
            StudentNumber("S001"), ends_on=date(2025, 12, 31)
        )
        uow.enrolments.upsert_many(
            [
                ClassEnrolment(
                    student_number="S001",
                    academic_year_code=YEAR_CODE,
                    class_code="3B",
                    starts_on=date(2026, 1, 1),
                )
            ]
        )
        uow.commit()

    autumn = _timetable(client, registrar, handle, "S001", FIRST_TERM).json()
    spring = _timetable(client, registrar, handle, "S001", SECOND_TERM).json()

    assert autumn["class_code"] == "3A"
    assert [(l["day_of_week"], l["period_number"]) for l in autumn["lessons"]] == [
        ("sunday", 1),
        ("monday", 3),
    ]

    assert spring["class_code"] == "3B"
    assert [(l["day_of_week"], l["period_number"]) for l in spring["lessons"]] == [
        ("monday", 1)
    ]


def test_a_class_with_no_grid_yet_is_not_the_same_as_no_class(
    client: TestClient, registrar: dict[str, str], week: None
) -> None:
    """The two empty answers a parent must never be given interchangeably.

    S001 has a room in Term 2 and nobody has laid out its week — a fact about the school.
    A child with no placement at all has no room to ask about — a fact about the child.
    Rendered identically, the first tells a parent her daughter has no lessons when the
    truth is that nobody has typed them in yet.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    # She is in 3A all year in this fixture, and 3A has no Term 2 lessons.
    no_grid = _timetable(client, registrar, handle, "S001", SECOND_TERM).json()
    assert no_grid["class_code"] == "3A"
    assert no_grid["lessons"] == []
    # The grid itself is still there, so a client can draw an empty week rather than nothing.
    assert [p["period_number"] for p in no_grid["periods"]] == [1, 2, 3]

    with SqlAlchemyUnitOfWork() as uow:
        uow.enrolments.close_open_enrolment(
            StudentNumber("S002"), ends_on=date(2025, 9, 2)
        )
        uow.commit()

    no_class = _timetable(client, registrar, handle, "S002", SECOND_TERM)
    assert no_class.status_code == 200, no_class.text
    assert no_class.json()["class_code"] is None
    assert no_class.json()["academic_year_code"] is None
    assert no_class.json()["lessons"] == []


def test_a_guardian_cannot_read_the_timetable_of_a_child_who_is_not_hers(
    client: TestClient, registrar: dict[str, str], week: None
) -> None:
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    refused = _timetable(client, registrar, brother, "S002", FIRST_TERM)
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "student_number"


def test_a_restricted_guardian_is_refused_her_own_linked_child_s_timetable(
    client: TestClient, registrar: dict[str, str], week: None
) -> None:
    """The custody restriction has to hold here too.

    The brother IS linked to S001; the sheet said `can view records: no`. Where a child is
    at eleven on Tuesday is, if anything, the most sensitive of the three reads to hand an
    adult a court order has barred.
    """
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    refused = _timetable(client, registrar, brother, "S001", FIRST_TERM)
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "student_number"


def test_an_unknown_child_and_someone_else_s_child_look_identical_on_the_timetable(
    client: TestClient, registrar: dict[str, str], week: None
) -> None:
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    not_hers = _timetable(client, registrar, brother, "S002", FIRST_TERM).json()["detail"]
    no_such = _timetable(client, registrar, brother, "S999", FIRST_TERM).json()["detail"]

    assert not_hers["code"] == no_such["code"]
    assert not_hers["message"] == no_such["message"]


# ---------------------------------------------------------------------------
# The same guard, on the classroom
# ---------------------------------------------------------------------------
#
# Fourth near-copy of the refusal block, for the reason the second and third exist: a rule
# holds only where it is enforced. What is NOT a copy is everything above the refusals —
# this route answers three questions at once (which room, which subjects, which teachers)
# and each has its own way of being wrong.
#
# The staffing assertions carry the weight here. `teacher_class_sections` is unique on
# (teacher, class, subject) and only one of its two write paths checks for a second teacher
# on one subject, so "exactly one teacher per subject" is an assumption the schema does not
# support — and the payload must not quietly hold a co-teacher back. It is also the first
# parent-facing route to touch staff data at all, so the absence of a teacher's contact
# details is asserted rather than trusted to the response model staying lean.


def _teacher(
    client: TestClient,
    headers: dict[str, str],
    staff_number: str,
    *,
    name_ar: str,
    name_en: str,
    subject: str,
    classes: list[str],
    is_active: bool = True,
):
    """Create a teacher and place them in rooms, through the school-manager route.

    Through the API rather than the repository because the assignment is the thing under
    test at one remove: a fixture that wrote the rows directly could seed a shape the real
    write path cannot produce, and then this suite would be asserting against fiction.
    """
    return client.put(
        f"/v1/schools/MAIN/teachers/{staff_number}",
        json={
            "full_name_ar": name_ar,
            "full_name_en": name_en,
            "email": "staff@example.test",
            "phone": "+201000000000",
            "is_active": is_active,
            "assignments": [
                {
                    "academic_year_code": YEAR_CODE,
                    "subject_code": subject,
                    "year_level_code": "3",
                    "class_codes": classes,
                }
            ],
        },
        headers=headers,
    )


@pytest.fixture
def classroom(client: TestClient, registrar: dict[str, str], week: None) -> None:
    """Two subjects on the rung, and four teachers arranged to catch four mistakes.

    `week` already put MATH on rung 3 and created 3A and 3B. This adds SCI, then:

      T-MATH   MATH in 3A          the ordinary case
      T-SCI    SCI  in 3A          a second subject, so ordering is observable
      T-CO     SCI  in 3A          a CO-TEACHER — the row the schema allows and one write
                                   path forbids, which must not be silently dropped
      T-OTHER  MATH in 3B          another room, which must not leak into 3A's answer
      T-GONE   MATH in 3A inactive left the school, and must not be offered to a parent
    """
    assert client.post(
        "/v1/subjects",
        json={
            "code": "SCI",
            "academic_year_code": YEAR_CODE,
            "name_en": "Science",
            "name_ar": "العلوم",
        },
        headers=registrar,
    ).status_code == 201
    assert client.put(
        "/v1/subject-assignments",
        json={
            "academic_year_code": YEAR_CODE,
            "subject_code": "SCI",
            "year_level_code": "3",
            "assigned": True,
        },
        headers=registrar,
    ).status_code == 204

    for staff, name_ar, name_en, subject, classes, active in (
        ("T-MATH", "أ. سامي", "Mr Sami", "MATH", ["3A"], True),
        ("T-SCI", "أ. هدى", "Ms Huda", "SCI", ["3A"], True),
        ("T-CO", "أ. منى", "Ms Mona", "SCI", ["3A"], True),
        ("T-OTHER", "أ. خالد", "Mr Khaled", "MATH", ["3B"], True),
        ("T-GONE", "أ. فريد", "Mr Farid", "MATH", ["3A"], False),
    ):
        created = _teacher(
            client, registrar, staff,
            name_ar=name_ar, name_en=name_en, subject=subject,
            classes=classes, is_active=active,
        )
        assert created.status_code == 200, created.text


def _classroom(
    client: TestClient, headers: dict[str, str], handle: str, student: str, term: str
):
    return client.get(
        f"/v1/guardians/by-id/{handle}/students/{student}/classroom",
        params={"term": term},
        headers=headers,
    )


def test_a_guardian_reads_her_child_s_room_its_subjects_and_its_staff(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    """One request, three answers, and the class resolved from the child.

    The parent supplies a student and a term. `3A` on the response is the school having
    resolved her placement — nothing in the request could have named a room.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    answer = _classroom(client, registrar, handle, "S001", FIRST_TERM)
    assert answer.status_code == 200, answer.text
    body = answer.json()

    # Which class — the NAME is the answer a parent recognises; the code is an internal key.
    assert body["class_code"] == "3A"
    assert body["class_name_ar"] == "الثالث أ"
    assert body["year_level_name_ar"] == "السنة 3"

    # Which subjects — the rung's board, in the school's own order.
    assert [s["code"] for s in body["subjects"]] == ["MATH", "SCI"]
    assert [s["name_ar"] for s in body["subjects"]] == ["الرياضيات", "العلوم"]

    # Who teaches them.
    assert {(t["subject_code"], t["full_name_ar"]) for t in body["teachers"]} == {
        ("MATH", "أ. سامي"),
        ("SCI", "أ. هدى"),
        ("SCI", "أ. منى"),
    }


def test_a_second_teacher_of_one_subject_is_not_silently_dropped(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    """The schema permits a co-teacher, so the answer has to admit one.

    `teacher_class_sections` is unique on (teacher, class, subject); two teachers of one
    subject in one room violates nothing, and the guard against it lives in exactly one of
    the two write paths. A response shaped as one teacher per subject — or a client calling
    `.first()` — hides a real person from the parent asking who teaches her daughter.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    teachers = _classroom(client, registrar, handle, "S001", FIRST_TERM).json()["teachers"]

    science = [t["full_name_ar"] for t in teachers if t["subject_code"] == "SCI"]
    assert sorted(science) == ["أ. منى", "أ. هدى"]


def test_another_room_s_teacher_does_not_leak_into_this_one(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    """The read is keyed on the room, so 3B's maths teacher is not 3A's.

    Both rooms are on the same rung and teach the same subject, which is exactly the shape
    a query that filtered by rung instead of by room would answer identically — and wrongly.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    teachers = _classroom(client, registrar, handle, "S001", FIRST_TERM).json()["teachers"]

    assert "أ. خالد" not in {t["full_name_ar"] for t in teachers}


def test_a_teacher_who_has_left_is_not_offered_to_a_parent(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    """Worse than reporting nobody: a parent would go and ask for them by name."""
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    teachers = _classroom(client, registrar, handle, "S001", FIRST_TERM).json()["teachers"]

    assert "أ. فريد" not in {t["full_name_ar"] for t in teachers}


def test_no_staff_contact_details_reach_a_parent(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    """The response model is the privacy boundary, and this is what asserts it.

    Every teacher in the fixture was created WITH an email and a phone, so their absence
    here is a property of the projection rather than of the fixture. A parent needs to know
    who teaches their child, not how to reach a member of staff directly — and the internal
    staff number is no more a parent's business than a database id.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    body = _classroom(client, registrar, handle, "S001", FIRST_TERM).json()

    leaked = {"email", "phone", "staff_number", "username", "user_id"}
    for teacher in body["teachers"]:
        assert leaked.isdisjoint(teacher.keys()), teacher
    # The values too, not only the keys: a field renamed on the way out would pass the
    # check above while still carrying the number.
    rendered = str(body)
    assert "staff@example.test" not in rendered
    assert "+201000000000" not in rendered


def test_a_child_with_no_placement_has_no_room_rather_than_an_empty_one(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    """`class_code: null` is a fact about the child; empty lists beside a class are not.

    S002 is withdrawn before Term 2 opens, so no placement covers it. That is not a missing
    record — she had left — and it must stay distinguishable from a room whose subjects or
    staffing nobody has entered.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    with SqlAlchemyUnitOfWork() as uow:
        uow.enrolments.close_open_enrolment(
            StudentNumber("S002"), ends_on=date(2025, 9, 2)
        )
        uow.commit()

    gone = _classroom(client, registrar, handle, "S002", SECOND_TERM)
    assert gone.status_code == 200, gone.text
    body = gone.json()
    assert body["class_code"] is None
    assert body["class_name_ar"] is None
    assert body["academic_year_code"] is None
    assert body["subjects"] == []
    assert body["teachers"] == []


def test_the_room_is_the_one_she_sat_in_for_that_term(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    """Invariant 2 again, on the route that answers "which class is my child in".

    A parent asking in June about Term 1 is asking which room those marks were earned in.
    Reading her current placement would rename her autumn.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    with SqlAlchemyUnitOfWork() as uow:
        uow.enrolments.close_open_enrolment(
            StudentNumber("S001"), ends_on=date(2025, 12, 31)
        )
        uow.enrolments.upsert_many(
            [
                ClassEnrolment(
                    student_number="S001",
                    academic_year_code=YEAR_CODE,
                    class_code="3B",
                    starts_on=date(2026, 1, 1),
                )
            ]
        )
        uow.commit()

    autumn = _classroom(client, registrar, handle, "S001", FIRST_TERM).json()
    spring = _classroom(client, registrar, handle, "S001", SECOND_TERM).json()

    assert autumn["class_code"] == "3A"
    assert autumn["class_name_ar"] == "الثالث أ"
    assert spring["class_code"] == "3B"
    # And the staffing follows the room, so the answer changes with it.
    assert "أ. خالد" in {t["full_name_ar"] for t in spring["teachers"]}
    assert "أ. سامي" not in {t["full_name_ar"] for t in spring["teachers"]}


def test_a_guardian_cannot_read_the_classroom_of_a_child_who_is_not_hers(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    refused = _classroom(client, registrar, brother, "S002", FIRST_TERM)
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "student_number"


def test_a_restricted_guardian_is_refused_her_own_linked_child_s_classroom(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    """The court order has to hold on the fourth route as well as the first three."""
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    refused = _classroom(client, registrar, brother, "S001", FIRST_TERM)
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "student_number"


def test_an_unknown_child_and_someone_else_s_child_look_identical_on_the_classroom(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    _upload(client, registrar)
    brother = _handle_for(client, registrar, "+201005554444")

    not_hers = _classroom(client, registrar, brother, "S002", FIRST_TERM).json()["detail"]
    no_such = _classroom(client, registrar, brother, "S999", FIRST_TERM).json()["detail"]

    assert not_hers["code"] == no_such["code"]
    assert not_hers["message"] == no_such["message"]


def test_an_unknown_term_is_refused_on_the_classroom_too(
    client: TestClient, registrar: dict[str, str], classroom: None
) -> None:
    """A typo must not render as a child with no class, no subjects and no teachers."""
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    refused = _classroom(client, registrar, handle, "S001", "not-a-term")
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "term_code"


def test_an_unknown_term_is_refused_rather_than_answered_empty(
    client: TestClient, registrar: dict[str, str], week: None
) -> None:
    """A typo must not render as a week with no lessons in it.

    "There is no such term" and "your daughter has nothing timetabled" are different
    answers, and only one of them is something the caller can fix.
    """
    _upload(client, registrar)
    handle = _handle_for(client, registrar, "+201001234567")

    refused = _timetable(client, registrar, handle, "S001", "not-a-term")
    assert refused.status_code == 404
    assert refused.json()["detail"]["field"] == "term_code"
