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
from sis.domain.structure import AcademicYear, ClassSection, School, YearLevel
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
