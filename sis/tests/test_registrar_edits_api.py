"""The direct-edit surface: one child, one class, one subject, over HTTP.

Every write these routes perform used to require a spreadsheet. The point of the suite is
not that the new routes work — that much a smoke test would show — but that they did not buy
their convenience by giving up the two rules the schema is shaped around:

**Invariant 2, placement is a dated membership.** `test_a_transfer_leaves_term_one_marks_in
_the_old_class` is the one to read first. It moves a child mid-year through the new transfer
route and then asserts her Term 1 report card still says 3A. A direct-edit route that
rewrote `class_section_id` would pass every other test in this file and fail that one, and
the failure it prevents is a completed, marked term re-printing under a class the child had
never entered.

**A subject belongs to a year.** `test_the_same_code_in_two_years_is_two_subjects` pins the
consequence of the change deliberately: `MATH` in two years is two rows, so the catalogue is
per-year and a marks upload resolves against the year of its term. This is the property the
old global catalogue had instead of cross-year comparability, and it is worth a test that
states which one this service chose.

The rest is the ordinary contract: an upsert reports 201 then 200, a PATCH of one field
leaves the others alone, and a child who has left keeps every record attached to her.
"""
from datetime import date

import pytest
from fastapi.testclient import TestClient

from sis.tests.conftest import registrar_headers
from sis.domain.structure import AcademicYear, ClassSection, School, YearLevel
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

YEAR = "2025-2026"
NEXT_YEAR = "2026-2027"
TERM = "2026-T1"


def _seed_two_years_and_two_classes() -> None:
    """Two years, one rung, and 3A/3B inside the first. The floor every test here needs."""
    with SqlAlchemyUnitOfWork() as uow:
        uow.schools.upsert_many([School(code="MAIN", name_en="Main School", name_ar="المدرسة")])
        uow.academic_years.upsert_many(
            [
                AcademicYear(
                    code=YEAR,
                    school_code="MAIN",
                    name_en="2025-2026",
                    name_ar="٢٠٢٥-٢٠٢٦",
                    starts_on=date(2025, 9, 1),
                    ends_on=date(2026, 6, 30),
                    is_current=True,
                ),
                AcademicYear(
                    code=NEXT_YEAR,
                    school_code="MAIN",
                    name_en="2026-2027",
                    name_ar="٢٠٢٦-٢٠٢٧",
                    starts_on=date(2026, 9, 1),
                    ends_on=date(2027, 6, 30),
                    is_current=False,
                ),
            ]
        )
        uow.year_levels.upsert_many(
            [YearLevel(code="3", school_code="MAIN", name_en="Year 3", name_ar="السنة 3", display_order=3)]
        )
        uow.class_sections.upsert_many(
            [
                ClassSection(
                    code="3A",
                    academic_year_code=YEAR,
                    year_level_code="3",
                    name_en="Year 3 A",
                    name_ar="السنة 3 أ",
                ),
                ClassSection(
                    code="3B",
                    academic_year_code=YEAR,
                    year_level_code="3",
                    name_en="Year 3 B",
                    name_ar="السنة 3 ب",
                ),
            ]
        )
        uow.commit()


@pytest.fixture()
def registrar() -> dict[str, str]:
    """The header every call carries, and it is checked again.

    This fixture used to send a literal that nothing verified, with a note that it was
    decoration until the key check came back. It is back: `sis/api/deps.py` authenticates
    a presented key against the school's own `api_keys` table, so this now resolves to the
    stored registrar key the suite seeds and these writes are made by a caller that
    genuinely holds registrar scope.
    """
    return registrar_headers()


@pytest.fixture()
def seeded(client: TestClient) -> TestClient:
    """The client, with structure already on file. `client` migrates; this fills it."""
    _seed_two_years_and_two_classes()
    return client


# ---------------------------------------------------------------------------
# One child
# ---------------------------------------------------------------------------


def test_a_child_is_created_then_corrected_through_the_same_route(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """201 the first time, 200 the second, and the second corrects the name.

    One route for "add" and "fix" is deliberate: a registrar who has to know in advance
    which of the two they are doing will pick wrong, and the wrong pick used to mean either
    a duplicate child or a 409 they had no way to act on.
    """
    created = seeded.post(
        "/v1/students",
        json={
            "student_number": "10432",
            "full_name_ar": "سارة محمد",
            "full_name_en": "Sara Mohamd",  # as typed, with the mistake
        },
        headers=registrar,
    )
    assert created.status_code == 201, created.text

    fixed = seeded.post(
        "/v1/students",
        json={
            "student_number": "10432",
            "full_name_ar": "سارة محمد",
            "full_name_en": "Sara Mohamed",
        },
        headers=registrar,
    )
    assert fixed.status_code == 200, fixed.text
    assert fixed.json()["full_name_en"] == "Sara Mohamed"

    read = seeded.get("/v1/students/10432", headers=registrar)
    assert read.status_code == 200
    assert read.json()["full_name_en"] == "Sara Mohamed"
    assert read.json()["full_name_ar"] == "سارة محمد"


def test_a_patch_of_one_field_leaves_the_others_alone(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """The failure this prevents: correcting the English spelling blanking the Arabic name.

    A PATCH that sent the whole record would do exactly that, because the form the
    registrar edited only had one field on it.
    """
    seeded.post(
        "/v1/students",
        json={
            "student_number": "10432",
            "full_name_ar": "سارة محمد",
            "full_name_en": "Sara Mohamd",
        },
        headers=registrar,
    )

    patched = seeded.patch(
        "/v1/students/10432", json={"full_name_en": "Sara Mohamed"}, headers=registrar
    )
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert body["full_name_en"] == "Sara Mohamed"
    assert body["full_name_ar"] == "سارة محمد"
    assert body["is_active"] is True


def test_patching_an_unknown_child_is_a_404_rather_than_creating_her(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """A PATCH at a number nobody has is a typo, and inventing the child hides it."""
    missing = seeded.patch(
        "/v1/students/99999", json={"full_name_en": "Nobody"}, headers=registrar
    )
    assert missing.status_code == 404, missing.text
    assert missing.json()["detail"]["field"] == "student_number"


def test_a_child_who_leaves_is_deactivated_and_keeps_her_record(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """There is no delete. Her marks and placements are still true statements."""
    seeded.post(
        "/v1/students",
        json={"student_number": "10432", "full_name_en": "Sara", "full_name_ar": "سارة"},
        headers=registrar,
    )
    left = seeded.patch("/v1/students/10432", json={"is_active": False}, headers=registrar)
    assert left.status_code == 200, left.text
    assert left.json()["is_active"] is False

    # Still readable by number...
    assert seeded.get("/v1/students/10432", headers=registrar).status_code == 200
    # ...and out of the picker a registrar uses to place somebody today.
    search = seeded.get("/v1/students?q=10432", headers=registrar)
    assert search.json()["count"] == 0
    with_left = seeded.get(
        "/v1/students?q=10432&include_inactive=true", headers=registrar
    )
    assert with_left.json()["count"] == 1


def test_a_blank_search_returns_nothing_rather_than_the_whole_school(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """The most common accidental request on the screen, answered with silence."""
    for number in ("10432", "10433"):
        seeded.post(
            "/v1/students",
            json={"student_number": number, "full_name_en": "Child", "full_name_ar": "طفل"},
            headers=registrar,
        )
    assert seeded.get("/v1/students?q=", headers=registrar).json()["count"] == 0
    assert seeded.get("/v1/students?q=Child", headers=registrar).json()["count"] == 2


# ---------------------------------------------------------------------------
# Placement, and the invariant it exists to protect
# ---------------------------------------------------------------------------


def _place(client: TestClient, headers: dict[str, str], number: str, code: str, day: date):
    return client.post(
        f"/v1/students/{number}/placements",
        json={
            "academic_year_code": YEAR,
            "class_code": code,
            "starts_on": day.isoformat(),
        },
        headers=headers,
    )


def test_a_child_is_placed_and_appears_on_that_register(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    seeded.post(
        "/v1/students",
        json={"student_number": "10432", "full_name_en": "Sara", "full_name_ar": "سارة"},
        headers=registrar,
    )
    placed = _place(seeded, registrar, "10432", "3A", date(2025, 9, 1))
    assert placed.status_code == 201, placed.text
    assert placed.json()["is_open"] is True

    register = seeded.get(
        f"/v1/classes/3A/students?academic_year={YEAR}&on=2025-10-01", headers=registrar
    )
    assert register.status_code == 200
    assert [row["student_number"] for row in register.json()["students"]] == ["10432"]


def test_a_transfer_leaves_term_one_marks_in_the_old_class(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """The important test in this file.

    A child in 3A from September moves to 3B in March. Afterwards:

      * she is on 3B's register in April,
      * she is *not* on 3A's register in April,
      * she IS still on 3A's register in October, and
      * no single day belongs to both placements.

    A transfer implemented as an UPDATE of the class code would fail the third assertion
    and, worse, would silently rewrite what her Term 1 report card says. That is the
    history rewrite invariant 2 exists to forbid, and this is the route most likely to be
    tempted into it.
    """
    seeded.post(
        "/v1/students",
        json={"student_number": "10432", "full_name_en": "Sara", "full_name_ar": "سارة"},
        headers=registrar,
    )
    _place(seeded, registrar, "10432", "3A", date(2025, 9, 1))

    moved = seeded.post(
        "/v1/students/10432/transfer",
        json={
            "academic_year_code": YEAR,
            "to_class_code": "3B",
            "on_date": "2026-03-01",
        },
        headers=registrar,
    )
    assert moved.status_code == 200, moved.text
    body = moved.json()
    assert body["closed"]["class_code"] == "3A"
    # Her last day in 3A is the day before her first in 3B: no day is in both.
    assert body["closed"]["ends_on"] == "2026-02-28"
    assert body["opened"]["class_code"] == "3B"
    assert body["opened"]["starts_on"] == "2026-03-01"

    def register(code: str, day: str) -> list[str]:
        response = seeded.get(
            f"/v1/classes/{code}/students?academic_year={YEAR}&on={day}", headers=registrar
        )
        assert response.status_code == 200, response.text
        return [row["student_number"] for row in response.json()["students"]]

    assert register("3B", "2026-04-01") == ["10432"]
    assert register("3A", "2026-04-01") == []
    # The assertion that matters: October is still 3A, forever.
    assert register("3A", "2025-10-01") == ["10432"]
    assert register("3B", "2025-10-01") == []

    history = seeded.get("/v1/students/10432/placements", headers=registrar)
    assert history.status_code == 200
    assert history.json()["count"] == 2


def test_ending_a_placement_uses_her_last_day_not_the_day_after(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """Off by one here puts a child on a register for a day she had already left."""
    seeded.post(
        "/v1/students",
        json={"student_number": "10432", "full_name_en": "Sara", "full_name_ar": "سارة"},
        headers=registrar,
    )
    _place(seeded, registrar, "10432", "3A", date(2025, 9, 1))

    ended = seeded.patch(
        "/v1/students/10432/placements/current",
        json={"ends_on": "2026-01-15"},
        headers=registrar,
    )
    assert ended.status_code == 200, ended.text
    assert ended.json()["ends_on"] == "2026-01-15"
    assert ended.json()["is_open"] is False

    def on(day: str) -> list[str]:
        response = seeded.get(
            f"/v1/classes/3A/students?academic_year={YEAR}&on={day}", headers=registrar
        )
        return [row["student_number"] for row in response.json()["students"]]

    assert on("2026-01-15") == ["10432"]  # her last day: still there
    assert on("2026-01-16") == []  # the day after: gone


def test_ending_a_placement_she_does_not_have_is_a_404(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    seeded.post(
        "/v1/students",
        json={"student_number": "10432", "full_name_en": "Sara", "full_name_ar": "سارة"},
        headers=registrar,
    )
    missing = seeded.patch(
        "/v1/students/10432/placements/current",
        json={"ends_on": "2026-01-15"},
        headers=registrar,
    )
    assert missing.status_code == 404, missing.text


# ---------------------------------------------------------------------------
# One class
# ---------------------------------------------------------------------------


def test_one_class_is_added_to_a_year_without_regenerating_it(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """November's extra section, which the generator cannot express without a rebuild."""
    added = seeded.post(
        "/v1/structure/classes",
        json={
            "code": "3C",
            "academic_year_code": YEAR,
            "year_level_code": "3",
            "name_en": "Year 3 C",
            "name_ar": "السنة 3 ج",
            "capacity": 24,
        },
        headers=registrar,
    )
    assert added.status_code == 201, added.text

    listed = seeded.get(f"/v1/structure/classes?academic_year={YEAR}", headers=registrar)
    assert sorted(row["code"] for row in listed.json()) == ["3A", "3B", "3C"]

    again = seeded.post(
        "/v1/structure/classes",
        json={
            "code": "3C",
            "academic_year_code": YEAR,
            "year_level_code": "3",
            "name_en": "Year 3 Falcons",
            "name_ar": "السنة 3 ج",
        },
        headers=registrar,
    )
    assert again.status_code == 200, again.text


def test_a_class_with_no_name_in_either_script_is_refused(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """A nameless class reaches a parent as a report card line with a mark and no class."""
    refused = seeded.post(
        "/v1/structure/classes",
        json={"code": "3Z", "academic_year_code": YEAR, "year_level_code": "3"},
        headers=registrar,
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["field"] == "name_en"


def test_a_capacity_of_zero_is_kept_and_not_read_as_unset(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """A section that admits nobody is a thing a registrar can mean, like a mark of 0."""
    added = seeded.post(
        "/v1/structure/classes",
        json={
            "code": "3Z",
            "academic_year_code": YEAR,
            "year_level_code": "3",
            "name_en": "Year 3 Z",
            "capacity": 0,
        },
        headers=registrar,
    )
    assert added.status_code == 201, added.text
    assert added.json()["capacity"] == 0

    listed = seeded.get(f"/v1/structure/classes?academic_year={YEAR}", headers=registrar)
    section = [row for row in listed.json() if row["code"] == "3Z"][0]
    assert section["capacity"] == 0


def test_renaming_a_class_keeps_every_child_in_it(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """Invariant 6 over HTTP: the code is identity, the name is a label."""
    seeded.post(
        "/v1/students",
        json={"student_number": "10432", "full_name_en": "Sara", "full_name_ar": "سارة"},
        headers=registrar,
    )
    _place(seeded, registrar, "10432", "3A", date(2025, 9, 1))

    renamed = seeded.patch(
        f"/v1/structure/classes/3A?academic_year={YEAR}",
        json={"name_en": "Falcons"},
        headers=registrar,
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name_en"] == "Falcons"
    assert renamed.json()["code"] == "3A"

    register = seeded.get(
        f"/v1/classes/3A/students?academic_year={YEAR}&on=2025-10-01", headers=registrar
    )
    assert [row["student_number"] for row in register.json()["students"]] == ["10432"]


# ---------------------------------------------------------------------------
# Subjects, per year
# ---------------------------------------------------------------------------


def test_the_same_code_in_two_years_is_two_subjects(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """The consequence of year-scoping, stated as a test.

    Both posts are 201: the second is a creation, not a collision. That is the property
    the change bought, and the property it gave up is the other side of the same fact —
    these two rows are not the same subject, so a mark on one is not comparable to a mark
    on the other, and nothing in this service will pretend otherwise.
    """
    first = seeded.post(
        "/v1/subjects",
        json={"code": "MATH", "academic_year_code": YEAR, "name_en": "Mathematics"},
        headers=registrar,
    )
    assert first.status_code == 201, first.text

    second = seeded.post(
        "/v1/subjects",
        json={"code": "MATH", "academic_year_code": NEXT_YEAR, "name_en": "Maths"},
        headers=registrar,
    )
    assert second.status_code == 201, second.text

    this_year = seeded.get(f"/v1/subjects?academic_year={YEAR}", headers=registrar)
    next_year = seeded.get(f"/v1/subjects?academic_year={NEXT_YEAR}", headers=registrar)
    assert [row["name_en"] for row in this_year.json()] == ["Mathematics"]
    assert [row["name_en"] for row in next_year.json()] == ["Maths"]


def test_reposting_a_subject_in_the_same_year_relabels_it(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    seeded.post(
        "/v1/subjects",
        json={"code": "SCI", "academic_year_code": YEAR, "name_en": "Sciense"},
        headers=registrar,
    )
    again = seeded.post(
        "/v1/subjects",
        json={"code": "SCI", "academic_year_code": YEAR, "name_en": "Science"},
        headers=registrar,
    )
    assert again.status_code == 200, again.text
    listed = seeded.get(f"/v1/subjects?academic_year={YEAR}", headers=registrar)
    assert [row["name_en"] for row in listed.json()] == ["Science"]


def test_a_subject_in_a_year_that_does_not_exist_is_refused_by_field(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """Named as `academic_year_code`, so the message lands under the right box on the form."""
    refused = seeded.post(
        "/v1/subjects",
        json={"code": "ART", "academic_year_code": "1999-2000", "name_en": "Art"},
        headers=registrar,
    )
    assert refused.status_code == 404, refused.text
    assert refused.json()["detail"]["field"] == "academic_year_code"


def test_retiring_a_subject_hides_it_unless_asked_for(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """Retire, never delete: a mark stated against it still needs a heading."""
    seeded.post(
        "/v1/subjects",
        json={"code": "ART", "academic_year_code": YEAR, "name_en": "Art"},
        headers=registrar,
    )
    retired = seeded.post(
        "/v1/subjects",
        json={
            "code": "ART",
            "academic_year_code": YEAR,
            "name_en": "Art",
            "is_active": False,
        },
        headers=registrar,
    )
    assert retired.status_code == 200, retired.text

    active = seeded.get(f"/v1/subjects?academic_year={YEAR}", headers=registrar)
    assert active.json() == []
    everything = seeded.get(
        f"/v1/subjects?academic_year={YEAR}&include_inactive=true", headers=registrar
    )
    assert [row["code"] for row in everything.json()] == ["ART"]


def test_an_unknown_year_is_a_404_rather_than_an_empty_catalogue(
    seeded: TestClient, registrar: dict[str, str]
) -> None:
    """A typo and a year with no subjects render identically on screen; only one is a bug."""
    missing = seeded.get("/v1/subjects?academic_year=1999-2000", headers=registrar)
    assert missing.status_code == 404, missing.text
