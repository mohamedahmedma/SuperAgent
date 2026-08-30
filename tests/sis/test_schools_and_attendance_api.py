"""Schools, stages, the daily register, and a child's own details — over HTTP.

Four features arrived together and each has one property worth more than the rest of its
tests put together:

**Schools are a boundary.** `test_two_schools_each_run_their_own_Y1_and_3A` is the reason
this table exists. Two branches with a rung and a class of the same code must stay separate,
and a register that showed one school's child in the other's classroom is the failure the
whole change is meant to prevent.

**A stage is a grouping and nothing else.** It orders a long ladder and carries no rules —
so the test asserts the order and asserts that a mistyped stage is refused, because a rung
silently landing in `unspecified` is a rung missing from the secondary group on every screen.

**An unmarked day is not an absence.** `test_a_child_nobody_marked_is_null_and_not_absent`
is the attendance equivalent of "a blank is not a zero". The register returns every child
placed that day, and the ones nobody marked come back with `state: null` — a third value,
which a client rendering as absent would use to accuse a child nobody looked at.

**An age is computed, never stored.** The response carries `date_of_birth` and an `age`
derived from it, and there is no age column anywhere: a stored age is right for one year and
silently wrong afterwards.
"""
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.sis.conftest import registrar_headers
from sis.domain.structure import AcademicYear, ClassSection, School, YearLevel
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork

NC = "NC"
MD = "MD"
NC_YEAR = "NC-2025-2026"
MD_YEAR = "MD-2025-2026"


def _seed_two_schools() -> None:
    """Two branches, each with its own `Y1` rung and its own `3A` class.

    The same codes on both sides deliberately. Anything that leaks across schools shows up
    here as a child on the wrong register, which is the failure worth catching.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.schools.upsert_many(
            [
                School(code=NC, name_en="Nasr City", name_ar="مدينة نصر"),
                School(code=MD, name_en="Maadi", name_ar="المعادي"),
            ]
        )
        uow.academic_years.upsert_many(
            [
                AcademicYear(
                    code=NC_YEAR,
                    school_code=NC,
                    name_en="2025-2026 Nasr City",
                    name_ar="٢٠٢٥",
                    starts_on=date(2025, 9, 1),
                    ends_on=date(2026, 6, 30),
                    is_current=True,
                ),
                AcademicYear(
                    code=MD_YEAR,
                    school_code=MD,
                    name_en="2025-2026 Maadi",
                    name_ar="٢٠٢٥",
                    starts_on=date(2025, 9, 1),
                    ends_on=date(2026, 6, 30),
                    is_current=True,
                ),
            ]
        )
        uow.year_levels.upsert_many(
            [
                YearLevel(
                    code="Y1",
                    school_code=NC,
                    name_en="Year 1",
                    name_ar="السنة ١",
                    display_order=1,
                    stage="primary",
                ),
                # The same rung code in the other school. This is what the migration made
                # possible and what the unique constraint now allows.
                YearLevel(
                    code="Y1",
                    school_code=MD,
                    name_en="Year 1",
                    name_ar="السنة ١",
                    display_order=1,
                    stage="primary",
                ),
            ]
        )
        uow.class_sections.upsert_many(
            [
                ClassSection(
                    code="3A",
                    academic_year_code=NC_YEAR,
                    year_level_code="Y1",
                    name_en="Nasr City 3A",
                    name_ar="٣أ",
                ),
                ClassSection(
                    code="3A",
                    academic_year_code=MD_YEAR,
                    year_level_code="Y1",
                    name_en="Maadi 3A",
                    name_ar="٣أ",
                ),
            ]
        )
        uow.commit()


@pytest.fixture()
def registrar() -> dict[str, str]:
    """The stored registrar key, verified for real — see `sis/api/deps.py`."""
    return registrar_headers()


@pytest.fixture()
def two_schools(client: TestClient) -> TestClient:
    _seed_two_schools()
    return client


def _add_child(client: TestClient, headers, number: str, name: str, **extra) -> None:
    body = {"student_number": number, "full_name_en": name, "full_name_ar": name}
    body.update(extra)
    response = client.post("/v1/students", json=body, headers=headers)
    assert response.status_code in (200, 201), response.text


def _place(client, headers, number: str, year: str, klass: str, day: str) -> None:
    response = client.post(
        f"/v1/students/{number}/placements",
        json={"academic_year_code": year, "class_code": klass, "starts_on": day},
        headers=headers,
    )
    assert response.status_code == 201, response.text


# ---------------------------------------------------------------------------
# Schools
# ---------------------------------------------------------------------------


def test_schools_are_listed_and_created_through_one_upsert(
    client: TestClient, registrar: dict[str, str]
) -> None:
    created = client.post(
        "/v1/schools",
        json={"code": "NC", "name_en": "Nasr City", "name_ar": "مدينة نصر"},
        headers=registrar,
    )
    assert created.status_code == 201, created.text

    again = client.post(
        "/v1/schools",
        json={"code": "NC", "name_en": "Nasr City Branch", "name_ar": "مدينة نصر"},
        headers=registrar,
    )
    assert again.status_code == 200, again.text

    listed = client.get("/v1/schools", headers=registrar)
    assert [row["name_en"] for row in listed.json()] == ["Nasr City Branch"]


def test_school_creation_stores_stage_two_configuration(
    client: TestClient, registrar: dict[str, str]
) -> None:
    response = client.post(
        "/v1/schools",
        json={
            "code": "CFG",
            "name_en": "Configured School",
            "name_ar": "مدرسة مهيأة",
            "language_type": "both",
            "kg_grade_count": 2,
            "primary_grade_count": 4,
            "preparatory_grade_count": 0,
            "secondary_grade_count": 0,
            "term_count": 3,
            "working_days": ["sunday", "monday", "wednesday"],
        },
        headers=registrar,
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["language_type"] == "both"
    assert body["kg_grade_count"] == 2
    assert body["primary_grade_count"] == 4
    assert body["preparatory_grade_count"] == 0
    assert body["secondary_grade_count"] == 0
    assert body["term_count"] == 3
    assert body["working_days"] == ["sunday", "monday", "wednesday"]


def test_school_creation_validates_levels_terms_and_grade_limits(
    client: TestClient, registrar: dict[str, str]
) -> None:
    base = {
        "code": "BAD",
        "name_en": "Invalid School",
        "name_ar": "مدرسة غير صالحة",
        "language_type": "arabic",
        "kg_grade_count": 0,
        "primary_grade_count": 0,
        "preparatory_grade_count": 0,
        "secondary_grade_count": 0,
        "term_count": 2,
        "working_days": ["sunday"],
    }
    assert client.post("/v1/schools", json=base, headers=registrar).status_code == 422
    assert client.post(
        "/v1/schools", json={**base, "kg_grade_count": 4}, headers=registrar
    ).status_code == 422
    assert client.post(
        "/v1/schools", json={**base, "kg_grade_count": 1, "term_count": 4}, headers=registrar
    ).status_code == 422


def test_closing_a_branch_keeps_the_configuration_it_was_created_with(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """A POST that mentions only labels must not reset a school to the creation defaults.

    Closing a branch goes through the same upsert as creating one, so a body that says
    `is_active: false` and nothing else used to carry Pydantic's defaults — four stages, two
    terms, five working days — over a primary-only, three-term school.
    """
    configured = {
        "code": "KEEP",
        "name_en": "Primary Only",
        "name_ar": "ابتدائي فقط",
        "language_type": "arabic",
        "kg_grade_count": 0,
        "primary_grade_count": 6,
        "preparatory_grade_count": 0,
        "secondary_grade_count": 0,
        "term_count": 3,
        "working_days": ["saturday", "sunday", "monday"],
    }
    assert client.post("/v1/schools", json=configured, headers=registrar).status_code == 201

    closed = client.post(
        "/v1/schools",
        json={
            "code": "KEEP",
            "name_en": "Primary Only",
            "name_ar": "ابتدائي فقط",
            "is_active": False,
        },
        headers=registrar,
    )
    assert closed.status_code == 200, closed.text
    body = closed.json()
    assert body["is_active"] is False
    assert body["language_type"] == "arabic"
    assert body["primary_grade_count"] == 6
    assert body["kg_grade_count"] == 0
    assert body["term_count"] == 3
    assert body["working_days"] == ["saturday", "sunday", "monday"]

    listed = client.get("/v1/schools?include_inactive=true", headers=registrar).json()
    stored = next(row for row in listed if row["code"] == "KEEP")
    assert stored["term_count"] == 3
    assert stored["working_days"] == ["saturday", "sunday", "monday"]


def test_a_stated_configuration_still_overwrites_the_stored_one(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Carrying omissions forward is not the same as refusing edits."""
    client.post(
        "/v1/schools",
        json={
            "code": "EDIT",
            "name_en": "Editable",
            "name_ar": "قابلة للتعديل",
            "term_count": 2,
            "working_days": ["sunday"],
        },
        headers=registrar,
    )
    updated = client.post(
        "/v1/schools",
        json={
            "code": "EDIT",
            "name_en": "Editable",
            "name_ar": "قابلة للتعديل",
            "term_count": 3,
        },
        headers=registrar,
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["term_count"] == 3
    # Untouched by an edit that did not mention it.
    assert updated.json()["working_days"] == ["sunday"]


def test_disabled_school_level_cannot_be_added(
    client: TestClient, registrar: dict[str, str]
) -> None:
    created = client.post(
        "/v1/schools",
        json={
            "code": "PRI",
            "name_en": "Primary Only",
            "name_ar": "ابتدائي فقط",
            "language_type": "arabic",
            "kg_grade_count": 0,
            "primary_grade_count": 6,
            "preparatory_grade_count": 0,
            "secondary_grade_count": 0,
            "term_count": 2,
            "working_days": ["sunday"],
        },
        headers=registrar,
    )
    assert created.status_code == 201, created.text
    refused = client.post(
        "/v1/structure/levels",
        json={
            "code": "KG1", "school_code": "PRI", "name_en": "KG 1",
            "name_ar": "كي جي ١", "display_order": 1, "stage": "garden",
        },
        headers=registrar,
    )
    assert refused.status_code == 422, refused.text


def test_a_closed_branch_is_hidden_but_not_deleted(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Closing a school keeps every register and mark taken in the years it ran."""
    client.post(
        "/v1/schools",
        json={"code": "OLD", "name_en": "Old Branch", "name_ar": "قديم"},
        headers=registrar,
    )
    client.post(
        "/v1/schools",
        json={
            "code": "OLD",
            "name_en": "Old Branch",
            "name_ar": "قديم",
            "is_active": False,
        },
        headers=registrar,
    )
    assert client.get("/v1/schools", headers=registrar).json() == []
    everything = client.get("/v1/schools?include_inactive=true", headers=registrar)
    assert [row["code"] for row in everything.json()] == ["OLD"]


def test_two_schools_each_run_their_own_Y1_and_3A(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """The point of the whole change: the same codes at two branches stay separate.

    A child enrolled in Nasr City's 3A must not appear on Maadi's 3A register. If the school
    boundary leaks anywhere — in the rung lookup, the class lookup or the roster query — this
    is where it shows, as a child at a school she has never attended.
    """
    nc_levels = two_schools.get(f"/v1/schools/{NC}/levels", headers=registrar).json()
    md_levels = two_schools.get(f"/v1/schools/{MD}/levels", headers=registrar).json()
    assert [row["code"] for row in nc_levels] == ["Y1"]
    assert [row["code"] for row in md_levels] == ["Y1"]
    assert nc_levels[0]["school_code"] == NC
    assert md_levels[0]["school_code"] == MD

    _add_child(two_schools, registrar, "NC-1", "Nasr City Child")
    _place(two_schools, registrar, "NC-1", NC_YEAR, "3A", "2025-09-01")

    def register(year: str) -> list[str]:
        response = two_schools.get(
            f"/v1/classes/3A/students?academic_year={year}&on=2025-10-01",
            headers=registrar,
        )
        assert response.status_code == 200, response.text
        return [row["student_number"] for row in response.json()["students"]]

    assert register(NC_YEAR) == ["NC-1"]
    assert register(MD_YEAR) == [], "a child leaked across the school boundary"


def test_a_class_is_wired_to_its_own_schools_rung(two_schools: TestClient) -> None:
    """Each branch's `3A` points at *its own* `Y1`, not at whichever the database returned.

    The regression this pins: class sections resolved their rung by bare code, and rung
    codes are unique per school rather than globally (`uq_year_levels_school_code`). With
    a `Y1` at both branches the lookup collapsed them into one id, so generating a ladder
    for a newly created school attached its classes to the *other* school's rungs.

    What that looked like to a registrar is the reason it went unnoticed for so long: the
    rung codes printed on screen were identical either way, so the only visible symptom was
    a brand-new school — one where nobody had created a single class — reporting the main
    school's class counts.

    Asserted against the stored foreign key rather than through the API, because every
    screen renders the rung's *code*, and both schools' codes read `Y1`. The wrong wiring
    is invisible in the response and unambiguous in the row.
    """
    with SqlAlchemyUnitOfWork() as uow:
        session = uow._session
        assert session is not None
        rows = session.execute(
            text(
                """
                SELECT year_school.code AS rung_school, class_school.code AS class_school
                FROM class_sections
                JOIN year_levels ON class_sections.year_level_id = year_levels.id
                JOIN schools AS year_school ON year_levels.school_id = year_school.id
                JOIN academic_years ON class_sections.academic_year_id = academic_years.id
                JOIN schools AS class_school ON academic_years.school_id = class_school.id
                """
            )
        ).all()

    assert rows, "the fixture seeded no classes, so this proves nothing"
    crossed = [row for row in rows if row.rung_school != row.class_school]
    assert not crossed, (
        "a class is attached to another school's rung: "
        + ", ".join(f"class at {row.class_school} -> rung at {row.rung_school}" for row in crossed)
    )


def test_a_new_school_starts_with_no_classes_of_its_own(two_schools: TestClient) -> None:
    """An empty branch reports zero classes even while another branch has a full ladder.

    A guard rather than a reproduction, and worth being precise about which: this passed
    even with the rung lookup bug above, because `list_for_year` filters on the academic
    year code and *that* is globally unique, so the wrongly-wired rung never changed which
    year a class was listed under. The reported symptom — a newly created school showing
    the main school's class count — therefore comes from the client rather than from here.

    It stays because it pins the property the split depends on: a school with a ladder and
    no classes answers "no classes", whatever the neighbouring branches hold.
    """
    with SqlAlchemyUnitOfWork() as uow:
        uow.schools.upsert_many(
            [School(code="ALX", name_en="Alexandria", name_ar="الإسكندرية")]
        )
        uow.academic_years.upsert_many(
            [
                AcademicYear(
                    code="ALX-2025-2026",
                    school_code="ALX",
                    name_en="2025-2026 Alexandria",
                    name_ar="٢٠٢٥",
                    starts_on=date(2025, 9, 1),
                    ends_on=date(2026, 6, 30),
                    is_current=True,
                )
            ]
        )
        # The same rung code the other two branches already use, which is exactly the
        # condition that made the old lookup ambiguous.
        uow.year_levels.upsert_many(
            [
                YearLevel(
                    code="Y1",
                    school_code="ALX",
                    name_en="Year 1",
                    name_ar="السنة ١",
                    display_order=1,
                    stage="primary",
                )
            ]
        )
        uow.commit()

    with SqlAlchemyUnitOfWork() as uow:
        assert list(uow.class_sections.list_for_year("ALX-2025-2026")) == []
        # And the schools that *do* have a class still have exactly the one each.
        assert len(list(uow.class_sections.list_for_year(NC_YEAR))) == 1
        assert len(list(uow.class_sections.list_for_year(MD_YEAR))) == 1


def test_the_years_route_narrows_to_one_school(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """And returns no ladder at all without a school, rather than a merged one."""
    everything = two_schools.get("/v1/structure/years", headers=registrar).json()
    assert sorted(y["code"] for y in everything["academic_years"]) == [MD_YEAR, NC_YEAR]
    assert everything["year_levels"] == [], "two schools' rungs must not be merged"

    one = two_schools.get(f"/v1/structure/years?school={NC}", headers=registrar).json()
    assert [y["code"] for y in one["academic_years"]] == [NC_YEAR]
    assert [level["school_code"] for level in one["year_levels"]] == [NC]


def test_a_year_naming_a_school_that_does_not_exist_is_refused_by_field(
    client: TestClient, registrar: dict[str, str]
) -> None:
    refused = client.post(
        "/v1/academic-years",
        json={
            "code": "2025-2026",
            "school_code": "NOPE",
            "name_en": "y",
            "name_ar": "y",
            "starts_on": "2025-09-01",
            "ends_on": "2026-06-30",
        },
        headers=registrar,
    )
    assert refused.status_code == 404, refused.text
    assert refused.json()["detail"]["field"] == "school_code"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def test_the_ladder_is_grouped_youngest_stage_first(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """Garden, primary, preparatory, secondary — then anything unclassified, last.

    Unclassified goes last rather than first: a rung nobody has grouped yet belongs at the
    bottom of the list, not above the kindergarten.
    """
    for code, stage, order in (
        ("Y10", "secondary", 10),
        ("KG1", "garden", 1),
        ("Y8", "preparatory", 8),
        ("MISC", "unspecified", 99),
    ):
        response = two_schools.post(
            "/v1/structure/levels",
            json={
                "code": code,
                "school_code": NC,
                "name_en": code,
                "name_ar": code,
                "display_order": order,
                "stage": stage,
            },
            headers=registrar,
        )
        assert response.status_code == 201, response.text

    levels = two_schools.get(f"/v1/schools/{NC}/levels", headers=registrar).json()
    assert [row["code"] for row in levels] == ["KG1", "Y1", "Y8", "Y10", "MISC"]
    assert [row["stage"] for row in levels] == [
        "garden",
        "primary",
        "preparatory",
        "secondary",
        "unspecified",
    ]


def test_a_mistyped_stage_is_refused_rather_than_silently_unspecified(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """"secondry" would otherwise create a rung missing from the secondary group."""
    refused = two_schools.post(
        "/v1/structure/levels",
        json={
            "code": "Y11",
            "school_code": NC,
            "name_en": "Year 11",
            "name_ar": "١١",
            "stage": "secondry",
        },
        headers=registrar,
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["field"] == "stage"


def test_a_rung_is_reclassified_without_detaching_its_classes(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """A stage is a label, so correcting it moves the rung between groups and nothing else."""
    _add_child(two_schools, registrar, "NC-2", "Child Two")
    _place(two_schools, registrar, "NC-2", NC_YEAR, "3A", "2025-09-01")

    again = two_schools.post(
        "/v1/structure/levels",
        json={
            "code": "Y1",
            "school_code": NC,
            "name_en": "Year 1",
            "name_ar": "السنة ١",
            "display_order": 1,
            "stage": "garden",
        },
        headers=registrar,
    )
    assert again.status_code == 200, again.text
    assert again.json()["stage"] == "garden"

    register = two_schools.get(
        f"/v1/classes/3A/students?academic_year={NC_YEAR}&on=2025-10-01", headers=registrar
    )
    assert [row["student_number"] for row in register.json()["students"]] == ["NC-2"]


# ---------------------------------------------------------------------------
# The daily register
# ---------------------------------------------------------------------------


def test_a_child_nobody_marked_is_null_and_not_absent(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """The attendance equivalent of "a blank is not a zero", and the most important test here.

    Three children on the register, two marked. The third must come back with `state: null`
    and `is_marked: false`, and the counts must add up to two — not three. Folding her into
    absent accuses a child nobody looked at; folding her into present flatters the school.
    """
    for number in ("A-1", "A-2", "A-3"):
        _add_child(two_schools, registrar, number, f"Child {number}")
        _place(two_schools, registrar, number, NC_YEAR, "3A", "2025-09-01")

    taken = two_schools.put(
        f"/v1/classes/3A/attendance?academic_year={NC_YEAR}&on=2025-10-01",
        json={
            "entries": [
                {"student_number": "A-1", "state": "present"},
                {"student_number": "A-2", "state": "absent"},
            ]
        },
        headers=registrar,
    )
    assert taken.status_code == 200, taken.text
    body = taken.json()

    by_number = {row["student_number"]: row for row in body["students"]}
    assert by_number["A-1"]["state"] == "present"
    assert by_number["A-2"]["state"] == "absent"
    assert by_number["A-3"]["state"] is None
    assert by_number["A-3"]["is_marked"] is False

    assert body["size"] == 3
    assert body["unmarked"] == 1
    assert body["is_complete"] is False
    # Counted over the two that were marked, not the three on the register.
    assert body["counts"]["recorded"] == 2
    assert body["counts"]["present"] == 1
    assert body["counts"]["absent"] == 1


def test_the_register_holds_every_placed_child_even_before_anyone_marks_it(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """Built from the enrolments, not from the marks. Otherwise an untouched register is
    an empty class with perfect attendance."""
    _add_child(two_schools, registrar, "B-1", "Child B1")
    _place(two_schools, registrar, "B-1", NC_YEAR, "3A", "2025-09-01")

    fresh = two_schools.get(
        f"/v1/classes/3A/attendance?academic_year={NC_YEAR}&on=2025-10-02", headers=registrar
    )
    assert fresh.status_code == 200, fresh.text
    body = fresh.json()
    assert body["size"] == 1
    assert body["unmarked"] == 1
    assert body["counts"]["recorded"] == 0
    assert body["students"][0]["state"] is None


def test_taking_the_register_twice_corrects_it_rather_than_duplicating(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """A day holds one statement per child, so the second save is a correction."""
    _add_child(two_schools, registrar, "C-1", "Child C1")
    _place(two_schools, registrar, "C-1", NC_YEAR, "3A", "2025-09-01")

    url = f"/v1/classes/3A/attendance?academic_year={NC_YEAR}&on=2025-10-03"
    first = two_schools.put(
        url,
        json={"entries": [{"student_number": "C-1", "state": "absent"}]},
        headers=registrar,
    )
    assert first.json()["counts"]["absent"] == 1

    corrected = two_schools.put(
        url,
        json={"entries": [{"student_number": "C-1", "state": "present"}]},
        headers=registrar,
    )
    assert corrected.status_code == 200, corrected.text
    assert corrected.json()["counts"]["present"] == 1
    assert corrected.json()["counts"]["absent"] == 0
    assert corrected.json()["counts"]["recorded"] == 1, "a second mark was stacked"


def test_a_partial_register_leaves_the_unnamed_children_alone(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """Saving the twelve present so far must not mark the other twenty-eight absent."""
    for number in ("D-1", "D-2"):
        _add_child(two_schools, registrar, number, f"Child {number}")
        _place(two_schools, registrar, number, NC_YEAR, "3A", "2025-09-01")

    url = f"/v1/classes/3A/attendance?academic_year={NC_YEAR}&on=2025-10-06"
    two_schools.put(
        url,
        json={"entries": [{"student_number": "D-1", "state": "present"}]},
        headers=registrar,
    )
    two_schools.put(
        url,
        json={"entries": [{"student_number": "D-2", "state": "late"}]},
        headers=registrar,
    )
    body = two_schools.get(url, headers=registrar).json()
    marks = {row["student_number"]: row["state"] for row in body["students"]}
    assert marks == {"D-1": "present", "D-2": "late"}
    assert body["is_complete"] is True


def test_an_excused_absence_needs_a_reason(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """Without one it cannot be told apart from an ordinary absence marked by mistake."""
    _add_child(two_schools, registrar, "E-1", "Child E1")
    _place(two_schools, registrar, "E-1", NC_YEAR, "3A", "2025-09-01")

    url = f"/v1/classes/3A/attendance?academic_year={NC_YEAR}&on=2025-10-07"
    refused = two_schools.put(
        url,
        json={"entries": [{"student_number": "E-1", "state": "excused"}]},
        headers=registrar,
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["field"] == "note"

    accepted = two_schools.put(
        url,
        json={
            "entries": [
                {"student_number": "E-1", "state": "excused", "note": "medical appointment"}
            ]
        },
        headers=registrar,
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["counts"]["excused"] == 1


def test_marking_a_child_who_is_not_in_the_class_that_day_is_refused(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """A stale screen or the wrong class. Writing it files her under a room she had left."""
    _add_child(two_schools, registrar, "F-1", "Child F1")
    # Placed in Maadi, marked in Nasr City.
    _place(two_schools, registrar, "F-1", MD_YEAR, "3A", "2025-09-01")

    refused = two_schools.put(
        f"/v1/classes/3A/attendance?academic_year={NC_YEAR}&on=2025-10-08",
        json={"entries": [{"student_number": "F-1", "state": "present"}]},
        headers=registrar,
    )
    assert refused.status_code == 422, refused.text
    assert "F-1" in refused.json()["detail"]["message"]


def test_a_late_child_was_in_the_room_and_is_not_an_absence(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """Late is its own state: eleven late days is a conversation, and folding them into
    present makes the pattern invisible."""
    _add_child(two_schools, registrar, "G-1", "Child G1")
    _place(two_schools, registrar, "G-1", NC_YEAR, "3A", "2025-09-01")
    two_schools.put(
        f"/v1/classes/3A/attendance?academic_year={NC_YEAR}&on=2025-10-09",
        json={"entries": [{"student_number": "G-1", "state": "late"}]},
        headers=registrar,
    )
    record = two_schools.get("/v1/students/G-1/attendance", headers=registrar).json()
    assert record["counts"]["late"] == 1
    assert record["counts"]["present"] == 0
    assert record["counts"]["in_the_room"] == 1
    assert record["counts"]["away"] == 0


def test_a_child_attendance_record_carries_counts_and_no_rate(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """`recorded` is the only denominator this service can state honestly, so no rate is
    computed for the caller."""
    _add_child(two_schools, registrar, "H-1", "Child H1")
    _place(two_schools, registrar, "H-1", NC_YEAR, "3A", "2025-09-01")
    for day, state in (("2025-10-13", "present"), ("2025-10-14", "absent")):
        two_schools.put(
            f"/v1/classes/3A/attendance?academic_year={NC_YEAR}&on={day}",
            json={"entries": [{"student_number": "H-1", "state": state}]},
            headers=registrar,
        )

    record = two_schools.get("/v1/students/H-1/attendance", headers=registrar).json()
    assert record["counts"]["recorded"] == 2
    assert [day["on_date"] for day in record["days"]] == ["2025-10-13", "2025-10-14"]
    assert "rate" not in record["counts"]
    assert "percentage" not in record["counts"]

    bounded = two_schools.get(
        "/v1/students/H-1/attendance?from=2025-10-14&to=2025-10-14", headers=registrar
    ).json()
    assert bounded["counts"]["recorded"] == 1
    assert bounded["counts"]["absent"] == 1


def test_a_mark_keeps_the_class_she_was_in_that_day(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """The attendance twin of the marks invariant: October stays October's class."""
    two_schools.post(
        "/v1/structure/classes",
        json={
            "code": "3B",
            "academic_year_code": NC_YEAR,
            "year_level_code": "Y1",
            "name_en": "Nasr City 3B",
            "name_ar": "٣ب",
        },
        headers=registrar,
    )
    _add_child(two_schools, registrar, "I-1", "Child I1")
    _place(two_schools, registrar, "I-1", NC_YEAR, "3A", "2025-09-01")
    two_schools.put(
        f"/v1/classes/3A/attendance?academic_year={NC_YEAR}&on=2025-10-15",
        json={"entries": [{"student_number": "I-1", "state": "present"}]},
        headers=registrar,
    )
    two_schools.post(
        "/v1/students/I-1/transfer",
        json={
            "academic_year_code": NC_YEAR,
            "to_class_code": "3B",
            "on_date": "2026-03-01",
        },
        headers=registrar,
    )

    record = two_schools.get("/v1/students/I-1/attendance", headers=registrar).json()
    october = [day for day in record["days"] if day["on_date"] == "2025-10-15"][0]
    assert october["class_code"] == "3A", "a past register was rewritten by a transfer"


# ---------------------------------------------------------------------------
# The child's own details
# ---------------------------------------------------------------------------


def test_a_birth_date_is_stored_and_the_age_is_computed_from_it(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """No age column anywhere: a stored age is right for one year and wrong afterwards.

    The reference day comes from UTC, matching what the API layer uses to compute the age it
    returns. Taken from `date.today()` instead, this test failed for the first hours of each
    local day in any timezone ahead of UTC: the birthday it constructed was "today" locally
    and still "tomorrow" in UTC, so the age came back one lower. A test that fails between
    midnight and 3am is a test nobody trusts.
    """
    today_utc = datetime.now(UTC).date()
    ten_years_ago = today_utc.replace(year=today_utc.year - 10)
    _add_child(
        two_schools,
        registrar,
        "J-1",
        "Child J1",
        date_of_birth=ten_years_ago.isoformat(),
        contact_phone="+201001234567",
        contact_email="parent@example.test",
        address="12 Some Street",
    )
    body = two_schools.get("/v1/students/J-1", headers=registrar).json()
    assert body["date_of_birth"] == ten_years_ago.isoformat()
    assert body["age"] == 10
    assert body["contact_phone"] == "+201001234567"
    assert body["contact_email"] == "parent@example.test"
    assert body["address"] == "12 Some Street"


def test_no_birth_date_means_no_age_rather_than_zero(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """`age: null` is "nobody recorded her birthday"; `age: 0` is a statement about a baby."""
    _add_child(two_schools, registrar, "K-1", "Child K1")
    body = two_schools.get("/v1/students/K-1", headers=registrar).json()
    assert body["date_of_birth"] is None
    assert body["age"] is None


def test_a_birth_date_in_the_future_is_refused(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """A mistyped year, and every age computed from it afterwards is negative."""
    # Two days out, so the assertion does not depend on which side of midnight UTC is on.
    tomorrow = (datetime.now(UTC).date() + timedelta(days=2)).isoformat()
    refused = two_schools.post(
        "/v1/students",
        json={
            "student_number": "L-1",
            "full_name_en": "Child L1",
            "full_name_ar": "L1",
            "date_of_birth": tomorrow,
        },
        headers=registrar,
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["field"] == "date_of_birth"


def test_patching_contact_details_leaves_the_rest_of_the_record_alone(
    two_schools: TestClient, registrar: dict[str, str]
) -> None:
    """Omitted means "leave it alone" — including a birth date the form did not carry."""
    _add_child(
        two_schools,
        registrar,
        "M-1",
        "Child M1",
        date_of_birth="2015-05-05",
        contact_phone="+201000000000",
    )
    patched = two_schools.patch(
        "/v1/students/M-1", json={"contact_phone": "+201111111111"}, headers=registrar
    )
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert body["contact_phone"] == "+201111111111"
    assert body["date_of_birth"] == "2015-05-05", "a birth date was erased by omission"
    assert body["full_name_en"] == "Child M1"
