"""How a year gets its terms, and what a term without dates still does.

Two changes meet in this file, and each is only safe because of the other.

**A year's term sections follow the school's term count.** The school answered "how many
terms do you run" when it was created; asking again, per year, is the same question with a
second answer that can disagree with the first. So creating a year creates its terms — one,
two or three — and changing the count on the school brings the years it already has back
into line.

**Term dates are optional.** They have to be: the terms above are built in June, and a
school that has not fixed its calendar cannot supply six dates to get past a NOT NULL. What
makes that safe rather than merely permissive is that nothing invents a date to fill the
gap — an undated term resolves a child's class against the *year's* window, which is the
same rule one level up and is the last test in this file.

The rule that governs both is data preservation. Dropping a school from three terms to two
must not take a term of marks with it, and re-saving a year must not overwrite the dates a
registrar has since typed. Both have a test here, because both are the kind of loss nobody
notices until a report card is wrong.
"""
from datetime import date

import pytest
from fastapi.testclient import TestClient

from sis.domain.grades import SubjectGrade
from sis.domain.people import ClassEnrolment, Student
from sis.domain.value_objects import Percentage, StudentNumber, TermCode
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.conftest import registrar_headers

SCHOOL = "TERMS"
YEAR = "TERMS-2025"


@pytest.fixture()
def registrar() -> dict[str, str]:
    return registrar_headers()


def _school(client: TestClient, headers: dict[str, str], *, terms: int, code: str = SCHOOL):
    return client.post(
        "/v1/schools",
        json={
            "code": code,
            "name_en": "Term Count School",
            "name_ar": "مدرسة الفصول",
            "language_type": "both",
            "kg_grade_count": 0,
            "primary_grade_count": 1,
            "preparatory_grade_count": 0,
            "secondary_grade_count": 1,
            "term_count": terms,
            "working_days": ["sunday", "monday"],
        },
        headers=headers,
    )


def _year(client: TestClient, headers: dict[str, str], *, code: str = YEAR, school: str = SCHOOL):
    return client.post(
        "/v1/academic-years",
        json={
            "code": code,
            "school_code": school,
            "name_en": "2025/2026",
            "name_ar": "٢٠٢٥",
            "starts_on": "2025-09-01",
            "ends_on": "2026-06-30",
            "is_current": True,
        },
        headers=headers,
    )


def _terms(client: TestClient, headers: dict[str, str], year: str = YEAR) -> list[dict]:
    response = client.get(f"/v1/terms?academic_year={year}", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("count", [1, 2, 3])
def test_a_year_gets_exactly_the_terms_its_school_says_it_runs(
    client: TestClient, registrar: dict[str, str], count: int
) -> None:
    """One selected term makes one section, two make two, three make three."""
    assert _school(client, registrar, terms=count).status_code == 201
    created = _year(client, registrar)
    assert created.status_code == 201, created.text

    # The write reports what it did, so a caller need not re-read to find out.
    plan = created.json()["terms"]
    assert plan["term_count"] == count
    assert plan["created"] == [f"{YEAR}-T{n}" for n in range(1, count + 1)]
    assert plan["removed"] == [] and plan["kept"] == []

    terms = _terms(client, registrar)
    assert [term["code"] for term in terms] == [f"{YEAR}-T{n}" for n in range(1, count + 1)]
    assert [term["sequence"] for term in terms] == list(range(1, count + 1))
    # Named, so a registrar is not asked to type "Term 1" into a field called Term 1.
    assert [term["name_en"] for term in terms] == [f"Term {n}" for n in range(1, count + 1)]
    assert terms[0]["name_ar"] == "الفصل الأول"


def test_the_new_terms_carry_no_dates_and_say_so(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """The point of the stage: a term exists before its calendar does.

    `is_dated` is asserted alongside the two nulls because it is the field a client should
    branch on — a screen testing `starts_on` alone would call a half-filled term dated.
    """
    assert _school(client, registrar, terms=2).status_code == 201
    assert _year(client, registrar).status_code == 201

    for term in _terms(client, registrar):
        assert term["starts_on"] is None
        assert term["ends_on"] is None
        assert term["is_dated"] is False


def test_term_dates_are_filled_in_one_at_a_time_and_either_end_may_stand_alone(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Optional means optional, including "we know when it starts, not when it ends"."""
    assert _school(client, registrar, terms=2).status_code == 201
    assert _year(client, registrar).status_code == 201

    # A start with no end. Accepted, and honestly reported as not yet a window.
    half = client.post(
        "/v1/terms",
        json={
            "code": f"{YEAR}-T1",
            "academic_year_code": YEAR,
            "name_en": "Term 1",
            "name_ar": "الفصل الأول",
            "starts_on": "2025-09-15",
            "sequence": 1,
        },
        headers=registrar,
    )
    assert half.status_code == 200, half.text  # 200: the term already existed
    assert half.json()["starts_on"] == "2025-09-15"
    assert half.json()["ends_on"] is None
    assert half.json()["is_dated"] is False

    # Both ends. Now it is a window.
    whole = client.post(
        "/v1/terms",
        json={
            "code": f"{YEAR}-T1",
            "academic_year_code": YEAR,
            "name_en": "Term 1",
            "name_ar": "الفصل الأول",
            "starts_on": "2025-09-15",
            "ends_on": "2026-01-20",
            "sequence": 1,
        },
        headers=registrar,
    )
    assert whole.status_code == 200, whole.text
    assert whole.json()["is_dated"] is True

    # An inverted *stated* range is still refused — optional widened what may be absent,
    # not what may be wrong.
    inverted = client.post(
        "/v1/terms",
        json={
            "code": f"{YEAR}-T2",
            "academic_year_code": YEAR,
            "name_en": "Term 2",
            "name_ar": "الفصل الثاني",
            "starts_on": "2026-05-01",
            "ends_on": "2026-02-01",
            "sequence": 2,
        },
        headers=registrar,
    )
    assert inverted.status_code == 422, inverted.text


def test_resaving_a_year_leaves_the_dates_and_labels_someone_typed(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """The sync is idempotent, and idempotent means it does not overwrite either.

    A registrar corrects the year's Arabic name in November, months after dating its terms.
    An upsert that re-wrote every term would reset all of it, and nothing on the screen
    would say a thing had been lost.
    """
    assert _school(client, registrar, terms=2).status_code == 201
    assert _year(client, registrar).status_code == 201
    client.post(
        "/v1/terms",
        json={
            "code": f"{YEAR}-T1",
            "academic_year_code": YEAR,
            "name_en": "First Semester",
            "name_ar": "الفصل الدراسي الأول",
            "starts_on": "2025-09-15",
            "ends_on": "2026-01-20",
            "sequence": 1,
        },
        headers=registrar,
    )

    again = _year(client, registrar)
    assert again.status_code == 200, again.text  # already existed
    assert again.json()["terms"]["created"] == []

    kept = _terms(client, registrar)[0]
    assert kept["name_en"] == "First Semester"
    assert kept["starts_on"] == "2025-09-15"
    assert kept["ends_on"] == "2026-01-20"


def test_raising_the_school_term_count_adds_a_section_to_the_years_on_file(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """The year follows the school, not only at the moment the year is created."""
    assert _school(client, registrar, terms=2).status_code == 201
    assert _year(client, registrar).status_code == 201
    assert len(_terms(client, registrar)) == 2

    raised = _school(client, registrar, terms=3)
    assert raised.status_code == 200, raised.text
    assert [plan["created"] for plan in raised.json()["terms"]] == [[f"{YEAR}-T3"]]

    assert [term["code"] for term in _terms(client, registrar)] == [
        f"{YEAR}-T1",
        f"{YEAR}-T2",
        f"{YEAR}-T3",
    ]


def test_lowering_the_term_count_drops_an_empty_section(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Downward too, while there is nothing to lose."""
    assert _school(client, registrar, terms=3).status_code == 201
    assert _year(client, registrar).status_code == 201
    assert len(_terms(client, registrar)) == 3

    lowered = _school(client, registrar, terms=2)
    assert lowered.status_code == 200, lowered.text
    assert [plan["removed"] for plan in lowered.json()["terms"]] == [[f"{YEAR}-T3"]]
    assert [term["code"] for term in _terms(client, registrar)] == [
        f"{YEAR}-T1",
        f"{YEAR}-T2",
    ]


def test_a_surplus_term_holding_marks_is_kept_and_reported(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """The one that matters. A term count is a setting; a mark is a fact about a child.

    Lowering the count must never be a delete of somebody's grades, so the surplus term
    stays, the year honestly shows three terms, and the response says which term survived
    and why the count and the screen disagree.
    """
    assert _school(client, registrar, terms=3).status_code == 201
    assert _year(client, registrar).status_code == 201
    _place_a_mark_in(client, registrar, term_code=f"{YEAR}-T3")

    lowered = _school(client, registrar, terms=2)
    assert lowered.status_code == 200, lowered.text
    plan = lowered.json()["terms"][0]
    assert plan["removed"] == []
    assert plan["kept"] == [f"{YEAR}-T3"]

    # Still on file, and so is the mark.
    assert [term["code"] for term in _terms(client, registrar)] == [
        f"{YEAR}-T1",
        f"{YEAR}-T2",
        f"{YEAR}-T3",
    ]
    report = client.get(f"/v1/students/S-1/grades?term={YEAR}-T3", headers=registrar)
    assert report.status_code == 200, report.text
    assert [line["percentage"] for line in report.json()["grades"]] == [0.0]


def _place_a_mark_in(client: TestClient, headers: dict[str, str], *, term_code: str) -> None:
    """One child, one class, one stated zero in the named term.

    A zero rather than a pass mark, because it is the value most easily lost by anything
    that confuses "no mark" with "a mark of nothing" — including a delete that checked for
    grades with a truth test.
    """
    assert client.post(
        "/v1/structure/levels",
        json={
            "code": "T-S1",
            "school_code": SCHOOL,
            "track_code": "AR",
            "name_en": "Secondary 1",
            "name_ar": "الأول الثانوي",
            "display_order": 1,
            "stage": "secondary",
        },
        headers=headers,
    ).status_code == 201
    assert client.post(
        "/v1/structure/classes",
        json={
            "code": "S1A",
            "academic_year_code": YEAR,
            "year_level_code": "T-S1",
            "name_en": "S1A",
            "name_ar": "S1A",
        },
        headers=headers,
    ).status_code == 201
    assert client.post(
        "/v1/subjects",
        json={
            "code": "PHYS",
            "academic_year_code": YEAR,
            "name_en": "Physics",
            "name_ar": "فيزياء",
        },
        headers=headers,
    ).status_code == 201

    with SqlAlchemyUnitOfWork() as unit:
        unit.students.upsert_many(
            [
                Student(
                    student_number="S-1",
                    full_name_en="A Child",
                    full_name_ar="طفل",
                    date_of_birth=date(2010, 5, 1),
                )
            ]
        )
        unit.enrolments.upsert_many(
            [
                ClassEnrolment(
                    student_number=StudentNumber("S-1"),
                    academic_year_code=YEAR,
                    class_code="S1A",
                    starts_on=date(2025, 9, 1),
                )
            ]
        )
        section_id = unit.class_sections.ids_for([(YEAR, "S1A")])[(YEAR, "S1A")]
        unit.grades.upsert_many(
            [
                SubjectGrade(
                    student_number=StudentNumber("S-1"),
                    subject_code="PHYS",
                    term_code=TermCode(term_code),
                    class_section_id=section_id,
                    class_code="S1A",
                    percentage=Percentage(0),
                )
            ]
        )
        unit.commit()


def test_an_undated_term_still_files_a_mark_under_the_right_class(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Invariant 2 survives the dates going away, which is the whole risk of this change.

    A term with no window cannot answer "which class was she in" on its own. The service
    asks the year's window instead — the period the term is unarguably inside — rather than
    guessing a narrower one. If this ever regressed to "no dates, no answer", every report
    card for an undated term would render with no class against it.
    """
    assert _school(client, registrar, terms=2).status_code == 201
    assert _year(client, registrar).status_code == 201
    _place_a_mark_in(client, registrar, term_code=f"{YEAR}-T1")

    undated = _terms(client, registrar)[0]
    assert undated["is_dated"] is False, "this test is meaningless if the term has dates"

    report = client.get(f"/v1/students/S-1/grades?term={YEAR}-T1", headers=registrar)
    assert report.status_code == 200, report.text
    assert report.json()["class_code"] == "S1A"


def test_the_year_names_its_school_its_tracks_and_its_classes(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """One read, so a screen cannot draw a school that existed at no instant.

    The grouping is by track because that is the axis a bilingual school reads along: two
    sections, one shared set of terms, separate ladders.
    """
    assert _school(client, registrar, terms=2).status_code == 201
    assert _year(client, registrar).status_code == 201
    for code, track in (("T-P1", "AR"), ("LG-P1", "LANG")):
        assert client.post(
            "/v1/structure/levels",
            json={
                "code": code,
                "school_code": SCHOOL,
                "track_code": track,
                "name_en": code,
                "name_ar": code,
                "display_order": 1,
                "stage": "primary",
            },
            headers=registrar,
        ).status_code == 201
    assert client.post(
        "/v1/structure/classes",
        json={
            "code": "P1A",
            "academic_year_code": YEAR,
            "year_level_code": "T-P1",
            "name_en": "P1A",
            "name_ar": "P1A",
        },
        headers=registrar,
    ).status_code == 201

    detail = client.get(f"/v1/academic-years/{YEAR}", headers=registrar)
    assert detail.status_code == 200, detail.text
    body = detail.json()

    assert body["year"]["code"] == YEAR
    assert body["school"]["code"] == SCHOOL
    assert body["school"]["term_count"] == 2
    assert [term["code"] for term in body["terms"]] == [f"{YEAR}-T1", f"{YEAR}-T2"]

    tracks = {group["track_code"]: group for group in body["tracks"]}
    assert set(tracks) == {"AR", "LANG"}
    assert [level["code"] for level in tracks["AR"]["year_levels"]] == ["T-P1"]
    assert tracks["AR"]["class_count"] == 1
    assert tracks["LANG"]["class_count"] == 0
    assert body["class_count"] == 1

    assert client.get("/v1/academic-years/NOPE", headers=registrar).status_code == 404


def test_a_year_of_a_second_school_gets_that_school_s_term_count(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """The count is read from the year's own school, not from whichever was touched last."""
    assert _school(client, registrar, terms=1, code="ONE").status_code == 201
    assert _school(client, registrar, terms=3, code="THREE").status_code == 201

    assert _year(client, registrar, code="ONE-2025", school="ONE").status_code == 201
    assert _year(client, registrar, code="THREE-2025", school="THREE").status_code == 201

    assert len(_terms(client, registrar, "ONE-2025")) == 1
    assert len(_terms(client, registrar, "THREE-2025")) == 3
