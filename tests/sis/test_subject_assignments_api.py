"""Where a subject is taught, over HTTP.

A subject has always belonged to a year. It now also belongs to a set of *rungs*, and the
whole value of that change is the narrowing it makes possible — so these tests are almost
all about what stops appearing, not about what appears.

Four properties, each of which is the reason a separate table exists at all:

**A subject appears only where it is assigned.** `GET /subjects?year_level=` is the read
every marks screen makes once it has a class in hand, and the assertion worth having is the
negative one: Physics assigned to Secondary is absent from Primary's answer. A test that
only checked Secondary would pass against a service that ignored the filter entirely.

**The two academic tracks are assigned separately.** A bilingual school's Arabic and
Languages sections own different rungs, so this falls out of the schema rather than being
enforced by a rule — which is precisely why it is asserted: a future change that hangs
assignments off a stage, or off a rung code shared between sections, would silently merge
the two catalogues and nothing else in the suite would notice.

**A duplicate is impossible and is not an error.** Two registrars dropping the same subject
on the same rung must produce one row, and the second must not be answered with a failure a
person has to read and dismiss.

**Un-assigning preserves the subject and its marks.** Taking Physics off Secondary is a
statement about next term's timetable. The subject row survives, and every grade already
awarded under it survives — that is the difference between this and retiring a subject, and
between both of those and a delete.
"""
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from sis.domain.grades import SubjectGrade
from sis.domain.people import ClassEnrolment, Student
from sis.domain.value_objects import Percentage, StudentNumber, TermCode
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.conftest import registrar_headers

SCHOOL = "BILING"
YEAR = "BILING-2025-2026"


@pytest.fixture()
def registrar() -> dict[str, str]:
    """The stored registrar key, verified for real — see `sis/api/deps.py`."""
    return registrar_headers()


def _school(client: TestClient, headers: dict[str, str]) -> None:
    """A bilingual school with one year, and one rung per section per stage.

    Rung codes carry their section (`AR-P1`, `LG-P1`) because a rung code is unique per
    school, not per section: two sections that both called their first primary year `P1`
    would be the same rung, and the separation these tests are about would be untestable
    rather than merely broken.
    """
    created = client.post(
        "/v1/schools",
        json={
            "code": SCHOOL,
            "name_en": "Bilingual School",
            "name_ar": "مدرسة ثنائية اللغة",
            "language_type": "both",
            "kg_grade_count": 0,
            "primary_grade_count": 1,
            "preparatory_grade_count": 0,
            "secondary_grade_count": 1,
            "term_count": 2,
            "working_days": ["sunday", "monday"],
        },
        headers=headers,
    )
    assert created.status_code == 201, created.text

    year = client.post(
        "/v1/academic-years",
        json={
            "code": YEAR,
            "school_code": SCHOOL,
            "name_en": "2025/2026",
            "name_ar": "٢٠٢٥",
            "starts_on": "2025-09-01",
            "ends_on": "2026-06-30",
            "is_current": True,
        },
        headers=headers,
    )
    assert year.status_code == 201, year.text

    for code, track, stage in (
        ("AR-P1", "AR", "primary"),
        ("AR-S1", "AR", "secondary"),
        ("LG-P1", "LANG", "primary"),
        ("LG-S1", "LANG", "secondary"),
    ):
        rung = client.post(
            "/v1/structure/levels",
            json={
                "code": code,
                "school_code": SCHOOL,
                "track_code": track,
                "name_en": code,
                "name_ar": code,
                "display_order": 1,
                "stage": stage,
            },
            headers=headers,
        )
        assert rung.status_code == 201, rung.text


def _subject(client: TestClient, headers: dict[str, str], code: str, order: int) -> None:
    made = client.post(
        "/v1/subjects",
        json={
            "code": code,
            "academic_year_code": YEAR,
            "name_en": code.title(),
            "name_ar": code,
            "display_order": order,
        },
        headers=headers,
    )
    assert made.status_code == 201, made.text


def _assign(
    client: TestClient,
    headers: dict[str, str],
    subject: str,
    rung: str,
    assigned: bool = True,
):
    return client.put(
        "/v1/subject-assignments",
        json={
            "academic_year_code": YEAR,
            "subject_code": subject,
            "year_level_code": rung,
            "assigned": assigned,
        },
        headers=headers,
    )


def _codes(client: TestClient, headers: dict[str, str], rung: str | None) -> list[str]:
    query = f"/v1/subjects?academic_year={YEAR}"
    if rung is not None:
        query += f"&year_level={rung}"
    response = client.get(query, headers=headers)
    assert response.status_code == 200, response.text
    return [row["code"] for row in response.json()]


def test_a_subject_appears_only_on_the_grades_it_is_assigned_to(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """The whole point of the stage, stated as the negative it is."""
    _school(client, registrar)
    for order, code in enumerate(("ARAB", "PHYS", "CHEM"), start=1):
        _subject(client, registrar, code, order)

    # Arabic is taught throughout; the sciences only in the secondary section.
    assert _assign(client, registrar, "ARAB", "AR-P1").status_code == 204
    assert _assign(client, registrar, "ARAB", "AR-S1").status_code == 204
    assert _assign(client, registrar, "PHYS", "AR-S1").status_code == 204
    assert _assign(client, registrar, "CHEM", "AR-S1").status_code == 204

    assert _codes(client, registrar, "AR-S1") == ["ARAB", "PHYS", "CHEM"]
    # The assertion the feature exists for: Primary does not inherit Secondary's sciences.
    assert _codes(client, registrar, "AR-P1") == ["ARAB"]
    # A rung nobody has assigned anything to teaches nothing, rather than everything.
    assert _codes(client, registrar, "LG-P1") == []

    # And the unnarrowed question still has its old answer: this is the year's catalogue,
    # which is a different question from what any one rung teaches.
    assert sorted(_codes(client, registrar, None)) == ["ARAB", "CHEM", "PHYS"]


def test_the_two_academic_tracks_are_assigned_separately(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Same school, same stage, same subject — and two independent answers."""
    _school(client, registrar)
    _subject(client, registrar, "PHYS", 1)

    assert _assign(client, registrar, "PHYS", "AR-S1").status_code == 204

    assert _codes(client, registrar, "AR-S1") == ["PHYS"]
    assert _codes(client, registrar, "LG-S1") == []

    # The board reports the track on each row, so a client can group without a second read.
    board = client.get(f"/v1/subject-assignments?academic_year={YEAR}", headers=registrar)
    assert board.status_code == 200, board.text
    assert [(row["year_level_code"], row["track_code"]) for row in board.json()] == [
        ("AR-S1", "AR")
    ]

    # Assigning it in the other section leaves the first alone rather than moving it.
    assert _assign(client, registrar, "PHYS", "LG-S1").status_code == 204
    assert _codes(client, registrar, "AR-S1") == ["PHYS"]
    assert _codes(client, registrar, "LG-S1") == ["PHYS"]


def test_assigning_twice_produces_one_row_and_no_error(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Idempotent, and idempotent *quietly* — a board can drop twice without apologising."""
    _school(client, registrar)
    _subject(client, registrar, "PHYS", 1)

    for _ in range(3):
        assert _assign(client, registrar, "PHYS", "AR-S1").status_code == 204

    assert _codes(client, registrar, "AR-S1") == ["PHYS"]

    # Counted in the table rather than through the listing: the listing would answer
    # `["PHYS"]` for one row or for three, which is the bug this asserts against.
    with SqlAlchemyUnitOfWork() as unit:
        rows = unit._session.execute(
            text("SELECT COUNT(*) FROM subject_year_levels")
        ).scalar_one()
    assert rows == 1

    # Un-assigning is idempotent in the same way, and does not fail on an absent row.
    assert _assign(client, registrar, "PHYS", "AR-S1", assigned=False).status_code == 204
    assert _assign(client, registrar, "PHYS", "AR-S1", assigned=False).status_code == 204
    assert _codes(client, registrar, "AR-S1") == []


def test_unassigning_keeps_the_subject_and_every_mark_stated_under_it(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """Requirement 5, at the sharpest point: the timetable changes, the record does not.

    A school that drops Physics from Secondary next year has not un-awarded the marks its
    children earned in it this year. Retiring the subject would not delete them either —
    the difference is that un-assigning does not even hide the subject from the catalogue.
    """
    _school(client, registrar)
    _subject(client, registrar, "PHYS", 1)
    assert _assign(client, registrar, "PHYS", "AR-S1").status_code == 204

    term = client.post(
        "/v1/terms",
        json={
            "code": "BILING-T1",
            "academic_year_code": YEAR,
            "name_en": "Term 1",
            "name_ar": "الفصل الأول",
            "starts_on": "2025-09-01",
            "ends_on": "2026-01-15",
            "sequence": 1,
        },
        headers=registrar,
    )
    assert term.status_code == 201, term.text
    section = client.post(
        "/v1/structure/classes",
        json={
            "code": "S1A",
            "academic_year_code": YEAR,
            "year_level_code": "AR-S1",
            "name_en": "S1A",
            "name_ar": "S1A",
        },
        headers=registrar,
    )
    assert section.status_code == 201, section.text

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
                # A genuine zero: the value most easily lost by anything that treats a
                # missing mark and a mark of nothing as the same fact.
                SubjectGrade(
                    student_number=StudentNumber("S-1"),
                    subject_code="PHYS",
                    term_code=TermCode("BILING-T1"),
                    class_section_id=section_id,
                    class_code="S1A",
                    percentage=Percentage(0),
                )
            ]
        )
        unit.commit()

    assert _assign(client, registrar, "PHYS", "AR-S1", assigned=False).status_code == 204

    # Gone from the rung...
    assert _codes(client, registrar, "AR-S1") == []
    # ...still in the year's catalogue, still active...
    catalogue = client.get(f"/v1/subjects?academic_year={YEAR}", headers=registrar).json()
    assert [(row["code"], row["is_active"]) for row in catalogue] == [("PHYS", True)]
    # ...and the mark is untouched. A real zero, which is the value most easily lost.
    marks = client.get("/v1/students/S-1/grades?term=BILING-T1", headers=registrar)
    assert marks.status_code == 200, marks.text
    stated = [row for row in marks.json()["grades"] if row["subject_code"] == "PHYS"]
    assert len(stated) == 1
    assert stated[0]["percentage"] == 0.0
    assert stated[0]["is_graded"] is True


def test_an_assignment_cannot_reach_across_schools_or_invent_a_grade(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """The pair is resolved through the year's school, so neither half can be borrowed."""
    _school(client, registrar)
    _subject(client, registrar, "PHYS", 1)

    # A rung of another school. `MAIN`/`Y1` is a rung nobody in BILING can name.
    other = client.post(
        "/v1/schools",
        json={
            "code": "OTHER",
            "name_en": "Other School",
            "name_ar": "مدرسة أخرى",
            "language_type": "arabic",
            "kg_grade_count": 0,
            "primary_grade_count": 1,
            "preparatory_grade_count": 0,
            "secondary_grade_count": 0,
            "term_count": 2,
            "working_days": ["sunday"],
        },
        headers=registrar,
    )
    assert other.status_code == 201, other.text
    assert client.post(
        "/v1/structure/levels",
        json={
            "code": "OTHER-P1",
            "school_code": "OTHER",
            "track_code": "AR",
            "name_en": "P1",
            "name_ar": "P1",
            "display_order": 1,
            "stage": "primary",
        },
        headers=registrar,
    ).status_code == 201

    refused = _assign(client, registrar, "PHYS", "OTHER-P1")
    assert refused.status_code == 404, refused.text
    assert refused.json()["detail"]["field"] == "year_level_code"

    # An invented rung fails the same way, rather than writing a row nobody can see.
    assert _assign(client, registrar, "PHYS", "NOPE").status_code == 404
    # As does an invented subject.
    assert _assign(client, registrar, "NOPE", "AR-S1").status_code == 404


def test_an_unknown_year_or_grade_is_refused_rather_than_answered_empty(
    client: TestClient, registrar: dict[str, str]
) -> None:
    """A typo and a rung that teaches nothing read identically once the answer is `[]`."""
    _school(client, registrar)

    missing_year = client.get("/v1/subject-assignments?academic_year=NOPE", headers=registrar)
    assert missing_year.status_code == 404, missing_year.text

    missing_rung = client.get(
        f"/v1/subjects?academic_year={YEAR}&year_level=NOPE", headers=registrar
    )
    assert missing_rung.status_code == 404, missing_rung.text
    assert missing_rung.json()["detail"]["field"] == "year_level"
