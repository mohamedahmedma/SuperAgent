"""Two schools, two databases, and the header that chooses between them.

This is the isolation itself, asserted the only way that means anything: with two real
migrated databases and a request that names one of them. Everything else in this suite
runs single-school, which is the default and stays the default — see `sis.tenancy`.

**What makes this different from a filter test.** A `WHERE school_id = ...` can be tested
by asserting that the other school's rows are absent from a response, and such a test
passes just as happily when the filter is missing but the other school happens to have no
matching rows. Here the other school's rows are in a *different file*, so a query that
forgot to scope reaches nothing at all. The assertions below are written to fail loudly if
the header is ever ignored: each school holds a child with the same `student_number`, and
they are told apart only by which database answered.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sis import tenancy
from sis.api.deps import SCHOOL_HEADER
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.conftest import MD, NC

# `two_databases` and the two school codes live in `conftest.py`: `test_authentication.py`
# proves a *key* does not cross between these files, and one copy of the tenancy wiring is
# what stops the two suites drifting apart about what a split estate looks like.

#: The same number at both branches. A child is a person and her number is the school's own
#: identifier for her, so two schools using `S-1` is ordinary — and it is exactly the case a
#: leak would render indistinguishable.
SHARED_NUMBER = "S-1"


def _seed(school_code: str, name: str) -> None:
    """One child, in one school's database, through the same repositories every route uses."""
    from sis.domain.people import Student

    with SqlAlchemyUnitOfWork(school_code=school_code) as uow:
        uow.students.upsert_many(
            [
                Student(
                    student_number=SHARED_NUMBER,
                    full_name_en=name,
                    full_name_ar=name,
                )
            ]
        )
        uow.commit()


def test_each_school_reads_only_its_own_database(two_databases: dict[str, str]) -> None:
    """The same student number in both files resolves to a different child in each.

    If the header were ignored and both requests answered from one database, the two names
    would be equal — so this fails on a leak rather than merely failing to prove one.
    """
    _seed(NC, "Nasr City Child")
    _seed(MD, "Maadi Child")

    with SqlAlchemyUnitOfWork(school_code=NC) as uow:
        nc_child = uow.students.get(SHARED_NUMBER)
    with SqlAlchemyUnitOfWork(school_code=MD) as uow:
        md_child = uow.students.get(SHARED_NUMBER)

    assert nc_child is not None and md_child is not None
    assert nc_child.full_name_en == "Nasr City Child"
    assert md_child.full_name_en == "Maadi Child"


def test_a_child_at_one_school_is_absent_from_the_others_database(
    two_databases: dict[str, str],
) -> None:
    """Not filtered out — absent. The row is in another file entirely."""
    _seed(NC, "Only At Nasr City")

    with SqlAlchemyUnitOfWork(school_code=MD) as uow:
        assert uow.students.get(SHARED_NUMBER) is None
        assert list(uow.students.search("Only")) == []


def test_an_unknown_school_is_refused_rather_than_defaulted(
    two_databases: dict[str, str],
) -> None:
    """A school this process does not serve resolves to nothing, never to the first one."""
    with pytest.raises(tenancy.UnknownSchool):
        SqlAlchemyUnitOfWork(school_code="NOPE").__enter__()


def test_the_registry_refuses_a_school_with_no_database(monkeypatch) -> None:
    """Named in `SIS_SCHOOLS` and nowhere else is a deploy-time failure, not a 3am one."""
    monkeypatch.setenv(tenancy.SCHOOLS_VAR, "AAA,BBB")
    monkeypatch.setenv(f"{tenancy.DATABASE_URL_PREFIX}_AAA", "sqlite:///./a.db")
    monkeypatch.delenv(f"{tenancy.DATABASE_URL_PREFIX}_BBB", raising=False)
    tenancy.reset_registry_cache()
    try:
        with pytest.raises(tenancy.TenancyMisconfigured) as refusal:
            tenancy.get_registry()
        assert "BBB" in str(refusal.value)
    finally:
        tenancy.reset_registry_cache()


def test_two_codes_folding_to_one_env_suffix_are_refused(monkeypatch) -> None:
    """`NC-1` and `NC.1` both read `SIS_DATABASE_URL_NC_1`, so they would share a file.

    Refused at startup rather than resolved, because the failure it prevents is the one
    this whole design exists to prevent — two schools' rows in one database — arriving by
    way of a punctuation choice nobody thought was significant.
    """
    monkeypatch.setenv(tenancy.SCHOOLS_VAR, "NC-1,NC.1")
    monkeypatch.setenv(f"{tenancy.DATABASE_URL_PREFIX}_NC_1", "sqlite:///./one.db")
    tenancy.reset_registry_cache()
    try:
        with pytest.raises(tenancy.TenancyMisconfigured) as refusal:
            tenancy.get_registry()
        assert "suffix" in str(refusal.value)
    finally:
        tenancy.reset_registry_cache()


def test_single_school_mode_ignores_the_header(client: TestClient) -> None:
    """With `SIS_SCHOOLS` unset the header is not required and not honoured.

    The property that keeps a laptop, this suite and every unsplit deployment working: the
    header is inert until an estate is actually split, so nothing has to be rolled out in
    lockstep.
    """
    assert not tenancy.get_registry().is_multi_school
    without = client.get("/v1/schools")
    with_header = client.get("/v1/schools", headers={SCHOOL_HEADER: "ANYTHING"})
    assert without.status_code == 200
    assert with_header.status_code == 200
    assert without.json() == with_header.json()


# ---------------------------------------------------------------------------
# The same isolation, over HTTP, through the real application
# ---------------------------------------------------------------------------


@pytest.fixture()
def split_client(two_databases: dict[str, str]) -> Iterator[TestClient]:
    """The real app, booted against two school databases.

    Boots through the lifespan, so the startup schema gate runs for real — and in
    multi-school mode that gate checks *every* school, which is the check that turns "the
    migration succeeded for four schools and failed for the fifth" into a startup failure
    naming the fifth rather than a 500 in front of its registrar.

    Presents no credential, because this service no longer takes one — see
    `test_authentication.py`. What is under test here is unaffected either way: separation
    is the connection a request is answered on, not who asked.
    """
    from sis.app import app

    with TestClient(app) as test_client:
        yield test_client


def _create_school(client: TestClient, code: str, name: str) -> None:
    response = client.post(
        "/v1/schools",
        json={"code": code, "name_en": name, "name_ar": name},
        headers={SCHOOL_HEADER: code},
    )
    assert response.status_code in (200, 201), response.text


def test_over_http_each_school_sees_only_itself(split_client: TestClient) -> None:
    """Written through the API, read back through the API, and the two never mix."""
    _create_school(split_client, NC, "Nasr City")
    _create_school(split_client, MD, "Maadi")

    nc = split_client.get("/v1/schools", headers={SCHOOL_HEADER: NC}).json()
    md = split_client.get("/v1/schools", headers={SCHOOL_HEADER: MD}).json()

    assert [row["code"] for row in nc] == [NC]
    assert [row["code"] for row in md] == [MD]


def test_over_http_a_student_does_not_cross(split_client: TestClient) -> None:
    """The same student number, present at one school and simply absent at the other."""
    _create_school(split_client, NC, "Nasr City")
    _create_school(split_client, MD, "Maadi")

    created = split_client.post(
        "/v1/students",
        json={
            "student_number": SHARED_NUMBER,
            "full_name_en": "Nasr City Child",
            "full_name_ar": "طفلة",
        },
        headers={SCHOOL_HEADER: NC},
    )
    assert created.status_code in (200, 201), created.text

    assert split_client.get(
        f"/v1/students/{SHARED_NUMBER}", headers={SCHOOL_HEADER: NC}
    ).status_code == 200
    assert split_client.get(
        f"/v1/students/{SHARED_NUMBER}", headers={SCHOOL_HEADER: MD}
    ).status_code == 404


def test_a_request_naming_no_school_is_refused(split_client: TestClient) -> None:
    """422 and a message naming the header, rather than one school's data by default."""
    response = split_client.get("/v1/schools")
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert SCHOOL_HEADER in detail["message"]
    assert detail["field"] == "school_code"


def test_a_request_naming_an_unknown_school_is_a_404(split_client: TestClient) -> None:
    """And carries the stable error code, not the school code the caller made up.

    `SisError.code` is what a client branches on. An earlier version of `UnknownSchool`
    assigned the school to that attribute, so every unrecognised school produced a
    different machine-readable code and no client could match on any of them.
    """
    response = split_client.get("/v1/schools", headers={SCHOOL_HEADER: "NOPE"})
    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "unknown_reference"


def test_the_header_normalises_like_every_other_school_code(
    split_client: TestClient,
) -> None:
    """` nc ` and `NC` name one school, matching what the value objects do everywhere."""
    _create_school(split_client, NC, "Nasr City")
    response = split_client.get("/v1/schools", headers={SCHOOL_HEADER: " nc "})
    assert response.status_code == 200, response.text
    assert [row["code"] for row in response.json()] == [NC]
