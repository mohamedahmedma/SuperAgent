"""Test fixtures for the SIS.

Environment is set before any `sis` module is imported. `sis.config` reads lazily and
caches, so a variable set after the first `get_settings()` is ignored — and the symptom
of getting that wrong is a suite that quietly migrates and then wipes the developer's
real `./sis.db`. A file-backed temporary SQLite rather than `sqlite://`: an in-memory
database lives inside one connection, so tests pass alone and fail together.

**The schema is built by running alembic, never by `create_all`.** Invariant 8 says
alembic owns the schema, and a suite that built tables from `Base.metadata` would test a
schema no deployment has ever had — every migration bug (a missing index, a column added
to the model and not to a revision, a constraint alembic cannot express under SQLite's
batch mode) passes here and fails in production. Running the real migration once per
session and copying the resulting file per test keeps that proof and still gives every
test a clean database. It also means `SIS_SKIP_MIGRATION_CHECK` is deliberately left
unset: the app's startup gate runs for real in the `client` fixture, so a revision added
without regenerating the head would fail the suite at boot rather than mid-request.

The `Fake*` repositories below exist so a service can be tested with no database at all.
That is the point of the ports layer: "one bad row must not discard the good ones" is a
rule about a use case, and a test that needs an engine, a migration and a temp file to
assert it will be slow enough that nobody writes the other twenty cases.
"""
import os
import tempfile

_TMPDIR = tempfile.mkdtemp(prefix="sis-tests-")
_LIVE_DB = os.path.join(_TMPDIR, "sis-test.db")
_TEMPLATE_DB = os.path.join(_TMPDIR, "template.db")
os.environ["SIS_DATABASE_URL"] = f"sqlite:///{_LIVE_DB}"

import csv  # noqa: E402
from hashlib import sha1  # noqa: E402
import io  # noqa: E402
import shutil  # noqa: E402
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import UTC, date, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Final, Protocol  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from openpyxl import Workbook  # noqa: E402

from sis.application.dto import ParseResult  # noqa: E402
from sis.application.ports.repositories import (  # noqa: E402
    ClassSectionKey,
    EnrolmentKey,
    GradeKey,
)
from sis.application.ports.unit_of_work import UnitOfWork  # noqa: E402
from sis.config import get_settings, reset_settings_cache  # noqa: E402
from sis.domain.access import AccessAttempt  # noqa: E402
from sis.domain.auth import ApiKey, Scope  # noqa: E402
from sis.domain.errors import UnknownReference  # noqa: E402
from sis.domain.grades import SubjectGrade  # noqa: E402
from sis.domain.guardians import (  # noqa: E402
    Guardian,
    RelationshipType,
    StudentGuardian,
)
from sis.domain.imports import ImportBatch, ImportRow, RowOutcome  # noqa: E402
from sis.domain.people import ClassEnrolment, Student  # noqa: E402
from sis.domain.structure import (
    School,  # noqa: E402
    AcademicYear,
    ClassSection,
    Subject,
    Term,
    YearLevel,
)
from sis.domain.value_objects import (  # noqa: E402
    AcademicYearCode,
    ClassCode,
    Percentage,
    Phone,
    StudentNumber,
    SubjectCode,
    TermCode,
    YearCode,
)
from sis import tenancy  # noqa: E402
from sis.infrastructure.db.session import reset_engine  # noqa: E402
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork  # noqa: E402

_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "sis" / "alembic.ini"

REGISTRAR_KEY = "registrar-fixture-key-0000000000"
READER_KEY = "reader-fixture-key-00000000000000"


# ---------------------------------------------------------------------------
# File builders. Uploads are bytes plus a filename — never a path — so a test
# never has to create, find or clean up a temp file to exercise an import.
# ---------------------------------------------------------------------------


def xlsx_bytes(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]] = (),
    *,
    sheet_title: str = "Sheet1",
) -> bytes:
    """A one-worksheet workbook, in memory.

    A `None` cell is written as a genuinely empty cell rather than an empty string,
    because invariant 1 lives or dies on that distinction: the parser must see "nothing
    was stated" and produce `None`, and a fixture that wrote `""` — or worse, `0` —
    would let a parser that invents a failing grade pass its own test.
    """
    book = Workbook()
    sheet = book.active
    sheet.title = sheet_title
    sheet.append(list(headers))
    for row in rows:
        sheet.append(list(row))
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def csv_bytes(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]] = (),
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> bytes:
    """The same sheet as CSV. `encoding` is a parameter because schools export cp1256."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\r\n")
    writer.writerow(list(headers))
    for row in rows:
        writer.writerow(["" if cell is None else cell for cell in row])
    return buffer.getvalue().encode(encoding)


# ---------------------------------------------------------------------------
# The migrated database
# ---------------------------------------------------------------------------


def _claim_database_url() -> None:
    """Point `SIS_DATABASE_URL` back at this suite's database.

    Set at import (line 30) and re-asserted here, because the variable is
    process-global and this is not the only suite that wants one. `tests/` contains a
    cross-service journey test that names its own SIS database at module scope, and
    pytest imports every collected module before running anything — so in a session
    covering both, whichever imported last silently owns the variable.

    That produced a failure with no useful message: alembic migrated the OTHER suite's
    file, and the template copy then failed with `FileNotFoundError` on a path nobody
    had written. Re-asserting is a line; deducing that from the traceback is an evening.

    A fixture that needs a specific database claims it rather than trusting the
    environment it inherited. The journey test does the same for itself.
    """
    os.environ["SIS_DATABASE_URL"] = f"sqlite:///{_LIVE_DB}"
    reset_settings_cache()


def _run_alembic_upgrade() -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    _claim_database_url()  # env.py resolves the URL through sis.config
    command.upgrade(AlembicConfig(str(_ALEMBIC_INI)), "head")


@pytest.fixture(scope="session")
def _migrated_template() -> str:
    """Run the real migration once, then keep the resulting file as a template.

    Copying a migrated file is not the same shortcut as `create_all`: the schema under
    test is still exactly what `alembic upgrade head` produced, including the
    `alembic_version` row the app's startup gate checks. What it buys is the per-test
    reset — dropping and re-migrating for every test costs seconds each, and a suite
    that slow gets run less often than the code changes.
    """
    _run_alembic_upgrade()
    reset_engine()
    shutil.copyfile(_LIVE_DB, _TEMPLATE_DB)
    return _TEMPLATE_DB


@pytest.fixture(autouse=True)
def database(_migrated_template: str) -> Iterator[str]:
    """A clean migrated database per test, at the URL the environment already names.

    Autouse so it is torn down before any fixture that opened a connection: on Windows
    an open SQLite handle makes the file unreplaceable, and the failure reads as a
    permission error in whichever test happens to run next rather than as leaked state.
    """
    # Per test, not once per session: another suite in the same run may have pointed the
    # variable elsewhere between one test and the next. See `_claim_database_url`.
    _claim_database_url()
    reset_engine()
    reset_settings_cache()
    shutil.copyfile(_migrated_template, _LIVE_DB)
    yield get_settings().database_url
    reset_engine()


@pytest.fixture()
def uow() -> Iterator[UnitOfWork]:
    """One entered transaction against the real schema. Commits are the test's business."""
    with SqlAlchemyUnitOfWork() as unit:
        yield unit


@pytest.fixture()
def uow_factory() -> Callable[[], UnitOfWork]:
    """For services whose steps are separate transactions — preview, then commit."""
    return SqlAlchemyUnitOfWork


@pytest.fixture()
def api_keys() -> dict[str, str]:
    """Two stored keys, minted through the same hashing the API authenticates with.

    Written to the database rather than injected past `require_registrar`, because the
    scopes are the thing under test: `registrar` does not imply `reader` in this service,
    and a fixture that overrode the dependency would make every authorisation test agree
    with itself instead of with `deps.py`.
    """
    from sis.api.deps import hash_api_key, key_prefix

    now = datetime.now(UTC)
    with SqlAlchemyUnitOfWork() as unit:
        for raw, scope in ((REGISTRAR_KEY, Scope.REGISTRAR), (READER_KEY, Scope.READER)):
            unit.api_keys.add(
                ApiKey(
                    prefix=key_prefix(raw),
                    key_hash=hash_api_key(raw),
                    label=f"test {scope.value}",
                    scope=scope,
                    is_active=True,
                    expires_at=None,
                    created_at=now,
                )
            )
        unit.commit()
    return {"registrar": REGISTRAR_KEY, "reader": READER_KEY}


def registrar_headers() -> dict[str, str]:
    return {"X-API-Key": REGISTRAR_KEY}


def reader_headers() -> dict[str, str]:
    return {"X-API-Key": READER_KEY}


# ---------------------------------------------------------------------------
# A split estate — two schools, two databases
# ---------------------------------------------------------------------------

#: Codes for the two-school fixtures below. Short and deliberately alike, so anything that
#: leaks across schools shows up as a row in the wrong file rather than as a plausible one.
NC = "NC"
MD = "MD"

@pytest.fixture()
def two_databases(_migrated_template: str) -> Iterator[dict[str, str]]:
    """One migrated database per school, and the registry pointing at both.

    Lives here rather than in `test_physical_separation.py` because two suites need it:
    that one proves data does not cross, and `test_authentication.py` proves a *key* does
    not either. One copy of the tenancy wiring, so the two cannot drift apart about what a
    split estate looks like.

    Built from the same template the single-school fixture uses, so both schools are on
    exactly the schema `alembic upgrade head` produces — which is the property this whole
    design leans on: same schema in every file, so one migration run per school and nothing
    that has to be kept in sync by hand.
    """
    directory = tempfile.mkdtemp(prefix="sis-schools-")
    urls = {}
    for code in (NC, MD):
        path = os.path.join(directory, f"{code}.db")
        shutil.copyfile(_migrated_template, path)
        urls[code] = f"sqlite:///{path}"

    previous = {name: os.environ.get(name) for name in (
        tenancy.SCHOOLS_VAR,
        f"{tenancy.DATABASE_URL_PREFIX}_{NC}",
        f"{tenancy.DATABASE_URL_PREFIX}_{MD}",
    )}
    os.environ[tenancy.SCHOOLS_VAR] = f"{NC},{MD}"
    os.environ[f"{tenancy.DATABASE_URL_PREFIX}_{NC}"] = urls[NC]
    os.environ[f"{tenancy.DATABASE_URL_PREFIX}_{MD}"] = urls[MD]
    tenancy.reset_registry_cache()
    reset_engine()

    yield urls

    # Engines first: on Windows an open SQLite handle makes the file unremovable, and the
    # failure surfaces in whichever test runs next rather than here.
    reset_engine()
    tenancy.reset_registry_cache()
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    tenancy.reset_registry_cache()
    shutil.rmtree(directory, ignore_errors=True)


#: The day every SIS API test is run "on", unless it moves the clock itself.
#:
#: It sits inside the 2026-2027 academic year the fixtures build, on the first day of it,
#: which is what the roster and placement dates in the suites are written against.
SUITE_TODAY: Final = date(2026, 9, 1)


class Clock:
    """A `today` the suite owns, and can move.

    `take_register` refuses a day that has not happened and freezes a day that has, so the
    calendar is not decoration in these tests: it decides whether a write is allowed at
    all. Reading the wall clock instead would mean the only writable day is the real one,
    which is how the attendance suites ended up with every date rewritten to match the
    afternoon they were last touched — green that day, red the next morning.
    """

    def __init__(self, today: date) -> None:
        self._today = today
        self._apps: list[Any] = []

    def __call__(self) -> date:
        return self._today

    def set(self, day: date | str) -> None:
        """Move to `day`, so a test can be Tuesday and then Wednesday."""
        self._today = date.fromisoformat(day) if isinstance(day, str) else day

    def install(self, app: Any) -> None:
        """Pin this clock on `app`.

        Taken per-app rather than once on the module singleton because some suites build
        their own with `create_app()`, and an override installed on someone else's app is
        a fixture that silently does nothing.
        """
        from sis.api.deps import get_today

        app.dependency_overrides[get_today] = self
        self._apps.append(app)

    def uninstall(self) -> None:
        from sis.api.deps import get_today

        for app in self._apps:
            app.dependency_overrides.pop(get_today, None)
        self._apps.clear()


@pytest.fixture()
def clock() -> Iterator[Clock]:
    """Pins the register's idea of today, and hands the test the dial.

    This is the one dependency the API suites override, and the reason is the opposite of
    the usual one: overriding it is what makes the test agree with `deps.py` rather than
    with itself. Everything else — the key lookup, the service factory, the unit of work —
    stays the code that ships; only the calendar is held still, because a rule about
    "today" cannot otherwise be stated by a suite that runs on every day but one.
    """
    from sis.app import app

    moving = Clock(SUITE_TODAY)
    moving.install(app)
    try:
        yield moving
    finally:
        moving.uninstall()


@pytest.fixture()
def client(api_keys: dict[str, str], clock: Clock) -> Iterator[TestClient]:
    """The real app over the migrated database, lifespan and startup gate included.

    Carries the stored **registrar** key by default, so every test that is about
    something else authenticates for real rather than through a dependency override. The
    difference matters: an override would make each of these tests agree with itself
    instead of with `deps.py`, and the day the stored-key path broke they would all still
    pass.

    A test that is about authentication sends its own header — `reader_headers()` for the
    scope split, or `headers={}` to present nothing — and the per-request value wins over
    this default.
    """
    from sis.app import app

    with TestClient(app, headers=registrar_headers()) as test_client:
        yield test_client


@pytest.fixture()
def unauthenticated_client(api_keys: dict[str, str]) -> Iterator[TestClient]:
    """The same app, presenting no credential at all.

    Exists so "this route refuses an anonymous caller" is asserted against the real
    dependency rather than assumed. Keys are still seeded, so a refusal here means the
    header was missing and not that the database was empty.
    """
    from sis.app import app

    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# In-memory repositories — one per port, no database anywhere below this line
# ---------------------------------------------------------------------------


class _Coded(Protocol):
    @property
    def code(self) -> object: ...


class _InMemory:
    """Shared storage and the snapshot pair the fake unit of work rolls back with."""

    def __init__(self) -> None:
        self._rows: dict[Any, Any] = {}

    def snapshot(self) -> dict[Any, Any]:
        # Shallow is enough and is the whole reason every entity here is a frozen
        # dataclass: nothing already stored can be mutated in place, so restoring the
        # mapping restores the state.
        return dict(self._rows)

    def restore(self, snapshot: dict[Any, Any]) -> None:
        self._rows.clear()
        self._rows.update(snapshot)


class _ByCode[EntityT: _Coded](_InMemory):
    """Keyed by `str(code)`, matching the port's rule that returned keys are strings."""

    def __init__(self, entities: Sequence[EntityT] = ()) -> None:
        super().__init__()
        for entity in entities:
            self._rows[str(entity.code)] = entity

    def get(self, code: object) -> EntityT | None:
        return self._rows.get(str(code))

    def get_many(self, codes: Collection[object]) -> Mapping[str, EntityT]:
        return {str(code): self._rows[str(code)] for code in codes if str(code) in self._rows}

    def _upsert(self, entities: Sequence[EntityT]) -> dict[str, bool]:
        created: dict[str, bool] = {}
        for entity in entities:
            key = str(entity.code)
            created[key] = key not in self._rows
            self._rows[key] = entity
        return created

    def all(self) -> tuple[EntityT, ...]:
        return tuple(self._rows.values())


class FakeAcademicYearRepository(_ByCode[AcademicYear]):
    def list_all(self) -> Sequence[AcademicYear]:
        return sorted(self.all(), key=lambda year: year.starts_on, reverse=True)

    def current(self) -> AcademicYear | None:
        return next((year for year in self.all() if year.is_current), None)

    def set_current(self, code: AcademicYearCode) -> AcademicYear:
        key = str(code)
        if key not in self._rows:
            raise UnknownReference(f"no academic year {key}", field="academic_year_code")
        # Cleared and set together: a moment with two current years answers every
        # "this year's classes" screen wrongly, and nothing logs that it happened.
        for other, year in list(self._rows.items()):
            self._rows[other] = replace(year, is_current=other == key)
        return self._rows[key]

    def upsert_many(self, years: Sequence[AcademicYear]) -> Mapping[str, bool]:
        return self._upsert(years)


class FakeYearLevelRepository(_ByCode[YearLevel]):
    def list_all(self) -> Sequence[YearLevel]:
        return sorted(self.all(), key=lambda level: level.sort_key)

    def upsert_many(self, levels: Sequence[YearLevel]) -> Mapping[str, bool]:
        return self._upsert(levels)


class FakeClassSectionRepository(_InMemory):
    """Keyed by `(academic_year_code, code)`: `3A` is a different room every September."""

    def __init__(self, sections: Sequence[ClassSection] = ()) -> None:
        super().__init__()
        self._ids: dict[ClassSectionKey, int] = {}
        for section in sections:
            self._store(section)

    def _store(self, section: ClassSection) -> bool:
        key = section.identity
        created = key not in self._rows
        self._rows[key] = section
        self._ids.setdefault(key, len(self._ids) + 1)
        return created

    def snapshot(self) -> dict[Any, Any]:
        return {"rows": dict(self._rows), "ids": dict(self._ids)}

    def restore(self, snapshot: dict[Any, Any]) -> None:
        self._rows.clear()
        self._rows.update(snapshot["rows"])
        self._ids.clear()
        self._ids.update(snapshot["ids"])

    def get(
        self, academic_year_code: AcademicYearCode, code: ClassCode
    ) -> ClassSection | None:
        return self._rows.get((str(academic_year_code), str(code)))

    def get_many(
        self, keys: Collection[ClassSectionKey]
    ) -> Mapping[ClassSectionKey, ClassSection]:
        wanted = {(str(year), str(code)) for year, code in keys}
        return {key: value for key, value in self._rows.items() if key in wanted}

    def list_for_year(
        self,
        academic_year_code: AcademicYearCode,
        *,
        year_level_code: YearCode | None = None,
    ) -> Sequence[ClassSection]:
        year = str(academic_year_code)
        level = None if year_level_code is None else str(year_level_code)
        found = [
            section
            for section in self._rows.values()
            if str(section.academic_year_code) == year
            and (level is None or str(section.year_level_code) == level)
        ]
        return sorted(found, key=lambda section: str(section.code))

    def upsert_many(
        self, sections: Sequence[ClassSection]
    ) -> Mapping[ClassSectionKey, bool]:
        return {section.identity: self._store(section) for section in sections}

    def rename(
        self,
        academic_year_code: AcademicYearCode,
        code: ClassCode,
        *,
        name_en: str | None = None,
        name_ar: str | None = None,
    ) -> ClassSection:
        key = (str(academic_year_code), str(code))
        section = self._rows.get(key)
        if section is None:
            raise UnknownReference(f"no class {key[1]} in {key[0]}", field="class_code")
        # Labels only. The key is untouched, so the surrogate id and every enrolment and
        # grade pointing at this section survive the rename (invariant 6).
        renamed = section.renamed(name_en=name_en, name_ar=name_ar)
        self._rows[key] = renamed
        return renamed

    def ids_for(
        self, keys: Collection[ClassSectionKey]
    ) -> Mapping[ClassSectionKey, int]:
        wanted = {(str(year), str(code)) for year, code in keys}
        return {key: value for key, value in self._ids.items() if key in wanted}


class FakeTermRepository(_ByCode[Term]):
    def list_for_year(self, academic_year_code: AcademicYearCode) -> Sequence[Term]:
        year = str(academic_year_code)
        found = [term for term in self.all() if str(term.academic_year_code) == year]
        return sorted(found, key=lambda term: term.sort_key)

    def upsert_many(self, terms: Sequence[Term]) -> Mapping[str, bool]:
        return self._upsert(terms)

    def set_closed(self, code: TermCode, *, is_closed: bool) -> Term:
        term = self._rows.get(str(code))
        if term is None:
            raise UnknownReference(f"no term {code}", field="term_code")
        self._rows[str(code)] = replace(term, is_closed=is_closed)
        return self._rows[str(code)]


class FakeSubjectRepository(_InMemory):
    """Keyed by `(academic year, code)`, because that pair is a subject's identity.

    Deliberately not built on `_ByCode` like its siblings. Keying this fake by code alone
    would let a test store `MATH` in two years and silently get one row back, which is the
    exact confusion the real schema now refuses — a fake that is more permissive than the
    database it stands in for makes the tests that pass against it worthless.
    """

    def __init__(self, subjects: Sequence[Subject] = ()) -> None:
        super().__init__()
        for subject in subjects:
            self._rows[self._key(subject)] = subject

    @staticmethod
    def _key(subject: Subject) -> tuple[str, str]:
        return (str(subject.academic_year_code), str(subject.code))

    def get(self, code: object, academic_year_code: object) -> Subject | None:
        return self._rows.get((str(academic_year_code), str(code)))

    def get_many(
        self, codes: Collection[object], academic_year_code: object
    ) -> Mapping[str, Subject]:
        year = str(academic_year_code)
        found = {}
        for code in codes:
            subject = self._rows.get((year, str(code)))
            if subject is not None:
                found[str(code)] = subject
        return found

    def list_for_year(
        self, academic_year_code: object, *, include_inactive: bool = False
    ) -> Sequence[Subject]:
        year = str(academic_year_code)
        found = [
            subject
            for (subject_year, _), subject in self._rows.items()
            if subject_year == year and (include_inactive or subject.is_active)
        ]
        return sorted(found, key=lambda subject: subject.sort_key)

    def upsert_many(self, subjects: Sequence[Subject]) -> Mapping[str, bool]:
        created: dict[str, bool] = {}
        for subject in subjects:
            key = self._key(subject)
            created[str(subject.code)] = key not in self._rows
            self._rows[key] = subject
        return created

    def all(self) -> tuple[Subject, ...]:
        return tuple(self._rows.values())


class FakeStudentRepository(_InMemory):
    def __init__(self, students: Sequence[Student] = ()) -> None:
        super().__init__()
        for student in students:
            self._rows[str(student.student_number)] = student

    def get(self, student_number: StudentNumber) -> Student | None:
        return self._rows.get(str(student_number))

    def get_many(
        self, student_numbers: Collection[StudentNumber]
    ) -> Mapping[str, Student]:
        wanted = {str(number) for number in student_numbers}
        return {key: value for key, value in self._rows.items() if key in wanted}

    def search(
        self, query: str, *, limit: int = 50, include_inactive: bool = False
    ) -> Sequence[Student]:
        needle = query.strip().casefold()
        found = [
            student
            for student in self._rows.values()
            if (include_inactive or student.is_active)
            and needle
            in f"{student.student_number} {student.full_name_en} {student.full_name_ar}".casefold()
        ]
        return sorted(found, key=lambda s: str(s.student_number))[:limit]

    def upsert_many(self, students: Sequence[Student]) -> Mapping[str, bool]:
        created: dict[str, bool] = {}
        for student in students:
            key = str(student.student_number)
            created[key] = key not in self._rows
            self._rows[key] = student
        return created

    def set_active(self, student_number: StudentNumber, *, is_active: bool) -> Student:
        key = str(student_number)
        student = self._rows.get(key)
        if student is None:
            raise UnknownReference(f"no student {key}", field="student_number")
        self._rows[key] = replace(student, is_active=is_active)
        return self._rows[key]


class FakeEnrolmentRepository(_InMemory):
    """Placements keyed including `starts_on` — a return to 3A is a second fact.

    Holds a reference to the class-section fake rather than a copy of the sections,
    because `class_section_on` has to resolve the placement it finds into the section a
    test may have renamed since; a snapshot taken at construction would answer with the
    old label and nothing would report a discrepancy.
    """

    def __init__(
        self,
        sections: FakeClassSectionRepository,
        enrolments: Sequence[ClassEnrolment] = (),
    ) -> None:
        super().__init__()
        self._sections = sections
        for enrolment in enrolments:
            self._rows[self._key(enrolment)] = enrolment

    @staticmethod
    def _key(enrolment: ClassEnrolment) -> EnrolmentKey:
        return (
            str(enrolment.student_number),
            str(enrolment.academic_year_code),
            str(enrolment.class_code),
            enrolment.starts_on,
        )

    def _for(self, student_id: object) -> list[ClassEnrolment]:
        number = str(student_id)
        found = [e for e in self._rows.values() if str(e.student_number) == number]
        return sorted(found, key=lambda e: e.starts_on)

    def class_section_on(
        self, student_id: StudentNumber, on_date: date
    ) -> ClassSection | None:
        for enrolment in self._for(student_id):
            if enrolment.covers(on_date):
                return self._sections.get(
                    enrolment.academic_year_code, enrolment.class_code
                )
        return None

    def class_sections_on(
        self, student_ids: Collection[StudentNumber], on_date: date
    ) -> Mapping[str, ClassSection]:
        found: dict[str, ClassSection] = {}
        for student_id in student_ids:
            section = self.class_section_on(student_id, on_date)
            # Absent, never guessed: a child with no placement that day is not in 3A.
            if section is not None:
                found[str(student_id)] = section
        return found

    def open_enrolment(self, student_id: StudentNumber) -> ClassEnrolment | None:
        return next((e for e in self._for(student_id) if e.is_open), None)

    def list_for_student(self, student_id: StudentNumber) -> Sequence[ClassEnrolment]:
        return self._for(student_id)

    def list_for_students(
        self, student_ids: Collection[StudentNumber]
    ) -> Mapping[str, Sequence[ClassEnrolment]]:
        found = {str(number): self._for(number) for number in student_ids}
        return {key: value for key, value in found.items() if value}

    def roster_on(
        self,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        on_date: date,
    ) -> Sequence[ClassEnrolment]:
        year, code = str(academic_year_code), str(class_code)
        found = [
            enrolment
            for enrolment in self._rows.values()
            if str(enrolment.academic_year_code) == year
            and str(enrolment.class_code) == code
            and enrolment.covers(on_date)
        ]
        return sorted(found, key=lambda e: str(e.student_number))

    def close_open_enrolment(
        self, student_id: StudentNumber, *, ends_on: date
    ) -> ClassEnrolment | None:
        open_row = self.open_enrolment(student_id)
        if open_row is None:
            return None
        # The class is never rewritten — only the end date. Overwriting it is the
        # history rewrite invariant 2 forbids.
        closed = replace(open_row, ends_on=ends_on)
        self._rows[self._key(open_row)] = closed
        return closed

    def upsert_many(
        self, enrolments: Sequence[ClassEnrolment]
    ) -> Mapping[EnrolmentKey, bool]:
        created: dict[EnrolmentKey, bool] = {}
        for enrolment in enrolments:
            key = self._key(enrolment)
            created[key] = key not in self._rows
            self._rows[key] = enrolment
        return created


class FakeGuardianRepository(_InMemory):
    """Adults keyed by *every* number they hold, mirroring the real lookup.

    Keying on all of them rather than only the primary is the whole point: the SQL
    repository resolves a guardian through `guardian_phones`, so a fake that matched only
    primaries would let "a second upload quoting her WhatsApp line finds the same woman"
    pass here and fail against a database.
    """

    def __init__(self, guardians: Sequence[Guardian] = ()) -> None:
        super().__init__()
        for guardian in guardians:
            self._store(guardian)

    def _store(self, guardian: Guardian) -> bool:
        created = guardian.identity not in self._rows
        self._rows[guardian.identity] = guardian
        return created

    def _owner_of(self, phone: object) -> Guardian | None:
        number = str(phone)
        return next(
            (g for g in self._rows.values() if g.reachable_on(number)), None
        )

    def get(self, phone: object) -> Guardian | None:
        return self._owner_of(phone)

    def get_many(self, phones: Collection[object]) -> Mapping[str, Guardian]:
        found: dict[str, Guardian] = {}
        for phone in phones:
            owner = self._owner_of(phone)
            if owner is not None:
                found[str(phone)] = owner
        return found

    def primary_phone_for(self, public_id: str) -> object | None:
        """The inverse of the fake's derived handle, found by scanning.

        A scan rather than a reverse map because the fake is small and a second index is a
        second thing to keep in step with `upsert_many`.
        """
        for guardian in self._rows.values():
            if self.public_id_for(guardian.primary_phone) == public_id:
                return guardian.primary_phone
        return None

    def public_id_for(self, phone: object) -> str | None:
        """A stable handle per guardian, derived from her identity so it survives a reload.

        Derived rather than random on purpose: the real repository reads a stored column,
        so a fake that minted a fresh value per call would let a test pass while the
        service under test held two different handles for one woman.
        """
        owner = self._owner_of(phone)
        return None if owner is None else "fake-" + sha1(owner.identity.encode()).hexdigest()

    def upsert_many(self, guardians: Sequence[Guardian]) -> Mapping[str, bool]:
        created: dict[str, bool] = {}
        for guardian in guardians:
            # Match on any known number, as the real one does, so a row whose primary is
            # new but whose alternate is on file updates rather than duplicating.
            existing = next(
                (g for g in self._rows.values() if any(g.reachable_on(p) for p in guardian.phones)),
                None,
            )
            if existing is None:
                created[guardian.identity] = True
                self._store(guardian)
                continue
            created[guardian.identity] = False
            # Numbers accumulate and the stored primary stays primary — moving it would
            # change the identity every stored link names.
            phones = list(existing.phones)
            known = {str(p) for p in phones}
            for phone in guardian.phones:
                if str(phone) not in known:
                    phones.append(phone)
                    known.add(str(phone))
            self._rows[existing.identity] = replace(
                guardian, phones=tuple(phones)
            )
        return created

    def all(self) -> tuple[Guardian, ...]:
        return tuple(self._rows.values())


class FakeStudentGuardianRepository(_InMemory):
    """Links keyed by `(student_number, guardian_phone)` — `StudentGuardian.identity`.

    Holds a reference to the guardian fake rather than a copy, because resolving "is this
    link the same adult as that phone" has to see numbers added since construction.
    """

    def __init__(
        self,
        guardians: FakeGuardianRepository,
        links: Sequence[StudentGuardian] = (),
    ) -> None:
        super().__init__()
        self._guardians = guardians
        for link in links:
            self._rows[link.identity] = link

    def list_for_student(self, student_number: object) -> Sequence[StudentGuardian]:
        number = str(student_number)
        found = [link for link in self._rows.values() if str(link.student_number) == number]
        return tuple(
            sorted(found, key=lambda l: (not l.is_primary_contact, str(l.guardian_phone)))
        )

    def list_for_students(
        self, student_numbers: Collection[object]
    ) -> Mapping[str, Sequence[StudentGuardian]]:
        found = {str(number): self.list_for_student(number) for number in student_numbers}
        # Absent, never an empty sequence: a child with no guardians is not the same
        # answer as one this query did not ask about.
        return {number: links for number, links in found.items() if links}

    def list_students_for_guardian(
        self, phone: object, *, viewable_only: bool = True
    ) -> Sequence[StudentGuardian]:
        owner = self._guardians.get(phone)
        if owner is None:
            return ()
        found = [
            link
            for link in self._rows.values()
            if owner.reachable_on(link.guardian_phone)
            and (not viewable_only or link.can_view_records)
        ]
        return tuple(sorted(found, key=lambda l: str(l.student_number)))

    def upsert_many(
        self, links: Sequence[StudentGuardian]
    ) -> Mapping[tuple[str, str], bool]:
        created: dict[tuple[str, str], bool] = {}
        for link in links:
            key = link.identity
            created[key] = key not in self._rows
            self._rows[key] = link
        return created

    def unlink(self, student_number: object, phone: object) -> bool:
        key = (str(student_number), str(phone))
        return self._rows.pop(key, None) is not None

    def all(self) -> tuple[StudentGuardian, ...]:
        return tuple(self._rows.values())


class FakeGradeRepository(_InMemory):
    def __init__(self, grades: Sequence[SubjectGrade] = ()) -> None:
        super().__init__()
        for grade in grades:
            self._rows[grade.identity] = grade

    def get(
        self,
        student_number: StudentNumber,
        subject_code: SubjectCode,
        term_code: TermCode,
    ) -> SubjectGrade | None:
        return self._rows.get((str(student_number), str(subject_code), str(term_code)))

    def get_many(self, keys: Collection[GradeKey]) -> Mapping[GradeKey, SubjectGrade]:
        wanted = {tuple(str(part) for part in key) for key in keys}
        return {key: value for key, value in self._rows.items() if key in wanted}

    def list_for_student(
        self, student_number: StudentNumber, *, term_code: TermCode | None = None
    ) -> Sequence[SubjectGrade]:
        number = str(student_number)
        term = None if term_code is None else str(term_code)
        found = [
            grade
            for grade in self._rows.values()
            if str(grade.student_number) == number
            and (term is None or str(grade.term_code) == term)
        ]
        # Subject `display_order` is a property of the subject table, which this fake
        # cannot see; code order is stable and enough for a unit test to assert against.
        return sorted(found, key=lambda g: str(g.subject_code))

    def list_for_class(
        self,
        class_section_id: int,
        term_code: TermCode,
        *,
        subject_code: SubjectCode | None = None,
    ) -> Sequence[SubjectGrade]:
        term = str(term_code)
        subject = None if subject_code is None else str(subject_code)
        found = [
            grade
            for grade in self._rows.values()
            if grade.class_section_id == class_section_id
            and str(grade.term_code) == term
            and (subject is None or str(grade.subject_code) == subject)
        ]
        return sorted(found, key=lambda g: (str(g.student_number), str(g.subject_code)))

    def upsert_many(self, grades: Sequence[SubjectGrade]) -> Mapping[GradeKey, bool]:
        created: dict[GradeKey, bool] = {}
        for grade in grades:
            # Stored exactly as handed over, `percentage=None` included. Substituting a
            # zero here would make an ungraded child look like a failing one.
            created[grade.identity] = grade.identity not in self._rows
            self._rows[grade.identity] = grade
        return created


class FakeImportBatchRepository(_InMemory):
    """Batches and their rows in one mapping, stored as immutable tuples.

    Tuples rather than lists so the unit of work's shallow snapshot is a real snapshot:
    a rolled-back `replace_rows` must not leave the outcomes of an abandoned commit
    visible on a batch the caller believes is still merely previewed.
    """

    def add(self, batch: ImportBatch, rows: Sequence[ImportRow]) -> None:
        self._rows[batch.batch_id] = (batch, tuple(rows))

    def get(self, batch_id: str) -> ImportBatch | None:
        found = self._rows.get(batch_id)
        return None if found is None else found[0]

    def save(self, batch: ImportBatch) -> ImportBatch:
        _, rows = self._rows.get(batch.batch_id, (None, ()))
        self._rows[batch.batch_id] = (batch, rows)
        return batch

    def list_rows(
        self,
        batch_id: str,
        *,
        outcomes: Collection[RowOutcome] | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> Sequence[ImportRow]:
        rows = self._matching(batch_id, outcomes)
        return rows[offset:] if limit is None else rows[offset : offset + limit]

    def count_rows(
        self, batch_id: str, *, outcomes: Collection[RowOutcome] | None = None
    ) -> int:
        return len(self._matching(batch_id, outcomes))

    def _matching(
        self, batch_id: str, outcomes: Collection[RowOutcome] | None
    ) -> tuple[ImportRow, ...]:
        _, rows = self._rows.get(batch_id, (None, ()))
        rows = tuple(sorted(rows, key=lambda row: row.line_number))
        if outcomes is None:
            return rows
        wanted = set(outcomes)
        return tuple(row for row in rows if row.outcome in wanted)

    def replace_rows(self, batch_id: str, rows: Sequence[ImportRow]) -> None:
        batch, _ = self._rows.get(batch_id, (None, ()))
        if batch is not None:
            self._rows[batch_id] = (batch, tuple(rows))

    def delete_expired(self, now: datetime) -> int:
        stale = [
            batch_id
            for batch_id, (batch, _) in self._rows.items()
            if batch.is_expired(now)
        ]
        for batch_id in stale:
            del self._rows[batch_id]
        return len(stale)


class FakeApiKeyRepository(_InMemory):
    def __init__(self, keys: Sequence[ApiKey] = ()) -> None:
        super().__init__()
        for key in keys:
            self._rows[key.prefix] = key

    def get_by_prefix(self, prefix: str) -> ApiKey | None:
        return self._rows.get(prefix)

    def list_all(self) -> Sequence[ApiKey]:
        return sorted(self._rows.values(), key=lambda key: key.created_at, reverse=True)

    def add(self, key: ApiKey) -> ApiKey:
        self._rows[key.prefix] = key
        return key

    def revoke(self, prefix: str) -> ApiKey | None:
        key = self._rows.get(prefix)
        if key is None:
            return None
        self._rows[prefix] = replace(key, is_active=False)
        return self._rows[prefix]

    def touch(self, prefix: str, *, at: datetime) -> None:
        key = self._rows.get(prefix)
        if key is not None:
            self._rows[prefix] = replace(key, last_used_at=at)

    def has_any(self) -> bool:
        return bool(self._rows)


class FakeAccessAuditRepository(_InMemory):
    """Appended to, read back, and never edited — the port has no other methods.

    Keyed by insertion order rather than by any field: an audit has no natural key, two
    identical attempts a second apart are two rows, and collapsing them would hide exactly
    the pattern the table exists to show — a run of refusals against one handle.
    """

    def __init__(self) -> None:
        super().__init__()
        self._next = 0

    def record(self, attempt: AccessAttempt) -> None:
        self._rows[self._next] = attempt
        self._next += 1

    def recent(
        self,
        *,
        guardian_public_id: str | None = None,
        student_number: str | None = None,
        allowed: bool | None = None,
        limit: int = 100,
    ) -> Sequence[AccessAttempt]:
        found = [
            attempt
            for _, attempt in sorted(self._rows.items(), reverse=True)
            if (not guardian_public_id or attempt.guardian_public_id == guardian_public_id)
            and (not student_number or attempt.student_number == student_number)
            and (allowed is None or attempt.allowed is allowed)
        ]
        return found[:limit]


class FakeUnitOfWork:
    """Every repository, one transaction, no database.

    `__exit__` restores each repository's snapshot unless `commit()` was called, which is
    the one behaviour a fake must not fake. A unit of work whose rollback did nothing
    would let a service that forgot to commit — or that raised after two hundred writes —
    pass every test and lose half a roster in production, because the fake's assertions
    would read the writes the real transaction discarded.

    `commits` and `rollbacks` are counters rather than flags so a test can assert that a
    preview committed *once*, not merely that it committed.
    """

    def __init__(self) -> None:
        self.academic_years = FakeAcademicYearRepository()
        self.year_levels = FakeYearLevelRepository()
        self.class_sections = FakeClassSectionRepository()
        self.terms = FakeTermRepository()
        self.subjects = FakeSubjectRepository()
        self.students = FakeStudentRepository()
        self.enrolments = FakeEnrolmentRepository(self.class_sections)
        self.guardians = FakeGuardianRepository()
        self.student_guardians = FakeStudentGuardianRepository(self.guardians)
        self.grades = FakeGradeRepository()
        self.imports = FakeImportBatchRepository()
        self.api_keys = FakeApiKeyRepository()
        self.access_audit = FakeAccessAuditRepository()
        self.commits = 0
        self.rollbacks = 0
        self._committed = False
        self._snapshots: dict[str, Any] = {}

    def _repositories(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (name, getattr(self, name))
            for name in (
                "academic_years",
                "year_levels",
                "class_sections",
                "terms",
                "subjects",
                "students",
                "enrolments",
                "guardians",
                "student_guardians",
                "grades",
                "imports",
                "api_keys",
                "access_audit",
            )
        )

    def __enter__(self) -> "FakeUnitOfWork":
        self._committed = False
        self._snapshots = {name: repo.snapshot() for name, repo in self._repositories()}
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._committed:
            self.rollback()
        # Returns None: an exception must keep travelling up to the layer that turns it
        # into a status code.

    def commit(self) -> None:
        self._committed = True
        self.commits += 1
        self._snapshots = {name: repo.snapshot() for name, repo in self._repositories()}

    def rollback(self) -> None:
        self.rollbacks += 1
        for name, repo in self._repositories():
            if name in self._snapshots:
                repo.restore(self._snapshots[name])


class StubParser:
    """A parser that returns what a test handed it, or raises what it handed instead.

    One class for both parser ports: they differ only in row type, and structural typing
    means a stub does not have to pick. `calls` records `(content, filename)` so a test
    can prove the service passed the upload through untouched rather than re-reading it.
    """

    def __init__(self, result: ParseResult[Any] | Exception) -> None:
        self._result = result
        self.calls: list[tuple[bytes, str]] = []

    def parse(self, content: bytes, filename: str) -> ParseResult[Any]:
        self.calls.append((content, filename))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


@pytest.fixture()
def fake_uow() -> FakeUnitOfWork:
    """An empty in-memory unit of work — the default way to test a service."""
    return FakeUnitOfWork()


@pytest.fixture()
def fake_uow_factory(fake_uow: FakeUnitOfWork) -> Callable[[], FakeUnitOfWork]:
    """The *same* unit of work each call, so writes survive between a preview and a commit."""
    return lambda: fake_uow


@pytest.fixture()
def seeded_fakes(fake_uow: FakeUnitOfWork) -> FakeUnitOfWork:
    """One year, two levels, 3A and 3B, two terms, two subjects, two children.

    Layla transfers 3A -> 3B in March, which is the fixture the whole of invariant 2
    hangs on: asked for a day in Term 1 she must still resolve to 3A, and asked for
    today, 3B. A seed where nobody ever moved would let a `students.class_code`
    implementation pass every test in the suite.
    """
    year = AcademicYear(
        code=AcademicYearCode("2025-2026"),
        school_code="MAIN",
        name_en="2025/2026",
        name_ar="٢٠٢٥/٢٠٢٦",
        starts_on=date(2025, 9, 1),
        ends_on=date(2026, 6, 30),
        is_current=True,
    )
    level = YearLevel(code=YearCode("Y3"), school_code="MAIN", name_en="Year 3", name_ar="السنة الثالثة", display_order=3)
    sections = [
        ClassSection(
            code=ClassCode(code_text),
            academic_year_code=year.code,
            year_level_code=level.code,
            name_en=f"Year 3 {code_text[-1]}",
            name_ar=f"الثالث {code_text[-1]}",
        )
        for code_text in ("3A", "3B")
    ]
    terms = [
        Term(
            code=TermCode("2026-T1"),
            academic_year_code=year.code,
            name_en="Term 1",
            name_ar="الفصل الأول",
            starts_on=date(2025, 9, 1),
            ends_on=date(2025, 12, 20),
            sequence=1,
        ),
        Term(
            code=TermCode("2026-T2"),
            academic_year_code=year.code,
            name_en="Term 2",
            name_ar="الفصل الثاني",
            starts_on=date(2026, 1, 10),
            ends_on=date(2026, 6, 10),
            sequence=2,
        ),
    ]
    # Both in `year`, because a subject belongs to one: the fixture's marks are stated in
    # this year's terms, and a subject filed under any other year would not resolve for them.
    subjects = [
        Subject(
            code=SubjectCode("MATH"),
            academic_year_code=year.code,
            name_en="Mathematics",
            name_ar="الرياضيات",
            display_order=1,
        ),
        Subject(
            code=SubjectCode("ARB"),
            academic_year_code=year.code,
            name_en="Arabic",
            name_ar="اللغة العربية",
            display_order=2,
        ),
    ]
    students = [
        Student(
            student_number=StudentNumber("S-1001"),
            full_name_en="Layla Hassan",
            full_name_ar="ليلى حسن",
        ),
        Student(
            student_number=StudentNumber("S-2002"),
            full_name_en="Omar Khaled",
            full_name_ar="عمر خالد",
        ),
    ]
    enrolments = [
        ClassEnrolment(
            student_number=students[0].student_number,
            academic_year_code=year.code,
            class_code=ClassCode("3A"),
            starts_on=date(2025, 9, 1),
            ends_on=date(2026, 3, 15),
        ),
        ClassEnrolment(
            student_number=students[0].student_number,
            academic_year_code=year.code,
            class_code=ClassCode("3B"),
            starts_on=date(2026, 3, 16),
        ),
        ClassEnrolment(
            student_number=students[1].student_number,
            academic_year_code=year.code,
            class_code=ClassCode("3A"),
            starts_on=date(2025, 9, 1),
        ),
    ]

    fake_uow.schools.upsert_many([School(code="MAIN", name_en="Main School", name_ar="المدرسة")])
    fake_uow.academic_years.upsert_many([year])
    fake_uow.year_levels.upsert_many([level])
    fake_uow.class_sections.upsert_many(sections)
    fake_uow.terms.upsert_many(terms)
    fake_uow.subjects.upsert_many(subjects)
    # One family, shaped to exercise the two things a single-guardian seed cannot: a
    # mother shared by both children (so "the same adult on two rows is one row in the
    # database" is under test by default), and a sibling whose access is restricted (so
    # every read has one link it must not return to a parent).
    guardians = [
        Guardian(
            phones=(Phone("+201001234567"), Phone("+201119998888")),
            full_name_ar="فاطمة علي",
            full_name_en="Fatma Ali",
        ),
        Guardian(phones=(Phone("+201002223333"),), full_name_en="Hassan Mahmoud"),
        Guardian(phones=(Phone("+201005554444"),), full_name_en="Karim Hassan"),
    ]
    links = [
        StudentGuardian(
            student_number=students[0].student_number,
            guardian_phone=Phone("+201001234567"),
            relationship_type=RelationshipType.MOTHER,
            is_primary_contact=True,
            can_view_records=True,
        ),
        StudentGuardian(
            student_number=students[0].student_number,
            guardian_phone=Phone("+201002223333"),
            relationship_type=RelationshipType.FATHER,
            can_view_records=True,
        ),
        # Restricted on purpose: on file as a contact, reads nothing.
        StudentGuardian(
            student_number=students[0].student_number,
            guardian_phone=Phone("+201005554444"),
            relationship_type=RelationshipType.SIBLING,
            relationship_label="big brother",
            can_view_records=False,
            restriction_note="court order 2026/114",
        ),
        # The same mother, on the second child.
        StudentGuardian(
            student_number=students[1].student_number,
            guardian_phone=Phone("+201001234567"),
            relationship_type=RelationshipType.MOTHER,
            is_primary_contact=True,
            can_view_records=True,
        ),
    ]

    fake_uow.students.upsert_many(students)
    fake_uow.enrolments.upsert_many(enrolments)
    fake_uow.guardians.upsert_many(guardians)
    fake_uow.student_guardians.upsert_many(links)

    section_ids = fake_uow.class_sections.ids_for([s.identity for s in sections])
    fake_uow.grades.upsert_many(
        [
            SubjectGrade(
                student_number=students[0].student_number,
                subject_code=SubjectCode("MATH"),
                term_code=TermCode("2026-T1"),
                class_section_id=section_ids[("2025-2026", "3A")],
                class_code=ClassCode("3A"),
                percentage=Percentage(88.5),
            ),
            # Ungraded on purpose: `None`, not 0.0. Every reporting test needs one row
            # that is awaiting a mark, or invariant 1 is never actually exercised.
            SubjectGrade(
                student_number=students[0].student_number,
                subject_code=SubjectCode("ARB"),
                term_code=TermCode("2026-T1"),
                class_section_id=section_ids[("2025-2026", "3A")],
                class_code=ClassCode("3A"),
                percentage=None,
            ),
            # A real zero, beside the blank above: the pair is what proves a query
            # distinguishes "scored nothing" from "not marked yet".
            SubjectGrade(
                student_number=students[1].student_number,
                subject_code=SubjectCode("MATH"),
                term_code=TermCode("2026-T1"),
                class_section_id=section_ids[("2025-2026", "3A")],
                class_code=ClassCode("3A"),
                percentage=Percentage(0.0),
            ),
        ]
    )
    return fake_uow


__all__ = [
    "READER_KEY",
    "REGISTRAR_KEY",
    "FakeAcademicYearRepository",
    "FakeApiKeyRepository",
    "FakeClassSectionRepository",
    "FakeEnrolmentRepository",
    "FakeGradeRepository",
    "FakeGuardianRepository",
    "FakeImportBatchRepository",
    "FakeStudentGuardianRepository",
    "FakeStudentRepository",
    "FakeSubjectRepository",
    "FakeTermRepository",
    "FakeUnitOfWork",
    "FakeYearLevelRepository",
    "StubParser",
    "csv_bytes",
    "reader_headers",
    "registrar_headers",
    "xlsx_bytes",
]
