"""SQLAlchemy repositories for the school's skeleton: years, levels, classes, terms, subjects.

Three rules govern every method in this module.

**A SQLAlchemy object never leaves it.** Everything returned is one of the frozen
dataclasses of `sis.domain.structure`, built column by column. Handing a service an ORM
row instead looks identical in a test and fails in production: the row belongs to a
session, so reading it after the request's session closed raises `DetachedInstanceError`,
and the relationships on these models are `lazy="raise"` exactly so that the alternative
mistake — a lazy load per row — is loud rather than a class list that quietly takes nine
seconds. The deeper reason is decision 7: a service holding an ORM row can assign to
`row.code`, and changing a code detaches every enrolment and grade hanging off it.

**Bulk methods cost a fixed number of statements, never one per row.** Each `upsert_many`
below states its own query count in a comment. A roster or a generated structure carries
hundreds of rows, and a repository that loops is how one upload becomes a thousand round
trips and a registrar watching a spinner.

**Flags that own a dedicated method are never written by an upsert.** `is_current` and
`is_closed` are set by `set_current` and `set_closed` and are excluded from every
ON CONFLICT update clause. Invariant 3 makes structure generation re-runnable, and a
re-run carrying `is_current=False` on last year's row — or `is_closed=False` on a term the
school froze in December — would silently undo an administrative act nobody re-performed.
`Subject.is_active` is the deliberate exception: `SubjectRepository` has no `set_active`,
so upsert is the only path by which a subject can be retired.
"""
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Select, delete, select, true, tuple_, update
from sqlalchemy.dialects.postgresql import insert as _postgresql_insert
from sqlalchemy.dialects.sqlite import insert as _sqlite_insert
from sqlalchemy.orm import Session

from sis.application.ports.repositories import ClassSectionKey, GradeSubjects
from sis.domain.errors import DuplicateCode, UnknownReference
from sis.domain.timetable import TimetableEntry, TimetablePeriod, TimetableSlot
from sis.domain.structure import (
    AcademicTrack,
    AcademicYear,
    ClassSection,
    School,
    SchoolLanguage,
    Subject,
    Term,
    WorkingDay,
    YearLevel,
)
from sis.domain.value_objects import (
    AcademicYearCode,
    ClassCode,
    SchoolCode,
    SubjectCode,
    TermCode,
    YearCode,
)
from sis.infrastructure.db import models

# Rows per INSERT statement. Sized against SQLite's 32766-parameter ceiling (3.32+, and
# ON CONFLICT already requires 3.24+), leaving room for the widest row here: 500 x ~10
# bound parameters. A batch larger than this costs one extra statement, not one per row.
_CHUNK_ROWS = 500


def _utcnow() -> datetime:
    """Timestamps written by hand, because Core writes skip the models' Python defaults."""
    return datetime.now(timezone.utc)


def _dialect_insert(session: Session) -> Any:
    """The dialect's INSERT, which is the only one carrying `on_conflict_do_update`.

    Upsert is not in the generic construct because it is not standard SQL, so the choice
    has to be made here rather than expressed once. An unsupported dialect fails on the
    first write with a sentence naming the problem, instead of silently falling back to a
    read-modify-write loop that races two concurrent imports against each other.
    """
    name = session.get_bind().dialect.name
    if name == "postgresql":
        return _postgresql_insert
    if name == "sqlite":
        return _sqlite_insert
    raise NotImplementedError(
        f"idempotent upsert is implemented for postgresql and sqlite, not {name!r}"
    )


def _write_upsert(
    session: Session,
    model: type[Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    conflict_on: Sequence[str],
    update_columns: Sequence[str],
) -> None:
    """Insert or update every row, one statement per `_CHUNK_ROWS`, keyed on natural identity.

    Rows are de-duplicated on the conflict key with the last occurrence winning — the same
    result a sequential apply would leave. That is not tidiness: PostgreSQL aborts the
    whole statement with "ON CONFLICT DO UPDATE command cannot affect row a second time"
    when one INSERT names a key twice, and SQLite instead applies both, so a file listing
    `3A` on two lines would fail an import on one database and succeed on the other.
    """
    deduplicated = {tuple(row[name] for name in conflict_on): dict(row) for row in rows}
    ordered = list(deduplicated.values())
    insert = _dialect_insert(session)
    for start in range(0, len(ordered), _CHUNK_ROWS):
        statement = insert(model).values(ordered[start : start + _CHUNK_ROWS])
        session.execute(
            statement.on_conflict_do_update(
                index_elements=list(conflict_on),
                set_={name: getattr(statement.excluded, name) for name in update_columns},
            )
        )
    _sync(session)


def _existing_keys(
    session: Session,
    model: type[Any],
    rows: Sequence[Mapping[str, Any]],
    conflict_on: Sequence[str],
) -> set[tuple[Any, ...]]:
    """Which natural keys are already on file — one statement whatever the row count.

    The filter is one `IN` per key column rather than a row-value `(a, b) IN ((...))`,
    which is not portable across the dialects this service runs on. A per-column filter
    over-matches — it selects every *combination* of the values asked for — so the result
    is intersected with the keys actually requested before it is returned.
    """
    columns = [getattr(model, name) for name in conflict_on]
    wanted = {tuple(row[name] for name in conflict_on) for row in rows}
    filters = [
        column.in_({row[name] for row in rows})
        for column, name in zip(columns, conflict_on, strict=True)
    ]
    found = session.execute(select(*columns).where(*filters)).all()
    return {tuple(row) for row in found} & wanted


def bulk_upsert(
    session: Session,
    model: type[Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    conflict_on: Sequence[str],
    update_columns: Sequence[str],
) -> set[tuple[Any, ...]]:
    """Upsert `rows`, returning the keys that already existed. Two statements per batch.

    The returned set is what every port's `Mapping[key, bool]` is built from: created is
    "not in this set". It is read *before* the write for the obvious reason — afterwards
    every key exists and the distinction is gone — which is also why an idempotent re-run
    can report "40 already present" without a second pass over the table.
    """
    if not rows:
        return set()
    existing = _existing_keys(session, model, rows, conflict_on)
    _write_upsert(
        session, model, rows, conflict_on=conflict_on, update_columns=update_columns
    )
    return existing


def _sync(session: Session) -> None:
    """Drop cached ORM state after a Core write.

    Core statements bypass the identity map entirely, so an entity loaded earlier in this
    session keeps the attributes it had before the UPDATE. Without this, a `get()` that
    follows an `upsert_many()` in the same request returns the labels the class had before
    the rename while the database holds the new ones.
    """
    session.expire_all()


def _require(mapping: Mapping[str, int], code: str, field: str) -> int:
    """Resolve a code to a surrogate id, or say which reference failed."""
    resolved = mapping.get(code)
    if resolved is None:
        raise UnknownReference(f"no {field.replace('_', ' ')} {code!r} on file", field=field)
    return resolved


def _require_level(
    mapping: Mapping[tuple[str, str], int], year_code: str, level_code: str
) -> int:
    """Resolve a rung within one academic year's school, or say so naming both.

    The message names the year as well as the rung because "no year level 'Y1' on file"
    is baffling to a registrar looking at a screen that lists Y1 — the rung exists, just
    not at the school this year belongs to.
    """
    resolved = mapping.get((year_code, level_code))
    if resolved is None:
        raise UnknownReference(
            f"no year level {level_code!r} at the school that owns academic year "
            f"{year_code!r}",
            field="year_level_code",
        )
    return resolved


def _ids_by_code(session: Session, model: type[Any], codes: Collection[str]) -> dict[str, int]:
    """One statement resolving many codes to surrogate ids; absent codes are simply missing."""
    if not codes:
        return {}
    rows = session.execute(
        select(model.code, model.id).where(model.code.in_(set(codes)))
    ).all()
    return {code: identifier for code, identifier in rows}


def _level_ids_by_year(
    session: Session,
    year_codes: Collection[str],
    level_codes: Collection[str],
) -> dict[tuple[str, str], int]:
    """Resolve rung codes *within the school that owns each academic year*.

    Rung codes are unique per school, not globally (`uq_year_levels_school_code`): "Y1"
    exists at every branch. `_ids_by_code` keys purely on the code string, so with two
    schools in one database it collapses every branch's `Y1` into whichever row the
    database returned last — and structure generation for a new school then wired its
    classes to *another school's* rungs. The visible symptom was a school where nobody
    had created a class reporting the main school's class counts.

    Keying on `(academic_year_code, year_level_code)` closes that: the academic year names
    the school, so the pair names exactly one rung. Still one statement, so the "four
    statements regardless of size" property of `upsert_many` is unchanged.

    Physical separation makes the collision impossible in a deployed multi-school estate —
    the branches are in different files. This matters for the database that still holds
    several schools: a deployment before the split, and any single-database install.
    """
    if not year_codes or not level_codes:
        return {}
    rows = session.execute(
        select(models.AcademicYear.code, models.YearLevel.code, models.YearLevel.id)
        .join(models.School, models.AcademicYear.school_id == models.School.id)
        .join(models.YearLevel, models.YearLevel.school_id == models.School.id)
        .where(
            models.AcademicYear.code.in_(set(year_codes)),
            models.YearLevel.code.in_(set(level_codes)),
        )
    ).all()
    return {(year_code, level_code): identifier for year_code, level_code, identifier in rows}


# ---------------------------------------------------------------------------
# ORM row -> domain entity. The only place both vocabularies appear.
# ---------------------------------------------------------------------------


def _to_school(row: models.School) -> School:
    return School(
        code=row.code,
        name_en=row.name_en,
        name_ar=row.name_ar,
        is_active=row.is_active,
        language_type=row.language_type,
        kg_grade_count=row.kg_grade_count,
        primary_grade_count=row.primary_grade_count,
        preparatory_grade_count=row.preparatory_grade_count,
        secondary_grade_count=row.secondary_grade_count,
        term_count=row.term_count,
        working_days=tuple(WorkingDay(day) for day in row.working_days.split(",") if day),
    )


def _to_year(row: models.AcademicYear, school_code: str) -> AcademicYear:
    """The school code is passed in, never read off `row.school`.

    Same reason as every other parent code in this module: the relationship is not eagerly
    loaded, so touching it would emit a SELECT per year. The selects below join `schools`
    and hand the code down.
    """
    return AcademicYear(
        code=row.code,
        school_code=school_code,
        name_en=row.name_en,
        name_ar=row.name_ar,
        starts_on=row.starts_on,
        ends_on=row.ends_on,
        is_current=row.is_current,
    )


def _to_level(row: models.YearLevel, school_code: str, track_code: str | None = None) -> YearLevel:
    return YearLevel(
        code=row.code,
        school_code=school_code,
        name_en=row.name_en,
        name_ar=row.name_ar,
        display_order=row.display_order,
        stage=row.stage,
        track_code=track_code,
    )


def _to_section(
    row: models.ClassSection, academic_year_code: str, year_level_code: str
) -> ClassSection:
    """The year and level codes are passed in, never read off the relationships.

    `ClassSection.academic_year` is `lazy="raise"`, so touching it here would raise on
    every single row; the selects below join both parents and hand their codes down. That
    is the point of the raise: the alternative spelling compiles and issues two extra
    queries per class.
    """
    return ClassSection(
        code=row.code,
        academic_year_code=academic_year_code,
        year_level_code=year_level_code,
        name_en=row.name_en,
        name_ar=row.name_ar,
        capacity=row.capacity,
    )


def _to_term(row: models.Term, academic_year_code: str) -> Term:
    return Term(
        code=row.code,
        academic_year_code=academic_year_code,
        name_en=row.name_en,
        name_ar=row.name_ar,
        starts_on=row.starts_on,
        ends_on=row.ends_on,
        sequence=row.sequence,
        is_closed=row.is_closed,
    )


def _to_subject(row: models.Subject, academic_year_code: str) -> Subject:
    """The year code is passed in rather than read off `row.academic_year`.

    Same reason as `_to_term`: touching the relationship would emit a `SELECT` per subject,
    so a report card's worth of columns would be a query each. Every caller here already
    knows the year — it is what they narrowed the lookup by.
    """
    return Subject(
        code=row.code,
        academic_year_code=academic_year_code,
        name_en=row.name_en,
        name_ar=row.name_ar,
        display_order=row.display_order,
        is_active=row.is_active,
    )


class SqlAlchemyAcademicYearRepository:
    """`AcademicYearRepository` over SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, code: AcademicYearCode) -> AcademicYear | None:
        # No school argument: a year code is globally unique, so it names exactly one year
        # at exactly one school. That is what keeps every caller of this method — and every
        # route taking `?academic_year=` — unchanged by the arrival of multiple schools.
        found = self._session.execute(
            select(models.AcademicYear, models.School.code)
            .join(models.School)
            .where(models.AcademicYear.code == str(code))
        ).first()
        return None if found is None else _to_year(found[0], found[1])

    def get_many(self, codes: Collection[AcademicYearCode]) -> Mapping[str, AcademicYear]:
        if not codes:
            return {}
        rows = self._session.execute(
            select(models.AcademicYear, models.School.code)
            .join(models.School)
            .where(models.AcademicYear.code.in_({str(code) for code in codes}))
        ).all()
        return {row[0].code: _to_year(row[0], row[1]) for row in rows}

    def list_all(self, school_code: SchoolCode | None = None) -> Sequence[AcademicYear]:
        """Every year, newest first; one school's when asked for.

        The filter is optional rather than required because the school picker itself needs
        the unfiltered list to know which schools have years at all.
        """
        statement = (
            select(models.AcademicYear, models.School.code)
            .join(models.School)
            .order_by(
                models.AcademicYear.starts_on.desc(), models.AcademicYear.code.desc()
            )
        )
        if school_code is not None:
            statement = statement.where(models.School.code == str(school_code))
        return [_to_year(row[0], row[1]) for row in self._session.execute(statement).all()]

    def current(self, school_code: SchoolCode | None = None) -> AcademicYear | None:
        """The flagged year. `.first()` on purpose — see `set_current` for why two can exist.

        Per school when asked: two branches each have a current year, and they are not
        required to be the same one — a school that starts its year a fortnight later is
        still mid-changeover while the other has moved on.
        """
        statement = (
            select(models.AcademicYear, models.School.code)
            .join(models.School)
            .where(models.AcademicYear.is_current.is_(True))
            .order_by(models.AcademicYear.starts_on.desc())
        )
        if school_code is not None:
            statement = statement.where(models.School.code == str(school_code))
        found = self._session.execute(statement).first()
        return None if found is None else _to_year(found[0], found[1])

    def set_current(self, code: AcademicYearCode) -> AcademicYear:
        """Clear the old flag and set the new one in a single UPDATE.

        The schema deliberately does not make `is_current` unique, because enforcing it
        would force the rollover into two statements — and between two statements the
        school has either no current year or two of them, so every "this year's classes"
        screen loading in that window answers wrongly. Setting the column to a *predicate*
        (`is_current = (code = :code)`) closes the window: one row gains the flag exactly
        as the other loses it, and the WHERE clause keeps the write to those two rows.
        """
        wanted = str(code)
        # The school's code comes back with the row rather than from a second read: a year
        # is only ever a year *of a school*, so `_to_year` needs it, and fetching it here
        # costs a join on a query that was already running.
        found = self._session.execute(
            select(models.AcademicYear, models.School.code)
            .join(models.School)
            .where(models.AcademicYear.code == wanted)
        ).first()
        if found is None:
            raise UnknownReference(
                f"no academic year {wanted!r} on file", field="academic_year_code"
            )
        # Read before the write; `_sync` expires the row afterwards.
        year = _to_year(found[0], found[1])
        self._session.execute(
            update(models.AcademicYear)
            .where(
                models.AcademicYear.is_current.is_(True)
                | (models.AcademicYear.code == wanted)
            )
            .values(
                is_current=(models.AcademicYear.code == wanted), updated_at=_utcnow()
            )
        )
        _sync(self._session)
        return replace(year, is_current=True)

    def upsert_many(self, years: Sequence[AcademicYear]) -> Mapping[str, bool]:
        # Two statements: one SELECT of the codes already on file, one INSERT .. ON
        # CONFLICT DO UPDATE (per 500-row chunk). Never one per year.
        if not years:
            return {}
        school_ids = _ids_by_code(
            self._session, models.School, {str(year.school_code) for year in years}
        )
        now = _utcnow()
        rows = [
            {
                "code": str(year.code),
                "school_id": _require(
                    school_ids, str(year.school_code), "school_code"
                ),
                "name_en": year.name_en,
                "name_ar": year.name_ar,
                "starts_on": year.starts_on,
                "ends_on": year.ends_on,
                "is_current": year.is_current,
                "created_at": now,
                "updated_at": now,
            }
            for year in years
        ]
        existing = bulk_upsert(
            self._session,
            models.AcademicYear,
            rows,
            conflict_on=("code",),
            # `created_at` stays out: rewriting it on every idempotent re-run would make
            # "when was this year set up" mean "when did generation last run". `school_id`
            # stays out for a stronger reason — moving a year to another school would carry
            # every class, term, subject and mark in it across, under a school that never
            # taught them.
            update_columns=("name_en", "name_ar", "starts_on", "ends_on", "updated_at"),
        )
        return {row["code"]: (row["code"],) not in existing for row in rows}


class SqlAlchemyYearLevelRepository:
    """`YearLevelRepository` over SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, code: YearCode, school_code: SchoolCode) -> YearLevel | None:
        # The school is required here and not for an academic year, and the asymmetry is
        # deliberate: a rung code is unique only within a school, because "Year 1" exists at
        # every branch. Callers always have the school — they reached the rung through a
        # year, which names one.
        found = self._session.execute(
            select(models.YearLevel, models.School.code, models.EducationalSystem.code)
            .join(models.School, models.YearLevel.school_id == models.School.id)
            .outerjoin(
                models.EducationalSystem,
                models.YearLevel.educational_system_id == models.EducationalSystem.id,
            )
            .where(
                models.YearLevel.code == str(code),
                models.School.code == str(school_code),
            )
        ).first()
        return None if found is None else _to_level(found[0], found[1], found[2])

    def get_many(
        self, codes: Collection[YearCode], school_code: SchoolCode
    ) -> Mapping[str, YearLevel]:
        if not codes:
            return {}
        rows = self._session.execute(
            select(models.YearLevel, models.School.code, models.EducationalSystem.code)
            .join(models.School, models.YearLevel.school_id == models.School.id)
            .outerjoin(
                models.EducationalSystem,
                models.YearLevel.educational_system_id == models.EducationalSystem.id,
            )
            .where(
                models.YearLevel.code.in_({str(code) for code in codes}),
                models.School.code == str(school_code),
            )
        ).all()
        return {row[0].code: _to_level(row[0], row[1], row[2]) for row in rows}

    def list_for_school(self, school_code: SchoolCode) -> Sequence[YearLevel]:
        """One school's ladder, grouped by stage and ordered within it.

        Ordered by the stored columns, never by code: lexicographically "Y10" precedes "Y9",
        and a school with ten rungs then prints a list no parent recognises. Stage leads
        because that is how the ladder is read — the garden block, then primary — and the
        index `ix_year_levels_school_stage` exists for exactly this ORDER BY.
        """
        rows = self._session.execute(
            select(models.YearLevel, models.School.code, models.EducationalSystem.code)
            .join(models.School, models.YearLevel.school_id == models.School.id)
            .outerjoin(
                models.EducationalSystem,
                models.YearLevel.educational_system_id == models.EducationalSystem.id,
            )
            .where(models.School.code == str(school_code))
            .order_by(
                models.YearLevel.stage,
                models.YearLevel.display_order,
                models.YearLevel.code,
            )
        ).all()
        # Sorted again in Python, on `sort_key`, because the stage order a school reads in
        # (garden, primary, preparatory, secondary) is not the alphabetical order SQL just
        # produced. The SQL ORDER BY still earns its keep: it makes the result deterministic
        # and lets the index serve the scan.
        return sorted(
            (_to_level(row[0], row[1], row[2]) for row in rows), key=lambda level: level.sort_key
        )

    def upsert_many(self, levels: Sequence[YearLevel]) -> Mapping[str, bool]:
        # Three statements, as `TermRepository.upsert_many`: resolve the schools, read which
        # (school, code) pairs exist, write.
        if not levels:
            return {}
        school_ids = _ids_by_code(
            self._session, models.School, {str(level.school_code) for level in levels}
        )
        system_rows = self._session.execute(
            select(models.EducationalSystem.school_id, models.EducationalSystem.code,
                   models.EducationalSystem.id)
            .where(models.EducationalSystem.school_id.in_(set(school_ids.values())))
        ).all()
        system_ids = {(school_id, code): identifier for school_id, code, identifier in system_rows}
        now = _utcnow()
        rows = [
            {
                "code": str(level.code),
                "school_id": _require(
                    school_ids, str(level.school_code), "school_code"
                ),
                "stage": str(level.stage),
                "educational_system_id": (
                    system_ids.get((school_ids[str(level.school_code)], level.track_code))
                    if level.track_code else None
                ),
                "name_en": level.name_en,
                "name_ar": level.name_ar,
                "display_order": level.display_order,
                "created_at": now,
            }
            for level in levels
        ]
        existing = bulk_upsert(
            self._session,
            models.YearLevel,
            rows,
            # The identity swap: the same rung code in another school inserts rather than
            # colliding, which is the whole point of scoping rungs by school.
            conflict_on=("school_id", "code"),
            # `stage` is updatable — it is a label a registrar corrects, and correcting it
            # moves the rung between groups on screen and does nothing else. `school_id` is
            # not, for the reason a year's is not: it would carry every class and mark under
            # the rung into a school that never taught them.
            update_columns=(
                "name_en", "name_ar", "display_order", "stage", "educational_system_id"
            ),
        )
        # Keyed on the pair, matching `conflict_on`. Checking `(code,)` against a set of
        # `(school_id, code)` tuples never matches, so every rung would be reported as
        # newly created — and structure generation would announce a full ladder built on
        # every re-run, which is invariant 3 reported as broken while working correctly.
        return {
            row["code"]: (row["school_id"], row["code"]) not in existing for row in rows
        }


# Class sections carry their parents' codes, and the relationships are `lazy="raise"`, so
# every read joins both parents in the same statement. Declared once so no query path can
# forget the join and discover it as a raise at runtime.
_SECTION_ROWS: Select[Any] = (
    select(models.ClassSection, models.AcademicYear.code, models.YearLevel.code)
    .join(models.AcademicYear, models.ClassSection.academic_year_id == models.AcademicYear.id)
    .join(models.YearLevel, models.ClassSection.year_level_id == models.YearLevel.id)
)


def _one_per_key(pairs: Iterable[tuple[ClassSectionKey, Any]]) -> dict[ClassSectionKey, Any]:
    """Collapse rows to one per `(academic_year_code, class_code)`, refusing to guess.

    The unique key in the schema is the *triple* including the year level, because uniform
    generation emits section codes like `A`..`H` under every level. So a bare "3A" within a
    year can legitimately match two rows, and the ports here take only the pair. Picking
    one — `.first()`, on an ordering nobody chose — enrols a child in the wrong room and
    prints her marks under a class she never sat in, with no error anywhere. Raising is
    loud and the registrar can qualify the code; guessing is silent and wrong.
    """
    resolved: dict[ClassSectionKey, Any] = {}
    for key, value in pairs:
        if key in resolved:
            raise DuplicateCode(
                f"class code {key[1]!r} is not unique within academic year {key[0]!r}: "
                "two year levels use it, so this lookup has no single answer",
                field="class_code",
            )
        resolved[key] = value
    return resolved


class SqlAlchemyClassSectionRepository:
    """`ClassSectionRepository` over SQLAlchemy. Identity is `(year, code)`; see `_one_per_key`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(
        self, academic_year_code: AcademicYearCode, code: ClassCode
    ) -> ClassSection | None:
        found = self.get_many([(str(academic_year_code), str(code))])
        return found.get((str(academic_year_code), str(code)))

    def get_many(
        self, keys: Collection[ClassSectionKey]
    ) -> Mapping[ClassSectionKey, ClassSection]:
        rows = self._section_rows(keys)
        return _one_per_key(
            ((year_code, section.code), _to_section(section, year_code, level_code))
            for section, year_code, level_code in rows
        )

    def list_for_year(
        self,
        academic_year_code: AcademicYearCode,
        *,
        year_level_code: YearCode | None = None,
    ) -> Sequence[ClassSection]:
        statement = _SECTION_ROWS.where(
            models.AcademicYear.code == str(academic_year_code)
        ).order_by(models.YearLevel.display_order, models.ClassSection.code)
        if year_level_code is not None:
            statement = statement.where(models.YearLevel.code == str(year_level_code))
        return [
            _to_section(section, year_code, level_code)
            for section, year_code, level_code in self._session.execute(statement).all()
        ]

    def ids_for(self, keys: Collection[ClassSectionKey]) -> Mapping[ClassSectionKey, int]:
        # One statement for the whole roster. A grade import resolves a surrogate id per
        # row, and doing that per row is the query storm this method exists to collapse.
        rows = self._section_rows(keys)
        return _one_per_key(
            ((year_code, section.code), section.id) for section, year_code, _ in rows
        )

    def rename(
        self,
        academic_year_code: AcademicYearCode,
        code: ClassCode,
        *,
        name_en: str | None = None,
        name_ar: str | None = None,
    ) -> ClassSection:
        """Labels only. The UPDATE below cannot touch `code` — invariant 6 as a statement."""
        key = (str(academic_year_code), str(code))
        rows = self._section_rows([key])
        found = _one_per_key(
            ((year_code, section.code), (section, year_code, level_code))
            for section, year_code, level_code in rows
        ).get(key)
        if found is None:
            raise UnknownReference(
                f"no class {key[1]!r} in academic year {key[0]!r}", field="class_code"
            )
        section, year_code, level_code = found
        # Renaming through the domain object, not around it: `renamed()` rejects a change
        # that would leave the class blank in both languages, which reaches a parent as a
        # report card line carrying a mark and no class beside it.
        renamed = _to_section(section, year_code, level_code).renamed(
            name_en=name_en, name_ar=name_ar
        )
        self._session.execute(
            update(models.ClassSection)
            .where(models.ClassSection.id == section.id)
            .values(
                name_en=renamed.name_en, name_ar=renamed.name_ar, updated_at=_utcnow()
            )
        )
        _sync(self._session)
        return renamed

    def upsert_many(
        self, sections: Sequence[ClassSection]
    ) -> Mapping[ClassSectionKey, bool]:
        """The one path structure generation writes through — uniform or per-year alike.

        Four statements regardless of size: resolve the academic years, resolve the year
        levels, read which sections already exist, write. Invariant 3 falls out of the last
        two: the write collides on `(academic_year_id, year_level_id, code)` and updates,
        and the read taken beforehand is what lets the caller say "40 already present".

        Existence is judged on that triple, but the port keys its answer by the *pair*
        `(year, code)` — and uniform generation emits `A`..`H` under every level, so one
        pair can name several rows. The entry then reports whether this call created any of
        them, which is the only honest reading of a key that cannot tell them apart.
        """
        if not sections:
            return {}
        year_ids = _ids_by_code(
            self._session,
            models.AcademicYear,
            {str(section.academic_year_code) for section in sections},
        )
        # Keyed by (year, rung) rather than by rung alone: a bare rung code names a
        # different rung at every school. See `_level_ids_by_year`.
        level_ids = _level_ids_by_year(
            self._session,
            {str(section.academic_year_code) for section in sections},
            {str(section.year_level_code) for section in sections},
        )
        now = _utcnow()
        rows: list[dict[str, Any]] = []
        keys: list[ClassSectionKey] = []
        for section in sections:
            year_code = str(section.academic_year_code)
            rows.append(
                {
                    "academic_year_id": _require(year_ids, year_code, "academic_year_code"),
                    "year_level_id": _require_level(
                        level_ids, year_code, str(section.year_level_code)
                    ),
                    "code": str(section.code),
                    "name_en": section.name_en,
                    "name_ar": section.name_ar,
                    "capacity": section.capacity,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            keys.append((year_code, str(section.code)))

        existing = bulk_upsert(
            self._session,
            models.ClassSection,
            rows,
            conflict_on=("academic_year_id", "year_level_id", "code"),
            # `is_active` stays out: a section retired by a registrar must not be revived
            # by the next idempotent re-run of generation.
            update_columns=("name_en", "name_ar", "capacity", "updated_at"),
        )
        created: dict[ClassSectionKey, bool] = {}
        for key, row in zip(keys, rows, strict=True):
            triple = (row["academic_year_id"], row["year_level_id"], row["code"])
            created[key] = created.get(key, False) or triple not in existing
        return created

    def _section_rows(self, keys: Collection[ClassSectionKey]) -> Sequence[Any]:
        """The joined rows for many `(year, code)` pairs — one statement, over-matched.

        Same portability trade as `_existing_keys`: two `IN` filters rather than a
        row-value comparison, so the result may hold pairs nobody asked for. Callers key
        the result by the pair and read only what they requested.
        """
        if not keys:
            return []
        statement = _SECTION_ROWS.where(
            models.AcademicYear.code.in_({year for year, _ in keys}),
            models.ClassSection.code.in_({code for _, code in keys}),
        )
        wanted = set(keys)
        return [
            row
            for row in self._session.execute(statement).all()
            if (row[1], row[0].code) in wanted
        ]


# Terms carry their academic year's code, and `Term.academic_year` is `lazy="raise"`.
_TERM_ROWS: Select[Any] = select(models.Term, models.AcademicYear.code).join(
    models.AcademicYear, models.Term.academic_year_id == models.AcademicYear.id
)


class SqlAlchemyTermRepository:
    """`TermRepository` over SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, code: TermCode) -> Term | None:
        row = self._session.execute(
            _TERM_ROWS.where(models.Term.code == str(code))
        ).first()
        return None if row is None else _to_term(row[0], row[1])

    def get_many(self, codes: Collection[TermCode]) -> Mapping[str, Term]:
        if not codes:
            return {}
        rows = self._session.execute(
            _TERM_ROWS.where(models.Term.code.in_({str(code) for code in codes}))
        ).all()
        return {term.code: _to_term(term, year_code) for term, year_code in rows}

    def list_for_year(self, academic_year_code: AcademicYearCode) -> Sequence[Term]:
        rows = self._session.execute(
            _TERM_ROWS.where(models.AcademicYear.code == str(academic_year_code)).order_by(
                models.Term.sequence, models.Term.code
            )
        ).all()
        return [_to_term(term, year_code) for term, year_code in rows]

    def set_closed(self, code: TermCode, *, is_closed: bool) -> Term:
        """Freeze or reopen. Refusing writes against a closed term is the service's job."""
        row = self._session.execute(
            _TERM_ROWS.where(models.Term.code == str(code))
        ).first()
        if row is None:
            raise UnknownReference(f"no term {str(code)!r} on file", field="term_code")
        term = _to_term(row[0], row[1])  # read before the write; `_sync` expires the row
        self._session.execute(
            update(models.Term)
            .where(models.Term.code == str(code))
            .values(is_closed=is_closed, updated_at=_utcnow())
        )
        _sync(self._session)
        return replace(term, is_closed=is_closed)

    def delete_if_unused(self, code: TermCode) -> bool:
        """Delete the term unless a mark is stated against it. See the port for the rule.

        One statement, so the check and the delete cannot be separated by another
        registrar's upload: the `NOT EXISTS` runs inside the DELETE rather than before it.
        `rowcount` then answers which of the two outcomes happened, without a second read.

        Only `subject_grades` is consulted, because it is the only table that points at a
        term. Attendance is dated, not termed, and nothing else references one — if that
        changes, this is the query that has to learn about it, and the foreign key will
        say so loudly rather than letting the delete succeed.
        """
        grades_here = (
            select(models.SubjectGrade.id)
            .join(models.Term, models.SubjectGrade.term_id == models.Term.id)
            .where(models.Term.code == str(code))
            .exists()
        )
        result = self._session.execute(
            delete(models.Term).where(models.Term.code == str(code), ~grades_here)
        )
        _sync(self._session)
        return bool(result.rowcount)

    def upsert_many(self, terms: Sequence[Term]) -> Mapping[str, bool]:
        # Three statements: resolve the academic years, read which term codes exist, write.
        if not terms:
            return {}
        year_ids = _ids_by_code(
            self._session,
            models.AcademicYear,
            {str(term.academic_year_code) for term in terms},
        )
        now = _utcnow()
        rows = [
            {
                "code": str(term.code),
                "academic_year_id": _require(
                    year_ids, str(term.academic_year_code), "academic_year_code"
                ),
                "name_en": term.name_en,
                "name_ar": term.name_ar,
                "starts_on": term.starts_on,
                "ends_on": term.ends_on,
                "sequence": term.sequence,
                "is_closed": term.is_closed,
                "created_at": now,
                "updated_at": now,
            }
            for term in terms
        ]
        existing = bulk_upsert(
            self._session,
            models.Term,
            rows,
            conflict_on=("code",),
            # `academic_year_id` is updatable because upsert is the only correction path
            # for a term filed under the wrong year; `is_closed` is not, because reopening
            # a frozen term is an act a re-run of generation must never perform silently.
            update_columns=(
                "academic_year_id",
                "name_en",
                "name_ar",
                "starts_on",
                "ends_on",
                "sequence",
                "updated_at",
            ),
        )
        return {row["code"]: (row["code"],) not in existing for row in rows}


class SqlAlchemySubjectRepository:
    """`SubjectRepository` over SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(
        self, code: SubjectCode, academic_year_code: AcademicYearCode
    ) -> Subject | None:
        row = self._session.execute(
            select(models.Subject)
            .join(models.AcademicYear)
            .where(
                models.Subject.code == str(code),
                models.AcademicYear.code == str(academic_year_code),
            )
        ).scalar_one_or_none()
        return None if row is None else _to_subject(row, str(academic_year_code))

    def get_many(
        self, codes: Collection[SubjectCode], academic_year_code: AcademicYearCode
    ) -> Mapping[str, Subject]:
        # Retired subjects are returned too: a grade sheet naming a subject the school has
        # since dropped must resolve it, or that term's report card prints a mark with no
        # heading. Narrowed to one year, which is what makes the code a unique key here.
        if not codes:
            return {}
        rows = self._session.execute(
            select(models.Subject)
            .join(models.AcademicYear)
            .where(
                models.Subject.code.in_({str(code) for code in codes}),
                models.AcademicYear.code == str(academic_year_code),
            )
        ).scalars()
        return {row.code: _to_subject(row, str(academic_year_code)) for row in rows}

    def list_for_year(
        self,
        academic_year_code: AcademicYearCode,
        *,
        include_inactive: bool = False,
        year_level_code: YearCode | None = None,
    ) -> Sequence[Subject]:
        statement = (
            select(models.Subject)
            .join(models.AcademicYear)
            .where(models.AcademicYear.code == str(academic_year_code))
            .order_by(models.Subject.display_order, models.Subject.code)
        )
        if year_level_code is not None:
            # An inner join, not a filter on a left join: a rung with no assignment
            # teaches nothing, and answering that with the whole catalogue is the exact
            # leak this stage exists to close. Narrowed to the year's own school, because
            # rung codes are unique per school and `P1` exists at every branch.
            statement = (
                statement.join(
                    models.SubjectYearLevel,
                    models.SubjectYearLevel.subject_id == models.Subject.id,
                )
                .join(
                    models.YearLevel,
                    models.YearLevel.id == models.SubjectYearLevel.year_level_id,
                )
                .where(
                    models.YearLevel.code == str(year_level_code),
                    models.YearLevel.school_id == models.AcademicYear.school_id,
                )
            )
        if not include_inactive:
            statement = statement.where(models.Subject.is_active.is_(True))
        return [
            _to_subject(row, str(academic_year_code))
            for row in self._session.execute(statement).scalars()
        ]

    def upsert_many(self, subjects: Sequence[Subject]) -> Mapping[str, bool]:
        # Three statements, as `TermRepository.upsert_many`: resolve the years, read which
        # (year, code) pairs exist, write.
        if not subjects:
            return {}
        year_ids = _ids_by_code(
            self._session,
            models.AcademicYear,
            {str(subject.academic_year_code) for subject in subjects},
        )
        now = _utcnow()
        rows = [
            {
                "code": str(subject.code),
                "academic_year_id": _require(
                    year_ids, str(subject.academic_year_code), "academic_year_code"
                ),
                "name_en": subject.name_en,
                "name_ar": subject.name_ar,
                "display_order": subject.display_order,
                "is_active": subject.is_active,
                "created_at": now,
            }
            for subject in subjects
        ]
        existing = bulk_upsert(
            self._session,
            models.Subject,
            rows,
            # The identity swap, in the one place it changes behaviour: the same code in a
            # different year now inserts instead of colliding.
            conflict_on=("academic_year_id", "code"),
            # `academic_year_id` is deliberately absent from the updatable columns, unlike
            # a term's. Moving a subject between years would carry every mark stated
            # against it into a year the school did not teach it in — and since the code is
            # free to exist in both years, the correction a registrar wants is to create it
            # in the right year, not to drag this row across.
            #
            # `is_active` *is* written: this port has no `set_active`, so upsert is the only
            # way a subject can ever be retired.
            update_columns=("name_en", "name_ar", "display_order", "is_active"),
        )
        return {
            row["code"]: (row["academic_year_id"], row["code"]) not in existing
            for row in rows
        }

    def assignments_for_year(
        self, academic_year_code: AcademicYearCode
    ) -> Sequence[GradeSubjects]:
        """The whole board in one query, grouped in Python rather than in the database.

        Retired subjects are included. A rung that taught `LATIN` until the school dropped
        it still taught it, and hiding the assignment would make the board disagree with
        the marks already filed under that rung.
        """
        rows = self._session.execute(
            select(models.YearLevel.code, models.EducationalSystem.code, models.Subject)
            .join(
                models.SubjectYearLevel,
                models.SubjectYearLevel.year_level_id == models.YearLevel.id,
            )
            .join(models.Subject, models.Subject.id == models.SubjectYearLevel.subject_id)
            .join(
                models.AcademicYear,
                models.AcademicYear.id == models.Subject.academic_year_id,
            )
            .outerjoin(
                models.EducationalSystem,
                models.YearLevel.educational_system_id == models.EducationalSystem.id,
            )
            .where(
                models.AcademicYear.code == str(academic_year_code),
                # A rung of another branch can never be assigned through this repository,
                # but the join is stated anyway: it makes the answer true by construction
                # rather than by trusting every past and future writer.
                models.YearLevel.school_id == models.AcademicYear.school_id,
            )
            .order_by(
                models.YearLevel.code,
                models.Subject.display_order,
                models.Subject.code,
            )
        ).all()
        assigned: dict[str, tuple[str | None, list[Subject]]] = {}
        for level_code, track_code, row in rows:
            _, subjects = assigned.setdefault(level_code, (track_code, []))
            subjects.append(_to_subject(row, str(academic_year_code)))
        return [
            GradeSubjects(year_level_code=code, track_code=track, subjects=tuple(subjects))
            for code, (track, subjects) in assigned.items()
        ]

    def _assignment_ids(
        self,
        code: SubjectCode,
        academic_year_code: AcademicYearCode,
        school_code: SchoolCode,
        year_level_code: YearCode,
    ) -> tuple[int, int]:
        """Resolve both sides at once, so an assignment across schools cannot be written.

        The rung is reached *through* the year's school rather than looked up on its own:
        rung codes are unique per school, so `P1` resolves at every branch, and a lookup
        that did not go through the year would happily attach the main school's Physics to
        the annexe's Secondary 1.
        """
        row = self._session.execute(
            select(models.Subject.id, models.YearLevel.id)
            .join(
                models.AcademicYear,
                models.Subject.academic_year_id == models.AcademicYear.id,
            )
            .join(models.School, models.AcademicYear.school_id == models.School.id)
            .join(models.YearLevel, models.YearLevel.school_id == models.School.id)
            .where(
                models.Subject.code == str(code),
                models.AcademicYear.code == str(academic_year_code),
                models.School.code == str(school_code),
                models.YearLevel.code == str(year_level_code),
            )
        ).one_or_none()
        if row is None:
            raise UnknownReference(
                f"no subject {code} and year level {year_level_code} in "
                f"{academic_year_code} at school {school_code}",
                field="year_level_code",
            )
        return row[0], row[1]

    def assign_to_level(
        self,
        code: SubjectCode,
        academic_year_code: AcademicYearCode,
        school_code: SchoolCode,
        year_level_code: YearCode,
    ) -> bool:
        subject_id, level_id = self._assignment_ids(
            code, academic_year_code, school_code, year_level_code
        )
        # Read-then-write rather than an upsert, because the answer this returns is the
        # point: the board wants to know whether the drop changed anything. The unique
        # constraint is still what makes a duplicate impossible — this check only decides
        # which of "created" and "already there" to report, inside one transaction.
        exists = self._session.scalar(
            select(models.SubjectYearLevel.id).where(
                models.SubjectYearLevel.subject_id == subject_id,
                models.SubjectYearLevel.year_level_id == level_id,
            )
        )
        if exists is not None:
            return False
        self._session.add(
            models.SubjectYearLevel(subject_id=subject_id, year_level_id=level_id)
        )
        self._session.flush()
        return True

    def unassign_from_level(
        self,
        code: SubjectCode,
        academic_year_code: AcademicYearCode,
        school_code: SchoolCode,
        year_level_code: YearCode,
    ) -> None:
        subject_id, level_id = self._assignment_ids(
            code, academic_year_code, school_code, year_level_code
        )
        # Deletes the association row and nothing else. `subject_grades` points at the
        # subject, not at this table, so every mark already awarded on this rung survives
        # a registrar changing their mind about next year's timetable.
        self._session.execute(
            delete(models.SubjectYearLevel).where(
                models.SubjectYearLevel.subject_id == subject_id,
                models.SubjectYearLevel.year_level_id == level_id,
            )
        )



def _to_period(row: models.TimetablePeriod, school_code: str) -> TimetablePeriod:
    """The school code is passed in rather than read off the relationship.

    Same reason as `_to_term`: touching `row.school` would emit a SELECT per period, so
    drawing a seven-period day would cost seven queries for a code the caller already has.
    """
    return TimetablePeriod(
        school_code=school_code,
        period_number=row.period_number,
        name_en=row.name_en,
        name_ar=row.name_ar,
        starts_at=row.starts_at,
        ends_at=row.ends_at,
        is_teaching=row.is_teaching,
    )


def _to_entry(
    row: models.TimetableEntry,
    *,
    class_code: str,
    term_code: str,
    academic_year_code: str,
    subject_code: str | None,
    staff_number: str | None,
) -> TimetableEntry:
    """Every code comes in from the join that loaded the row, for the reason above."""
    return TimetableEntry(
        slot=TimetableSlot(
            class_code=class_code,
            term_code=term_code,
            day_of_week=row.day_of_week,
            period_number=row.period_number,
        ),
        academic_year_code=academic_year_code,
        subject_code=subject_code,
        teacher_staff_number=staff_number,
    )


class SqlAlchemyTimetableRepository:
    """`TimetableRepository` over SQLAlchemy.

    Every read resolves its codes in the same query as the row, so drawing a week is one
    round trip rather than one per lesson. Every write resolves the ids it needs in bulk
    for the same reason — laying out a thirty-five-slot grid is one statement's worth of
    lookups and one upsert, not seventy queries.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- The school's day ---------------------------------------------------

    def list_periods(self, school_code: SchoolCode) -> Sequence[TimetablePeriod]:
        rows = self._session.execute(
            select(models.TimetablePeriod)
            .join(models.School)
            .where(models.School.code == str(school_code))
            .order_by(models.TimetablePeriod.period_number)
        ).scalars()
        return [_to_period(row, str(school_code)) for row in rows]

    def replace_periods(
        self, school_code: SchoolCode, periods: Sequence[TimetablePeriod]
    ) -> Sequence[TimetablePeriod]:
        """Set the school's whole day at once. See the port for why it is whole-grid.

        Deletes what the new grid does not name, then upserts the rest — in that order and
        in one transaction, so a school shrinking from eight periods to seven never has a
        moment with both. A period that lessons are timetabled into is not deleted here:
        the foreign key would not stop it (entries reference the *number*, not the row),
        so the service checks before calling and this method stays a plain write.
        """
        school_id = _require(
            _ids_by_code(self._session, models.School, {str(school_code)}),
            str(school_code),
            "school_code",
        )
        wanted = {period.period_number for period in periods}
        self._session.execute(
            delete(models.TimetablePeriod).where(
                models.TimetablePeriod.school_id == school_id,
                models.TimetablePeriod.period_number.not_in(wanted) if wanted else true(),
            )
        )
        if periods:
            now = _utcnow()
            bulk_upsert(
                self._session,
                models.TimetablePeriod,
                [
                    {
                        "school_id": school_id,
                        "period_number": period.period_number,
                        "name_en": period.name_en,
                        "name_ar": period.name_ar,
                        "starts_at": period.starts_at,
                        "ends_at": period.ends_at,
                        "is_teaching": period.is_teaching,
                        "created_at": now,
                        "updated_at": now,
                    }
                    for period in periods
                ],
                conflict_on=("school_id", "period_number"),
                update_columns=(
                    "name_en",
                    "name_ar",
                    "starts_at",
                    "ends_at",
                    "is_teaching",
                    "updated_at",
                ),
            )
        _sync(self._session)
        return self.list_periods(school_code)

    # -- The lessons --------------------------------------------------------

    def _entry_rows(self) -> Select:
        """One SELECT shape for every read, so the two cannot disagree about the joins.

        The subject and the teacher are outer joins because both are nullable and mean
        something when absent: a free period, and a lesson nobody is assigned to yet. An
        inner join would silently drop exactly the rows this stage creates most of.
        """
        return (
            select(
                models.TimetableEntry,
                models.ClassSection.code,
                models.Term.code,
                models.AcademicYear.code,
                models.Subject.code,
                models.Teacher.staff_number,
            )
            .join(
                models.ClassSection,
                models.TimetableEntry.class_section_id == models.ClassSection.id,
            )
            .join(models.Term, models.TimetableEntry.term_id == models.Term.id)
            .join(
                models.AcademicYear,
                models.TimetableEntry.academic_year_id == models.AcademicYear.id,
            )
            .outerjoin(
                models.Subject, models.TimetableEntry.subject_id == models.Subject.id
            )
            .outerjoin(
                models.Teacher, models.TimetableEntry.teacher_id == models.Teacher.id
            )
        )

    def list_entries(
        self,
        academic_year_code: AcademicYearCode,
        *,
        class_code: ClassCode | None = None,
        term_code: TermCode | None = None,
        year_level_code: YearCode | None = None,
    ) -> Sequence[TimetableEntry]:
        statement = self._entry_rows().where(
            models.AcademicYear.code == str(academic_year_code)
        )
        if class_code is not None:
            statement = statement.where(models.ClassSection.code == str(class_code))
        if term_code is not None:
            statement = statement.where(models.Term.code == str(term_code))
        if year_level_code is not None:
            # Joined through the class rather than through the entry: a lesson has no rung
            # of its own, it has a room, and the room is on exactly one rung.
            statement = statement.join(
                models.YearLevel, models.ClassSection.year_level_id == models.YearLevel.id
            ).where(models.YearLevel.code == str(year_level_code))
        # Ordered as a grid is read. `day_of_week` sorts alphabetically here, which is not
        # the school's week — the caller that knows the school reorders it, because only
        # the school knows whether its week starts on Saturday or Sunday.
        rows = self._session.execute(
            statement.order_by(
                models.ClassSection.code,
                models.Term.sequence,
                models.TimetableEntry.period_number,
            )
        ).all()
        return [
            _to_entry(
                row[0],
                class_code=row[1],
                term_code=row[2],
                academic_year_code=row[3],
                subject_code=row[4],
                staff_number=row[5],
            )
            for row in rows
        ]

    def upsert_entries(
        self, entries: Sequence[TimetableEntry]
    ) -> Mapping[tuple, bool]:
        """Insert or update by slot. See the port: the slot is the identity.

        Four bulk lookups and one upsert, whatever the size of the grid. The alternative —
        resolving a class, a term and a subject per lesson — turns laying out one class's
        week into a hundred queries, and a whole school's into thousands.
        """
        if not entries:
            return {}
        year_ids = _ids_by_code(
            self._session,
            models.AcademicYear,
            {str(entry.academic_year_code) for entry in entries},
        )
        term_ids = _ids_by_code(
            self._session, models.Term, {str(entry.slot.term_code) for entry in entries}
        )
        # Classes are keyed by `(year, code)` rather than by code alone: `3A` names a
        # different room of children every September, and resolving it globally is how a
        # lesson lands on last year's class.
        section_ids = self._session.execute(
            select(models.ClassSection.code, models.AcademicYear.code, models.ClassSection.id)
            .join(
                models.AcademicYear,
                models.ClassSection.academic_year_id == models.AcademicYear.id,
            )
            .where(
                models.ClassSection.code.in_(
                    {str(entry.slot.class_code) for entry in entries}
                ),
                models.AcademicYear.code.in_(
                    {str(entry.academic_year_code) for entry in entries}
                ),
            )
        ).all()
        sections = {(code, year): identifier for code, year, identifier in section_ids}
        # Subjects are per-year too, for the same reason and with the same failure mode.
        subject_rows = self._session.execute(
            select(models.Subject.code, models.AcademicYear.code, models.Subject.id)
            .join(
                models.AcademicYear,
                models.Subject.academic_year_id == models.AcademicYear.id,
            )
            .where(
                models.Subject.code.in_(
                    {
                        str(entry.subject_code)
                        for entry in entries
                        if entry.subject_code is not None
                    }
                    or {""}
                ),
                models.AcademicYear.code.in_(
                    {str(entry.academic_year_code) for entry in entries}
                ),
            )
        ).all()
        subjects = {(code, year): identifier for code, year, identifier in subject_rows}

        now = _utcnow()
        rows = []
        for entry in entries:
            year_code = str(entry.academic_year_code)
            class_code = str(entry.slot.class_code)
            section_id = sections.get((class_code, year_code))
            if section_id is None:
                raise UnknownReference(
                    f"no class {class_code} in {year_code}", field="class_code"
                )
            subject_id = None
            if entry.subject_code is not None:
                subject_id = subjects.get((str(entry.subject_code), year_code))
                if subject_id is None:
                    raise UnknownReference(
                        f"no subject {entry.subject_code} in {year_code}",
                        field="subject_code",
                    )
            rows.append(
                {
                    "class_section_id": section_id,
                    "academic_year_id": _require(
                        year_ids, year_code, "academic_year_code"
                    ),
                    "term_id": _require(
                        term_ids, str(entry.slot.term_code), "term_code"
                    ),
                    "day_of_week": str(entry.slot.day_of_week),
                    "period_number": entry.slot.period_number,
                    "subject_id": subject_id,
                    # Never set in this stage. Left explicit rather than omitted so the
                    # upsert's column list is the table's, and adding it later is a change
                    # to one line here rather than a debugging session.
                    "teacher_id": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )

        existing = bulk_upsert(
            self._session,
            models.TimetableEntry,
            rows,
            conflict_on=(
                "class_section_id",
                "term_id",
                "day_of_week",
                "period_number",
            ),
            # `class_section_id`, `term_id`, `day_of_week` and `period_number` are the
            # slot and are therefore not updatable: changing one of them is moving a
            # lesson, which is a delete and an insert, not an edit.
            update_columns=("subject_id", "teacher_id", "updated_at"),
        )
        return {
            entry.slot.key: (
                row["class_section_id"],
                row["term_id"],
                row["day_of_week"],
                row["period_number"],
            )
            not in existing
            for entry, row in zip(entries, rows, strict=True)
        }

    def delete_entries(self, slots: Sequence[TimetableSlot]) -> int:
        """Clear these slots. See the port on why clearing differs from scheduling nothing."""
        if not slots:
            return 0
        # One statement per distinct (class, term) pair rather than one per slot: a whole
        # class's week clears in one round trip, and a school's in a handful.
        removed = 0
        by_pair: dict[tuple[str, str], list[TimetableSlot]] = {}
        for slot in slots:
            by_pair.setdefault(
                (str(slot.class_code), str(slot.term_code)), []
            ).append(slot)
        for (class_code, term_code), group in by_pair.items():
            result = self._session.execute(
                delete(models.TimetableEntry).where(
                    models.TimetableEntry.id.in_(
                        select(models.TimetableEntry.id)
                        .join(
                            models.ClassSection,
                            models.TimetableEntry.class_section_id
                            == models.ClassSection.id,
                        )
                        .join(
                            models.Term, models.TimetableEntry.term_id == models.Term.id
                        )
                        .where(
                            models.ClassSection.code == class_code,
                            models.Term.code == term_code,
                            tuple_(
                                models.TimetableEntry.day_of_week,
                                models.TimetableEntry.period_number,
                            ).in_(
                                [
                                    (str(slot.day_of_week), slot.period_number)
                                    for slot in group
                                ]
                            ),
                        )
                    )
                )
            )
            removed += result.rowcount or 0
        _sync(self._session)
        return removed


class SqlAlchemySchoolRepository:
    """`SchoolRepository` over SQLAlchemy. The outermost scope in the service.

    Small, because a school is a name and a boundary and nothing else. What it does not
    have is a delete: closing a branch is `is_active = False`, and the RESTRICT on every
    year and rung pointing at it means the database refuses the alternative anyway. The
    registers and marks of the years it ran are still true statements.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, code: SchoolCode) -> School | None:
        row = self._session.execute(
            select(models.School).where(models.School.code == str(code))
        ).scalar_one_or_none()
        return None if row is None else _to_school(row)

    def get_many(self, codes: Collection[SchoolCode]) -> Mapping[str, School]:
        if not codes:
            return {}
        rows = self._session.execute(
            select(models.School).where(
                models.School.code.in_({str(code) for code in codes})
            )
        ).scalars()
        return {row.code: _to_school(row) for row in rows}

    def list_all(self, *, include_inactive: bool = False) -> Sequence[School]:
        """Every school, by code. Closed branches only when asked for by name."""
        statement = select(models.School).order_by(models.School.code)
        if not include_inactive:
            statement = statement.where(models.School.is_active.is_(True))
        return [_to_school(row) for row in self._session.execute(statement).scalars()]

    def list_tracks(self, school_code: SchoolCode) -> Sequence[AcademicTrack]:
        rows = self._session.execute(
            select(models.EducationalSystem, models.School.code)
            .join(models.School)
            .where(
                models.School.code == str(school_code),
                models.EducationalSystem.is_active.is_(True),
            )
            .order_by(models.EducationalSystem.display_order, models.EducationalSystem.code)
        ).all()
        return [
            AcademicTrack(
                code=row.code,
                school_code=code,
                language_type=(
                    SchoolLanguage.ARABIC if row.kind == "arabic" else SchoolLanguage.LANGUAGES
                ),
                name_en=row.name_en,
                name_ar=row.name_ar,
                display_order=row.display_order,
                is_active=row.is_active,
            )
            for row, code in rows
        ]

    def sync_tracks(self, school: School) -> None:
        school_id = _require(
            _ids_by_code(self._session, models.School, {str(school.code)}),
            str(school.code),
            "school_code",
        )
        wanted = []
        if school.language_type in (SchoolLanguage.ARABIC, SchoolLanguage.BOTH):
            wanted.append(("AR", "arabic", "Arabic", "العربية"))
        if school.language_type in (SchoolLanguage.LANGUAGES, SchoolLanguage.BOTH):
            wanted.append(("LANG", "language", "Languages", "اللغات"))
        wanted_codes = {item[0] for item in wanted}
        self._session.execute(
            update(models.EducationalSystem)
            .where(
                models.EducationalSystem.school_id == school_id,
                models.EducationalSystem.code.not_in(wanted_codes),
            )
            .values(is_active=False)
        )
        now = _utcnow()
        bulk_upsert(
            self._session,
            models.EducationalSystem,
            [
                {
                    "school_id": school_id,
                    "code": code,
                    "kind": kind,
                    "name_en": name_en,
                    "name_ar": name_ar,
                    "display_order": order,
                    "is_active": True,
                    "created_at": now,
                }
                for order, (code, kind, name_en, name_ar) in enumerate(wanted, start=1)
            ],
            conflict_on=("school_id", "code"),
            update_columns=("kind", "name_en", "name_ar", "display_order", "is_active"),
        )

    def upsert_many(self, schools: Sequence[School]) -> Mapping[str, bool]:
        # Two statements, as `AcademicYearRepository.upsert_many`.
        if not schools:
            return {}
        now = _utcnow()
        rows = [
            {
                "code": str(school.code),
                "name_en": school.name_en,
                "name_ar": school.name_ar,
                "is_active": school.is_active,
                "language_type": school.language_type.value,
                "kg_grade_count": school.kg_grade_count,
                "primary_grade_count": school.primary_grade_count,
                "preparatory_grade_count": school.preparatory_grade_count,
                "secondary_grade_count": school.secondary_grade_count,
                "term_count": school.term_count,
                "working_days": ",".join(day.value for day in school.working_days),
                "created_at": now,
            }
            for school in schools
        ]
        existing = bulk_upsert(
            self._session,
            models.School,
            rows,
            conflict_on=("code",),
            # `is_active` is written here because this port has no `set_active`, exactly as
            # for a subject: upsert is the only way a branch can be closed or reopened.
            update_columns=(
                "name_en", "name_ar", "is_active", "language_type",
                "kg_grade_count", "primary_grade_count", "preparatory_grade_count",
                "secondary_grade_count", "term_count", "working_days",
            ),
        )
        return {row["code"]: (row["code"],) not in existing for row in rows}

__all__ = [
    "SqlAlchemyAcademicYearRepository",
    "SqlAlchemyClassSectionRepository",
    "SqlAlchemySchoolRepository",
    "SqlAlchemySubjectRepository",
    "SqlAlchemyTermRepository",
    "SqlAlchemyYearLevelRepository",
    "bulk_upsert",
]
