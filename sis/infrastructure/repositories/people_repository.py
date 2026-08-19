"""Children and their time-bounded placements, over SQLAlchemy.

Implements `StudentRepository` and `EnrolmentRepository` from
`sis.application.ports.repositories`. Nothing here returns an ORM row: every method
hands back a `sis.domain` object, because the moment a service holds a
`models.ClassEnrolment` it holds a live session, and the unit test that was supposed to
assert a rule needs a database again.

**The vocabulary gap this module exists to close.** The ports speak `StudentNumber` and
`(academic_year_code, class_code)`; the tables join on surrogate integers. Translating
in both directions is this layer's entire job, and it is why every bulk method starts by
resolving codes to ids in one query rather than per row.

**Statement counts are part of the contract, not an optimisation.** A roster upload
carries hundreds of rows, and the per-row `SELECT`/`INSERT` shape turns one upload into
a thousand round trips — a registrar watching a spinner while a five-second import takes
four minutes. Every bulk method here states its statement count above its body, and none
of them grows with the number of rows.

**Nothing here commits.** The transaction belongs to the unit of work (invariant 4): a
repository that committed per row would leave row 140's failure on top of 139 written
enrolments, which is the half-imported roster no screen reports as broken.
"""
from collections.abc import Collection, Iterator, Mapping, Sequence
from datetime import date
from typing import Any

from sqlalchemy import Select, insert, or_, select, update
from sqlalchemy.orm import Session

from sis.application.ports.repositories import EnrolmentKey
from sis.domain.errors import DuplicateCode, UnknownReference
from sis.domain.people import ClassEnrolment, Student
from sis.domain.structure import ClassSection
from sis.domain.value_objects import AcademicYearCode, ClassCode, StudentNumber
from sis.infrastructure.db import models

# SQLite binds one parameter per element of an `IN` list and refuses past
# SQLITE_MAX_VARIABLE_NUMBER — 999 on builds distributions still ship. A two-thousand
# child roster would then fail on the *shape* of the query rather than on any of its
# data, so every `IN` below is chunked. "One query" in this module means one per chunk.
_IN_CHUNK = 400


def _chunked(values: Sequence[Any]) -> Iterator[Sequence[Any]]:
    """Split an `IN` list into batches SQLite will accept."""
    for start in range(0, len(values), _IN_CHUNK):
        yield values[start : start + _IN_CHUNK]


def _like_term(query: str) -> str:
    """Escape the registrar's typing before it becomes a LIKE pattern.

    `%` and `_` are wildcards, so a search box that passes them through turns the single
    keystroke `_` into "every child in the school" — which reads as a broken filter
    rather than as a wildcard anybody asked for.
    """
    escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _to_student(row: models.Student) -> Student:
    """One ORM row -> one domain `Student`."""
    return Student(
        student_number=row.student_number,
        full_name_ar=row.full_name_ar,
        full_name_en=row.full_name_en,
        is_active=row.is_active,
    )


# The columns a domain `ClassSection` is built from. Selected explicitly because the
# `class_sections` relationships are `lazy="raise"`: touching `enrolment.class_section`
# would raise instead of quietly emitting the per-row query this module exists to avoid.
_SECTION_COLUMNS = (
    models.ClassSection.code.label("class_code"),
    models.AcademicYear.code.label("academic_year_code"),
    models.YearLevel.code.label("year_level_code"),
    models.ClassSection.name_en,
    models.ClassSection.name_ar,
    models.ClassSection.capacity,
)


def _to_section(row: Any) -> ClassSection:
    return ClassSection(
        code=row.class_code,
        academic_year_code=row.academic_year_code,
        year_level_code=row.year_level_code,
        name_en=row.name_en,
        name_ar=row.name_ar,
        capacity=row.capacity,
    )


def _to_enrolment(row: Any) -> ClassEnrolment:
    return ClassEnrolment(
        student_number=row.student_number,
        academic_year_code=row.academic_year_code,
        class_code=row.class_code,
        starts_on=row.starts_on,
        ends_on=row.ends_on,
    )


def _join_section(stmt: Select[Any]) -> Select[Any]:
    """Attach the section, its year and its level to a statement already on enrolments."""
    return stmt.join(
        models.ClassSection,
        models.ClassSection.id == models.ClassEnrolment.class_section_id,
    ).join(
        models.AcademicYear,
        models.AcademicYear.id == models.ClassSection.academic_year_id,
    ).join(
        models.YearLevel,
        models.YearLevel.id == models.ClassSection.year_level_id,
    )


def _enrolment_query() -> Select[Any]:
    """Enrolments with the codes a domain `ClassEnrolment` needs, in one statement."""
    return _join_section(
        select(
            models.ClassEnrolment.id.label("enrolment_id"),
            models.Student.student_number,
            models.ClassSection.code.label("class_code"),
            models.AcademicYear.code.label("academic_year_code"),
            models.ClassEnrolment.starts_on,
            models.ClassEnrolment.ends_on,
        )
        .select_from(models.ClassEnrolment)
        .join(models.Student, models.Student.id == models.ClassEnrolment.student_id)
    )


class SqlAlchemyStudentRepository:
    """`StudentRepository` over a session. Children only — placement is the sibling below."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, student_number: StudentNumber) -> Student | None:
        row = self._session.scalars(
            select(models.Student).where(
                models.Student.student_number == str(student_number)
            )
        ).first()
        return _to_student(row) if row is not None else None

    def get_many(
        self, student_numbers: Collection[StudentNumber]
    ) -> Mapping[str, Student]:
        wanted = sorted({str(number) for number in student_numbers})
        found: dict[str, Student] = {}
        for chunk in _chunked(wanted):
            for row in self._session.scalars(
                select(models.Student).where(models.Student.student_number.in_(chunk))
            ):
                found[row.student_number] = _to_student(row)
        return found

    def search(
        self, query: str, *, limit: int = 50, include_inactive: bool = False
    ) -> Sequence[Student]:
        """Type-ahead over number and both spellings.

        `ilike` rather than `like` because a registrar types "ahmed" looking for "Ahmed";
        against `full_name_ar` it is a no-op, which is correct — Arabic has no case — and
        not an accident. The leading wildcard means `ix_students_name_*` cannot serve
        this scan; those indexes serve the ordered listing. A school-sized table makes
        that acceptable, and `limit` caps what crosses the wire.
        """
        if not query.strip():
            return []
        term = _like_term(query)
        stmt = select(models.Student).where(
            or_(
                models.Student.student_number.ilike(term, escape="\\"),
                models.Student.full_name_ar.ilike(term, escape="\\"),
                models.Student.full_name_en.ilike(term, escape="\\"),
            )
        )
        if not include_inactive:
            stmt = stmt.where(models.Student.is_active.is_(True))
        stmt = stmt.order_by(models.Student.student_number).limit(limit)
        return [_to_student(row) for row in self._session.scalars(stmt)]

    def upsert_many(self, students: Sequence[Student]) -> Mapping[str, bool]:
        """Insert or update by student number; `True` marks the ones this call created.

        Three statements for a roster of any size: one `SELECT` of the numbers already on
        file, then at most one executemany `INSERT` and one executemany `UPDATE`. A
        re-run that changes nothing issues **one** — the `SELECT` alone.

        That last part is invariant 3 held honestly rather than nominally. Rows whose
        names and active flag already match are left out of the `UPDATE` payload, so
        re-importing September's roster in March does not stamp `updated_at` on nine
        hundred children and destroy the only column that can answer "whose details
        changed in this import".
        """
        if not students:
            return {}

        # Last write wins within one file. A roster naming the same child twice is a
        # typo, and sending both rows to the `INSERT` trips the unique index and rejects
        # the whole upload — the "one bad row discards the good ones" failure of
        # invariant 4, caused here rather than in the file.
        incoming: dict[str, Student] = {
            str(student.student_number): student for student in students
        }

        existing: dict[str, models.Student] = {}
        for chunk in _chunked(sorted(incoming)):
            for row in self._session.scalars(
                select(models.Student).where(models.Student.student_number.in_(chunk))
            ):
                existing[row.student_number] = row

        to_insert: list[dict[str, Any]] = []
        to_update: list[dict[str, Any]] = []
        created: dict[str, bool] = {}
        for number, student in incoming.items():
            row = existing.get(number)
            if row is None:
                created[number] = True
                to_insert.append(
                    {
                        "student_number": number,
                        "full_name_ar": student.full_name_ar,
                        "full_name_en": student.full_name_en,
                        "is_active": student.is_active,
                    }
                )
                continue
            created[number] = False
            if (
                row.full_name_ar != student.full_name_ar
                or row.full_name_en != student.full_name_en
                or row.is_active != student.is_active
            ):
                to_update.append(
                    {
                        "id": row.id,
                        "full_name_ar": student.full_name_ar,
                        "full_name_en": student.full_name_en,
                        "is_active": student.is_active,
                    }
                )

        if to_insert:
            self._session.execute(insert(models.Student), to_insert)
        if to_update:
            self._session.execute(update(models.Student), to_update)
        return created

    def set_active(self, student_number: StudentNumber, *, is_active: bool) -> Student:
        row = self._session.scalars(
            select(models.Student).where(
                models.Student.student_number == str(student_number)
            )
        ).first()
        if row is None:
            raise UnknownReference(
                f"no student on file with number {student_number}",
                field="student_number",
            )
        row.is_active = is_active
        # Flush, never commit: the write must be visible to the rest of this transaction
        # and durable only when the unit of work says so.
        self._session.flush()
        return _to_student(row)


class SqlAlchemyEnrolmentRepository:
    """`EnrolmentRepository` over a session — invariant 2 as queries against date windows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def class_section_on(
        self, student_id: StudentNumber, on_date: date
    ) -> ClassSection | None:
        """Which class this child was in on `on_date`, or `None`.

        The window test is `starts_on <= on_date AND (ends_on IS NULL OR ends_on >=
        on_date)`: an open placement covers every day from its start onwards, which is
        why `ends_on IS NULL` must be spelled out rather than compared. SQL's three-valued
        logic makes `ends_on >= :on` evaluate to NULL — not TRUE — for the open row, so
        the naive predicate silently excludes exactly the child's *current* class and
        answers "she is in no class" about a child sitting in one.

        `ORDER BY starts_on DESC LIMIT 1` because the schema cannot enforce non-overlap
        between two bounded windows (see `models.ClassEnrolment`), and an ambiguous day
        must resolve the same way every time it is asked: the placement that began most
        recently. An arbitrary `.first()` would let two screens print two classes for one
        child on one day, which is worse than either answer alone.

        Served by `ix_class_enrolments_student_on_date (student_id, starts_on, ends_on)`
        — the leading `student_id` narrows to a handful of rows and the dates resolve the
        window inside the index. This is the hottest query in the service: it runs once
        per grade written and once per report-card line rendered.
        """
        stmt = (
            _join_section(
                select(*_SECTION_COLUMNS)
                .select_from(models.ClassEnrolment)
                .join(
                    models.Student,
                    models.Student.id == models.ClassEnrolment.student_id,
                )
            )
            .where(
                models.Student.student_number == str(student_id),
                models.ClassEnrolment.starts_on <= on_date,
                or_(
                    models.ClassEnrolment.ends_on.is_(None),
                    models.ClassEnrolment.ends_on >= on_date,
                ),
            )
            .order_by(models.ClassEnrolment.starts_on.desc())
            .limit(1)
        )
        row = self._session.execute(stmt).first()
        return _to_section(row) if row is not None else None

    def class_sections_on(
        self, student_ids: Collection[StudentNumber], on_date: date
    ) -> Mapping[str, ClassSection]:
        """`class_section_on` for many children — same window test, one query per chunk.

        Ordered by `starts_on` ascending so the latest-starting placement overwrites the
        earlier one in the dict: the same "most recent wins" tie-break as the singular
        method, reached by assignment order instead of `LIMIT`. Children with no
        placement on that day are simply absent — never mapped to a guessed class.
        """
        wanted = sorted({str(number) for number in student_ids})
        resolved: dict[str, ClassSection] = {}
        for chunk in _chunked(wanted):
            stmt = (
                _join_section(
                    select(models.Student.student_number, *_SECTION_COLUMNS)
                    .select_from(models.ClassEnrolment)
                    .join(
                        models.Student,
                        models.Student.id == models.ClassEnrolment.student_id,
                    )
                )
                .where(
                    models.Student.student_number.in_(chunk),
                    models.ClassEnrolment.starts_on <= on_date,
                    or_(
                        models.ClassEnrolment.ends_on.is_(None),
                        models.ClassEnrolment.ends_on >= on_date,
                    ),
                )
                .order_by(models.ClassEnrolment.starts_on)
            )
            for row in self._session.execute(stmt):
                resolved[row.student_number] = _to_section(row)
        return resolved

    def open_enrolment(self, student_id: StudentNumber) -> ClassEnrolment | None:
        # At most one open row per student is enforced by the partial unique index
        # `uq_class_enrolments_open_per_student`, so this needs no tie-break.
        row = self._session.execute(
            _enrolment_query().where(
                models.Student.student_number == str(student_id),
                models.ClassEnrolment.ends_on.is_(None),
            )
        ).first()
        return _to_enrolment(row) if row is not None else None

    def list_for_student(self, student_id: StudentNumber) -> Sequence[ClassEnrolment]:
        rows = self._session.execute(
            _enrolment_query()
            .where(models.Student.student_number == str(student_id))
            .order_by(models.ClassEnrolment.starts_on)
        )
        return [_to_enrolment(row) for row in rows]

    def list_for_students(
        self, student_ids: Collection[StudentNumber]
    ) -> Mapping[str, Sequence[ClassEnrolment]]:
        """Placement histories in bulk, so the overlap check runs in memory.

        The rule itself stays in `ClassEnrolment.conflicts_with`. Expressing it in SQL
        here would move the one rule invariant 2 protects into a place no unit test can
        reach without a database. Children with no placements are absent from the
        mapping rather than mapped to an empty list.
        """
        wanted = sorted({str(number) for number in student_ids})
        grouped: dict[str, list[ClassEnrolment]] = {}
        for chunk in _chunked(wanted):
            rows = self._session.execute(
                _enrolment_query()
                .where(models.Student.student_number.in_(chunk))
                .order_by(models.Student.student_number, models.ClassEnrolment.starts_on)
            )
            for row in rows:
                grouped.setdefault(row.student_number, []).append(_to_enrolment(row))
        return grouped

    def roster_on(
        self,
        academic_year_code: AcademicYearCode,
        class_code: ClassCode,
        on_date: date,
    ) -> Sequence[ClassEnrolment]:
        """The register of one class as of one day.

        Note the key the port gives us: `(year, class code)`. The unique key on
        `class_sections` is the *triple* including the year level, so two levels of one
        year may legitimately both own a section coded `A`. Every enrolment in every
        matching section is returned rather than one room picked arbitrarily — an
        ambiguous code should produce a register a human can see is doubled, not a
        register that silently omits half the children.

        Served by `ix_class_enrolments_section_window (class_section_id, starts_on,
        ends_on)`, with the same open-ended window test as `class_section_on`.
        """
        rows = self._session.execute(
            _enrolment_query()
            .where(
                models.AcademicYear.code == str(academic_year_code),
                models.ClassSection.code == str(class_code),
                models.ClassEnrolment.starts_on <= on_date,
                or_(
                    models.ClassEnrolment.ends_on.is_(None),
                    models.ClassEnrolment.ends_on >= on_date,
                ),
            )
            .order_by(models.Student.student_number)
        )
        return [_to_enrolment(row) for row in rows]

    def close_open_enrolment(
        self, student_id: StudentNumber, *, ends_on: date
    ) -> ClassEnrolment | None:
        """End the child's open placement on `ends_on`; `None` when she had none.

        The closed placement is built as a domain object *before* the `UPDATE` runs, so
        an end date preceding the start raises `InvalidDateRange` from
        `ClassEnrolment.__post_init__` rather than from
        `ck_class_enrolments_dates_ordered`. The distinction is not cosmetic: a failed
        CHECK aborts the transaction, which during an import takes every good row already
        written down with it, and reports itself as an IntegrityError naming a constraint
        instead of a message naming the cell.

        Only `ends_on` is written. A transfer is this call followed by a new enrolment,
        never a rewrite of `class_section_id` on the row that is already there — that
        edit is precisely the history rewrite invariant 2 exists to forbid.
        """
        row = self._session.execute(
            _enrolment_query().where(
                models.Student.student_number == str(student_id),
                models.ClassEnrolment.ends_on.is_(None),
            )
        ).first()
        if row is None:
            return None

        closed = ClassEnrolment(
            student_number=row.student_number,
            academic_year_code=row.academic_year_code,
            class_code=row.class_code,
            starts_on=row.starts_on,
            ends_on=ends_on,
        )
        self._session.execute(
            update(models.ClassEnrolment)
            .where(models.ClassEnrolment.id == row.enrolment_id)
            .values(ends_on=ends_on)
        )
        return closed

    def upsert_many(
        self, enrolments: Sequence[ClassEnrolment]
    ) -> Mapping[EnrolmentKey, bool]:
        """Insert or update placements in bulk; `True` marks the ones this call created.

        Five statements for a roster of any size: student numbers -> ids, class codes ->
        section ids, the placements already on file, then at most one executemany
        `INSERT` and one executemany `UPDATE`.

        Matched on `(student_id, class_section_id, starts_on)` — the columns of
        `uq_class_enrolment_placement` — so committing the same roster twice writes the
        same placements instead of stacking a second copy that would then read as an
        overlap. `reason` is written on insert only: re-importing a roster must not
        overwrite "transfer" with the empty string and erase why a child moved.

        The student rows are locked (`FOR UPDATE`) before the existing placements are
        read. Two concurrent commits of the same child would otherwise each read a state
        that the other's insert has already invalidated, and both would open a placement;
        the partial unique index catches the double-*open* case, but nothing catches two
        overlapping bounded windows. Under SQLite the lock is a no-op — its whole-file
        write lock already serialises this.
        """
        if not enrolments:
            return {}

        incoming: dict[EnrolmentKey, ClassEnrolment] = {
            (
                str(enrolment.student_number),
                str(enrolment.academic_year_code),
                str(enrolment.class_code),
                enrolment.starts_on,
            ): enrolment
            for enrolment in enrolments
        }
        student_ids = self._student_ids(sorted({key[0] for key in incoming}))
        section_ids = self._section_ids({(key[1], key[2]) for key in incoming})

        existing: dict[tuple[int, int, date], models.ClassEnrolment] = {}
        for chunk in _chunked(sorted(student_ids.values())):
            for row in self._session.scalars(
                select(models.ClassEnrolment).where(
                    models.ClassEnrolment.student_id.in_(chunk)
                )
            ):
                existing[(row.student_id, row.class_section_id, row.starts_on)] = row

        to_insert: list[dict[str, Any]] = []
        to_update: list[dict[str, Any]] = []
        created: dict[EnrolmentKey, bool] = {}
        for key, enrolment in incoming.items():
            number, year_code, class_code, starts_on = key
            student_pk = student_ids.get(number)
            if student_pk is None:
                raise UnknownReference(
                    f"no student on file with number {number}", field="student_number"
                )
            section_pk = section_ids.get((year_code, class_code))
            if section_pk is None:
                raise UnknownReference(
                    f"no class {class_code} in academic year {year_code}",
                    field="class_code",
                )

            row = existing.get((student_pk, section_pk, starts_on))
            if row is None:
                created[key] = True
                to_insert.append(
                    {
                        "student_id": student_pk,
                        "class_section_id": section_pk,
                        "starts_on": starts_on,
                        "ends_on": enrolment.ends_on,
                        "reason": "",
                    }
                )
                continue
            created[key] = False
            # Only a genuinely different end date is written, so a re-import leaves
            # `updated_at` alone on placements nobody changed.
            if row.ends_on != enrolment.ends_on:
                to_update.append({"id": row.id, "ends_on": enrolment.ends_on})

        if to_insert:
            self._session.execute(insert(models.ClassEnrolment), to_insert)
        if to_update:
            self._session.execute(update(models.ClassEnrolment), to_update)
        return created

    def _student_ids(self, numbers: Sequence[str]) -> dict[str, int]:
        """Student numbers -> surrogate ids, with the write lock described above."""
        found: dict[str, int] = {}
        for chunk in _chunked(numbers):
            rows = self._session.execute(
                select(models.Student.student_number, models.Student.id)
                .where(models.Student.student_number.in_(chunk))
                .with_for_update()
            )
            for number, student_id in rows:
                found[number] = student_id
        return found

    def _section_ids(
        self, keys: Collection[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """`(year code, class code)` -> section id, refusing an ambiguous match.

        `class_sections` is unique on `(year, level, code)`, so a school whose section
        codes are bare letters can hold a `Y3/A` and a `Y4/A` in one year, and a roster
        naming "A" resolves to neither. Taking `.first()` of that would enrol the child
        in whichever room the planner happened to return — silently, and in the half of
        cases where it is wrong, permanently. `DuplicateCode` names the real problem:
        the code the file uses is not unique enough to place anybody.
        """
        year_codes = sorted({year for year, _ in keys})
        class_codes = sorted({code for _, code in keys})
        candidates: dict[tuple[str, str], list[int]] = {}
        for chunk in _chunked(class_codes):
            rows = self._session.execute(
                select(
                    models.AcademicYear.code,
                    models.ClassSection.code,
                    models.ClassSection.id,
                )
                .select_from(models.ClassSection)
                .join(
                    models.AcademicYear,
                    models.AcademicYear.id == models.ClassSection.academic_year_id,
                )
                .where(
                    models.AcademicYear.code.in_(year_codes),
                    models.ClassSection.code.in_(chunk),
                )
            )
            for year_code, class_code, section_id in rows:
                candidates.setdefault((year_code, class_code), []).append(section_id)

        resolved: dict[tuple[str, str], int] = {}
        for key in keys:
            matches = candidates.get(key, [])
            if len(matches) > 1:
                raise DuplicateCode(
                    f"class code {key[1]} names {len(matches)} sections in academic "
                    f"year {key[0]}; it cannot identify one on its own",
                    field="class_code",
                )
            if matches:
                resolved[key] = matches[0]
        return resolved


__all__ = ["SqlAlchemyEnrolmentRepository", "SqlAlchemyStudentRepository"]
