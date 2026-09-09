"""`GradeRepository` over SQLAlchemy: stated marks, keyed by codes, stored by ids.

The domain's `SubjectGrade` names a student, a subject and a term by *code*; the table
joins them by surrogate id. That translation is this module's whole job, and it is done
in bulk — three lookup queries for a whole upload, never one per row — because a grade
sheet is hundreds of rows and a per-row resolve turns one upload into a query storm the
registrar watches as a spinner.

**Invariant 1 lives on both sides of this file.** On write, a `percentage` of `None` is
put in the values dict as `None` and reaches the column as SQL NULL; it is never
defaulted, coalesced or skipped. On read, the test is `is None` and never truthiness —
`Percentage(0)` is falsy-looking and an earned zero read as "not graded yet" is the same
lie as a blank read as zero, told in the other direction. Nothing here averages,
weights or drops anything (decision 5): what a teacher stated is what is stored.

Nothing here commits. The transaction boundary belongs to whoever composed the request;
a repository that commits mid-import is how half a roster lands and the rest does not.
"""
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from sqlalchemy import ColumnElement, Select, insert, select, update
from sqlalchemy.orm import Session

from sis.application.ports.repositories import GradeKey
from sis.domain.errors import UnknownReference
from sis.domain.grades import SubjectGrade
from sis.infrastructure.audit import actor_context
from sis.domain.value_objects import (
    Percentage,
    StudentNumber,
    SubjectCode,
    TermCode,
)
from sis.infrastructure.db import models


class SqlAlchemyGradeRepository:
    """Stated marks in SQL. One row per `(student, subject, term)`, enforced by the DB."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- reads ---------------------------------------------------------------

    def get(
        self,
        student_number: StudentNumber,
        subject_code: SubjectCode,
        term_code: TermCode,
    ) -> SubjectGrade | None:
        stmt = _joined().where(
            models.Student.student_number == student_number.value,
            models.Subject.code == subject_code.value,
            models.Term.code == term_code.value,
        )
        row = self._session.execute(stmt).first()
        return None if row is None else _to_domain(row)

    def get_many(self, keys: Collection[GradeKey]) -> Mapping[GradeKey, SubjectGrade]:
        """The marks already on file among `keys` — what a preview calls a restatement."""
        wanted = set(keys)
        if not wanted:
            return {}
        stmt = _joined().where(
            models.Student.student_number.in_({key[0] for key in wanted}),
            models.Subject.code.in_({key[1] for key in wanted}),
            models.Term.code.in_({key[2] for key in wanted}),
        )
        found: dict[GradeKey, SubjectGrade] = {}
        for row in self._session.execute(stmt):
            key: GradeKey = (row[1], row[2], row[3])
            # Three separate INs describe the cross product of the codes asked about,
            # which is a superset of the triples actually wanted; the intersection
            # happens here. Row-value `IN ((a,b,c), ...)` would express it exactly and
            # is not portable across the dialects this service is expected to run on.
            if key in wanted:
                found[key] = _to_domain(row)
        return found

    def list_for_student(
        self, student_number: StudentNumber, *, term_code: TermCode | None = None
    ) -> Sequence[SubjectGrade]:
        stmt = _joined().where(models.Student.student_number == student_number.value)
        if term_code is not None:
            stmt = stmt.where(models.Term.code == term_code.value)
        stmt = stmt.order_by(
            models.Term.sequence, models.Subject.display_order, models.Subject.code
        )
        return [_to_domain(row) for row in self._session.execute(stmt)]

    def list_for_class(
        self,
        class_section_id: int,
        term_code: TermCode,
        *,
        subject_code: SubjectCode | None = None,
    ) -> Sequence[SubjectGrade]:
        """The grade sheet, in the order it is printed: by child, then by subject column."""
        stmt = _joined().where(
            models.SubjectGrade.class_section_id == class_section_id,
            models.Term.code == term_code.value,
        )
        if subject_code is not None:
            stmt = stmt.where(models.Subject.code == subject_code.value)
        stmt = stmt.order_by(
            models.Student.student_number,
            models.Subject.display_order,
            models.Subject.code,
        )
        return [_to_domain(row) for row in self._session.execute(stmt)]

    # -- writes --------------------------------------------------------------

    def upsert_many(self, grades: Sequence[SubjectGrade]) -> Mapping[GradeKey, bool]:
        """Write the stated figures; `True` marks the keys this call created.

        Five statements for any number of grades: three code->id resolutions, one
        existence scan, then one executemany insert and one executemany update. The
        existence scan is what makes the call idempotent — committing the same sheet
        twice updates the same rows and reports every key as `False`, rather than
        colliding on `uq_subject_grade_identity` and losing the batch.
        """
        if not grades:
            return {}

        # Last statement wins for a repeated key. A file that lists the same child twice
        # for one subject would otherwise send two INSERTs in one executemany, and the
        # unique constraint would discard every good row in the batch with them
        # (decision 4): the duplicate is a row-level fact, not a batch-level failure.
        latest: dict[GradeKey, SubjectGrade] = {grade.identity: grade for grade in grades}

        students = self._resolve(
            models.Student.student_number,
            models.Student.id,
            {key[0] for key in latest},
            "student",
            "student_number",
        )
        subjects = self._resolve(
            models.Subject.code,
            models.Subject.id,
            {key[1] for key in latest},
            "subject",
            "subject_code",
        )
        terms = self._resolve(
            models.Term.code,
            models.Term.id,
            {key[2] for key in latest},
            "term",
            "term_code",
        )
        self._validate_academic_context(latest.values(), subjects, terms)
        existing = self._existing_ids(students.values(), subjects.values(), terms.values())
        existing_rows = {
            row.id: row
            for row in self._session.scalars(
                select(models.SubjectGrade).where(
                    models.SubjectGrade.id.in_(list(existing.values()) or [-1])
                )
            )
        }

        created: dict[GradeKey, bool] = {}
        to_insert: list[dict[str, Any]] = []
        to_update: list[dict[str, Any]] = []
        for key, grade in latest.items():
            triple = (students[key[0]], subjects[key[1]], terms[key[2]])
            actor_user_id, actor = actor_context.get()
            stated: dict[str, Any] = {
                "class_section_id": grade.class_section_id,
                # Invariant 1, at the only point where it can be broken silently: the
                # key is always present and its value is `None` for an ungraded row, so
                # the column is written NULL. Omitting the key on a None would leave the
                # previous mark standing on an update, and `or 0.0` here would fabricate
                # a failing grade nobody awarded.
                "percentage": None if grade.percentage is None else grade.percentage.value,
                "points": grade.points,
                "max_points": grade.max_points,
            }
            row_id = existing.get(triple)
            if row_id is None:
                to_insert.append(
                    {
                        "student_id": triple[0],
                        "subject_id": triple[1],
                        "term_id": triple[2],
                        "recorded_by": actor,
                        **stated,
                    }
                )
                created[key] = True
            else:
                # `remark` and `recorded_by` are deliberately absent: a re-import
                # restates the figure, and rewriting them to the column default would
                # erase the teacher's note and the audit trail of who entered the mark.
                to_update.append({"id": row_id, "recorded_by": actor, **stated})
                previous = existing_rows[row_id]
                old_values = {
                    "percentage": previous.percentage, "points": previous.points,
                    "max_points": previous.max_points, "recorded_by": previous.recorded_by,
                }
                new_values = {**stated, "recorded_by": actor, "student_id": triple[0]}
                if old_values != new_values:
                    self._session.add(models.AuditLog(
                        actor_user_id=actor_user_id, actor=actor, action="grade_edit",
                        entity_type="SubjectGrade", entity_id=str(row_id),
                        old_values=old_values, new_values=new_values,
                    ))
                created[key] = False

        if to_insert:
            self._session.execute(insert(models.SubjectGrade), to_insert)
        if to_update:
            self._session.execute(update(models.SubjectGrade), to_update)
        self._session.flush()
        return created

    def _validate_academic_context(
        self,
        grades: Collection[SubjectGrade],
        subjects: Mapping[str, int],
        terms: Mapping[str, int],
    ) -> None:
        """Refuse a mark whose class, subject, and term describe different years.

        Foreign keys prove that each id exists; they cannot prove that the four ids belong
        to the same academic context.  This bulk lookup is the server-side half of that
        invariant and remains authoritative for imports and direct API calls alike.
        """
        section_ids = {grade.class_section_id for grade in grades}
        sections = {
            row.id: (row.academic_year_id, row.year_level_id)
            for row in self._session.scalars(
                select(models.ClassSection).where(models.ClassSection.id.in_(section_ids))
            )
        }
        subject_years = dict(self._session.execute(
            select(models.Subject.id, models.Subject.academic_year_id)
            .where(models.Subject.id.in_(subjects.values()))
        ).all())
        term_years = dict(self._session.execute(
            select(models.Term.id, models.Term.academic_year_id)
            .where(models.Term.id.in_(terms.values()))
        ).all())
        allowed_pairs = set(self._session.execute(
            select(models.SubjectYearLevel.subject_id, models.SubjectYearLevel.year_level_id)
            .where(models.SubjectYearLevel.subject_id.in_(subjects.values()))
        ).all())
        for grade in grades:
            section = sections.get(grade.class_section_id)
            subject_id = subjects[str(grade.subject_code)]
            term_id = terms[str(grade.term_code)]
            if section is None:
                raise UnknownReference("no class section for this grade", field="class_section_id")
            year_id, level_id = section
            if subject_years[subject_id] != year_id or term_years[term_id] != year_id:
                raise UnknownReference(
                    "grade subject, term, and class must belong to the same academic year",
                    field="academic_year_code",
                )
            # Legacy imports may predate subject-to-grade configuration entirely. Once a
            # subject has any configured grades, that configuration is authoritative.
            subject_pairs = {pair for pair in allowed_pairs if pair[0] == subject_id}
            if subject_pairs and (subject_id, level_id) not in subject_pairs:
                raise UnknownReference(
                    "subject is not configured for the class grade level",
                    field="subject_code",
                )

    # -- internals -----------------------------------------------------------

    def _resolve(
        self,
        code_column: ColumnElement[str],
        id_column: ColumnElement[int],
        codes: set[str],
        noun: str,
        field: str,
    ) -> dict[str, int]:
        """Codes to surrogate ids in one query, refusing the ones nothing resolves.

        Raising beats returning a partial map: an unresolved code silently dropped here
        is a mark that vanishes between the preview a registrar approved and the table.
        """
        if not codes:
            return {}
        rows = self._session.execute(
            select(code_column, id_column).where(code_column.in_(codes))
        ).all()
        found = {code: identifier for code, identifier in rows}
        missing = sorted(codes - found.keys())
        if missing:
            raise UnknownReference(
                f"no {noun} on file for: {', '.join(missing)}", field=field
            )
        return found

    def _existing_ids(
        self,
        student_ids: Collection[int],
        subject_ids: Collection[int],
        term_ids: Collection[int],
    ) -> dict[tuple[int, int, int], int]:
        """Which `(student, subject, term)` triples already have a row, and its id."""
        stmt = select(
            models.SubjectGrade.id,
            models.SubjectGrade.student_id,
            models.SubjectGrade.subject_id,
            models.SubjectGrade.term_id,
        ).where(
            models.SubjectGrade.student_id.in_(student_ids),
            models.SubjectGrade.subject_id.in_(subject_ids),
            models.SubjectGrade.term_id.in_(term_ids),
        )
        return {
            (student_id, subject_id, term_id): row_id
            for row_id, student_id, subject_id, term_id in self._session.execute(stmt)
        }


def _joined() -> Select[Any]:
    """A grade with the four codes that name it, in one row and one query.

    Every relationship on the model is `lazy="raise"`, on purpose: without these joins a
    listing of a class would emit four extra SELECTs per grade to render one line.
    """
    return (
        select(
            models.SubjectGrade,
            models.Student.student_number,
            models.Subject.code,
            models.Term.code,
            models.ClassSection.code,
        )
        .join(models.Student, models.Student.id == models.SubjectGrade.student_id)
        .join(models.Subject, models.Subject.id == models.SubjectGrade.subject_id)
        .join(models.Term, models.Term.id == models.SubjectGrade.term_id)
        .join(
            models.ClassSection,
            models.ClassSection.id == models.SubjectGrade.class_section_id,
        )
    )


def _to_domain(row: Any) -> SubjectGrade:
    grade, student_number, subject_code, term_code, class_code = row
    return SubjectGrade(
        student_number=student_number,
        subject_code=subject_code,
        term_code=term_code,
        class_section_id=grade.class_section_id,
        class_code=class_code,
        # `is None`, never `if grade.percentage`. A stored 0.0 is an earned zero and is
        # falsy; a truthiness test here would hand every one of them back as "not graded
        # yet" — invariant 1 broken on the read path instead of the write path.
        percentage=None if grade.percentage is None else Percentage(grade.percentage),
        points=grade.points,
        max_points=grade.max_points,
    )


__all__ = ["SqlAlchemyGradeRepository"]
