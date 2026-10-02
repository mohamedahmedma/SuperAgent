"""The daily register, over SQLAlchemy.

Two reads and one write, and the shapes are chosen from what the screens actually ask:

    marks_for_class(section, day)      the register a form teacher is filling in
    marks_for_student(student, range)  the attendance block on a child's card
    upsert_many(marks)                 a whole class saved in one pass

`upsert_many` writes the day for a whole class in a fixed number of statements, because the
alternative is one round trip per child and a teacher watching a spinner while forty rows go
out one at a time. It upserts on `(student_id, on_date)` — the columns of
`uq_attendance_student_day` — so taking the register twice on the same morning corrects the
marks rather than colliding on the second attempt and losing the batch.

Nothing here writes a row to mean "unmarked". A child with no mark for a day has no row for
that day, and every count this module produces reports how many days it counted, so a rate
is never divided by a denominator that includes days nobody looked at. That is invariant 1 —
a blank is not a zero — one column over.
"""
from collections.abc import Collection, Mapping, Sequence
from datetime import date, datetime, timezone

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from sis.domain.attendance import AttendanceMark, AttendanceState
from sis.domain.value_objects import StudentNumber
from sis.infrastructure.db import models
from sis.infrastructure.repositories.structure_repository import bulk_upsert

__all__ = ["SqlAlchemyAttendanceRepository"]


def _utcnow() -> datetime:
    """Written by hand, because Core writes skip the models' Python defaults."""
    return datetime.now(timezone.utc)


def _joined() -> Select:
    """Every read joins the child and her class, so a mark carries the codes it needs.

    The relationships on these models are `lazy="raise"`, and deliberately: a register of
    forty children would otherwise be forty extra SELECTs for the student numbers alone.
    """
    return (
        select(
            models.Attendance,
            models.Student.student_number,
            models.ClassSection.code,
        )
        .join(models.Student, models.Student.id == models.Attendance.student_id)
        .join(
            models.ClassSection,
            models.ClassSection.id == models.Attendance.class_section_id,
        )
    )


def _to_domain(row: object) -> AttendanceMark:
    mark, student_number, class_code = row  # type: ignore[misc]  # a Row of three
    return AttendanceMark(
        student_number=student_number,
        on_date=mark.on_date,
        state=mark.state,
        class_section_id=mark.class_section_id,
        class_code=class_code,
        note=mark.note,
    )


class SqlAlchemyAttendanceRepository:
    """`AttendanceRepository` over a session. Never commits; the unit of work does."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def marks_for_class(
        self, class_section_id: int, on_date: date
    ) -> Mapping[str, AttendanceMark]:
        """The marks taken for one class on one day, keyed by student number.

        A mapping rather than a list because the caller is merging it into a register: the
        children come from the enrolment query, and this answers "what, if anything, was
        recorded for her". A child missing from the result is a child nobody marked, which
        the screen has to show as blank rather than as present.
        """
        rows = self._session.execute(
            _joined().where(
                models.Attendance.class_section_id == class_section_id,
                models.Attendance.on_date == on_date,
            )
        ).all()
        return {row[1]: _to_domain(row) for row in rows}

    def marks_for_student(
        self,
        student_number: StudentNumber,
        *,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> Sequence[AttendanceMark]:
        """One child's marks, oldest first, optionally bounded.

        Both bounds are inclusive. A half-open range would be the more usual choice and the
        wrong one here: a registrar asking for a term asks from its first day to its last,
        and those are the two dates the term itself states.
        """
        statement = _joined().where(
            models.Student.student_number == str(student_number)
        )
        if from_date is not None:
            statement = statement.where(models.Attendance.on_date >= from_date)
        if to_date is not None:
            statement = statement.where(models.Attendance.on_date <= to_date)
        statement = statement.order_by(models.Attendance.on_date)
        return [_to_domain(row) for row in self._session.execute(statement).all()]

    def marks_for_students(
        self,
        student_numbers: Collection[StudentNumber],
        *,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> Sequence[AttendanceMark]:
        """The same read for many children in one statement, for a class-wide summary."""
        if not student_numbers:
            return []
        statement = _joined().where(
            models.Student.student_number.in_(
                {str(number) for number in student_numbers}
            )
        )
        if from_date is not None:
            statement = statement.where(models.Attendance.on_date >= from_date)
        if to_date is not None:
            statement = statement.where(models.Attendance.on_date <= to_date)
        statement = statement.order_by(
            models.Student.student_number, models.Attendance.on_date
        )
        return [_to_domain(row) for row in self._session.execute(statement).all()]

    def upsert_many(
        self, marks: Sequence[AttendanceMark], *, recorded_by: str = ""
    ) -> Mapping[tuple[str, date], bool]:
        """Write a day's register; `True` marks the entries this call created.

        Three statements for a class of any size: student numbers to ids, the pairs already
        on file, then one executemany upsert. Matched on `(student_id, on_date)`, so a
        second save of the same morning corrects the marks instead of failing on the unique
        constraint and losing the whole register.

        `created_at` is deliberately excluded from the update columns: it answers "when was
        this register first taken", and rewriting it on a correction in June would make every
        row look as though the morning register had been taken in June.
        """
        if not marks:
            return {}

        numbers = {str(mark.student_number) for mark in marks}
        student_ids = {
            row[1]: row[0]
            for row in self._session.execute(
                select(models.Student.id, models.Student.student_number).where(
                    models.Student.student_number.in_(numbers)
                )
            ).all()
        }
        missing = sorted(numbers - set(student_ids))
        if missing:
            from sis.domain.errors import UnknownReference

            raise UnknownReference(
                "no student on file with number " + ", ".join(missing),
                field="student_number",
            )

        now = _utcnow()
        rows = [
            {
                "student_id": student_ids[str(mark.student_number)],
                "class_section_id": mark.class_section_id,
                "on_date": mark.on_date,
                "state": str(
                    mark.state.value
                    if isinstance(mark.state, AttendanceState)
                    else mark.state
                ),
                "note": mark.note,
                "recorded_by": recorded_by,
                "created_at": now,
                "updated_at": now,
            }
            for mark in marks
        ]

        existing = bulk_upsert(
            self._session,
            models.Attendance,
            rows,
            conflict_on=("student_id", "on_date"),
            # `class_section_id` *is* updatable, unlike the parent ids elsewhere in this
            # service. It has to be: a child transferred mid-year and re-marked for a day
            # she spent in the new class must have that day filed under the new class, and
            # this is a per-day statement rather than a span whose history matters.
            update_columns=(
                "class_section_id",
                "state",
                "note",
                "recorded_by",
                "updated_at",
            ),
        )

        by_id = {value: key for key, value in student_ids.items()}
        return {
            (by_id[row["student_id"]], row["on_date"]): (
                row["student_id"],
                row["on_date"],
            )
            not in existing
            for row in rows
        }
