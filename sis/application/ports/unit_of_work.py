"""The transaction boundary: every repository a service can reach, and one commit.

`commit()` is the only place a transaction commits. A service therefore cannot
half-apply an import: it does its two hundred writes through the repositories below and
then commits once, so the batch lands whole or not at all. The failure this forecloses
is a commit-per-row loop, where row 140 raising `NotEnrolled` leaves 139 enrolments
written and 61 missing -- a half-imported roster that no screen reports as broken,
because every row it *did* write looks perfectly correct. Recovering from that means a
human diffing the spreadsheet against the database.

Repositories are attributes of the unit of work rather than separate injected objects
for the same reason: two repositories obtained independently can sit on two connections,
and then "write the student, then write her enrolment" is two transactions with a window
between them. Reaching them through one unit of work makes sharing a transaction the
only thing that is easy to do.

Leaving the context manager without calling `commit()` rolls back. Default-rollback is
what makes the guarantee hold under exceptions rather than under discipline: a service
that raises halfway through leaves nothing behind without having remembered to catch
anything.

This is a Protocol, so a service's tests supply a fake unit of work holding in-memory
repositories, with `commit()` a flag it flips. That is the point of the whole layer --
the rule "one bad row must not discard the good ones" is testable without a database.
"""
from types import TracebackType
from typing import Protocol

from sis.application.ports.repositories import (
    AcademicYearRepository,
    ApiKeyRepository,
    AttendanceRepository,
    EnrolmentRepository,
    ClassSectionRepository,
    GuardianRepository,
    ImportBatchRepository,
    StudentGuardianRepository,
    StudentRepository,
    GradeRepository,
    SubjectRepository,
    SchoolRepository,
    TermRepository,
    YearLevelRepository,
)


class UnitOfWork(Protocol):
    """One transaction, entered as a context manager, exposing every repository."""

    # The outermost scope. Everything below belongs to one school, reached through an
    # academic year or a rung; a student is the one thing that does not, because a child is
    # a person rather than a school's property.
    schools: SchoolRepository

    # Academic structure. Separate repositories rather than one "structure" repository
    # because generation walks years -> levels -> sections and each step queries for what
    # already exists; a merged interface would grow a method per level anyway.
    academic_years: AcademicYearRepository
    year_levels: YearLevelRepository
    class_sections: ClassSectionRepository
    terms: TermRepository
    subjects: SubjectRepository

    # People and their time-bounded placements (decision 2). `enrolments` is a first-class
    # repository, not a collection hanging off a student, because the query that matters
    # -- "who was in 3A during Term 1" -- starts from the class and the date, not the child.
    students: StudentRepository
    enrolments: EnrolmentRepository

    # The adults responsible for those children, and the many-to-many between them. Two
    # repositories rather than one for the same reason `students` and `enrolments` are
    # split: a guardian is a person who outlives any particular link, and the question
    # asked most often -- "which children may this number see" -- starts from the link.
    guardians: GuardianRepository
    student_guardians: StudentGuardianRepository

    # The daily register. A first-class repository rather than a collection on a student
    # for the same reason `enrolments` is: the question asked every morning — "who is in this
    # room today" — starts from the class and the date, not from the child.
    attendance: AttendanceRepository

    grades: GradeRepository

    # Preview batches and their rows. Written in the same transaction as the data a
    # commit applies, so a batch can never be marked committed by a transaction that
    # then failed to write the rows it claimed to have applied.
    imports: ImportBatchRepository

    api_keys: ApiKeyRepository

    def __enter__(self) -> "UnitOfWork":
        """Begin the transaction. The returned object is the one to use inside `with`."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Roll back unless `commit()` was called, and never swallow the exception.

        Returning `None` rather than a truthy value is load-bearing: a `TermClosed` or
        `ImportContentMismatch` must keep travelling up to the API layer that turns it
        into a status code. An `__exit__` that suppressed it would leave the caller
        holding a rolled-back transaction and a success response.
        """
        ...

    def commit(self) -> None:
        """Make every write in this transaction durable, in one step."""
        ...

    def rollback(self) -> None:
        """Discard every write in this transaction.

        Explicit as well as automatic, for the service that decides a batch is
        unwritable after inspecting outcomes rather than by raising.
        """
        ...
