"""The SQLAlchemy `UnitOfWork`: one session, every repository, one commit.

This is the only object in the service that calls `commit()`, `rollback()` or `close()`
on a session. Repositories below it stage work and never end a transaction, so a service
that writes two hundred enrolments and then hits a bad row leaves nothing behind --
the guarantee invariant 4 rests on.

The session is created in `__enter__`, not in `__init__`, because a unit of work that is
constructed and never entered must not have taken a connection. FastAPI resolves
dependencies before the handler body runs, and a request rejected by auth or validation
would otherwise check out a pooled connection, hold it for the length of the request and
return it only when the garbage collector got round to the session. Under load that is a
pool exhaustion that looks like the database being slow.

The factory is a parameter so a test can point one unit of work at its own engine
without touching process-wide state; it defaults to the shared sessionmaker, resolved at
enter time so `reset_engine()` between tests is honoured.
"""
from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Final

from sqlalchemy.orm import Session

from sis.infrastructure.db.session import get_sessionmaker

if TYPE_CHECKING:
    from sis.application.ports.repositories import (
        AcademicYearRepository,
        ApiKeyRepository,
        ClassSectionRepository,
        EnrolmentRepository,
        GradeRepository,
        GuardianRepository,
        ImportBatchRepository,
        StudentGuardianRepository,
        StudentRepository,
        SubjectRepository,
        TermRepository,
        YearLevelRepository,
    )

# The attribute names the port promises. Kept as data so `__getattr__` can tell
# "you forgot the `with`" apart from "you misspelled the repository".
_REPOSITORY_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
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
    }
)


class SqlAlchemyUnitOfWork:
    """One database transaction, exposing the repositories that write inside it.

    Satisfies the `UnitOfWork` protocol structurally -- it does not inherit from it, so
    a service depending on the port stays free of any import from `infrastructure`.
    """

    # Annotations only, deliberately: the attributes exist between `__enter__` and
    # `__exit__` and nowhere else, so `__getattr__` below can catch misuse.
    academic_years: AcademicYearRepository
    year_levels: YearLevelRepository
    class_sections: ClassSectionRepository
    terms: TermRepository
    subjects: SubjectRepository
    students: StudentRepository
    enrolments: EnrolmentRepository
    guardians: GuardianRepository
    student_guardians: StudentGuardianRepository
    grades: GradeRepository
    imports: ImportBatchRepository
    api_keys: ApiKeyRepository

    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> "SqlAlchemyUnitOfWork":
        """Open the session and build the repositories bound to it."""
        if self._session is not None:
            # Re-entering would rebind the repositories to a second session while the
            # first one still holds uncommitted writes, and the outer `__exit__` would
            # then close the wrong one.
            raise RuntimeError("This unit of work is already open; use a new one.")

        factory = self._session_factory or get_sessionmaker()
        session = factory()
        self._session = session
        self._bind(session)
        # No explicit `begin()`: a SQLAlchemy 2.0 session starts its transaction on
        # first use, and calling `begin()` on one that already has it raises.
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Discard anything uncommitted, always release the connection, never swallow.

        The rollback is unconditional rather than guarded by a "did we commit" flag.
        After a successful `commit()` it discards only work done since that commit,
        which is the same rule stated once instead of twice -- and a flag is exactly the
        kind of state that goes stale when someone later adds a second commit or an
        early return. The close sits in `finally` because a rollback on a connection the
        database has already dropped raises, and a leaked connection is a worse outcome
        than the error that caused it.

        Returns `None` so `TermClosed` and friends keep travelling to the API layer that
        turns them into status codes.
        """
        session = self._session
        self._session = None
        self._unbind()
        if session is None:
            return

        try:
            session.rollback()
        finally:
            session.close()

    def commit(self) -> None:
        """Make every write staged in this transaction durable, in one step."""
        self._require_session().commit()

    def rollback(self) -> None:
        """Discard every write staged since the last commit."""
        self._require_session().rollback()

    def _bind(self, session: Session) -> None:
        """Instantiate every repository against the one open session."""
        # Imported here, not at module scope: alembic's `env.py` imports this package for
        # the metadata, and a top-level import would drag the whole repository layer --
        # and every model it touches -- into a migration run. It also keeps repositories
        # free to import from `sis.infrastructure.db` without a cycle.
        from sis.infrastructure import repositories as repo

        self.academic_years = repo.SqlAlchemyAcademicYearRepository(session)
        self.year_levels = repo.SqlAlchemyYearLevelRepository(session)
        self.class_sections = repo.SqlAlchemyClassSectionRepository(session)
        self.terms = repo.SqlAlchemyTermRepository(session)
        self.subjects = repo.SqlAlchemySubjectRepository(session)
        self.students = repo.SqlAlchemyStudentRepository(session)
        self.enrolments = repo.SqlAlchemyEnrolmentRepository(session)
        self.guardians = repo.SqlAlchemyGuardianRepository(session)
        self.student_guardians = repo.SqlAlchemyStudentGuardianRepository(session)
        self.grades = repo.SqlAlchemyGradeRepository(session)
        self.imports = repo.SqlAlchemyImportBatchRepository(session)
        self.api_keys = repo.SqlAlchemyApiKeyRepository(session)

    def _unbind(self) -> None:
        """Drop the repositories so a closed unit of work cannot still be written through."""
        for name in _REPOSITORY_ATTRIBUTES:
            self.__dict__.pop(name, None)

    def _require_session(self) -> Session:
        if self._session is None:
            raise RuntimeError(
                "Unit of work used outside its `with` block; "
                "commit and rollback need an open session."
            )
        return self._session

    def __getattr__(self, name: str) -> object:
        # Only reached when the attribute is absent, i.e. outside the `with` block.
        if name in _REPOSITORY_ATTRIBUTES:
            raise RuntimeError(
                f"{name!r} is only available inside "
                f"`with SqlAlchemyUnitOfWork() as uow:` -- the repositories exist for "
                "the life of the transaction."
            )
        raise AttributeError(name)


__all__ = ["SqlAlchemyUnitOfWork"]
