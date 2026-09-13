"""The SQLAlchemy `UnitOfWork`: one session, every repository, one commit.

Modelled on `sis/infrastructure/db/unit_of_work.py`. It is the only object in the
backend that commits, rolls back or closes a session; the repositories it builds stage
work and never end a transaction.

The session is opened in `__enter__`, not `__init__`, so a unit of work that is built and
never entered has not taken a pooled connection. The session factory is a parameter so a
test can point one unit of work at its own schema; by default it is the process-wide
`SessionLocal`, looked up at enter time.
"""
from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Final

from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from backend.application.ports.repositories import (
        ConversationRepository,
        DocumentPairRepository,
    )

# The attribute names the port promises, kept as data so `__getattr__` can tell "you
# forgot the `with`" apart from "you misspelled the repository".
_REPOSITORY_ATTRIBUTES: Final[frozenset[str]] = frozenset({"conversations", "document_pairs"})


class SqlAlchemyUnitOfWork:
    """One database transaction, exposing the repositories that write inside it.

    Satisfies the `UnitOfWork` protocol structurally rather than by inheritance, so code
    that depends on the port imports nothing from `infra`.
    """

    # Annotations only: the attributes exist between `__enter__` and `__exit__`.
    conversations: ConversationRepository
    document_pairs: DocumentPairRepository

    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> "SqlAlchemyUnitOfWork":
        if self._session is not None:
            # Re-entering would rebind the repositories to a second session while the
            # first still holds uncommitted writes.
            raise RuntimeError("This unit of work is already open; use a new one.")
        factory = self._session_factory
        if factory is None:
            from backend.infra.database import SessionLocal

            factory = SessionLocal
        self._session = factory()
        self._bind(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Discard anything uncommitted, always release the connection, never swallow.

        The rollback is unconditional: after a `commit()` it discards only what was
        staged since, which is the same rule without a "did we commit" flag to go stale.
        The close is in `finally` because rolling back on a connection the database has
        already dropped raises, and a leaked connection is worse than that error.
        """
        session, self._session = self._session, None
        for name in _REPOSITORY_ATTRIBUTES:
            self.__dict__.pop(name, None)
        if session is None:
            return
        try:
            session.rollback()
        finally:
            session.close()

    def commit(self) -> None:
        self._require_session().commit()

    def rollback(self) -> None:
        self._require_session().rollback()

    def _bind(self, session: Session) -> None:
        # Imported here so importing this module does not pull in every repository and
        # model — alembic's env.py and the app's startup gate only need the database.
        from backend.infra.repositories import (
            SqlAlchemyConversationRepository,
            SqlAlchemyDocumentPairRepository,
        )

        self.conversations = SqlAlchemyConversationRepository(session)
        self.document_pairs = SqlAlchemyDocumentPairRepository(session)

    def _require_session(self) -> Session:
        if self._session is None:
            raise RuntimeError(
                "Unit of work used outside its `with` block; commit and rollback need an "
                "open session."
            )
        return self._session

    def __getattr__(self, name: str) -> object:
        # Only reached when the attribute is absent, i.e. outside the `with` block.
        if name in _REPOSITORY_ATTRIBUTES:
            raise RuntimeError(
                f"{name!r} is only available inside `with SqlAlchemyUnitOfWork() as uow:` "
                "— the repositories exist for the life of the transaction."
            )
        raise AttributeError(name)


__all__ = ["SqlAlchemyUnitOfWork"]
