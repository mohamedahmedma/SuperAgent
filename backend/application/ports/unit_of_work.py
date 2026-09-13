"""The transaction boundary: every repository a service can reach, and one commit.

The same contract as `sis/application/ports/unit_of_work.py`. `commit()` is the only
place a transaction commits, and leaving the `with` block without it rolls back — so a
service that raises halfway leaves nothing behind without having had to remember to
catch anything.

Repositories are attributes of the unit of work rather than objects injected separately:
two repositories obtained independently can sit on two connections, and then "write the
conversation, then its messages" is two transactions with a window between them.
Reaching both through one unit of work makes sharing a transaction the easy thing.
"""
from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Protocol

from backend.application.ports.repositories import ConversationRepository


class UnitOfWork(Protocol):
    """One transaction, entered as a context manager, exposing every repository."""

    conversations: ConversationRepository

    def __enter__(self) -> "UnitOfWork":
        """Begin the transaction. Use the returned object inside the `with` block."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Roll back anything uncommitted, and never swallow the exception."""
        ...

    def commit(self) -> None:
        """Make every write in this transaction durable, in one step."""
        ...

    def rollback(self) -> None:
        """Discard every write staged since the last commit."""
        ...


#: What a service is given: a way to start a fresh transaction whenever it needs one.
UnitOfWorkFactory = Callable[[], UnitOfWork]
