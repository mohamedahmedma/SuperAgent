"""Which children a guardian may be told about.

**Authorisation is not this service's fact.** Which guardian may see which student is the
registrar's: entered from paperwork, amended by custody decisions, audited in `sis/`.
This port asks rather than remembers, and `sis/` re-checks the answer before returning a
mark — two independent refusals from one source of truth, instead of a second copy that
goes stale the first time a court order is applied to the other one.
"""
from __future__ import annotations

from typing import Protocol

from records.domain.people import PermittedStudent


class GuardianDirectory(Protocol):
    """The two authorization questions this service asks, and nothing else."""

    def children_of(
        self, guardian_id: str, *, school_code: str | None = None
    ) -> list[PermittedStudent]:
        """Every child this guardian may be told about. Empty when there are none.

        Empty is an ordinary answer: a parent whose only link carries a custody restriction
        has no children *to be told about*, and that is different from not being a parent.
        Both come back empty here on purpose — the distinction is one this service is not
        entitled to reveal, since a caller who could tell them apart could detect a
        restriction from the outside.
        """

    def permits(self, guardian_id: str, student_id: str) -> PermittedStudent | None:
        """That one child, if this guardian may be told about her. `None` otherwise."""


__all__ = ["GuardianDirectory"]
