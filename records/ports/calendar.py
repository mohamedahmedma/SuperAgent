"""The academic calendar, as far as this service needs it."""
from __future__ import annotations

from typing import Protocol

from records.domain.terms import SchoolTerm


class SchoolCalendar(Protocol):
    """The academic calendar, as far as this service needs it."""

    def term(self, code: str) -> SchoolTerm | None:
        """That named term, or `None` when the school has no such term."""

    def current_term(self) -> SchoolTerm | None:
        """The term we are in, or the most recent one when today falls in no term.

        The fallback is what a parent means. Asked in August, "how is she doing" is a
        question about the term that just ended, not an error — so a gap between years
        answers with the last term rather than refusing.
        """


__all__ = ["SchoolCalendar"]
