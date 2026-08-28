"""A calendar held in a list. The default when no SIS is configured."""
from __future__ import annotations

from records.domain.errors import CalendarUnavailable
from records.domain.terms import SchoolTerm


class FakeSchoolCalendar:
    """A calendar in a list. The default when no SIS is configured."""

    def __init__(self, terms: list[SchoolTerm] | None = None, *, unavailable: bool = False) -> None:
        self.terms = list(terms or [])
        self.unavailable = unavailable

    def _all(self) -> list[SchoolTerm]:
        if self.unavailable:
            raise CalendarUnavailable("The fake calendar is switched off.")
        return self.terms

    def term(self, code: str) -> SchoolTerm | None:
        return next((t for t in self._all() if t.code == str(code)), None)

    def current_term(self) -> SchoolTerm | None:
        terms = self._all()
        if not terms:
            return None
        current = [t for t in terms if t.is_current]
        return current[-1] if current else terms[-1]


__all__ = ["FakeSchoolCalendar"]
