"""Deterministic weeks, so the service and its tests need no live SIS.

Also the reference for what a correct adapter returns — particularly the two empty answers
a real one is most likely to flatten into each other: a child with no class this term, and
a class whose week nobody has laid out. See `records/domain/timetable.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from records.domain.errors import TimetableUnavailable
from records.domain.timetable import StudentTimetable, TimetableStatus


@dataclass
class FakeTimetables:
    """Weeks in a dict, keyed by `(student_ref, term)`. The default when no SIS is configured.

    A key that is absent answers `NO_CLASS`, which is what an unconfigured deployment should
    say: it knows of no placement, rather than inventing an empty week for a room it cannot
    name.
    """

    weeks: dict[tuple[str, str], StudentTimetable] = field(default_factory=dict)
    #: Set to raise instead of answering, so the honest-failure path can be tested without
    #: taking a real service down.
    unavailable: bool = False

    #: Every `(student_ref, term, guardian_ref)` this fixture was asked for, in order. A
    #: test asserts against it that the guardian actually reaches the backend — the whole
    #: point of that argument is lost silently if a route stops passing it, and nothing
    #: else would fail.
    asked: list[tuple[str, str, str]] = field(default_factory=list)

    def get_timetable(
        self, *, student_ref: str, term: str, guardian_ref: str = ""
    ) -> StudentTimetable:
        self.asked.append((student_ref, term, guardian_ref))
        if self.unavailable:
            raise TimetableUnavailable("FakeTimetables configured as unavailable")
        return self.weeks.get(
            (student_ref, term), StudentTimetable(status=TimetableStatus.NO_CLASS)
        )


__all__ = ["FakeTimetables"]
