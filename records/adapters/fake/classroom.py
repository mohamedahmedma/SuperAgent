"""Deterministic classrooms, so the service and its tests need no live SIS.

Also the reference for what a correct adapter returns — particularly the three empty
answers a real one is most likely to flatten into each other: a child with no placement, a
room whose subject board nobody has curated, and a room whose staffing nobody has entered.
See `records/domain/classroom.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from records.domain.classroom import ClassroomStatus, StudentClassroom
from records.domain.errors import ClassroomUnavailable


@dataclass
class FakeClassrooms:
    """Rooms in a dict, keyed by `(student_ref, term)`. The default when no SIS is configured.

    A key that is absent answers `NO_CLASS`, which is what an unconfigured deployment should
    say: it knows of no placement, rather than inventing a room it cannot name.
    """

    rooms: dict[tuple[str, str], StudentClassroom] = field(default_factory=dict)
    #: Set to raise instead of answering, so the honest-failure path can be tested without
    #: taking a real service down.
    unavailable: bool = False

    #: Every `(student_ref, term, guardian_ref)` this fixture was asked for, in order. A
    #: test asserts against it that the guardian actually reaches the backend — the whole
    #: point of that argument is lost silently if a route stops passing it, and nothing
    #: else would fail. It also proves the three parent-facing routes each ask ONCE.
    asked: list[tuple[str, str, str]] = field(default_factory=list)

    def get_classroom(
        self, *, student_ref: str, term: str, guardian_ref: str = ""
    ) -> StudentClassroom:
        self.asked.append((student_ref, term, guardian_ref))
        if self.unavailable:
            raise ClassroomUnavailable("FakeClassrooms configured as unavailable")
        return self.rooms.get(
            (student_ref, term), StudentClassroom(status=ClassroomStatus.NO_CLASS)
        )


__all__ = ["FakeClassrooms"]
