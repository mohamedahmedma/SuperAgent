"""Deterministic marks, so the service and its tests need no live system of record.

Also the reference for what a correct adapter returns — particularly a subject where
`percentage` and `academic_percentage` differ, which is the case a real adapter is most
likely to flatten into one number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from records.domain.errors import LmsUnavailable
from records.domain.marks import SubjectAttendance, SubjectGrade


@dataclass
class FakeLms:
    """Deterministic fixtures, so the service and its tests need no live LMS.

    Also the reference for what a correct adapter returns — particularly a subject
    where `percentage` and `academic_percentage` differ, which is the case a real
    adapter is most likely to flatten into one number.
    """

    grades: dict[tuple[str, str], list[SubjectGrade]] = field(default_factory=dict)
    attendance: dict[tuple[str, str], list[SubjectAttendance]] = field(default_factory=dict)
    # Set to raise instead of answering, so the honest-failure path can be tested
    # without taking a real service down.
    unavailable: bool = False

    #: Every `(student_ref, term, guardian_ref)` this fixture was asked for, in order.
    #: A test asserts against it that the guardian actually reaches the backend — the
    #: whole point of the argument is lost silently if a route stops passing it, and
    #: nothing else would fail.
    asked: list[tuple[str, str, str]] = field(default_factory=list)

    def get_subject_grades(
        self, *, student_ref: str, term: str, guardian_ref: str = ""
    ) -> list[SubjectGrade]:
        self.asked.append((student_ref, term, guardian_ref))
        if self.unavailable:
            raise LmsUnavailable("FakeLms configured as unavailable")
        return list(self.grades.get((student_ref, term), []))

    def get_subject_attendance(
        self, *, student_ref: str, term: str, guardian_ref: str = ""
    ) -> list[SubjectAttendance]:
        self.asked.append((student_ref, term, guardian_ref))
        if self.unavailable:
            raise LmsUnavailable("FakeLms configured as unavailable")
        return list(self.attendance.get((student_ref, term), []))


__all__ = ["FakeLms"]
