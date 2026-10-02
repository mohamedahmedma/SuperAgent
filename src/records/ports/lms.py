"""The system of record that holds the marks.

Everything LMS-shaped is behind this. Routes never import a Moodle symbol, never see a
web-service function name, never handle a Moodle error type — replacing the system of
record means writing one class, and the blast radius is one file under `adapters/`.
"""
from __future__ import annotations

from typing import Protocol

from records.domain.marks import SubjectAttendance, SubjectGrade


class LmsAdapter(Protocol):
    """What the facade needs from a system of record. Nothing more.

    Both calls take the SCHOOL's student reference — the number on a letter home — not
    an internal LMS id. That keeps the contract backend-agnostic and lets the facade key
    everything on the identifier a registrar can actually look up.

    ## `guardian_ref`, and why it is on the port

    Both calls also name the parent the read is **on behalf of**, and a backend that can
    use it is expected to.

    This used to say the adapter "is not an authorisation boundary and must never be asked
    to be one", because the facade had already decided. The facade still decides — nothing
    below removes that check — but deciding in one place and then asking the system of
    record a question that names no parent throws the answer away at the last hop. A
    backend that re-checks the link cannot, and a leaked adapter credential reaches one
    family instead of the school.

    So the rule is now the stronger one: **the subject travels all the way to the system of
    record.** Two independent refusals, made from the same registrar data, rather than one
    made here and trusted downstream. A backend with no notion of guardians ignores the
    argument; it is on the port so that forgetting to pass it is impossible rather than
    merely unlikely.
    """

    def get_subject_grades(
        self, *, student_ref: str, term: str, guardian_ref: str
    ) -> list[SubjectGrade]:
        """Every subject's result for one student in one term, read for one guardian."""
        ...

    def get_subject_attendance(
        self, *, student_ref: str, term: str, guardian_ref: str
    ) -> list[SubjectAttendance]:
        """Every subject's attendance for one student in one term, read for one guardian."""
        ...


__all__ = ["LmsAdapter"]
