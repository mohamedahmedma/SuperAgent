"""The system of record that holds the class, its subject board and its staff.

## Why this is its own port, and why it is only ONE port

The reasoning in [timetable.py](timetable.py) applies unchanged: `LmsAdapter` is "the system
of record that holds the **marks**", chosen by `RECORDS_LMS`, and a class roster is
structural school data like the academic calendar and the guardian links — wired from
`SIS_BASE_URL`, because a school's structure lives in its SIS whoever happens to hold its
gradebook.

What is new here is the count. Three capabilities reach this port — which class, which
subjects, which teachers — and they are one port with one method rather than three, because
all three are answers about the **same room**, resolved from the same time-bounded
placement. Three ports would ask the system of record three times and could be told about
three different rooms: a placement edited between two calls answers with one room's subjects
beside another room's staff. `week_for_student` states that rule for the timetable ("so a
placement edited between them cannot produce a week belonging to a room the child is no
longer in") and it is the same rule here, three times over.

So the facade asks once and projects. The three parent-facing routes over this are a
presentation decision; the read is not divisible.

## The subject travels to the system of record

Like every other port here, the call names the parent it reads **on behalf of**, and a
backend that can use it is expected to. The facade has already decided the guardian may be
told about this child; naming her again means the system of record decides it independently,
from the registrar's own data, on this request.

## The class is deliberately not an argument

A room is reached only through the placement a child holds for a term, and that resolution
belongs to the system of record, which owns the placements. A caller holding a class code
would eventually ask about a room the child has left. So this takes a student and a term,
exactly as the marks and timetable ports do, and never a class.
"""
from __future__ import annotations

from typing import Protocol

from records.domain.classroom import StudentClassroom


class StudentClassrooms(Protocol):
    """What the facade needs to answer "which class, which subjects, which teachers"."""

    def get_classroom(
        self, *, student_ref: str, term: str, guardian_ref: str
    ) -> StudentClassroom:
        """One child's room for one term, read on behalf of one guardian.

        Takes the SCHOOL's student reference — the number on a letter home — never an
        internal id, so the contract stays backend-agnostic and the facade keys everything
        on the identifier a registrar can look up.

        Returns a `StudentClassroom` whose `status` says whether there is a room at all. A
        child with no placement for the term is `NO_CLASS` and **not** an exception: she had
        left, or had not yet joined, and neither is a failure to read. An implementation
        that cannot reach its backend raises `UpstreamUnavailable` instead — "could not ask"
        is never "the answer is no".

        Note what the term does not buy. It resolves which ROOM she sat in; the staffing it
        comes back with is present tense, because the system of record's assignment table
        carries no term. An implementation must not pretend otherwise.
        """
        ...


__all__ = ["StudentClassrooms"]
