"""The system of record that holds the weekly plan.

## Why this is its own port and not a method on `LmsAdapter`

`LmsAdapter` is documented as "the system of record that holds the **marks**", and
`RECORDS_LMS` chooses it. A timetable is not a mark: it is structural school data, like the
academic calendar and the guardian links, and those two already have their own ports for
exactly this reason — they are wired from `SIS_BASE_URL` rather than from `RECORDS_LMS`,
because a school's structure lives in its SIS whoever happens to hold its gradebook.

Folding the week into `LmsAdapter` would tie it to that choice, so a school running Moodle
for marks would lose its timetable — a capability its SIS holds perfectly well. It would
also oblige every future marks backend to implement a schedule it has no concept of, or to
refuse it, and a port whose implementations must decline half of it is a port describing two
things.

So: third structural port, same shape as its neighbours. One question, asked on behalf of
one parent.

## The subject travels to the system of record

Like the marks port, the call names the parent it reads **on behalf of**, and a backend that
can use it is expected to. The facade has already decided the guardian may be told about
this child; naming her again means the system of record decides it independently, from the
registrar's own data, on this request. Two refusals rather than one made here and trusted
downstream — so a leaked credential reaches one family instead of the school.

## The class is deliberately not an argument

A timetable is a statement about a *room*, and a child reaches one only through the
placement she holds for a term. That resolution belongs to the system of record, which owns
the placements: it is a time-bounded fact that changes mid-year, and a caller holding a
class code would eventually ask for the week of a room the child has left. So this port
takes a student and a term, exactly as the marks port does, and never a class.
"""
from __future__ import annotations

from typing import Protocol

from records.domain.timetable import StudentTimetable


class StudentTimetables(Protocol):
    """What the facade needs in order to answer "what does her week look like". Nothing more."""

    def get_timetable(
        self, *, student_ref: str, term: str, guardian_ref: str
    ) -> StudentTimetable:
        """One child's week for one term, read on behalf of one guardian.

        Takes the SCHOOL's student reference — the number on a letter home — never an
        internal id, so the contract stays backend-agnostic and the facade keys everything
        on the identifier a registrar can look up.

        Returns a `StudentTimetable` whose `status` says which answer this is. A child with
        no placement for the term is `NO_CLASS` and **not** an exception: she had left, or
        had not yet joined, and neither is a failure to read. An implementation that cannot
        reach its backend raises `UpstreamUnavailable` instead — "could not ask" is never
        "the answer is no".
        """
        ...


__all__ = ["StudentTimetables"]
