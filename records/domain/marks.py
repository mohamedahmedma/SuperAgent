"""One subject's result and one subject's register, as a system of record states them.

Values, not rows. This service stores neither: it asks, reshapes, and forgets. What each
adapter owes the rest of the facade is one of these, and the two invariants below are the
whole reason they are worth a module.

**Two percentages, and they are not interchangeable.** A school that grades attendance
produces two legitimate answers to "how is she doing in maths". `percentage` is the
official course total, attendance and all; `academic_percentage` is the assessments alone.
A parent almost always means the second; the school's record is the first. Measured on a
real Moodle instance: 65% against 80% for the same child.

**Nothing is silently absent.** A missing figure carries a reason. A model handed `null`
will narrate a plausible one; a model handed `"points_not_percentage"` will not.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectGrade:
    """One subject's result, as the system of record already computed it.

    Two percentages, because a school that grades attendance produces two legitimate
    answers to "how is she doing in maths" and they are not interchangeable:

    `percentage` is the official course total, attendance and all. `academic_percentage`
    is the assessments alone. A parent almost always means the second; the school's
    record is the first. Neither is a rounding of the other — measured on a real
    instance, 65% against 80% for the same child.

    `academic_percentage` is None when it could not be derived EXACTLY — the course uses
    a weighted or drop-lowest scheme that cannot be reconstructed from points.
    `academic_unavailable` says which. It is never an approximation, because a consumer
    that ignores a caveat flag puts the approximation in front of a parent.
    """

    course_ref: str
    #: The subject's name as the backend reports it. Singular because a flat course
    #: course title, whatever a teacher typed, rarely bilingual — and on that path the
    #: school's own names come from `CourseBinding` rather than from here.
    subject_name: str
    percentage: float | None
    #: The Arabic name, when the backend keeps one. Empty where the backend has none.
    #:
    #: Separate rather than folded into `subject_name` because a backend that has both and
    #: must pick one always picks wrong for somebody: this school reads Arabic, and a
    #: report card headed "Mathematics" is one a parent has to decode. Where there is no
    #: binding to supply the school's own wording — see `GradeAssembler.assemble_unbound` —
    #: this is the only place it can come from.
    subject_name_ar: str = ""
    academic_percentage: float | None = None
    academic_unavailable: str = ""
    graded_count: int = 0
    excluded_count: int = 0
    pending_count: int = 0
    is_complete: bool = False
    # The gradebook's own category subtotals. Exact under every aggregation scheme,
    # so this is the reliable route to a partial subject grade when the derived
    # `academic_percentage` is unavailable.
    categories: tuple[dict, ...] = ()


@dataclass(frozen=True)
class SubjectAttendance:
    """One subject's attendance, as the system of record computed it.

    `percentage` is points-weighted, not a day count: the school's own status values
    decide what an excused absence or a late arrival costs. A student marked Excused
    throughout reads 50% on a default status set, where counting present days would say
    0% and accuse them of missing school they were excused from.

    None when no register has been taken — which is not zero. A child cannot be absent
    from a class nobody recorded.
    """

    course_ref: str
    subject_name: str
    percentage: float | None
    taken_sessions: int = 0
    by_status: tuple[dict, ...] = ()
    # Carried so a term-level figure can be aggregated CORRECTLY across subjects.
    # Averaging the per-subject percentages would weight a subject with two registers
    # the same as one with forty; summing points and maxima is what the LMS itself does
    # across activities.
    points: float = 0.0
    max_points: float = 0.0


__all__ = ["SubjectAttendance", "SubjectGrade"]
