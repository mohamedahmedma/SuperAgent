"""The swap seam.

Everything LMS-shaped is behind `LmsAdapter`. Routes never import a backend's symbol,
never see its web-service function names, never handle its error types. That is what
makes "replace the system of record later" a real option rather than an intention: the
blast radius of the swap is this one file.

WHAT THE PROTOCOL ASKS FOR, AND WHY.
It asks for one call per student per term, returning a figure the gradebook itself
computed. Its first shape was per-course lists of assignments that the facade then
aggregated, and measuring against a live instance showed that shape was unusable: the
exclusion flag was not exposed at all, so an excused assignment was indistinguishable
from a counted one, and re-deriving a percentage from the numbers that WERE exposed
gave 50% or 30% for a child genuinely on 90%.

So the rule is: the gradebook says what the number is, and this service transports it.
`records.grading` aggregates nothing — the arithmetic it used to do is arithmetic nobody
should be doing outside the gradebook.

Failures are normalised to one exception on purpose. `LmsUnavailable` is what makes the
honest-failure path possible end to end: the route turns it into a 503 with
`code: "lms_unavailable"`, and the agent turns that into "I cannot reach the school
records right now" — never into a plausible-sounding grade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class LmsUnavailable(RuntimeError):
    """The system of record could not be reached or did not answer usefully.

    Deliberately does not distinguish "down", "timed out" and "returned nonsense".
    All three mean the same thing to the parent, and collapsing them stops a caller
    from probing the LMS's health through this service.
    """


# ---------------------------------------------------------------------------
# What an adapter returns
# ---------------------------------------------------------------------------


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


class LmsAdapter(Protocol):
    #: Does this backend name its own subjects?
    #:
    #: `False` for a backend whose course list is flat and whose titles are whatever a
    #: teacher typed — the school decides what each course *is* and whether a parent may
    #: see it, and `CourseBinding` is where it says so.
    #:
    #: `True` for a system of record that already stores marks against the school's own
    #: subject codes. Requiring bindings there would mean re-entering the curriculum into
    #: this service, which is supposed to hold no data of its own, and dropping every
    #: subject until somebody did.
    #:
    #: Declared on the port rather than discovered with `isinstance`, so a route asks what
    #: a backend can do instead of which class it happens to be.
    reports_own_subjects: bool = False

    """What the facade needs from a system of record. Nothing more.

    Both calls take the SCHOOL's student reference — the number on a letter home — not
    an internal LMS id. That keeps the contract backend-agnostic and lets the facade key
    everything on the identifier a registrar can actually look up.

    The adapter is not an authorisation boundary and must never be asked to be one. The
    facade has already decided this guardian may see this student before either method
    is called.
    """

    def get_subject_grades(self, *, student_ref: str, term: str) -> list[SubjectGrade]:
        """Every subject's result for one student in one term."""
        ...

    def get_subject_attendance(self, *, student_ref: str, term: str) -> list[SubjectAttendance]:
        """Every subject's attendance for one student in one term."""
        ...


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeLms:
    #: Bindings apply — the fixture exists to exercise that path.
    reports_own_subjects = False

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

    def get_subject_grades(self, *, student_ref: str, term: str) -> list[SubjectGrade]:
        if self.unavailable:
            raise LmsUnavailable("FakeLms configured as unavailable")
        return list(self.grades.get((student_ref, term), []))

    def get_subject_attendance(self, *, student_ref: str, term: str) -> list[SubjectAttendance]:
        if self.unavailable:
            raise LmsUnavailable("FakeLms configured as unavailable")
        return list(self.attendance.get((student_ref, term), []))


# The process-wide adapter. `app.py` sets it at startup; tests override it. A
# module-level slot rather than a FastAPI dependency because it is genuinely process
# configuration, not per-request state.
_adapter: LmsAdapter | None = None


def set_adapter(adapter: LmsAdapter) -> None:
    global _adapter
    _adapter = adapter


def get_adapter() -> LmsAdapter:
    if _adapter is None:
        raise LmsUnavailable("No LMS adapter configured.")
    return _adapter
