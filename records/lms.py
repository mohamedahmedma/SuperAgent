"""The swap seam.

Everything LMS-shaped is behind `LmsAdapter`. Routes never import a Moodle symbol,
never see a web-service function name, never handle a Moodle error type. That is what
makes "replace Moodle later" a real option rather than an intention: the blast radius
of the swap is this one file.

Three implementations are anticipated:

* `FakeLms` — deterministic fixtures. Used by the tests and by local development, so
  the whole service runs with no Moodle at all.
* `MoodleAdapter` — the real one. Skeleton below; the web-service call shapes are
  marked because they must be verified against the school's actual Moodle build and
  plugin set rather than trusted from documentation.
* whatever replaces Moodle. It implements this protocol and nothing else changes.

Failures are normalised to one exception on purpose. `LmsUnavailable` is what makes
the honest-failure path possible end to end: the route turns it into a 503 with
`code: "lms_unavailable"`, and the agent turns that into "I cannot reach the school
records right now" — never into a plausible-sounding grade.
"""
from datetime import datetime
from typing import Protocol

from records.schemas import AssignmentGrade, AttendanceDay


class LmsUnavailable(RuntimeError):
    """The system of record could not be reached or did not answer usefully.

    Deliberately does not distinguish "down", "timed out" and "returned nonsense".
    All three mean the same thing to the parent, and collapsing them stops a caller
    from probing the LMS's health through this service.
    """


class LmsCourse:
    """One course as the adapter reports it, before binding to a subject and term."""

    def __init__(self, course_id: int, idnumber: str = "", fullname: str = "", teacher_name: str = ""):
        self.course_id = course_id
        self.idnumber = idnumber
        self.fullname = fullname
        self.teacher_name = teacher_name


class LmsAdapter(Protocol):
    """What the facade needs from a system of record. Nothing more.

    Note the shape of these calls: they take an LMS user id and a list of course ids
    that the *caller has already decided this guardian may see*. The adapter is not an
    authorisation boundary and must never be asked to be one.
    """

    def get_courses(self, *, lms_user_id: int) -> list[LmsCourse]:
        """Courses the student is enrolled in."""
        ...

    def get_assignment_grades(
        self, *, lms_user_id: int, course_id: int
    ) -> list[AssignmentGrade]:
        """Every assignment in one course for one student, already normalised.

        Normalising here rather than in the routes is what keeps the excused/missing
        distinction from leaking LMS-specific encodings into the rollup. Whatever
        Moodle calls an exemption, it arrives at `compute_rollup` as
        `SubmissionStatus.EXCUSED`.
        """
        ...

    def get_attendance(
        self, *, lms_user_id: int, course_ids: list[int], since: datetime | None = None
    ) -> list[AttendanceDay]:
        """Attendance sessions for one student across courses."""
        ...


class FakeLms:
    """Deterministic fixtures, so the service and its tests need no Moodle.

    Also the reference for what a correct adapter returns — particularly the presence
    of an excused assignment, which is the case a real adapter is most likely to get
    wrong and which the rollup tests depend on.
    """

    def __init__(self, courses: dict | None = None, grades: dict | None = None, attendance: dict | None = None):
        self._courses = courses or {}
        self._grades = grades or {}
        self._attendance = attendance or {}

    def get_courses(self, *, lms_user_id: int) -> list[LmsCourse]:
        return list(self._courses.get(lms_user_id, []))

    def get_assignment_grades(self, *, lms_user_id: int, course_id: int) -> list[AssignmentGrade]:
        return list(self._grades.get((lms_user_id, course_id), []))

    def get_attendance(
        self, *, lms_user_id: int, course_ids: list[int], since: datetime | None = None
    ) -> list[AttendanceDay]:
        days = list(self._attendance.get(lms_user_id, []))
        if since is not None:
            days = [d for d in days if d.date >= since]
        return days


class MoodleAdapter:
    """Moodle web services, wrapped.

    NOT YET IMPLEMENTED — the method bodies raise. This is the one piece of the plan
    that must be written against a live Moodle rather than from documentation, because
    the call shapes depend on the school's Moodle version and installed plugins.

    What needs verifying on the school's actual instance before writing these:

    * `gradereport_user_get_grade_items` is the richest read — it returns grade items
      with per-user grades and category structure in one call, which is far cheaper
      than walking `mod_assign_get_assignments` plus `mod_assign_get_grades`. Confirm
      it is enabled in the external service and that it exposes the exemption flag;
      if it does not, the excused/missing distinction has to come from
      `mod_assign_get_grades` and this whole adapter gets chattier.
    * Attendance comes from the `mod_attendance` plugin, not from core. Confirm the
      plugin is installed, that its web-service functions are exposed, and what its
      configured status codes actually are — schools customise them, and the mapping
      onto present/absent/late/excused is per-deployment.
    * The token must belong to a dedicated web-service user with only the functions
      above whitelisted. Not an admin token.

    Two things to build in from the first line, because retrofitting them hurts:
    caching (these calls are chatty and a parent asking three questions should not
    trigger thirty round trips), and a timeout that raises `LmsUnavailable` rather
    than hanging a chat turn.
    """

    def __init__(self, base_url: str, token: str, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def _call(self, function: str, **params):
        raise NotImplementedError(
            "MoodleAdapter is a skeleton. Implement against the school's Moodle instance; "
            "see the class docstring for what to verify first."
        )

    def get_courses(self, *, lms_user_id: int) -> list[LmsCourse]:
        raise NotImplementedError

    def get_assignment_grades(self, *, lms_user_id: int, course_id: int) -> list[AssignmentGrade]:
        raise NotImplementedError

    def get_attendance(
        self, *, lms_user_id: int, course_ids: list[int], since: datetime | None = None
    ) -> list[AttendanceDay]:
        raise NotImplementedError


# The process-wide adapter. `app.py` sets it at startup; tests override it with a
# FakeLms. A module-level slot rather than a FastAPI dependency because it is
# genuinely process configuration, not per-request state.
_adapter: LmsAdapter | None = None


def set_adapter(adapter: LmsAdapter) -> None:
    global _adapter
    _adapter = adapter


def get_adapter() -> LmsAdapter:
    if _adapter is None:
        raise LmsUnavailable("No LMS adapter configured.")
    return _adapter
