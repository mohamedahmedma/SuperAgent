"""Turning what the LMS reported into what the contract promises.

A deliberate layer rather than inline mapping in the routes, for three reasons:

**It is where the school's own rules are applied.** An unpublished course must not reach
a parent; a percentage becomes a letter according to school policy. Routes orchestrate,
the LMS reports, and the decisions in between live here where they can be read in one
place and tested without a database or a network.

**The drop that keeps a teacher's sandbox off a report has moved upstream.** This layer
used to match reported subjects against a table of published course bindings, because a
flat course list with free-text idnumbers gave it no other way to tell a real subject from
somebody's scratch course. The system of record stores marks against the school's own
subject codes now, so a subject it reports for a child in a term is one the school entered
against her — there is nothing to match and nothing to drop.

**It has no I/O**, so every rule below is testable as a pure function.

Nothing here calculates a grade. The LMS computed those; this classifies and reshapes.
"""
from __future__ import annotations

from records.grading import DEFAULT_POLICY, GradingPolicy
from records.lms import SubjectAttendance, SubjectGrade
from records.schemas import (
    AcademicGrade,
    AttendanceDay,
    CourseGrade,
    GradeCategory,
)


class GradeAssembler:
    """LMS subject results -> the grades contract.

    Constructed with a policy so a school's letter boundaries are injected rather than
    hardcoded, and so a test can assert against its own bands without editing a global.
    """

    def __init__(self, policy: GradingPolicy | None = None):
        self.policy = policy or DEFAULT_POLICY

    def assemble(self, subjects: list[SubjectGrade]) -> list[CourseGrade]:
        """Map the subjects the system of record reported, all of them.

        Every subject it returns is one the school entered against this child for this
        term, so there is nothing to match against and nothing to drop. The filter this
        replaces existed for a backend whose course list was flat and whose titles were
        whatever a teacher typed; that backend is gone, and keeping a table to protect
        against it meant keeping a database for a case that could no longer arise.

        Nothing is invented for the fields a binding used to supply. `course_id` is the
        subject code, because that is the identifier this backend answers about and a
        fabricated numeric id would be a key that resolves to nothing.
        """
        return [self._to_course_grade(subject) for subject in subjects]

    def _to_course_grade(self, subject: SubjectGrade) -> CourseGrade:
        letter, passed = self.policy.classify(subject.percentage)
        academic_letter, academic_passed = self.policy.classify(subject.academic_percentage)

        return CourseGrade(
            course_id=subject.course_ref,
            subject_code=subject.course_ref,
            # Each script from the backend that keeps it, falling back to the other rather
            # than to blank: a subject rendered with no name at all is worse than one
            # rendered in the wrong language, and this service holds no translation to
            # invent the missing side with.
            subject_name_ar=subject.subject_name_ar or subject.subject_name,
            subject_name_en=subject.subject_name or subject.subject_name_ar,
            computed_percentage=subject.percentage,
            letter_grade=letter,
            passed=passed,
            academic=AcademicGrade(
                percentage=subject.academic_percentage,
                letter_grade=academic_letter,
                passed=academic_passed,
                # `unavailable`, not `unavailable_reason`. This path spelled it the second
                # way and pydantic, which ignores unknown keyword arguments, dropped it
                # silently — so every parent on the SIS backend was shown a blank academic
                # figure with no explanation of why it could not be derived. The one shape
                # a caveat must never take is absent.
                unavailable=subject.academic_unavailable,
            ),
            # Carried through for the same reason: the gradebook's own category subtotals
            # are exact under every aggregation scheme, which makes them the reliable route
            # to a partial subject grade when the derived academic figure is unavailable.
            categories=[
                GradeCategory(
                    name=str(category.get("name") or ""),
                    percentage=category.get("percentage"),
                )
                for category in subject.categories
            ],
            graded_count=subject.graded_count,
            excused_count=subject.excluded_count,
            # `pending_count`, not `missing_count`. Two real and different fields on the
            # contract, and this path was filling the wrong one: "not marked yet" was
            # being reported as "she did not hand it in".
            pending_count=subject.pending_count,
            is_complete=subject.is_complete,
        )


class AttendanceAssembler:
    """LMS subject attendance -> the term-level attendance contract.

    The contract reports one figure for the term while the LMS answers per subject, so
    this aggregates — and the aggregation is the reason this is not a one-line map.
    """

    #: Statuses that mean the child was accounted for. Excused is here on purpose: a
    #: child off school with a doctor's note has not "missed" school in the sense a
    #: parent is asking about, and counting it against them turns a legitimate absence
    #: into an accusation.
    PRESENT_LIKE = ("present", "late", "excused")

    def visible(self, subjects: list[SubjectAttendance]) -> list[SubjectAttendance]:
        """Everything the system of record reported.

        Kept as a named step rather than inlined, because it is where a filter would go
        if a backend ever needed one again — and because the routes read better saying
        `visible(subjects)` than passing the raw list into four aggregations.

        There is nothing to filter today. Every subject reported is one the school
        recorded against this child for this term; the drop that keeps a half-configured
        course away from a parent happens upstream, where a subject with no register taken
        has no row at all.
        """
        return list(subjects)

    def term_percentage(self, subjects: list[SubjectAttendance]) -> float | None:
        """One attendance figure for the whole term.

        Sums points and maxima rather than averaging the per-subject percentages.
        Averaging would weight a subject with two registers taken equally against one
        with forty, so a single perfect-attendance elective could mask a term of
        absences in everything else.

        None when nothing has been marked anywhere — not zero. A child cannot be absent
        from classes nobody recorded.
        """
        max_points = sum(s.max_points for s in subjects)
        if max_points <= 0:
            return None
        return round((sum(s.points for s in subjects) / max_points) * 100, 2)

    def status_totals(self, subjects: list[SubjectAttendance]) -> dict[str, int]:
        """Session counts per status description, summed across subjects.

        Keyed on the DESCRIPTION rather than the acronym. Acronyms are school-configured
        and short — a school running two status sets may reuse a letter for something
        else entirely — whereas the description is what a parent would be told.
        """
        totals: dict[str, int] = {}
        for subject in subjects:
            for status in subject.by_status:
                label = str(status.get("description") or status.get("acronym") or "")
                if not label:
                    continue
                totals[label] = totals.get(label, 0) + int(status.get("count") or 0)
        return totals

    def counts(self, subjects: list[SubjectAttendance]) -> dict[str, int]:
        """The four totals the contract names, from whatever the school calls them.

        Matched on the English description rather than assumed positions in the status
        set. A school that has renamed or reordered its statuses still reports
        correctly; one running Arabic descriptions falls through to zero here and is
        served by `status_totals`, which passes its own labels through verbatim.
        """
        totals = self.status_totals(subjects)
        buckets = {"present": 0, "absent": 0, "late": 0, "excused": 0}

        for label, count in totals.items():
            lowered = label.strip().lower()
            for bucket in buckets:
                if lowered.startswith(bucket):
                    buckets[bucket] += count
                    break
        return buckets

    @staticmethod
    def recent_days(subjects: list[SubjectAttendance]) -> list[AttendanceDay]:
        """Per-day detail.

        Empty for now, and honestly so: `local_schoolapi` summarises server-side and
        does not return individual sessions, which is what keeps one child's register
        from arriving with their classmates' attached. A day-level view needs a new
        plugin endpoint scoped to one student, not a filter applied after the fact.
        """
        return []
