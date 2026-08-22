"""Turning what the LMS reported into what the contract promises.

A deliberate layer rather than inline mapping in the routes, for three reasons:

**It is where the school's own rules are applied.** An unpublished course must not reach
a parent; a percentage becomes a letter according to school policy. Routes orchestrate,
the LMS reports, and the decisions in between live here where they can be read in one
place and tested without a database or a network.

**It is the security-relevant step.** Matching LMS subjects against published bindings
is what stops a teacher's sandbox course appearing on a report. Doing that in a route,
between other work, is how it eventually gets skipped in the one route somebody adds
later.

**It has no I/O**, so every rule below is testable as a pure function.

Nothing here calculates a grade. The LMS computed those; this classifies and reshapes.
"""
from __future__ import annotations

from records.grading import DEFAULT_POLICY, GradingPolicy
from records.lms import SubjectAttendance, SubjectGrade
from records.models import CourseBinding
from records.schemas import (
    AcademicGrade,
    AttendanceDay,
    CourseGrade,
    GradeCategory,
)


def term_prefix(term_code: str) -> str:
    """The course-idnumber prefix identifying a term, e.g. "2026-T1" -> "2026-T1-".

    One function because the convention is load-bearing: it is how the LMS narrows a
    query, how bindings are matched, and how a term's courses are told apart. Spelling
    it inline in three places is how the trailing hyphen goes missing in one of them and
    "2026-T1" starts matching "2026-T10".
    """
    return f"{term_code}-"


def _index_bindings(bindings: list[CourseBinding]) -> dict[str, CourseBinding]:
    """Bindings keyed by the reference the LMS reports.

    Keyed on `lms_idnumber` first because that is the school's own course key and what
    the plugin returns; the numeric course id is the fallback for a binding recorded
    before idnumbers were in use.
    """
    index: dict[str, CourseBinding] = {}
    for binding in bindings:
        if binding.lms_idnumber:
            index[str(binding.lms_idnumber)] = binding
        index.setdefault(str(binding.lms_course_id), binding)
    return index


class GradeAssembler:
    """LMS subject results -> the grades contract.

    Constructed with a policy so a school's letter boundaries are injected rather than
    hardcoded, and so a test can assert against its own bands without editing a global.
    """

    def __init__(self, policy: GradingPolicy | None = None):
        self.policy = policy or DEFAULT_POLICY

    def assemble(
        self, subjects: list[SubjectGrade], bindings: list[CourseBinding]
    ) -> list[CourseGrade]:
        """Match reported subjects to published bindings and map them.

        A subject with no published binding is DROPPED, silently and deliberately. The
        LMS answers for every course a student is enrolled in; the school decides which
        of those a parent may see. A teacher's sandbox, a course still being built, a
        subject whose grades are not yet released — each is a real reason a course
        exists in the LMS and has no business on a report.

        The drop is not an error. There is nothing for a parent to be told about a
        course they were never meant to know exists.
        """
        index = _index_bindings(bindings)

        assembled: list[CourseGrade] = []
        for subject in subjects:
            binding = index.get(subject.course_ref)
            if binding is None:
                continue
            assembled.append(self._to_course_grade(subject, binding))
        return assembled

    def assemble_unbound(self, subjects: list[SubjectGrade]) -> list[CourseGrade]:
        """Map subjects a backend has already identified, with no binding table.

        For a system of record that stores marks against the school's own subject codes:
        every subject it reports is one the school entered against this child for this
        term, so there is nothing to match and nothing to drop.

        **The drop that protects a parent from a teacher's sandbox is not lost — it moved.**
        On the Moodle path it happens here, because Moodle answers for every course a
        student is enrolled in. On this path it happened upstream: a subject with no mark
        recorded for this child in this term has no row to return.

        Nothing is invented for the fields a binding used to supply. `course_id` is the
        subject code, because that is the identifier this backend answers about and a
        fabricated numeric id would be a key that resolves to nothing.
        """
        return [self._to_unbound_course_grade(subject) for subject in subjects]

    def _to_unbound_course_grade(self, subject: SubjectGrade) -> CourseGrade:
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
                unavailable_reason=subject.academic_unavailable,
            ),
            graded_count=subject.graded_count,
            excused_count=subject.excluded_count,
            missing_count=subject.pending_count,
            is_complete=subject.is_complete,
        )

    def _to_course_grade(self, subject: SubjectGrade, binding: CourseBinding) -> CourseGrade:
        letter, passed = self.policy.classify(subject.percentage)
        academic_letter, academic_passed = self.policy.classify(subject.academic_percentage)

        return CourseGrade(
            course_id=str(binding.lms_course_id),
            subject_code=binding.subject_code,
            # Names come from the BINDING, not from the LMS course title. The school
            # decides what a subject is called in each language; a Moodle fullname is
            # whatever the teacher typed, and is rarely bilingual.
            subject_name_ar=binding.subject_name_ar,
            subject_name_en=binding.subject_name_en or subject.subject_name,
            computed_percentage=subject.percentage,
            letter_grade=letter,
            passed=passed,
            academic=AcademicGrade(
                percentage=subject.academic_percentage,
                letter_grade=academic_letter,
                passed=academic_passed,
                unavailable=subject.academic_unavailable,
            ),
            categories=[
                GradeCategory(
                    name=str(category.get("name") or ""),
                    percentage=category.get("percentage"),
                )
                for category in subject.categories
            ],
            graded_count=subject.graded_count,
            excused_count=subject.excluded_count,
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

    def __init__(self, bindings: list[CourseBinding], *, unbound: bool = False):
        """`unbound` for a backend that names its own subjects.

        There is then no binding table to filter against, and nothing to filter: every
        subject such a backend reports is one the school recorded against this child for
        this term. The drop that keeps a teacher's sandbox away from a parent still
        happens — it happens upstream, where a subject with no register taken has no row.
        """
        self._index = _index_bindings(bindings)
        self._unbound = unbound

    def visible(self, subjects: list[SubjectAttendance]) -> list[SubjectAttendance]:
        """Only subjects with a published binding — same rule as grades.

        Everything, on the unbound path. Filtering against an empty index there would
        return nothing at all and report a term of registers as "no attendance recorded".
        """
        if self._unbound:
            return list(subjects)
        return [s for s in subjects if s.course_ref in self._index]

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
