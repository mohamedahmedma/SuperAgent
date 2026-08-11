"""The rules applied between what the LMS reports and what a parent is told.

No database, no network — the assembler is pure, which is why these can be exhaustive.

Two properties carry most of the weight. An unpublished course must never reach a
parent, and a percentage the system of record could not compute must never become a
letter grade.
"""
import pytest

from records.assembler import AttendanceAssembler, GradeAssembler, term_prefix
from records.grading import GradingPolicy
from records.lms import SubjectAttendance, SubjectGrade
from records.models import CourseBinding


def binding(idnumber: str = "2026-T1-G7A-MATH", course_id: int = 9001, **kwargs) -> CourseBinding:
    return CourseBinding(
        lms_course_id=course_id,
        lms_idnumber=idnumber,
        term_id=1,
        subject_code=kwargs.get("subject_code", "MATH"),
        subject_name_ar=kwargs.get("subject_name_ar", "الرياضيات"),
        subject_name_en=kwargs.get("subject_name_en", "Mathematics"),
        grade_level="G7",
        section="A",
        is_published=True,
    )


def subject(course_ref: str = "2026-T1-G7A-MATH", **kwargs) -> SubjectGrade:
    return SubjectGrade(
        course_ref=course_ref,
        subject_name=kwargs.pop("subject_name", "Mathematics"),
        percentage=kwargs.pop("percentage", 80.0),
        **kwargs,
    )


class TestTermPrefix:
    def test_it_appends_the_separator(self):
        assert term_prefix("2026-T1") == "2026-T1-"

    def test_the_separator_is_what_stops_a_term_matching_its_own_successor(self):
        # Without the trailing hyphen, "2026-T1" also prefixes "2026-T10".
        assert not "2026-T10-G7A-MATH".startswith(term_prefix("2026-T1"))


class TestGradeAssembly:
    def test_a_published_subject_is_returned(self):
        courses = GradeAssembler().assemble([subject()], [binding()])

        assert len(courses) == 1
        assert courses[0].computed_percentage == 80.0

    def test_a_subject_with_no_binding_is_dropped(self):
        """A teacher's sandbox, or a course still being built.

        The LMS answers for every course a student is enrolled in; the school decides
        which a parent may see. Dropping is silent and correct — there is nothing to
        tell a parent about a course they were never meant to know exists.
        """
        courses = GradeAssembler().assemble(
            [subject(course_ref="2026-T1-G7A-SANDBOX")], [binding()]
        )

        assert courses == []

    def test_matching_falls_back_to_the_numeric_course_id(self):
        """For a binding recorded before course idnumbers were in use."""
        courses = GradeAssembler().assemble([subject(course_ref="9001")], [binding()])

        assert len(courses) == 1

    def test_subject_names_come_from_the_binding_not_the_lms(self):
        """The school decides what a subject is called, in both languages.

        A Moodle course fullname is whatever the teacher typed and is rarely bilingual.
        """
        courses = GradeAssembler().assemble(
            [subject(subject_name="MATHS!! period 3 (Mr Ahmed)")], [binding()]
        )

        assert courses[0].subject_name_en == "Mathematics"
        assert courses[0].subject_name_ar == "الرياضيات"

    def test_both_percentages_are_carried_through(self):
        """The case the two figures exist for: attendance drags the official total down."""
        courses = GradeAssembler().assemble(
            [subject(percentage=65.0, academic_percentage=80.0)], [binding()]
        )

        assert courses[0].computed_percentage == 65.0
        assert courses[0].academic.percentage == 80.0

    def test_each_percentage_gets_its_own_letter(self):
        courses = GradeAssembler().assemble(
            [subject(percentage=65.0, academic_percentage=80.0)], [binding()]
        )

        assert courses[0].letter_grade == "D"
        assert courses[0].academic.letter_grade == "B"

    def test_an_underivable_academic_figure_gets_no_letter(self):
        """The rule that stops a caveat being ignored.

        When the course uses a weighting that cannot be re-derived, there is no
        percentage — so there must be no letter either. A letter with a null percentage
        beside it is exactly the shape a consumer renders as a grade.
        """
        courses = GradeAssembler().assemble(
            [subject(academic_percentage=None, academic_unavailable="aggregation_not_summable")],
            [binding()],
        )

        assert courses[0].academic.percentage is None
        assert courses[0].academic.letter_grade == ""
        assert courses[0].academic.passed is None
        assert courses[0].academic.unavailable == "aggregation_not_summable"

    def test_nothing_graded_is_not_a_fail(self):
        courses = GradeAssembler().assemble([subject(percentage=None)], [binding()])

        assert courses[0].computed_percentage is None
        assert courses[0].letter_grade == ""
        assert courses[0].passed is None

    def test_a_genuine_zero_is_a_fail(self):
        """The counterpart: zero is a mark a child can earn."""
        courses = GradeAssembler().assemble([subject(percentage=0.0)], [binding()])

        assert courses[0].computed_percentage == 0.0
        assert courses[0].letter_grade == "F"
        assert courses[0].passed is False

    def test_the_school_policy_is_injected_not_hardcoded(self):
        strict = GradingPolicy(
            letter_bands=((95.0, "A"), (85.0, "B"), (0.0, "F")), pass_threshold=85.0
        )
        courses = GradeAssembler(strict).assemble([subject(percentage=90.0)], [binding()])

        assert courses[0].letter_grade == "B"
        assert courses[0].passed is True

    def test_category_subtotals_are_carried_through(self):
        courses = GradeAssembler().assemble(
            [subject(categories=({"name": "Assessments", "percentage": 88.0},))],
            [binding()],
        )

        assert courses[0].categories[0].name == "Assessments"
        assert courses[0].categories[0].percentage == 88.0

    def test_counts_survive(self):
        courses = GradeAssembler().assemble(
            [subject(graded_count=3, excluded_count=1, pending_count=2, is_complete=False)],
            [binding()],
        )

        assert courses[0].graded_count == 3
        assert courses[0].excused_count == 1
        assert courses[0].pending_count == 2
        assert courses[0].is_complete is False


class TestAttendanceAssembly:
    def attendance(self, course_ref="2026-T1-G7A-MATH", **kwargs) -> SubjectAttendance:
        return SubjectAttendance(
            course_ref=course_ref,
            subject_name=kwargs.pop("subject_name", "Mathematics"),
            percentage=kwargs.pop("percentage", 100.0),
            **kwargs,
        )

    def test_an_unbound_subject_is_not_visible(self):
        assembler = AttendanceAssembler([binding()])

        assert assembler.visible([self.attendance(course_ref="2026-T1-G7A-SANDBOX")]) == []

    def test_the_term_figure_sums_points_rather_than_averaging(self):
        """The reason this is not a one-line map.

        Maths: 2 of 40 points across a full term. PE: 2 of 2, one perfect register.
        Averaging the percentages gives (5 + 100) / 2 = 52.5% and hides a term of
        absence behind a single elective. Summing gives 4/42 = 9.52%, which is what a
        parent needs to hear.
        """
        assembler = AttendanceAssembler([binding(), binding("2026-T1-G7A-PE", 9002)])
        subjects = [
            self.attendance(points=2.0, max_points=40.0),
            self.attendance(course_ref="2026-T1-G7A-PE", points=2.0, max_points=2.0),
        ]

        assert assembler.term_percentage(subjects) == 9.52

    def test_nothing_marked_anywhere_is_null_not_zero(self):
        assembler = AttendanceAssembler([binding()])

        assert assembler.term_percentage([self.attendance(points=0.0, max_points=0.0)]) is None

    def test_a_genuine_zero_is_zero(self):
        assembler = AttendanceAssembler([binding()])

        assert assembler.term_percentage([self.attendance(points=0.0, max_points=10.0)]) == 0.0

    def test_status_counts_are_summed_across_subjects(self):
        assembler = AttendanceAssembler([binding(), binding("2026-T1-G7A-PE", 9002)])
        subjects = [
            self.attendance(by_status=({"acronym": "P", "description": "Present", "count": 8},)),
            self.attendance(
                course_ref="2026-T1-G7A-PE",
                by_status=({"acronym": "P", "description": "Present", "count": 3},),
            ),
        ]

        assert assembler.counts(subjects)["present"] == 11

    def test_counts_match_on_description_not_position(self):
        """A school may rename or reorder its statuses; the meaning is in the label."""
        assembler = AttendanceAssembler([binding()])
        subjects = [
            self.attendance(by_status=(
                {"acronym": "X", "description": "Absent without notice", "count": 2},
                {"acronym": "Y", "description": "Excused by parent", "count": 1},
            ))
        ]

        counts = assembler.counts(subjects)
        assert counts["absent"] == 2
        assert counts["excused"] == 1

    def test_unrecognised_labels_still_reach_the_caller_verbatim(self):
        """A school running Arabic descriptions is not silently reported as zero."""
        assembler = AttendanceAssembler([binding()])
        subjects = [self.attendance(by_status=({"description": "حاضر", "count": 5},))]

        assert assembler.status_totals(subjects)["حاضر"] == 5

    def test_day_detail_is_empty_rather_than_invented(self):
        assembler = AttendanceAssembler([binding()])

        assert assembler.recent_days([self.attendance()]) == []
