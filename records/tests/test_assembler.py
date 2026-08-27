"""The rules applied between what the system of record reports and what a parent is told.

No database, no network — the assembler is pure, which is why these can be exhaustive.

The property that carries most of the weight: **a percentage the system of record could
not compute must never become a letter grade.** A letter with a null percentage beside it
is exactly the shape a consumer renders as a grade.

The other property this file used to assert — that an unpublished course never reaches a
parent — was about matching reported subjects against a table of published bindings. That
table is gone with the flat-course backend it existed for; the system of record stores
marks against the school's own subject codes, so a subject it reports for a child in a term
is one the school entered against her, and there is nothing to match or drop.
"""
import pytest

from records.assembler import AttendanceAssembler, GradeAssembler
from records.grading import GradingPolicy
from records.lms import SubjectAttendance, SubjectGrade


def subject(course_ref: str = "2026-T1-G7A-MATH", **kwargs) -> SubjectGrade:
    return SubjectGrade(
        course_ref=course_ref,
        subject_name=kwargs.pop("subject_name", "Mathematics"),
        percentage=kwargs.pop("percentage", 80.0),
        **kwargs,
    )


class TestGradeAssembly:
    def test_a_published_subject_is_returned(self):
        courses = GradeAssembler().assemble([subject()])

        assert len(courses) == 1
        assert courses[0].computed_percentage == 80.0

    def test_both_percentages_are_carried_through(self):
        """The case the two figures exist for: attendance drags the official total down."""
        courses = GradeAssembler().assemble(
            [subject(percentage=65.0, academic_percentage=80.0)]
        )

        assert courses[0].computed_percentage == 65.0
        assert courses[0].academic.percentage == 80.0

    def test_each_percentage_gets_its_own_letter(self):
        courses = GradeAssembler().assemble(
            [subject(percentage=65.0, academic_percentage=80.0)]
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
            [subject(academic_percentage=None, academic_unavailable="aggregation_not_summable")]
        )

        assert courses[0].academic.percentage is None
        assert courses[0].academic.letter_grade == ""
        assert courses[0].academic.passed is None
        assert courses[0].academic.unavailable == "aggregation_not_summable"

    def test_nothing_graded_is_not_a_fail(self):
        courses = GradeAssembler().assemble([subject(percentage=None)])

        assert courses[0].computed_percentage is None
        assert courses[0].letter_grade == ""
        assert courses[0].passed is None

    def test_a_genuine_zero_is_a_fail(self):
        """The counterpart: zero is a mark a child can earn."""
        courses = GradeAssembler().assemble([subject(percentage=0.0)])

        assert courses[0].computed_percentage == 0.0
        assert courses[0].letter_grade == "F"
        assert courses[0].passed is False

    def test_the_school_policy_is_injected_not_hardcoded(self):
        strict = GradingPolicy(
            letter_bands=((95.0, "A"), (85.0, "B"), (0.0, "F")), pass_threshold=85.0
        )
        courses = GradeAssembler(strict).assemble([subject(percentage=90.0)])

        assert courses[0].letter_grade == "B"
        assert courses[0].passed is True

    def test_category_subtotals_are_carried_through(self):
        courses = GradeAssembler().assemble(
            [subject(categories=({"name": "Assessments", "percentage": 88.0},))]
        )

        assert courses[0].categories[0].name == "Assessments"
        assert courses[0].categories[0].percentage == 88.0

    def test_counts_survive(self):
        courses = GradeAssembler().assemble(
            [subject(graded_count=3, excluded_count=1, pending_count=2, is_complete=False)]
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

    def test_the_term_figure_sums_points_rather_than_averaging(self):
        """The reason this is not a one-line map.

        Maths: 2 of 40 points across a full term. PE: 2 of 2, one perfect register.
        Averaging the percentages gives (5 + 100) / 2 = 52.5% and hides a term of
        absence behind a single elective. Summing gives 4/42 = 9.52%, which is what a
        parent needs to hear.
        """
        assembler = AttendanceAssembler()
        subjects = [
            self.attendance(points=2.0, max_points=40.0),
            self.attendance(course_ref="2026-T1-G7A-PE", points=2.0, max_points=2.0),
        ]

        assert assembler.term_percentage(subjects) == 9.52

    def test_nothing_marked_anywhere_is_null_not_zero(self):
        assembler = AttendanceAssembler()

        assert assembler.term_percentage([self.attendance(points=0.0, max_points=0.0)]) is None

    def test_a_genuine_zero_is_zero(self):
        assembler = AttendanceAssembler()

        assert assembler.term_percentage([self.attendance(points=0.0, max_points=10.0)]) == 0.0

    def test_status_counts_are_summed_across_subjects(self):
        assembler = AttendanceAssembler()
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
        assembler = AttendanceAssembler()
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
        assembler = AttendanceAssembler()
        subjects = [self.attendance(by_status=({"description": "حاضر", "count": 5},))]

        assert assembler.status_totals(subjects)["حاضر"] == 5

    def test_day_detail_is_empty_rather_than_invented(self):
        assembler = AttendanceAssembler()

        assert assembler.recent_days([self.attendance()]) == []
