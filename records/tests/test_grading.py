"""Term rollup arithmetic.

The excused-vs-zero tests are the ones that protect a real child's real grade. They
are written as comparisons rather than as fixed expected numbers, so they still mean
something if the school's policy changes.
"""
from records.grading import GradingPolicy, compute_rollup
from records.schemas import AssignmentGrade, SubmissionStatus


def _graded(assignment_id: str, percentage: float, category: str = "homework") -> AssignmentGrade:
    return AssignmentGrade(
        assignment_id=assignment_id,
        status=SubmissionStatus.GRADED,
        score=percentage,
        max_score=100.0,
        percentage=percentage,
        category=category,
    )


def _excused(assignment_id: str, category: str = "homework") -> AssignmentGrade:
    return AssignmentGrade(
        assignment_id=assignment_id, status=SubmissionStatus.EXCUSED, category=category
    )


def _missing(assignment_id: str, category: str = "homework") -> AssignmentGrade:
    return AssignmentGrade(
        assignment_id=assignment_id,
        status=SubmissionStatus.MISSING,
        max_score=100.0,
        category=category,
    )


def test_excused_work_leaves_the_denominator():
    """90% with an excused assignment is still 90%, not 45%."""
    rollup = compute_rollup([_graded("a1", 90.0), _excused("a2")])
    assert rollup.percentage == 90.0
    assert rollup.excused_count == 1


def test_excused_and_missing_are_not_the_same():
    """The single most consequential distinction in this file."""
    excused = compute_rollup([_graded("a1", 90.0), _excused("a2")])
    missing = compute_rollup([_graded("a1", 90.0), _missing("a2")])
    assert excused.percentage > missing.percentage


def test_missing_work_counts_as_zero():
    """One perfect assignment and one never handed in is 50%, not 100%."""
    rollup = compute_rollup([_graded("a1", 100.0), _missing("a2")])
    assert rollup.percentage == 50.0
    assert rollup.missing_count == 1


def test_all_excused_yields_no_grade_not_zero():
    """"No grade yet" is not a grade of zero, and must not be reported as one."""
    rollup = compute_rollup([_excused("a1"), _excused("a2")])
    assert rollup.percentage is None
    assert rollup.letter == ""
    assert rollup.passed is None


def test_empty_term_yields_no_grade():
    rollup = compute_rollup([])
    assert rollup.percentage is None
    assert rollup.is_complete is False


def test_ungraded_submission_does_not_move_the_figure():
    """Submitted-but-not-marked is undecided; forcing it either way is a lie."""
    without = compute_rollup([_graded("a1", 80.0)])
    with_pending = compute_rollup(
        [
            _graded("a1", 80.0),
            AssignmentGrade(assignment_id="a2", status=SubmissionStatus.SUBMITTED_UNGRADED),
        ]
    )
    assert without.percentage == with_pending.percentage
    assert with_pending.pending_count == 1
    # But the figure is explicitly not final, which is what lets the agent say "so far".
    assert with_pending.is_complete is False
    assert without.is_complete is True


def test_weighted_categories_renormalise_over_what_exists():
    """Mid-term, an unsat final must not be counted as 40% of zero."""
    policy = GradingPolicy(category_weights={"homework": 0.3, "midterm": 0.3, "final": 0.4})
    rollup = compute_rollup(
        [_graded("a1", 80.0, "homework"), _graded("a2", 90.0, "midterm")], policy
    )
    # 0.3*80 + 0.3*90 renormalised over the 0.6 of weight actually present = 85.
    assert rollup.percentage == 85.0


def test_letter_bands_and_pass_threshold_apply():
    rollup = compute_rollup([_graded("a1", 85.0)])
    assert rollup.letter == "B"
    assert rollup.passed is True

    failing = compute_rollup([_graded("a1", 41.0)])
    assert failing.letter == "F"
    assert failing.passed is False
