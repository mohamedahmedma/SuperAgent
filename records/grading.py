"""Term rollup — the arithmetic Moodle cannot do for us.

Moodle aggregates within a course. It has no term entity, so "the student's grade in
Mathematics this term" has no Moodle answer and must be computed here.

Two rules carry all the risk, and both are about the denominator:

**Excused work leaves the denominator.** It is not a zero. A child with three excused
assignments and 90% on everything else has 90%, not 60%. This is the single most
common way a home-built gradebook quietly harms a real student, and it is why
`SubmissionStatus.EXCUSED` is a distinct value rather than an absent score.

**Missing work stays in it, as a zero.** An unsubmitted assignment past its due date
is a real zero and must drag the average down, otherwise a child who submits one
perfect assignment and nothing else reads as a straight-A student.

Everything else — weights, letter boundaries, the pass mark — is school policy and
lives in `GradingPolicy` so it can be changed without touching this arithmetic.
"""
from dataclasses import dataclass, field

from records.schemas import AssignmentGrade, SubmissionStatus

# Assignments that count toward the figure at all, and how.
_COUNTS_WITH_SCORE = {SubmissionStatus.GRADED}
_COUNTS_AS_ZERO = {SubmissionStatus.MISSING}
# EXCUSED leaves the denominator; SUBMITTED_UNGRADED and NOT_DUE are simply not
# decided yet and would bias the figure in whichever direction they were forced.
_EXCLUDED = {
    SubmissionStatus.EXCUSED,
    SubmissionStatus.SUBMITTED_UNGRADED,
    SubmissionStatus.NOT_DUE,
}


@dataclass
class GradingPolicy:
    """School policy, not arithmetic. Safe to edit; safe to differ per school.

    `category_weights` maps a category name to its share of the final figure, e.g.
    `{"homework": 0.3, "midterm": 0.3, "final": 0.4}`. Left empty, every graded
    assignment is weighted by its own `max_score` — the "natural" aggregation, and a
    reasonable default before a school has told you its scheme.
    """

    category_weights: dict[str, float] = field(default_factory=dict)
    # Descending boundaries. First threshold the percentage meets or exceeds wins.
    letter_bands: list[tuple[float, str]] = field(
        default_factory=lambda: [(90.0, "A"), (80.0, "B"), (70.0, "C"), (60.0, "D"), (0.0, "F")]
    )
    pass_threshold: float = 60.0


@dataclass
class Rollup:
    percentage: float | None
    letter: str
    passed: bool | None
    graded_count: int
    excused_count: int
    missing_count: int
    pending_count: int
    is_complete: bool


def _weighted_by_category(
    assignments: list[AssignmentGrade], weights: dict[str, float]
) -> float | None:
    """Weighted mean across categories, renormalised over the categories present.

    Renormalisation matters mid-term: if the final exam has not happened, its 40%
    should not be counted as 40% of zero. Dropping absent categories and rescaling
    the rest is what makes an in-progress figure mean "how she is doing so far"
    rather than "how she would do if she skipped the final".
    """
    totals: dict[str, list[float]] = {}
    for item in assignments:
        if item.status in _EXCLUDED:
            continue
        bucket = totals.setdefault(item.category or "", [])
        if item.status in _COUNTS_AS_ZERO:
            bucket.append(0.0)
        elif item.percentage is not None:
            bucket.append(item.percentage)

    present = {name: scores for name, scores in totals.items() if scores and weights.get(name)}
    if not present:
        return None

    weight_sum = sum(weights[name] for name in present)
    if weight_sum <= 0:
        return None

    return sum(
        (sum(scores) / len(scores)) * weights[name] for name, scores in present.items()
    ) / weight_sum


def _natural(assignments: list[AssignmentGrade]) -> float | None:
    """Points earned over points possible, excused work excluded from both sides."""
    earned = 0.0
    possible = 0.0
    for item in assignments:
        if item.status in _EXCLUDED:
            continue
        if item.status in _COUNTS_AS_ZERO:
            # A real zero: it contributes nothing to the numerator and its full
            # weight to the denominator.
            possible += item.max_score or 0.0
            continue
        if item.score is None or item.max_score is None:
            continue
        earned += item.score
        possible += item.max_score

    if possible <= 0:
        return None
    return round((earned / possible) * 100.0, 2)


def compute_rollup(
    assignments: list[AssignmentGrade], policy: GradingPolicy | None = None
) -> Rollup:
    """Roll a term's assignments up into one subject figure.

    Returns `percentage=None` rather than `0.0` when nothing is gradeable yet. Zero
    is a grade a child can earn; "no grade yet" is not, and collapsing them tells a
    parent their child scored nothing when in truth the term has not started.
    """
    policy = policy or GradingPolicy()

    graded = sum(1 for a in assignments if a.status == SubmissionStatus.GRADED)
    excused = sum(1 for a in assignments if a.status == SubmissionStatus.EXCUSED)
    missing = sum(1 for a in assignments if a.status == SubmissionStatus.MISSING)
    pending = sum(
        1
        for a in assignments
        if a.status in (SubmissionStatus.SUBMITTED_UNGRADED, SubmissionStatus.NOT_DUE)
    )

    if policy.category_weights:
        percentage = _weighted_by_category(assignments, policy.category_weights)
        if percentage is not None:
            percentage = round(percentage, 2)
    else:
        percentage = _natural(assignments)

    letter = ""
    passed: bool | None = None
    if percentage is not None:
        for threshold, band in policy.letter_bands:
            if percentage >= threshold:
                letter = band
                break
        passed = percentage >= policy.pass_threshold

    return Rollup(
        percentage=percentage,
        letter=letter,
        passed=passed,
        graded_count=graded,
        excused_count=excused,
        missing_count=missing,
        pending_count=pending,
        # "Complete" means nothing is still awaiting a decision — the figure will not
        # move on its own. It is what lets the agent say "final" instead of "so far".
        is_complete=pending == 0 and len(assignments) > 0,
    )
