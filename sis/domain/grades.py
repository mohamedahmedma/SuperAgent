"""One mark, as a teacher stated it, for one child in one subject in one term.

This is the whole of decision 5: a grade is a *stated figure*. Nothing in this module
averages, weights, drops a lowest score or rolls a term up into a year. Those are school
policies that differ between schools and change between years, and a policy baked into
the stored record is a policy that cannot be corrected afterwards without rewriting
history. What is stored is what somebody wrote down.

The type carries no database identity and no timestamps. It is the value the services
are unit-tested against with fake repositories, so nothing here imports sqlalchemy,
fastapi or pydantic, and it must stay that way.
"""
import math
from dataclasses import dataclass

from sis.domain.errors import InvalidPercentage, ValidationError
from sis.domain.value_objects import (
    ClassCode,
    Percentage,
    StudentNumber,
    SubjectCode,
    TermCode,
)


@dataclass(frozen=True, slots=True)
class SubjectGrade:
    """A mark, or the recorded absence of one, attached to the class it was earned in.

    **`percentage=None` means NOT GRADED YET. It is not a low mark and it is not
    `Percentage(0)`.** These are categorically different facts about a child:

    * `Percentage(0)` -- a teacher looked at the work and awarded zero. The child sat
      the paper, or did not hand it in, and the school has decided that is worth
      nothing. It is a mark she earned.
    * `None` -- nobody has entered anything. The exam may be next week; the sheet may
      be on a desk. The school does not know what this child scored.

    Reporting the second as the first tells a parent something false about their child:
    that she scored zero in maths, when the truth is that maths has not been marked. A
    parent who is told that acts on it -- a conversation with the child, a call to the
    school, a tutor -- and the school's own record says the parent is wrong. That damage
    is not undone by later filling the mark in.

    The failure never arrives as a decision; it arrives as convenience. `sum(g.value for
    ...)`, `COALESCE(percentage, 0)`, a serialiser emitting `0` for a null, a chart
    library that will not plot a gap. Every one of those is a line somebody wrote to
    make a total add up. So the substitution is refused *here*, in the type, rather than
    trusted to each of them:

    * `percentage` is a `Percentage` or `None` and never a bare number -- a raw `0.0`
      arriving from a parser is rejected, not adopted, because that is exactly the value
      a `.get(col, 0)` produces for a missing column.
    * `is_graded` is the only supported way to ask, so no call site re-derives the
      distinction as a truth test and gets it wrong. `Percentage(0)` is a perfectly
      ordinary object and is truthy; `if grade.percentage:` is the bug this property
      exists to make unnecessary.
    * There is no default, no `or 0`, no `__float__`. A caller that genuinely wants a
      number for an ungraded row must say so by name, through `value_or`, and that call
      is greppable.

    `class_section_id` is carried on the grade rather than looked up from the student
    because of decision 2: placement is a time-bounded membership, so "which class is
    she in" has different answers on different days. A mark earned in 3A in Term 1 must
    still print under 3A after a March transfer to 3B, and it will not if the class is
    resolved at report time from the child's *current* row. `class_code` travels beside
    the id for the same reason a report needs a label a human recognises; the id is the
    identity, the code is what the registrar reads.
    """

    student_number: StudentNumber | str
    subject_code: SubjectCode | str
    term_code: TermCode | str
    class_section_id: int
    class_code: ClassCode | str
    percentage: Percentage | None = None
    # The raw figures behind the percentage, when the teacher stated them ("17 out of
    # 20"). Kept alongside rather than instead of the percentage so a correction can be
    # audited against what was actually written; never re-derived from each other.
    points: float | None = None
    max_points: float | None = None

    def __post_init__(self) -> None:
        # Codes only. `percentage` is pointedly *not* coerced: a bare number arriving
        # there is the "blank became 0.0" bug, and coercing it would adopt the very
        # value the check below exists to refuse.
        for name, kind in (
            ("student_number", StudentNumber),
            ("subject_code", SubjectCode),
            ("term_code", TermCode),
            ("class_code", ClassCode),
        ):
            raw = getattr(self, name)
            if not isinstance(raw, kind):
                object.__setattr__(self, name, kind(raw))
        if self.percentage is not None and not isinstance(self.percentage, Percentage):
            # A float here is the "blank became 0.0" bug arriving in person. Refuse it
            # at construction: by the time it is a stored row it is indistinguishable
            # from a mark somebody awarded.
            raise InvalidPercentage(
                "percentage must be a Percentage or None for not-graded-yet; a bare "
                f"number is not accepted (got {type(self.percentage).__name__})",
                field="percentage",
            )
        object.__setattr__(self, "points", _checked(self.points, "points"))
        object.__setattr__(self, "max_points", _checked(self.max_points, "max_points"))
        if self.max_points is not None and self.max_points <= 0:
            raise ValidationError(
                "max_points must be greater than zero; a mark out of nothing has no "
                "meaning and divides by zero downstream",
                field="max_points",
            )
        if self.points is not None and self.max_points is None:
            raise ValidationError(
                "points were stated without max_points; '17' is not a grade until the "
                "sheet says what it is out of",
                field="max_points",
            )
        if self.points is not None and self.points > self.max_points:  # type: ignore[operator]
            raise ValidationError(
                f"points ({self.points:g}) exceed max_points ({self.max_points:g})",
                field="points",
            )

    @property
    def is_graded(self) -> bool:
        """True only when a figure was stated. The one supported way to ask.

        Exists so that "has this been marked" is never re-derived at a call site as
        `if grade.percentage:` or `if grade.percentage.value:` -- the first is right by
        accident today and the second reads an earned zero as unmarked.
        """
        return self.percentage is not None

    @property
    def is_awaiting_grade(self) -> bool:
        """True when nobody has entered a mark. Reads plainly in report and UI code."""
        return self.percentage is None

    @property
    def is_zero(self) -> bool:
        """True only for an earned zero -- never for an ungraded row.

        The property that makes the distinction expressible: `is_zero` and
        `is_awaiting_grade` are never both true, and a screen that shows "0%" should be
        driven by this one.
        """
        return self.percentage is not None and self.percentage.value == 0.0

    @property
    def identity(self) -> tuple[str, str, str]:
        """The natural key of a mark: one figure per student, subject and term.

        Used by the import path to detect that a row restates a grade already on file
        rather than adding a second one, which is how a child ends up with two maths
        marks for Term 1 and every report picks whichever the query returned first.
        """
        return (
            self.student_number.value,
            self.subject_code.value,
            self.term_code.value,
        )

    def value_or(self, default: float) -> float:
        """The stated percentage, or `default` when there is none. Named on purpose.

        The only substitution path, so that `value_or(0.0)` is a visible, greppable
        decision made by one caller for one screen -- not a silent default buried in a
        serialiser where nobody can find it later.
        """
        return default if self.percentage is None else self.percentage.value

    def regraded(self, percentage: Percentage | None) -> "SubjectGrade":
        """A copy carrying a new mark, keeping the class the original was earned in.

        Accepts `None` deliberately: clearing a wrongly entered mark must return the row
        to "not graded yet", and the only alternative a registrar has otherwise is to
        type a zero.
        """
        return SubjectGrade(
            student_number=self.student_number,
            subject_code=self.subject_code,
            term_code=self.term_code,
            class_section_id=self.class_section_id,
            class_code=self.class_code,
            percentage=percentage,
            points=self.points if percentage is not None else None,
            max_points=self.max_points,
        )

    @classmethod
    def from_points(
        cls,
        *,
        student_number: StudentNumber | str,
        subject_code: SubjectCode | str,
        term_code: TermCode | str,
        class_section_id: int,
        class_code: ClassCode | str,
        points: float | None,
        max_points: float,
    ) -> "SubjectGrade":
        """Build one from "17 out of 20", restating that single figure as a percentage.

        Not aggregation, and not a policy: one stated mark expressed on the 0-100 scale
        the rest of the service compares on. It is a named constructor rather than a
        computed property so the conversion happens once, at the boundary, and cannot
        drift between a report and a screen.

        `points=None` yields an ungraded row that still remembers what it will be out
        of -- a sheet handed out but not yet marked -- and pointedly not a zero.
        """
        # Checked here rather than left to `__post_init__`, because the division below
        # happens *before* the constructor runs: an "out of 0" column would leave the
        # import with a raw ZeroDivisionError, which is not a `SisError` and so kills
        # the whole batch instead of rejecting the one bad row (decision 4).
        checked_max = _checked(max_points, "max_points")
        if checked_max is None or checked_max <= 0:
            raise ValidationError(
                "max_points must be greater than zero; a mark out of nothing has no "
                "meaning and divides by zero downstream",
                field="max_points",
            )
        return cls(
            student_number=student_number,
            subject_code=subject_code,
            term_code=term_code,
            class_section_id=class_section_id,
            class_code=class_code,
            percentage=(
                None if points is None else Percentage(points / checked_max * 100)
            ),
            points=points,
            max_points=checked_max,
        )

    def __str__(self) -> str:
        mark = "not graded" if self.percentage is None else f"{self.percentage}%"
        return f"{self.student_number} {self.subject_code} {self.term_code}: {mark}"


def _checked(raw: float | None, field: str) -> float | None:
    """Reject the non-numbers a spreadsheet produces, before they become a stored mark.

    NaN is checked explicitly because it compares false to everything, including itself:
    a NaN that reaches the database passes both `points > max` and `points <= max`, so
    every later validation silently answers "fine".
    """
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValidationError(
            f"{field} must be a number (got {type(raw).__name__})", field=field
        )
    number = float(raw)
    if not math.isfinite(number):
        raise ValidationError(
            f"{field} is not a finite number; an empty divisor produces this",
            field=field,
        )
    if number < 0:
        raise ValidationError(f"{field} may not be negative", field=field)
    return number
