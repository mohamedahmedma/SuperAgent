"""School grading POLICY. Not arithmetic.

This module used to aggregate assignments into a subject percentage. It no longer
does, and the deletion is the point: the system of record already computed that figure,
applying category weights, drop-lowest and exclusions, and any second implementation of
the same sum is a second answer waiting to contradict the first. Measured on a live
Moodle, the two plausible re-derivations gave 50% and 30% for a child genuinely on 90%.

What remains is the half the LMS genuinely does not own:

    the gradebook says WHAT the number is        -> Moodle, via local_schoolapi
    the school says WHAT THE NUMBER MEANS        -> here

Letter boundaries and the pass mark are school policy. Two schools reading the same
82% will disagree about whether it is a B or a B+, and neither is wrong — so this is
configuration, not calculation, and it belongs on our side of the seam.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Which of the two figures a school wants led with.
#:
#: BOTH are always returned; this only says which one answers "how is she doing in
#: maths" when the assistant has to pick one to say first. A school that grades
#: attendance and considers that part of the subject sets OFFICIAL; a school that
#: treats attendance as pastoral rather than academic sets ACADEMIC.
PRIMARY_ACADEMIC = "academic"
PRIMARY_OFFICIAL = "official"
_VALID_PRIMARY = (PRIMARY_ACADEMIC, PRIMARY_OFFICIAL)


def _primary_from_env() -> str:
    """Read the preference, falling back loudly rather than silently.

    A typo here would otherwise change which number a parent is told, quietly, which is
    the worst possible way for a configuration mistake to present.
    """
    raw = (os.getenv("RECORDS_PRIMARY_GRADE") or PRIMARY_ACADEMIC).strip().lower()
    if raw in _VALID_PRIMARY:
        return raw

    logger.warning(
        "RECORDS_PRIMARY_GRADE=%r is not one of %s — falling back to %r",
        raw, _VALID_PRIMARY, PRIMARY_ACADEMIC,
    )
    return PRIMARY_ACADEMIC


@dataclass(frozen=True)
class GradingPolicy:
    """How a school reads a percentage.

    Immutable: a policy that could be mutated after a report card was rendered would
    make the same stored figure mean different things on different days.

    Note what is NOT here any more. `category_weights` is gone, because weighting is
    aggregation and aggregation belongs to the gradebook. A school that wants
    "midterm 30%, final 40%" configures that in Moodle, where the teacher can see it
    and where it applies to the figure the school itself publishes — rather than here,
    where it would silently disagree with the gradebook.
    """

    #: Descending boundaries; the first threshold the percentage meets or exceeds wins.
    letter_bands: tuple[tuple[float, str], ...] = (
        (90.0, "A"), (80.0, "B"), (70.0, "C"), (60.0, "D"), (0.0, "F"),
    )
    pass_threshold: float = 60.0

    #: Which figure to lead with. Never which figure to RETURN — both always are.
    #:
    #: This exists because the right answer is a school decision that can change, and
    #: changing it should be a setting rather than a redeploy of the assistant's
    #: prompts. A school grading attendance as part of the subject leads with the
    #: official total; one treating it as pastoral leads with the academic figure.
    #:
    #: Defaults to academic because that is what a parent asking "how is she doing in
    #: maths" almost always means — but the school gets to disagree.
    primary_figure: str = PRIMARY_ACADEMIC

    def classify(self, percentage: float | None) -> tuple[str, bool | None]:
        """Turn a percentage into `(letter, passed)`.

        `None` in, `("", None)` out — never `("F", False)`. A child with nothing graded
        yet has not failed, and a report that says otherwise is worse than one that says
        nothing. This is the same null-is-not-zero rule the whole system runs on, at the
        one point where it would be easiest to lose.
        """
        if percentage is None:
            return "", None

        for threshold, band in self.letter_bands:
            if percentage >= threshold:
                return band, percentage >= self.pass_threshold

        # Only reachable if a school configures bands that do not reach zero.
        return "", percentage >= self.pass_threshold


#: The default a deployment gets when the school has not said otherwise. Named rather
#: than constructed inline so the fallback is greppable — "which policy produced this
#: letter?" should have one answer.
#:
#: Only `primary_figure` is environment-driven today. Letter bands and the pass mark
#: are deliberately not: they belong per-school in the database alongside terms, and an
#: env var holding a band table would be unreadable and unversioned. This is the
#: smallest configuration surface that answers the question actually being asked.
DEFAULT_POLICY = GradingPolicy(primary_figure=_primary_from_env())
