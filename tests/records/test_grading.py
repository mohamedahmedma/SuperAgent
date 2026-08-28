"""School grading policy — classification, not calculation.

This file used to test aggregation: turning a list of assignments into a subject
percentage. That code is gone, and so are those tests. The system of record computes
the figure now, applying weights, drop-lowest and exclusions, and the equivalent
assertions belong where the arithmetic does — inside the gradebook, not here.

What is left is the half a school owns rather than a gradebook: what a number MEANS.
Two schools reading 82% will disagree about whether it is a B or a B+, and neither is
wrong.
"""
from records.grading import (
    DEFAULT_POLICY,
    PRIMARY_ACADEMIC,
    PRIMARY_OFFICIAL,
    GradingPolicy,
    _primary_from_env,
)


class TestClassification:
    def test_a_percentage_maps_to_its_band(self):
        assert DEFAULT_POLICY.classify(95.0)[0] == "A"
        assert DEFAULT_POLICY.classify(85.0)[0] == "B"
        assert DEFAULT_POLICY.classify(75.0)[0] == "C"
        assert DEFAULT_POLICY.classify(65.0)[0] == "D"
        assert DEFAULT_POLICY.classify(10.0)[0] == "F"

    def test_a_boundary_belongs_to_the_higher_band(self):
        """Exactly 90 is an A. A child on the line is not rounded down."""
        assert DEFAULT_POLICY.classify(90.0)[0] == "A"
        assert DEFAULT_POLICY.classify(89.99)[0] == "B"

    def test_the_pass_mark_is_separate_from_the_letter(self):
        _, passed = DEFAULT_POLICY.classify(60.0)
        assert passed is True

        _, failed = DEFAULT_POLICY.classify(59.99)
        assert failed is False

    def test_no_grade_is_not_a_fail(self):
        """The rule the whole system runs on, at the point it is easiest to lose.

        A child with nothing graded yet has not failed. Returning ("F", False) here
        would put a fail on a report for a term that has not been marked.
        """
        letter, passed = DEFAULT_POLICY.classify(None)

        assert letter == ""
        assert passed is None

    def test_a_genuine_zero_is_a_fail(self):
        """The counterpart. Zero is a mark a child can earn, and it is not None."""
        letter, passed = DEFAULT_POLICY.classify(0.0)

        assert letter == "F"
        assert passed is False

    def test_a_school_can_set_its_own_bands(self):
        strict = GradingPolicy(
            letter_bands=((95.0, "Excellent"), (85.0, "Good"), (0.0, "Needs work")),
            pass_threshold=85.0,
        )

        assert strict.classify(90.0) == ("Good", True)
        assert strict.classify(80.0) == ("Needs work", False)

    def test_a_policy_cannot_be_mutated_after_the_fact(self):
        """A policy that could change would make a stored figure mean different things
        on different days."""
        import dataclasses

        try:
            DEFAULT_POLICY.pass_threshold = 50.0
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("GradingPolicy must be frozen")

    def test_bands_that_do_not_reach_zero_still_return_a_pass_verdict(self):
        """A misconfigured policy must not crash a parent's request.

        No band matches, so there is no letter — but whether the child passed is still
        answerable, and answering it is more useful than raising.
        """
        gappy = GradingPolicy(letter_bands=((50.0, "P"),), pass_threshold=50.0)

        assert gappy.classify(10.0) == ("", False)


class TestPrimaryFigure:
    """Which of the two figures a school leads with.

    Both are always returned; this only decides which the assistant says first. It is
    configuration because the right answer is a school decision that can change — and
    changing it should not mean redeploying the assistant's prompts.
    """

    def test_it_defaults_to_academic(self, monkeypatch):
        monkeypatch.delenv("RECORDS_PRIMARY_GRADE", raising=False)

        assert _primary_from_env() == PRIMARY_ACADEMIC

    def test_a_school_can_choose_the_official_total(self, monkeypatch):
        monkeypatch.setenv("RECORDS_PRIMARY_GRADE", "official")

        assert _primary_from_env() == PRIMARY_OFFICIAL

    def test_it_is_case_and_whitespace_tolerant(self, monkeypatch):
        monkeypatch.setenv("RECORDS_PRIMARY_GRADE", "  OFFICIAL  ")

        assert _primary_from_env() == PRIMARY_OFFICIAL

    def test_a_typo_falls_back_loudly_rather_than_silently(self, monkeypatch, caplog):
        """A typo must not quietly change which number a parent is told."""
        monkeypatch.setenv("RECORDS_PRIMARY_GRADE", "acadmic")

        with caplog.at_level("WARNING"):
            resolved = _primary_from_env()

        assert resolved == PRIMARY_ACADEMIC
        assert "acadmic" in caplog.text

    def test_the_policy_carries_it(self):
        assert GradingPolicy(primary_figure=PRIMARY_OFFICIAL).primary_figure == "official"
