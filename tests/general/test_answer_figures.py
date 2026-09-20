# -*- coding: utf-8 -*-
"""An answer may not state an amount its evidence does not contain.

The incident this exists for: asked for Year 3 fees, the assistant answered 88,000 EGP —
the FS1-FS2 row. The grader had approved the evidence and the evidence was right; the
answer read the wrong line of it. Nothing in the system looks at that, and no prompt can
be trusted to have settled it.

## Why this is allowed to exist when its predecessor was retired

A numeric grounding check was added (404139d) and removed again (f40f8c2), and the reason
it was removed is the reason this one is shaped as it is. The comparison was never the
problem; the EXTRACTOR was:

  * "45 حصة" passed as verified because 45 appears inside "07:45" — the check approving
    a figure it had not verified, which is the quieter failure and the worse one;
  * a correct timetable was withdrawn because "10:00" read as 10,000, the multiplier
    lookahead having truncated «الفيزياء» to «الف», the Arabic for "thousand".

Both live below three digits, and neither survives a whole-token comparison. So this
ignores one- and two-digit figures outright — times, dates, ordinals, small counts and
percentages all leave the check in one stroke — compares whole tokens rather than
substrings, and reads no figure's meaning at all. It is deliberately narrow: a check that
fires on a correct answer teaches a deployment to switch it off.

The other half is that records no longer have model-written figures to check. The tool
renders the grid and the model writes the sentence beside it, so the prose worth checking
is what a turn wrote from RETRIEVED CHUNKS — which is exactly where the fee incident was.
"""
import unittest
from unittest.mock import patch

from backend.chat.answer_checks import enforce_answer_figures, ungrounded_figures

FEE_TABLE = (
    "Grade | Egyptian (EGP) | International (EGP)\n"
    "Pre-K | 75,000 EGP | 85,000 EGP\n"
    "FS1-FS2 | 88,000 EGP | 98,000 EGP\n"
    "Y03-Y05 | 105,000 EGP | 115,000 EGP"
)


class TheIncidentThisExistsForTests(unittest.TestCase):
    def test_a_fee_from_the_wrong_row_is_caught(self):
        """Verbatim from the deployment: Year 3 answered with the FS1-FS2 figure. The
        figure IS in the corpus, which is why no other check sees it — but this asks a
        narrower question than "is it in the corpus", and 88,000 is not the Year 3 row."""
        answer = "مصاريف Year 3 هي 88,000 جنيه في السنة."
        evidence = "Y03-Y05 | 105,000 EGP | 115,000 EGP"
        self.assertEqual(["88000"], ungrounded_figures(answer, evidence))

    def test_the_right_fee_passes(self):
        answer = "مصاريف Year 3 هي 105,000 جنيه في السنة."
        self.assertEqual([], ungrounded_figures(answer, FEE_TABLE))

    def test_a_figure_written_in_arabic_digits_is_the_same_figure(self):
        """A model may answer in ١٠٥٬٠٠٠ where the corpus wrote 105,000."""
        self.assertEqual([], ungrounded_figures("المصاريف ١٠٥٬٠٠٠ جنيه", FEE_TABLE))

    def test_separators_do_not_make_two_figures_differ(self):
        self.assertEqual([], ungrounded_figures("the fee is 105000 EGP", FEE_TABLE))
        self.assertEqual([], ungrounded_figures("the fee is 105 000 EGP", FEE_TABLE))


class TheFaultsThatRetiredThePreviousCheckTests(unittest.TestCase):
    """Each of these withdrew a correct answer, or passed a wrong one, in production."""

    def test_a_time_is_not_read_as_a_price(self):
        """"10:00" read as 10,000 and a correct timetable was withdrawn. Two digits at a
        time, so this check never sees a clock at all."""
        answer = "الحصة الساعة 10:00 والفسحة 11:30"
        self.assertEqual([], ungrounded_figures(answer, "no numbers here at all"))

    def test_a_figure_is_never_satisfied_by_a_substring_of_another(self):
        """"45 حصة" passed as verified because 45 appears inside "07:45" — a check
        approving a figure it had not verified. Whole tokens only, and both are under
        three digits anyway, so neither reaches the comparison."""
        self.assertEqual([], ungrounded_figures("45 حصة", "the day starts at 07:45"))
        # And the whole-token rule itself, at a length the check does look at:
        self.assertEqual(["105"], ungrounded_figures("105 students", "the fee is 105,000"))

    def test_an_arabic_multiplier_word_is_not_interpreted(self):
        """The retired check parsed «ألف» as a thousand and turned a clock into a price.
        Nothing here reads a figure's meaning; "45 ألف" is the token 45."""
        self.assertEqual([], ungrounded_figures("المصاريف 45 ألف", "no figures"))

    def test_dates_and_percentages_are_left_alone(self):
        answer = "الترم يبدأ 15/09 وفيه خصم 7% للأخ التاني"
        self.assertEqual([], ungrounded_figures(answer, "no figures at all"))


class WhatItReportsAndWhatItReplacesTests(unittest.TestCase):
    class _Finalizer:
        def __init__(self, answer):
            self.answer = answer

    class _Plan:
        short_circuit = False

    def _trace(self, *texts):
        return {"retrieved_chunks": [{"text": t} for t in texts]}

    def _run(self, mode, answer, trace):
        from backend.profiles import get_profile

        with patch.object(get_profile().agent, "answer_figures_mode", mode):
            return enforce_answer_figures(self._Finalizer(answer), self._Plan(), trace)

    def test_observe_reports_without_touching_the_answer(self):
        """The shipped default. It cannot tell an invented figure from a derived one — an
        answer that sums two rows states a figure the evidence does not spell — so a
        deployment reads its own false-positive rate before it lets this replace
        anything."""
        replacement = self._run(
            "observe", "the fee is 88,000 EGP", self._trace("Y03-Y05 | 105,000 EGP")
        )
        self.assertEqual("", replacement)

    def test_enforce_replaces_the_answer(self):
        replacement = self._run(
            "enforce", "the fee is 88,000 EGP", self._trace("Y03-Y05 | 105,000 EGP")
        )
        self.assertTrue(replacement)

    def test_a_grounded_answer_is_never_replaced(self):
        self.assertEqual(
            "", self._run("enforce", "the fee is 105,000 EGP", self._trace(FEE_TABLE))
        )

    def test_off_does_nothing_at_all(self):
        self.assertEqual(
            "", self._run("off", "the fee is 88,000 EGP", self._trace("105,000 EGP"))
        )

    def test_a_turn_that_retrieved_nothing_is_not_checked(self):
        """Records answers and social turns reach here too. A turn with no chunks has no
        evidence to check against, and the figures in it came from somewhere this does
        not police — the tool renders a record's grid, the model does not retype it."""
        self.assertEqual(
            "", self._run("enforce", "she scored 87.5 and 91.0", {"retrieved_chunks": []})
        )

    def test_a_short_circuited_turn_is_not_checked(self):
        from backend.profiles import get_profile

        class _Social:
            short_circuit = True

        with patch.object(get_profile().agent, "answer_figures_mode", "enforce"):
            self.assertEqual(
                "",
                enforce_answer_figures(
                    self._Finalizer("2,500 hellos"), _Social := _Social(),
                    self._trace("nothing"),
                ),
            )

    def test_the_shipped_default_observes(self):
        from backend.profiles.registry import load_profile

        self.assertEqual("observe", load_profile("school").agent.answer_figures_mode)


if __name__ == "__main__":
    unittest.main()
