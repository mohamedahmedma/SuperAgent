"""Which day «بكره» is, and every way that answer can be wrong.

`backend/school_week.py` exists because a model has no clock. These cases are the
vocabulary and the arithmetic on their own, with the date injected — the tool-level
behaviour they feed is in `test_records_tool.py`.

Two properties run through the whole file:

  * A NAMED day never consults the clock. A timetable is a recurring weekly plan, so
    every Sunday holds the same lessons and dating one would only add a way to be wrong.

  * A RELATIVE day is arithmetic on the school's own date, and is asserted across a week
    boundary in both directions — the failure this protects against is an off-by-one that
    a test written on a Wednesday would never see.
"""
import unittest
from datetime import date

from backend.school_week import AskedDay, day_phrase, resolve_day

#: A Thursday. Chosen because every interesting neighbour is a different kind of day:
#: tomorrow is Friday (shut at a Sunday-to-Thursday school), the day after is Saturday
#: (a school day there), and yesterday is Wednesday (an ordinary one).
THURSDAY = date(2026, 9, 10)


class TheWordsParentsActuallyType(unittest.TestCase):
    """Egyptian dialect included, because that is what arrives.

    The question this feature was built for is «إيه حصص بكره؟», and «بكره» is not the
    word a phrasebook gives for tomorrow.
    """

    def test_tomorrow_in_dialect_and_in_standard_arabic(self):
        for text in ("بكره", "بكرة", "بكرا", "غدا", "غداً"):
            self.assertEqual(day_phrase(text), "tomorrow", text)

    def test_today_in_dialect_and_in_standard_arabic(self):
        for text in ("النهاردة", "النهارده", "اليوم"):
            self.assertEqual(day_phrase(text), "today", text)

    def test_the_arabic_week(self):
        expected = {
            "السبت": "saturday",
            "الأحد": "sunday",
            "الاثنين": "monday",
            "الاتنين": "monday",
            "الثلاثاء": "tuesday",
            "الأربعاء": "wednesday",
            "الخميس": "thursday",
            "الجمعة": "friday",
        }
        for text, day in expected.items():
            self.assertEqual(day_phrase(text), day, text)

    def test_english_reads_the_same_table(self):
        self.assertEqual(day_phrase("what does he have tomorrow?"), "tomorrow")
        self.assertEqual(day_phrase("lessons on Sunday"), "sunday")

    def test_a_day_is_found_inside_a_whole_question(self):
        """The phrase arrives surrounded by the rest of the message, never on its own."""
        self.assertEqual(day_phrase("ابني هياخد ايه بكره"), "tomorrow")
        self.assertEqual(day_phrase("عندها ايه يوم الأحد"), "sunday")

    def test_the_question_mark_is_not_part_of_the_day(self):
        """«بكرة؟» — no space before it, which is how the question actually arrives.

        Matching whole space-delimited tokens missed every question ending this way,
        which is most of them.
        """
        self.assertEqual(day_phrase("بنتي هتاخد إيه بكرة؟"), "tomorrow")
        self.assertEqual(day_phrase("حصص الأربعاء؟"), "wednesday")

    def test_a_question_naming_no_day_finds_none(self):
        """The whole-week question, which must stay the whole-week answer."""
        for text in ("إيه جدولها؟", "what is her timetable", "", "   "):
            self.assertEqual(day_phrase(text), "", repr(text))


class TheWordsThatLookLikeDays(unittest.TestCase):
    """Every one of these cost a wrong answer somewhere before it was handled."""

    def test_the_school_day_is_not_today(self):
        """«اليوم الدراسي» is "the school day" — the subject of the question, not its day.

        The longer phrase is listed as recognised-but-not-a-day, and longest-match-first
        is what stops it falling through to «اليوم».
        """
        self.assertEqual(day_phrase("اليوم الدراسي بيخلص امتى؟"), "")
        self.assertEqual(day_phrase("when does the school day end"), "")

    def test_the_day_after_tomorrow_beats_tomorrow(self):
        """«بعد بكره» contains «بكره», and the shorter reading is a day early."""
        self.assertEqual(day_phrase("حصصه بعد بكرة"), "day after tomorrow")
        self.assertEqual(day_phrase("بعد غد"), "day after tomorrow")

    def test_one_o_clock_is_not_sunday(self):
        """«الواحدة» contains the letters of «الأحد» and is a time, not a day.

        Whole tokens rather than substrings is the whole of the fix, and this is what
        it buys.
        """
        self.assertEqual(day_phrase("الساعة الواحدة"), "")


class ANamedDayNeverNeedsTheClock(unittest.TestCase):
    def test_a_named_day_resolves_to_itself(self):
        asked = resolve_day("الأحد", today=THURSDAY)
        self.assertEqual(asked, AskedDay(key="sunday", phrase="sunday", relative=""))

    def test_a_named_day_is_the_same_answer_whatever_the_date(self):
        """A timetable is a standing weekly plan, so every Sunday is the same Sunday."""
        keys = {
            resolve_day("sunday", today=THURSDAY.replace(day=10 + offset)).key
            for offset in range(7)
        }
        self.assertEqual(keys, {"sunday"})

    def test_a_named_day_reports_no_relative_wording(self):
        """Empty `relative` is what tells the template to say "Sunday" and not
        "tomorrow, Sunday" — the parent already said which day they meant."""
        self.assertEqual(resolve_day("Thursday", today=THURSDAY).relative, "")


class ARelativeDayIsArithmetic(unittest.TestCase):
    def test_tomorrow_from_a_thursday_is_friday(self):
        asked = resolve_day("بكره", today=THURSDAY)
        self.assertEqual(asked.key, "friday")
        # Carried so the answer can say "tomorrow" back to the parent who asked it.
        self.assertEqual(asked.relative, "tomorrow")

    def test_today_is_today(self):
        self.assertEqual(resolve_day("النهاردة", today=THURSDAY).key, "thursday")

    def test_yesterday_steps_back(self):
        self.assertEqual(resolve_day("امبارح", today=THURSDAY).key, "wednesday")

    def test_the_day_after_tomorrow_steps_two(self):
        self.assertEqual(resolve_day("بعد بكرة", today=THURSDAY).key, "saturday")

    def test_tomorrow_wraps_the_week_forwards(self):
        """A Saturday's tomorrow is Sunday — the wrap the modulo exists for."""
        saturday = date(2026, 9, 12)
        self.assertEqual(saturday.strftime("%A").lower(), "saturday")
        self.assertEqual(resolve_day("tomorrow", today=saturday).key, "sunday")

    def test_yesterday_wraps_the_week_backwards(self):
        monday = date(2026, 9, 14)
        self.assertEqual(monday.strftime("%A").lower(), "monday")
        self.assertEqual(resolve_day("yesterday", today=monday).key, "sunday")

    def test_every_day_of_one_week_resolves_to_its_own_name(self):
        """The index table against Python's own, for all seven — an off-by-one here is
        invisible on six days out of seven."""
        for offset in range(7):
            today = date(2026, 9, 7 + offset)
            self.assertEqual(
                resolve_day("today", today=today).key,
                today.strftime("%A").lower(),
            )


class NothingResolvesWhenNothingWasAsked(unittest.TestCase):
    def test_a_question_with_no_day_resolves_to_none(self):
        """None is what sends the caller back to the whole week."""
        self.assertIsNone(resolve_day("إيه جدولها؟", today=THURSDAY))
        self.assertIsNone(resolve_day("", today=THURSDAY))

    def test_a_canonical_phrase_resolves_as_readily_as_a_parent_s_words(self):
        """The planner hands over "tomorrow"; the model hands over «بكره». One lookup
        serves both, so the two paths cannot come to disagree about which day it is."""
        self.assertEqual(
            resolve_day("tomorrow", today=THURSDAY),
            resolve_day("بكره", today=THURSDAY),
        )
