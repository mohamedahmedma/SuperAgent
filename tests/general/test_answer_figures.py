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

So this ignores figures under three digits outright — times, dates, ordinals, small
counts and percentages all leave the check in one stroke — compares whole tokens rather
than substrings, and reads no figure's meaning at all.

## Why the cases below are the shapes they are

Every bound in the extractor was measured against the shipped corpus rather than guessed,
and the fixtures here are that corpus's own text. Extracting every run of digits from it
gives eighteen distinct amounts, all comma-grouped and four to six digits, and alongside
them a set of figures that are NOT amounts and must never be read as one:

    (+20) 100 000 0000   and   +201000000000      the same mobile, twice, one document
    EG00 0000 0000 0000 0000 0000 0000 0          an IBAN
    0000-0000-0000-0000                           an account number
    In Years 4, 7 and 9                           a list, not the amount 479
    17,600 EGP, 26,400 EGP, 26,400 EGP            a list that IS three amounts
    7:45 AM–2:30 PM,  15/09 - 18/12,  2024-2025   times, dates, an academic year
    Sep 2024,  Summer Camp 2026                   bare years
    a 7% discount,  Term 3 (30 %)                 rates

Each of those is a case below, because each of them is a way a correct answer could have
been accused of inventing a figure. A check that fires on a correct answer teaches a
deployment to switch it off, so where this can be wrong it is built to go quiet rather
than to report: an identifier yields nothing, an ungrouped run in the calendar band
yields nothing, a run whose separators do not group into thousands yields nothing.

What it cannot do is tell an invented figure from a DERIVED one — a sum, a discount
applied, a monthly figure. Those are the last class here, asserted as the false positives
they are, and they are why the shipped mode is `observe`.

The other half of the scope is that records no longer have model-written figures to
check. The tool renders the grid and the model writes the sentence beside it, so the
prose worth checking is what a turn wrote from RETRIEVED CHUNKS — which is exactly where
the fee incident was.
"""
import collections
import unittest
from unittest.mock import patch

from backend.chat.answer_checks import enforce_answer_figures, ungrounded_figures

# --------------------------------------------------------------------------------------
# Fixtures: the shipped corpus's own text, transcribed from the tables and paragraphs it
# indexes. Nothing here is invented — an expectation that holds against paraphrase would
# not tell us the check survives production.
# --------------------------------------------------------------------------------------

FEES = (
    "Grade | Egyptian (EGP) | International (EGP)\n"
    "Pre-K | 75,000 EGP | 85,000 EGP\n"
    "FS1–FS2 | 88,000 EGP | 98,000 EGP\n"
    "Y01–Y02 | 95,000 EGP | 105,000 EGP\n"
    "Y03–Y05 | 105,000 EGP | 115,000 EGP\n"
    "Y06 | 105,000 EGP | 115,000 EGP\n"
    "Y07–Y08 | 120,000 EGP | 130,000 EGP\n"
    "Y09–Y10 | 135,000 EGP | 145,000 EGP\n"
    "Y11–Y12 | 150,000 EGP | 160,000 EGP"
)

CAMP = (
    "Programme | Fee (per student / week) | Includes\n"
    "Half-Day Programme | 2,500 EGP | Morning STEAM sessions; Hands-on activities & play\n"
    "Full-Day Programme | 4,000 EGP | Full STEAM & recreation day\n"
    "Half-Week Programme | 3,000 EGP | Morning STEAM sessions"
)

INSTALMENTS = (
    "Example - FS1-FS2 Fees (Egyptian, EGP):\n"
    "Full Fees: 88,000 EGP\n"
    "Down Payment: 17,600 EGP\n"
    "Each subsequent payment: 17,600 EGP, 26,400 EGP, 26,400 EGP"
)

CONTACT = (
    "Hours: 7:45 AM–2:30 PM, Closed on Fridays & Saturdays\n"
    "Mobile Phone: (+20) 100 000 0000\n"
    "WhatsApp: (+20) 100 000 0000 - Not for updates\n"
    "Mobile Phone: +201000000000\n"
    "Account Number: 0000-0000-0000-0000\n"
    "IBAN Number: EG00 0000 0000 0000 0000 0000 0000 0"
)

CALENDAR = (
    "Semester(s) start dates:\n"
    "1st: 15/09 - 18/12\n"
    "2nd: 11/01 - 26/03\n"
    "3rd: 12/04 - 25/06\n"
    "Please note that the school year ends on 02nd July.\n"
    "school is running and offering its learning services since Sep 2024 (mock)\n"
    "All under Mock School ID EG-MOCK-001, issued on 15 September 2025."
)

DISCOUNTS = (
    "Sibling discounts:\n"
    "The school offers a 7% discount for second and subsequent children.\n"
    "Term 3 (30 %): 1st March\n"
    "40% for Term 2, paid upon acceptance."
)

#: Every distinct amount the corpus states. All eighteen are comma-grouped and none falls
#: in the calendar band, which is what makes the year rule free.
CORPUS_AMOUNTS = (
    "2,500", "3,000", "4,000", "17,600", "26,400", "75,000", "85,000", "88,000",
    "95,000", "98,000", "105,000", "115,000", "120,000", "130,000", "135,000",
    "145,000", "150,000", "160,000",
)

PRICED = "\n".join((FEES, CAMP, INSTALMENTS))
NOTHING = ""

Case = collections.namedtuple("Case", "name answer evidence expect why")


def case(name, answer, evidence, expect, why=""):
    return Case(name, answer, evidence, expect, why)


def _suite(class_name, doc, cases):
    """One named test per case, so a failure names the shape that broke, not an index."""
    seen = set()

    def method(spec):
        def test(self):
            self.assertEqual(
                sorted(spec.expect),
                ungrounded_figures(spec.answer, spec.evidence),
                "\n  answer:   %r\n  evidence: %r" % (spec.answer, (spec.evidence or "")[:160]),
            )

        test.__doc__ = spec.why or None
        return test

    attributes = {"__doc__": doc}
    for spec in cases:
        assert spec.name not in seen, "duplicate case name %r" % spec.name
        seen.add(spec.name)
        attributes["test_" + spec.name] = method(spec)
    return type(class_name, (unittest.TestCase,), attributes)


# --------------------------------------------------------------------------------------
# The incident
# --------------------------------------------------------------------------------------

TheIncidentThisExistsForTests = _suite(
    "TheIncidentThisExistsForTests",
    """The wrong row of the right table, which is the failure nothing else sees.

    The figure IS in the corpus, so no retrieval check and no grader can object to it.
    This asks a narrower question than "is it in the corpus": is it in the evidence THIS
    answer was written from.""",
    [
        case("a_fee_from_the_wrong_row_is_caught",
             "مصاريف Year 3 هي 88,000 جنيه في السنة.",
             "Y03–Y05 | 105,000 EGP | 115,000 EGP", ["88000"],
             "Verbatim from the deployment: Year 3 answered with the FS1-FS2 figure."),
        case("the_same_failure_in_english",
             "Year 3 costs 88,000 EGP per year.",
             "Y03–Y05 | 105,000 EGP | 115,000 EGP", ["88000"]),
        case("the_right_fee_passes", "مصاريف Year 3 هي 105,000 جنيه في السنة.", FEES, []),
        case("a_tier_the_evidence_does_not_cover",
             "Y11 fees are 150,000 EGP.",
             "Y03–Y05 | 105,000 EGP | 115,000 EGP", ["150000"],
             "Correct against the corpus, wrong against the evidence retrieved — and "
             "that is the reported case, because the turn had no way to know it."),
        case("the_camp_price_quoted_as_a_school_fee", "الرسوم 2,500 جنيه", FEES, ["2500"]),
        case("an_amount_no_row_states", "the fee is 111,000 EGP", FEES, ["111000"]),
        case("a_fee_missing_a_digit", "the fee is 10,500 EGP", FEES, ["10500"]),
        case("the_wrong_column_is_NOT_caught",
             "Y03 fees are 115,000 EGP for Egyptian families.",
             "Y03–Y05 | 105,000 EGP | 115,000 EGP", [],
             "A limit worth stating: both columns are on the row, so reading across it "
             "is invisible here. This catches the wrong LINE, not the wrong cell."),
    ],
)


# --------------------------------------------------------------------------------------
# Grounded: every amount the corpus states
# --------------------------------------------------------------------------------------

EveryAmountTheCorpusStatesTests = _suite(
    "EveryAmountTheCorpusStatesTests",
    """Each of the eighteen, quoted back with its evidence present. Any one of these
    firing is a correct answer withdrawn, which is the failure that retired the last
    check, so they are enumerated rather than sampled.""",
    [
        case("grounded_%s" % amount.replace(",", ""),
             "The fee is %s EGP per year." % amount, PRICED, [])
        for amount in CORPUS_AMOUNTS
    ],
)


# --------------------------------------------------------------------------------------
# One figure, many spellings
# --------------------------------------------------------------------------------------

TheSameFigureWrittenDifferentlyTests = _suite(
    "TheSameFigureWrittenDifferentlyTests",
    """A model does not have to copy the corpus's typography, and must not be accused of
    inventing a figure because it did not. Every one of these is 105,000, which the fee
    table states.""",
    [
        case("western_grouped", "the fee is 105,000 EGP", FEES, []),
        case("no_separator_at_all", "the fee is 105000 EGP", FEES, []),
        case("an_ascii_space", "the fee is 105 000 EGP", FEES, []),
        case("a_non_breaking_space", "the fee is 105 000 EGP", FEES, []),
        case("a_narrow_non_breaking_space", "the fee is 105 000 EGP", FEES, []),
        case("a_thin_space", "the fee is 105 000 EGP", FEES, []),
        case("the_arabic_thousands_separator", "الرسوم 105٬000 جنيه", FEES, []),
        case("an_arabic_comma", "الرسوم 105،000 جنيه", FEES, []),
        case("arabic_indic_digits", "المصاريف ١٠٥٬٠٠٠ جنيه", FEES, []),
        case("eastern_arabic_indic_digits", "المصاريف ۱۰۵٬۰۰۰ جنيه", FEES, []),
        case("digits_from_two_scripts_in_one_figure", "المصاريف 105٬٠٠٠ جنيه", FEES, []),
        case("wrapped_in_bidi_marks", "الرسوم ‏105,000‏ جنيه", FEES, [],
             "A right-to-left mark is invisible in the answer. Splitting a figure on one "
             "reports both halves against an answer that stated neither."),
        case("split_by_a_zero_width_space", "the fee is 105​000 EGP", FEES, []),
        case("grouped_answer_ungrouped_evidence", "105,000", "the fee is 105000 EGP", []),
        case("ungrouped_answer_grouped_evidence", "105000", "the fee is 105,000 EGP", []),
    ],
)


# --------------------------------------------------------------------------------------
# Figures that are not amounts
# --------------------------------------------------------------------------------------

FiguresThatAreNotAmountsTests = _suite(
    "FiguresThatAreNotAmountsTests",
    """Checked against NO evidence at all, so anything reported here is a false positive
    by construction. These are the corpus's own times, dates, labels, rates and counts —
    and the two faults that retired the previous check.""",
    [
        case("school_hours", "Hours: 7:45 AM–2:30 PM", NOTHING, [],
             "«10:00» read as 10,000 withdrew a correct timetable. Two digits at a time, "
             "so the check never sees a clock."),
        case("school_hours_in_arabic", "المدرسة من ٧:٤٥ صباحًا حتى ٢:٣٠ مساءً", NOTHING, []),
        case("an_after_school_range", "(7am-3:30 pm)", NOTHING, []),
        case("a_semester_start", "1st: 15/09 - 18/12", NOTHING, []),
        case("all_three_semesters",
             "1st: 15/09 - 18/12, 2nd: 11/01 - 26/03, 3rd: 12/04 - 25/06", NOTHING, []),
        case("the_sibling_discount", "a 7% discount for second and subsequent children",
             NOTHING, []),
        case("a_rate_with_a_space_before_the_sign", "Term 3 (30 %): 1st March", NOTHING, []),
        case("a_rate_on_a_term", "40% for Term 2, paid upon acceptance", NOTHING, []),
        case("a_three_digit_rate", "100% of our teachers are certified", NOTHING, [],
             "The lower bound drops 7% and 40% on length alone; 100% it does not, and an "
             "answer saying 100% of anything has not invented the amount one hundred."),
        case("an_arabic_percent_sign", "خصم ٧٪ للأخ التاني", NOTHING, []),
        case("a_three_digit_rate_in_arabic", "١٠٠٪ من المدرسين معتمدين", NOTHING, []),
        case("a_percent_sign_written_first", "٪١٠٠ من المدرسين معتمدين", NOTHING, [],
             "Arabic is written both ways round, so the sign is read on either side."),
        case("a_rate_next_to_an_amount", "خصم ٧٪ على 105,000 جنيه", FEES, [],
             "The rate leaves and the amount stays."),
        case("year_group_labels", "Y01–Y02 and Y11–Y12", NOTHING, []),
        case("foundation_stage_labels", "FS1–FS2 pupils", NOTHING, []),
        case("an_assessment_name", "the CAT4 assessment in Years 4, 7 and 9", NOTHING, [],
             "«4, 7» is two figures. Joining a list into 47 — or 479 — manufactures an "
             "amount to report against an answer that never stated one."),
        case("a_class_size", "classes of 18 students", NOTHING, []),
        case("an_ordinal_date", "the school year ends on 02nd July", NOTHING, []),
        case("a_decimal_grade", "she scored 87.5 in the assessment", NOTHING, [],
             "A full stop is not a separator: 87.5 is a grade, not 875."),
        case("two_digit_counts", "3 semesters and 12 subjects", NOTHING, []),
        case("a_country_code_alone", "reach us on (+20)", NOTHING, []),
        case("an_arabic_multiplier_word", "المصاريف 45 ألف", NOTHING, [],
             "The retired check parsed «ألف» as a thousand and turned a clock into a "
             "price. Nothing here reads a figure's meaning."),
        case("a_figure_inside_a_clock_is_not_a_figure", "45 حصة", "the day starts at 07:45", []),
        case("a_single_digit", "1 child", NOTHING, []),
        case("a_two_digit_figure", "99 places remain", NOTHING, []),
    ],
)


# --------------------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------------------

IdentifiersAreNotAmountsTests = _suite(
    "IdentifiersAreNotAmountsTests",
    """Phone numbers, an IBAN, an account number and a school ID. The corpus writes its
    mobile two ways on consecutive lines, so an answer that picks the other form than its
    evidence is the ordinary case, not the odd one.""",
    [
        case("a_spaced_mobile", "call (+20) 100 000 0000", NOTHING, []),
        case("an_unspaced_mobile", "call +201000000000", NOTHING, []),
        case("the_two_mobile_forms_are_never_compared",
             "You can reach the school on +201000000000.",
             "Mobile Phone: (+20) 100 000 0000", [],
             "Both forms are in the shipped corpus, one line apart. Reading either as an "
             "amount reports a correct answer."),
        case("and_the_other_way_round",
             "You can reach the school on (+20) 100 000 0000.",
             "Mobile Phone: +201000000000", []),
        case("a_whatsapp_line", "WhatsApp: (+20) 100 000 0000 - Not for updates", NOTHING, []),
        case("an_iban", "IBAN Number: EG00 0000 0000 0000 0000 0000 0000 0", NOTHING, []),
        case("an_iban_with_its_spaces_removed", "IBAN EG000000000000000000000000000",
             CONTACT, []),
        case("an_account_number", "Account Number: 0000-0000-0000-0000", CONTACT, []),
        case("an_account_number_without_hyphens", "Account Number: 0000000000000000",
             CONTACT, []),
        case("a_school_id_with_its_evidence", "All under Mock School ID EG-MOCK-001",
             CALENDAR, []),
        case("a_school_id_without_its_evidence", "Mock School ID EG-MOCK-001", NOTHING,
             ["001"],
             "Three digits is three digits. A short identifier IS checked, and grounds "
             "itself whenever its own line was retrieved."),
        case("a_url", "See https://aurexis.example/pre-k-opening", NOTHING, []),
        case("a_maps_link", "https://maps.aurexis.example/cairo-campus", NOTHING, []),
        case("an_email", "send your resume to careers@aurexis.example", NOTHING, []),
    ],
)


# --------------------------------------------------------------------------------------
# Years
# --------------------------------------------------------------------------------------

YearsAreNotAmountsTests = _suite(
    "YearsAreNotAmountsTests",
    """A model adding the year to a sentence whose evidence did not spell it was the
    largest remaining source of false alarms. Grouping tells a year from an amount: this
    corpus writes 2,500 for the one and 2026 for the other, never the reverse.""",
    [
        case("a_bare_year", "Future Leaders Summer Camp 2026", NOTHING, []),
        case("the_opening_year", "running since Sep 2024 (mock)", NOTHING, []),
        case("an_academic_year_range", "during its opening academic year 2024-2025",
             NOTHING, []),
        case("an_academic_year_with_a_slash", "the 2026/2027 academic year", NOTHING, []),
        case("a_deadline_carrying_a_year", "Payment to be completed by 30 June 2026",
             NOTHING, []),
        case("a_year_in_arabic", "العام الدراسي ٢٠٢٦", NOTHING, []),
        case("a_year_added_to_an_otherwise_grounded_answer",
             "The camp runs in summer 2026 and the half-day programme is 2,500 EGP.",
             CAMP, [],
             "The realistic shape: everything checkable is grounded and the model has "
             "added a year the chunk did not carry."),
        case("the_bottom_of_the_band", "the building dates from 1900", NOTHING, []),
        case("below_the_band", "the building dates from 1899", NOTHING, ["1899"]),
        case("the_top_of_the_band", "valid until 2099", NOTHING, []),
        case("above_the_band", "valid until 2100", NOTHING, ["2100"]),
        case("a_grouped_figure_in_the_band_is_still_an_amount",
             "the fee is 1,950 EGP", NOTHING, ["1950"],
             "Nobody writes a year as 2,026. Grouping is what keeps a small amount "
             "checkable while a bare year is not."),
    ],
)


# --------------------------------------------------------------------------------------
# Lists and ranges
# --------------------------------------------------------------------------------------

ListsAndRangesTests = _suite(
    "ListsAndRangesTests",
    """The corpus states three instalments on one line, separated by commas. A list of
    amounts must stay a list of amounts; a list of small figures must not become one
    large one.""",
    [
        case("the_instalment_line_verbatim",
             "Each subsequent payment: 17,600 EGP, 26,400 EGP, 26,400 EGP",
             INSTALMENTS, []),
        case("the_instalment_line_without_the_currency",
             "الدفعات: 17,600، 26,400، 26,400", INSTALMENTS, [],
             "A model restating the line in Arabic drops EGP from between the figures, "
             "leaving them separated only by a comma and a space."),
        case("a_list_that_invents_its_last_member",
             "الدفعات: 17,600، 26,400، 31,000", INSTALMENTS, ["31000"]),
        case("a_fee_range", "fees run from 75,000 to 160,000 EGP", FEES, []),
        case("a_fee_range_written_with_a_dash", "fees run 75,000-160,000 EGP", FEES, []),
        case("a_comma_list_of_year_groups", "the test is sat in Years 4, 7, 9", NOTHING, []),
        case("a_comma_list_of_terms", "payable across terms 1, 2, 3", NOTHING, []),
        case("a_comma_list_with_no_spaces", "terms 1,2,3", NOTHING, []),
        case("an_arabic_comma_list", "السنوات ٤، ٧، ٩", NOTHING, []),
        case("a_bulleted_list_of_amounts",
             "- Pre-K: 75,000 EGP\n- FS1–FS2: 88,000 EGP\n- Y03–Y05: 105,000 EGP",
             FEES, []),
    ],
)


# --------------------------------------------------------------------------------------
# Answer shape and length
# --------------------------------------------------------------------------------------

WhateverShapeTheAnswerTakesTests = _suite(
    "WhateverShapeTheAnswerTakesTests",
    """One word to many paragraphs, prose to markdown, Arabic to English. The check reads
    text, so the shape should not matter — these exist to prove it does not.""",
    [
        case("a_one_word_answer", "105,000", FEES, []),
        case("a_one_word_answer_that_is_wrong", "111,000", FEES, ["111000"]),
        case("a_one_sentence_answer", "Year 3 fees are 105,000 EGP per year.", FEES, []),
        case("a_multi_paragraph_answer",
             "أهلاً بيك! مصاريف Year 3 هي 105,000 جنيه في السنة للمصريين.\n\n"
             "لو حابب الكامب الصيفي، النص يوم بـ 2,500 جنيه في الأسبوع والفل داي "
             "بـ 4,000 جنيه.\n\n"
             "المصاريف بتتقسط: مقدم 17,600 جنيه وبعدها 26,400 جنيه.",
             PRICED, []),
        case("a_markdown_table_answer",
             "| Grade | Fee |\n|---|---|\n| Pre-K | 75,000 EGP |\n| Y03–Y05 | 105,000 EGP |",
             FEES, []),
        case("a_markdown_table_with_one_wrong_row",
             "| Grade | Fee |\n|---|---|\n| Pre-K | 75,000 EGP |\n| Y03–Y05 | 111,000 EGP |",
             FEES, ["111000"]),
        case("an_answer_in_bold", "The fee is **105,000 EGP** per year.", FEES, []),
        case("an_answer_with_no_figures_at_all", "نعم، بنوفر باص مدرسي للمناطق المحددة.",
             FEES, []),
        case("an_rtl_answer_with_latin_inside", "مصاريف Y03–Y05 هي 105,000 EGP.", FEES, []),
        case("a_very_long_grounded_answer",
             ("Y03–Y05 costs 105,000 EGP for Egyptian families. " * 200), FEES, []),
        case("one_wrong_figure_hidden_in_a_long_answer",
             ("Y03–Y05 costs 105,000 EGP for Egyptian families. " * 120)
             + "Y06 costs 111,000 EGP. "
             + ("Y03–Y05 costs 105,000 EGP for Egyptian families. " * 120),
             FEES, ["111000"]),
        case("figures_spread_across_lines",
             "Pre-K\n75,000\nFS1–FS2\n88,000\nY03–Y05\n105,000", FEES, []),
    ],
)


# --------------------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------------------

DegenerateInputTests = _suite(
    "DegenerateInputTests",
    """Empty, absent and whitespace on either side. A check that raises is a turn that
    500s, so every one of these has to return a list.""",
    [
        case("an_empty_answer", "", FEES, []),
        case("an_empty_evidence", "the fee is 105,000 EGP", "", ["105000"]),
        case("both_empty", "", "", []),
        case("a_whitespace_answer", "   \n\t  ", FEES, []),
        case("a_missing_answer", None, FEES, []),
        case("a_missing_evidence", "the fee is 105,000 EGP", None, ["105000"]),
        case("an_answer_with_no_digits", "Yes, we do.", FEES, []),
        case("an_evidence_with_no_digits", "105,000", "no figures here at all", ["105000"]),
        case("several_missing_figures_come_back_sorted",
             "111,000 then 222,000 then 133,000", FEES, ["111000", "133000", "222000"]),
        case("a_figure_stated_twice_is_reported_once",
             "111,000 EGP, yes — 111,000 EGP.", FEES, ["111000"]),
    ],
)


# --------------------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------------------

TheBoundsThemselvesTests = _suite(
    "TheBoundsThemselvesTests",
    """Three digits to seven. Below the floor sit the clock and the calendar; above the
    ceiling sit the phone book and the IBAN. The corpus's largest amount is 160,000.""",
    [
        case("two_digits_are_below_the_floor", "99 places", NOTHING, []),
        case("three_digits_reach_it", "999 places", NOTHING, ["999"]),
        case("exactly_one_hundred", "100 places", NOTHING, ["100"]),
        case("seven_digits_are_an_amount", "the endowment is 1,500,000 EGP", NOTHING,
             ["1500000"]),
        case("seven_digits_without_separators", "the endowment is 1500000 EGP", NOTHING,
             ["1500000"]),
        case("eight_digits_are_an_identifier", "reference 12,345,678", NOTHING, [],
             "Past the ceiling nothing is reported, which costs a very large amount its "
             "check and buys every phone number, IBAN and account number silence."),
        case("eight_digits_without_separators", "reference 12345678", NOTHING, []),
        case("leading_zeros_are_part_of_the_token", "room 007", NOTHING, ["007"]),
        case("all_zeros", "code 000", NOTHING, ["000"]),
        case("the_largest_amount_the_corpus_states", "Y11–Y12 is 160,000 EGP", FEES, []),
        case("one_order_of_magnitude_above_it", "Y11–Y12 is 1,600,000 EGP", FEES,
             ["1600000"]),
        case("a_figure_is_never_satisfied_by_a_substring_of_another",
             "105 students", "the fee is 105,000", ["105"],
             "Whole tokens. The substring match is what let 45 be satisfied by 07:45."),
    ],
)


# --------------------------------------------------------------------------------------
# Known false positives
# --------------------------------------------------------------------------------------

TheFalsePositivesItCannotAvoidTests = _suite(
    "TheFalsePositivesItCannotAvoidTests",
    """A derived figure is not in the evidence and is not invented either, and nothing
    that only compares digits can tell the two apart. Asserted rather than wished away:
    these are the reason the shipped mode is `observe` and not `enforce`, and the rate
    a deployment has to read before it changes that.""",
    [
        case("a_sum_of_two_rows", "two children in Y03 come to 210,000 EGP", FEES,
             ["210000"]),
        case("a_discount_applied",
             "with the 7% sibling discount that is 97,650 EGP",
             "\n".join((FEES, DISCOUNTS)), ["97650"]),
        case("a_term_share_computed", "40% of 105,000 is 42,000 EGP", FEES, ["42000"]),
        case("a_difference_between_tiers",
             "the international tier costs 10,000 EGP more", FEES, ["10000"]),
        case("a_monthly_figure", "105,000 EGP is about 8,750 EGP a month", FEES, ["8750"]),
        case("a_total_of_the_instalments",
             "three payments of 26,400 come to 79,200 EGP", INSTALMENTS, ["79200"]),
    ],
)


# --------------------------------------------------------------------------------------
# What it reports and what it replaces
# --------------------------------------------------------------------------------------

class WhatItReportsAndWhatItReplacesTests(unittest.TestCase):
    """The check's contract with the turn: which turns it looks at, and what it returns."""

    class _Finalizer:
        def __init__(self, answer):
            self.answer = answer

    class _Plan:
        short_circuit = False

    def _trace(self, *texts):
        return {"retrieved_chunks": [{"text": t} for t in texts]}

    def _run(self, mode, answer, trace, plan=None):
        from backend.profiles import get_profile

        with patch.object(get_profile().agent, "answer_figures_mode", mode):
            return enforce_answer_figures(
                self._Finalizer(answer), self._Plan() if plan is None else plan, trace
            )

    def test_observe_reports_without_touching_the_answer(self):
        """The shipped default. It cannot tell an invented figure from a derived one, so
        a deployment reads its own false-positive rate before it lets this replace
        anything."""
        self.assertEqual(
            "", self._run("observe", "the fee is 88,000 EGP", self._trace(FEES))
        )

    def test_enforce_replaces_the_answer(self):
        self.assertTrue(
            self._run("enforce", "the fee is 111,000 EGP", self._trace(FEES))
        )

    def test_enforce_serves_the_profiles_own_copy(self):
        from backend.profiles import get_profile

        replacement = self._run("enforce", "the fee is 111,000 EGP", self._trace(FEES))
        self.assertEqual(get_profile().user_copy.unverified_answer, replacement)

    def test_a_grounded_answer_is_never_replaced(self):
        self.assertEqual(
            "", self._run("enforce", "the fee is 105,000 EGP", self._trace(FEES))
        )

    def test_off_does_nothing_at_all(self):
        self.assertEqual(
            "", self._run("off", "the fee is 111,000 EGP", self._trace(FEES))
        )

    def test_evidence_is_every_chunk_the_turn_retrieved(self):
        """A figure grounded by the third chunk is grounded."""
        self.assertEqual(
            "", self._run("enforce", "the fee is 2,500 EGP",
                          self._trace(CONTACT, CALENDAR, CAMP))
        )

    def test_a_turn_that_retrieved_nothing_is_not_checked(self):
        """Records answers and social turns reach here too. A turn with no chunks has no
        evidence to check against, and the figures in it came from somewhere this does
        not police — the tool renders a record's grid, the model does not retype it."""
        self.assertEqual(
            "", self._run("enforce", "she scored 87.5 and 910.0", {"retrieved_chunks": []})
        )

    def test_a_turn_with_no_trace_at_all_is_not_checked(self):
        self.assertEqual("", self._run("enforce", "the fee is 111,000 EGP", None))

    def test_a_turn_with_no_plan_is_not_checked(self):
        from backend.profiles import get_profile

        with patch.object(get_profile().agent, "answer_figures_mode", "enforce"):
            self.assertEqual(
                "",
                enforce_answer_figures(
                    self._Finalizer("the fee is 111,000 EGP"), None, self._trace(FEES)
                ),
            )

    def test_a_short_circuited_turn_is_not_checked(self):
        class _Social:
            short_circuit = True

        self.assertEqual(
            "", self._run("enforce", "2,500 hellos", self._trace(FEES), plan=_Social())
        )

    def test_an_empty_answer_replaces_nothing(self):
        self.assertEqual("", self._run("enforce", "", self._trace(FEES)))

    def test_a_chunk_without_text_does_not_raise(self):
        self.assertEqual(
            "", self._run("observe", "the fee is 111,000 EGP",
                          {"retrieved_chunks": [{"score": 0.9}]})
        )

    def test_a_chunk_whose_text_is_not_a_string_does_not_raise(self):
        self.assertEqual(
            "", self._run("enforce", "the fee is 105000 EGP",
                          {"retrieved_chunks": [{"text": 105000}]})
        )

    def test_an_unknown_mode_observes_rather_than_enforces(self):
        """A profile typo must not start withdrawing answers."""
        self.assertEqual(
            "", self._run("watch", "the fee is 111,000 EGP", self._trace(FEES))
        )

    def test_the_shipped_default_observes(self):
        from backend.profiles.registry import load_profile

        self.assertEqual("observe", load_profile("school").agent.answer_figures_mode)


if __name__ == "__main__":
    unittest.main()
