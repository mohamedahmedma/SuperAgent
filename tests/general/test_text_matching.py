"""The matching lane: folding, light stemming, and the boundaries around them.

Every test here asserts CORRECT behaviour — a failure is a defect in
`backend/text_matching.py`, not in the test.

Three classes of thing are pinned, and they fail in different ways:

  - **Folding does its job.** Missing it means a parent who spells their own child's
    name correctly cannot be matched to the SIS row. Those cases are taken from the
    real roster in `sis.db` rather than invented, because the invented ones were all
    spelled consistently and the real ones are not.
  - **Folding does not do too much.** A light stemmer that quietly became a root
    stemmer would merge طالب with طلب and مدرسة with تدريس, and retrieval would get
    worse in a way no test would otherwise notice.
  - **The lossy lane stays out of the lossless one.** `sanitize_text` feeds the model
    and the display; `search_key` feeds BM25. If folded text ever reaches the first,
    answers start quoting Arabic that was never written.
"""
import sqlite3
import unittest
from pathlib import Path

from backend.text_matching import (
    arabic_stop_words_for_analyzer,
    fold,
    has_arabic,
    name_key,
    search_key,
)
from backend.text_normalization import sanitize_text

REPO_ROOT = Path(__file__).resolve().parents[2]


class FoldingTests(unittest.TestCase):
    """The orthographic variants Arabic writes the same word in."""

    def test_alef_forms_converge(self):
        self.assertEqual(fold("أحمد"), fold("احمد"))
        self.assertEqual(fold("إبراهيم"), fold("ابراهيم"))
        self.assertEqual(fold("آية"), fold("ايه"))

    def test_teh_marbuta_and_heh_converge(self):
        self.assertEqual(fold("فاطمة"), fold("فاطمه"))
        self.assertEqual(fold("سارة"), fold("ساره"))

    def test_alef_maksura_and_yeh_converge(self):
        self.assertEqual(fold("ليلى"), fold("ليلي"))
        self.assertEqual(fold("يسرى"), fold("يسري"))

    def test_diacritics_are_folded_away(self):
        self.assertEqual(fold("مُحَمَّد"), fold("محمد"))
        self.assertEqual(fold("الرِّياضِيّات"), fold("الرياضيات"))

    def test_folding_chains_onto_sanitize_text(self):
        """PDF damage is repaired before folding, not instead of it.

        Presentation forms and tatweel are `sanitize_text`'s job; folding runs after,
        so a name pasted out of a PDF and the same name typed normally converge.
        """
        self.assertEqual(fold("ﺍﻟﺮﺳﻮﻡ"), fold("الرسوم"))
        self.assertEqual(fold("الرســوم"), fold("الرسوم"))

    def test_latin_is_casefolded_so_one_key_serves_both_scripts(self):
        self.assertEqual(fold("Layla Ahmed"), fold("layla ahmed"))

    def test_empty_and_none_are_safe(self):
        self.assertEqual("", fold(""))
        self.assertEqual("", fold(None))
        self.assertEqual("", name_key(""))
        self.assertEqual("", search_key(""))

    def test_has_arabic(self):
        self.assertTrue(has_arabic("ما هي الرسوم"))
        self.assertTrue(has_arabic("fees الرسوم"))
        self.assertFalse(has_arabic("what are the fees"))
        self.assertFalse(has_arabic("2026"))


class LightStemmingTests(unittest.TestCase):
    """`search_key` merges surface forms of ONE word and nothing more."""

    def test_clitics_are_stripped(self):
        forms = ["الرسوم", "رسوم", "للرسوم", "بالرسوم"]
        self.assertEqual(1, len({search_key(word) for word in forms}), forms)

    def test_conjunction_before_the_article_is_stripped(self):
        """Snowball handles a bare و but not و+ال, and و is the commonest opening
        letter in written Arabic — the gap would exempt a large share of real tokens."""
        self.assertEqual(search_key("الرسوم"), search_key("والرسوم"))
        self.assertEqual(search_key("الرسوم"), search_key("فالرسوم"))
        self.assertEqual(search_key("المصاريف"), search_key("والمصاريف"))

    def test_conjunction_stripping_is_guarded_by_the_article(self):
        """و is only removed in front of ال. Removing it unconditionally would turn
        ولد into لد."""
        self.assertEqual("ولد", search_key("ولد"))
        self.assertEqual("وقت", search_key("وقت"))

    def test_distinct_words_stay_distinct(self):
        """The regression guard against light stemming becoming ROOT stemming.

        NLTK's ISRI stemmer reduces to the triliteral root and maps every one of these
        pairs onto one term. On a school corpus that means a question about an
        application (طلب) retrieves the answer about a student (طالب).
        """
        for left, right in [("طالب", "طلب"), ("مصاريف", "مصروف"), ("كتاب", "مكتب")]:
            with self.subTest(pair=(left, right)):
                self.assertNotEqual(search_key(left), search_key(right))

    def test_latin_is_left_for_the_milvus_analyzer(self):
        """English already gets lowercase + stop words + Porter one layer down, in
        `build_analyzer_params`. Stemming it again here would be duplicated work on a
        path that already behaves."""
        self.assertEqual("payments", search_key("payments"))
        self.assertEqual("2026", search_key("2026"))

    def test_mixed_script_text_keeps_both_halves(self):
        key = search_key("رسوم الـ IB Diploma 2026")
        self.assertIn("ib", key)
        self.assertIn("diploma", key)
        self.assertIn("2026", key)
        self.assertIn("رسوم", key)


class MeaningIsPreservedTests(unittest.TestCase):
    """Where the feminine ending changes WHAT a word is, not just who it refers to.

    Snowball strips the feminine ة, which is right far more often than it is wrong:
    طالب and طالبة are one noun about one kind of person, and a corpus writing one
    while a parent types the other should still match.

    A few words in this domain only look like that pair. مدرسة is a place and مدرس is
    a person. Sharing a BM25 term would make "مين مدرس الرياضيات؟" score against every
    chunk that merely says "the school" — which, in a school corpus, is nearly all of
    them.
    """

    def test_a_teacher_is_not_a_school(self):
        self.assertNotEqual(search_key("مدرس"), search_key("مدرسة"))

    def test_an_office_is_not_a_library(self):
        self.assertNotEqual(search_key("مكتب"), search_key("مكتبة"))

    def test_a_computer_is_not_a_calculator(self):
        self.assertNotEqual(search_key("حاسب"), search_key("حاسبة"))

    def test_a_protected_noun_still_normalises_across_its_prefixes(self):
        """Protection is against the SUFFIX being stripped, not against normalisation.
        Every spelling of "the school" must still reach one term."""
        for spelling in ["مدرسة", "المدرسة", "والمدرسة", "بالمدرسة", "للمدرسة", "مدرسه"]:
            with self.subTest(spelling=spelling):
                self.assertEqual(search_key("مدرسة"), search_key(spelling))

    def test_a_protected_noun_is_still_folded(self):
        """The ة/ه spelling difference must not survive, or the protection would trade
        one matching failure for another."""
        self.assertEqual(search_key("المدرسه"), search_key("المدرسة"))

    def test_genuine_gender_pairs_still_merge(self):
        """The general rule the exceptions above are carved out of. A corpus saying
        طالبة must answer a question asking about a طالب."""
        for masculine, feminine in [
            ("طالب", "طالبة"),
            ("معلم", "معلمة"),
            ("مدير", "مديرة"),
            ("ناظر", "ناظرة"),
        ]:
            with self.subTest(pair=(masculine, feminine)):
                self.assertEqual(search_key(masculine), search_key(feminine))

    def test_the_protected_list_stays_short(self):
        """Every entry is a word the stemmer now under-merges, so the list earns its
        place by being small. A growing one means the stemmer is being fought rather
        than corrected."""
        from backend.text_matching import _DISTINCT_FEMININE_NOUNS

        self.assertLessEqual(len(_DISTINCT_FEMININE_NOUNS), 8)


class EverydayParentVocabularyTests(unittest.TestCase):
    """The words parents actually type, in the forms they actually type them.

    Each group is one thing said several ways — definite, indefinite, with a
    conjunction or a preposition attached. All of them have to reach one BM25 term or
    the sparse half of retrieval misses a question it should have answered.
    """

    GROUPS = {
        "fees": ["الرسوم", "رسوم", "بالرسوم", "للرسوم", "والرسوم"],
        "expenses": ["المصاريف", "مصاريف", "بالمصاريف", "والمصاريف"],
        "attendance": ["الحضور", "حضور", "للحضور", "والحضور"],
        "absence": ["الغياب", "غياب", "بالغياب", "والغياب"],
        "uniform": ["الزي", "زي", "بالزي", "والزي"],
        "books": ["الكتب", "كتب", "بالكتب", "والكتب"],
        "exams": ["الامتحانات", "امتحانات", "بالامتحانات", "والامتحانات"],
        "grades": ["الدرجات", "درجات", "بالدرجات", "والدرجات"],
        "timetable": ["الجدول", "جدول", "بالجدول", "والجدول"],
        "holidays": ["الاجازات", "اجازات", "بالاجازات", "والاجازات"],
    }

    def test_each_group_reaches_one_term(self):
        for name, forms in self.GROUPS.items():
            with self.subTest(word=name):
                keys = {search_key(form) for form in forms}
                self.assertEqual(1, len(keys), f"{name} split into {keys}")

    def test_different_subjects_stay_different(self):
        """Merging surface forms must not merge topics: a question about fees should
        not retrieve the attendance policy."""
        keys = [search_key(forms[0]) for forms in self.GROUPS.values()]
        self.assertEqual(len(keys), len(set(keys)), "two unrelated topics share a term")

    def test_a_whole_question_keeps_its_content_words(self):
        """A realistic Egyptian question: the filler goes, the subject stays."""
        key = search_key("عايزة اعرف المصاريف كام للصف الرابع؟")
        self.assertIn("مصاريف", key)
        self.assertIn("رابع", key)

    def test_the_singular_and_the_plural_are_one_term(self):
        """A parent asks about «الاجازة» and the calendar is written about «الاجازات».

        Snowball takes the ت off a sound feminine plural but leaves the ا, so these
        arrived as two terms that never met until the suffix was stripped first.
        """
        for singular, plural in [
            ("اجازة", "اجازات"),
            ("درجة", "درجات"),
            ("رحلة", "رحلات"),
            ("امتحان", "امتحانات"),
            ("مدرسة", "مدارس"),
        ]:
            with self.subTest(pair=(singular, plural)):
                if singular == "مدرسة":
                    # A broken plural, listed here so the exception is explicit rather
                    # than an untested assumption: مدارس changes its internal vowels
                    # instead of taking a suffix, so no suffix stemmer reaches it.
                    self.assertNotEqual(search_key(singular), search_key(plural))
                else:
                    self.assertEqual(search_key(singular), search_key(plural))

    def test_stripping_the_plural_cannot_shrink_a_short_word(self):
        """بنات must not become بن, which would match most of the corpus."""
        for word in ["بنات", "لغات", "ستات"]:
            with self.subTest(word=word):
                self.assertGreaterEqual(len(search_key(word)), 3)

    def test_female_teachers_are_not_the_school(self):
        """The plural strip runs after the protected check, so مدرسات still stems
        normally rather than being caught by the school exception."""
        self.assertNotEqual(search_key("مدرسات"), search_key("المدرسة"))
        self.assertEqual(search_key("مدرس"), search_key("مدرسات"))

    def test_hamza_spelling_does_not_change_the_term(self):
        """Parents type أ and ا interchangeably, and so does the corpus."""
        for careless, careful in [
            ("الاجازة", "الأجازة"),
            ("الاسبوع", "الأسبوع"),
            ("الامتحان", "الإمتحان"),
        ]:
            with self.subTest(word=careful):
                self.assertEqual(search_key(careless), search_key(careful))


class NamesAreFoldedButNeverStemmedTests(unittest.TestCase):
    """A proper noun may be folded. Stemming one risks selecting a sibling."""

    def test_stemming_would_damage_these_names(self):
        """The premise of the whole `name_key` / `search_key` split.

        If this ever stops holding, the two functions could be merged. While it holds,
        merging them would make `resolve_child` able to match the wrong child.
        """
        for name in ["ليلى", "أميرة", "فاطمة"]:
            with self.subTest(name=name):
                self.assertNotEqual(name_key(name), search_key(name))

    def test_name_key_preserves_the_whole_name(self):
        self.assertEqual("ليلي", name_key("ليلى"))
        self.assertEqual("اميره", name_key("أميرة"))
        self.assertEqual("فاطمه", name_key("فاطمة"))


class StopWordTests(unittest.TestCase):
    """The Arabic stop list has to be in the form the analyzer actually sees."""

    def test_every_stop_word_is_stable_under_search_key(self):
        """The analyzer runs over text `search_key` produced. A stop word written in
        surface form would never match a token, and would fail silently."""
        for word in arabic_stop_words_for_analyzer():
            with self.subTest(word=word):
                self.assertEqual(word, search_key(word))

    def test_ali_is_not_a_stop_word(self):
        """Folding ى->ي makes the preposition على and the given name علي one string,
        and this deployment's SIS carries a child called علي عثمان. Stopping it would
        delete that child's name from the index."""
        self.assertNotIn(search_key("علي"), arabic_stop_words_for_analyzer())

    def test_school_vocabulary_is_not_stopped(self):
        stop = set(arabic_stop_words_for_analyzer())
        vocabulary = [
            "الرسوم", "المصاريف", "الحضور", "الغياب", "الزي", "الباص", "النتيجة",
            "الدرجات", "الامتحان", "المواد", "الصف", "الترم", "الكتب", "المدرسة",
        ]
        for word in vocabulary:
            with self.subTest(word=word):
                self.assertNotIn(search_key(word), stop)

    def test_the_list_reaches_the_analyzer(self):
        from backend.indexing.milvus_client import build_analyzer_params

        filters = build_analyzer_params()["filter"]
        stop_filter = next(f for f in filters if isinstance(f, dict) and f.get("type") == "stop")
        self.assertLessEqual(set(arabic_stop_words_for_analyzer()), set(stop_filter["stop_words"]))
        # English handling is untouched by the Arabic addition.
        self.assertIn("_english_", stop_filter["stop_words"])


class TheLossyLaneStaysOutOfTheLosslessOneTests(unittest.TestCase):
    """`sanitize_text` feeds the model and the display; folding must not leak into it."""

    def test_sanitize_text_still_preserves_diacritics(self):
        """Guards the boundary from the other side. Folding belongs in `text_matching`;
        the day it moves into `sanitize_text`, answers start quoting Arabic stripped of
        orthography the source actually had."""
        self.assertIn("َ", sanitize_text("مُحَمَّد"))

    def test_sanitize_text_still_preserves_hamza_and_teh_marbuta(self):
        self.assertEqual("أحمد", sanitize_text("أحمد"))
        self.assertEqual("فاطمة", sanitize_text("فاطمة"))

    def test_folding_is_lossy_and_that_is_the_difference(self):
        self.assertNotEqual(sanitize_text("أحمد"), fold("أحمد"))


class RealRosterSpellingsTests(unittest.TestCase):
    """Against the actual SIS rows, which are not spelled consistently.

    `sis.db` holds «ليلى أحمد» with a hamza next to «محمد احمد» without one, and
    «فاطمة» entered as «فاكمه». Before folding, only a byte-identical spelling matched,
    so a parent typing their own child's name correctly resolved to nobody.
    """

    @classmethod
    def setUpClass(cls):
        database = REPO_ROOT / "sis.db"
        if not database.exists():
            raise unittest.SkipTest("sis.db not present")
        with sqlite3.connect(database) as connection:
            cls.rows = list(connection.execute(
                "select full_name_ar, full_name_en from students"
            ))

    def _roster(self):
        from backend.chat.child_roster import _as_options

        return _as_options([
            {"student_id": str(index), "full_name_ar": arabic, "full_name_en": english}
            for index, (arabic, english) in enumerate(self.rows, 1)
        ])

    def test_correct_spelling_resolves_against_a_variant_row(self):
        from backend.chat.child_resolution import resolve_child

        for typed, expected in [
            ("محمد أحمد", "محمد احمد"),
            ("أميرة محمود", "اميره محمود"),
            ("يوسف إبراهيم", "يوسف ابراهيم"),
            ("ساره محمود", "سارة محمود"),
            ("سيد يسرى", "سيد يسري"),
        ]:
            with self.subTest(typed=typed):
                out = resolve_child(reference="named", child_name=typed, roster=self._roster())
                self.assertTrue(out.resolved, f"{typed} matched nobody")
                self.assertEqual(expected, out.label)

    def test_a_latin_first_name_resolves_to_an_arabic_row(self):
        from backend.chat.child_resolution import resolve_child

        out = resolve_child(reference="named", child_name="Layla", roster=self._roster())
        self.assertTrue(out.resolved)
        self.assertEqual("ليلى أحمد", out.label)

    def test_an_ambiguous_first_name_still_asks(self):
        """Folding must widen matching without weakening the uniqueness rule: four
        children carry أحمد, and picking one would show the wrong child's marks."""
        from backend.chat.child_resolution import resolve_child

        out = resolve_child(reference="named", child_name="أحمد", roster=self._roster())
        self.assertFalse(out.resolved)
        self.assertTrue(out.ask)
        self.assertGreater(len(out.options), 1)

    def test_a_typo_in_the_sis_row_is_not_papered_over(self):
        """«فاكمه» is ك for ط — a wrong consonant, not a spelling variant. Folding
        deliberately does not reach it, and a fuzzy matcher that did could select a
        sibling. The fix belongs in the SIS row."""
        from backend.chat.child_resolution import resolve_child

        out = resolve_child(reference="named", child_name="فاطمة أحمد", roster=self._roster())
        self.assertFalse(out.resolved)


if __name__ == "__main__":
    unittest.main()
