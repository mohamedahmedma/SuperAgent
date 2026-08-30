"""Bilingual document pairs: the row model, and the routing rule built on it.

The feature in one sentence: a document uploaded in both Arabic and English is ONE
entry, and a question is answered from the half matching the language it was asked in.

Every test asserts correct behaviour. The four routing cases in
`LanguageRoutingTests` are the specification — if one of them changes, the feature has
changed, not the test.
"""
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.chat.language import ARABIC, ENGLISH
from backend.db.models import DocumentPair
from backend.indexing import language_check, pair_store


def _memory_sessionmaker():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    DocumentPair.__table__.create(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class PairStoreTestCase(unittest.TestCase):
    """Each test gets its own empty in-memory table."""

    def setUp(self):
        self._patch = patch.object(pair_store, "SessionLocal", _memory_sessionmaker())
        self._patch.start()
        self.addCleanup(self._patch.stop)


class RowLifecycleTests(PairStoreTestCase):
    def test_attaching_one_side_creates_an_unpaired_row(self):
        row = pair_store.attach("", ENGLISH, "fees_en.docx", title="Fees")
        self.assertTrue(row["pair_id"])
        self.assertEqual("fees_en.docx", row["filename_en"])
        self.assertEqual("", row["filename_ar"])
        self.assertFalse(row["paired"], "one side is not a pair")

    def test_the_second_side_completes_the_same_row(self):
        """The reason pairs are a table: the Arabic half can arrive months later
        without re-uploading the English one."""
        first = pair_store.attach("", ENGLISH, "fees_en.docx", title="Fees")
        second = pair_store.attach(first["pair_id"], ARABIC, "fees_ar.docx")

        self.assertEqual(first["pair_id"], second["pair_id"])
        self.assertTrue(second["paired"])
        self.assertEqual(1, len(pair_store.list_pairs()), "a second row was created")

    def test_a_title_survives_the_second_upload(self):
        first = pair_store.attach("", ENGLISH, "fees_en.docx", title="Fees Policy")
        second = pair_store.attach(first["pair_id"], ARABIC, "fees_ar.docx")
        self.assertEqual("Fees Policy", second["title"])

    def test_a_title_defaults_to_the_filename_stem(self):
        row = pair_store.attach("", ENGLISH, "bus_routes.docx")
        self.assertEqual("bus_routes", row["title"])

    def test_a_file_belongs_to_at_most_one_row(self):
        """Re-uploading a file under a new entry MOVES it. Two rows claiming one file
        would make 'does this have a twin' answerable two ways."""
        first = pair_store.attach("", ENGLISH, "fees_en.docx")
        pair_store.attach(first["pair_id"], ARABIC, "fees_ar.docx")
        pair_store.attach("", ENGLISH, "fees_en.docx", title="Moved")

        holders = [r for r in pair_store.list_pairs() if r["filename_en"] == "fees_en.docx"]
        self.assertEqual(1, len(holders))
        self.assertEqual("Moved", holders[0]["title"])

    def test_replacing_the_same_side_of_the_same_row_is_stable(self):
        first = pair_store.attach("", ARABIC, "fees_ar.docx", title="Fees")
        again = pair_store.attach(first["pair_id"], ARABIC, "fees_ar.docx")
        self.assertEqual(first["pair_id"], again["pair_id"])
        self.assertEqual("fees_ar.docx", again["filename_ar"])

    def test_detaching_one_side_leaves_the_other(self):
        row = pair_store.attach("", ENGLISH, "fees_en.docx")
        pair_store.attach(row["pair_id"], ARABIC, "fees_ar.docx")

        remaining = pair_store.detach("fees_ar.docx")
        self.assertIsNotNone(remaining)
        self.assertEqual("", remaining["filename_ar"])
        self.assertEqual("fees_en.docx", remaining["filename_en"])
        self.assertFalse(remaining["paired"])

    def test_detaching_the_last_side_removes_the_row(self):
        """An empty row is not a document with no files — it is a row nobody can see or
        fill, because the upload form creates a new one."""
        pair_store.attach("", ENGLISH, "fees_en.docx")
        self.assertIsNone(pair_store.detach("fees_en.docx"))
        self.assertEqual([], pair_store.list_pairs())

    def test_detaching_an_unknown_file_is_a_no_op(self):
        self.assertIsNone(pair_store.detach("never_uploaded.docx"))

    def test_find_by_filename_matches_either_side(self):
        row = pair_store.attach("", ENGLISH, "fees_en.docx")
        pair_store.attach(row["pair_id"], ARABIC, "fees_ar.docx")

        for name in ("fees_ar.docx", "fees_en.docx"):
            with self.subTest(filename=name):
                self.assertEqual(row["pair_id"], pair_store.find_by_filename(name)["pair_id"])

    def test_an_unknown_language_is_rejected(self):
        with self.assertRaises(ValueError):
            pair_store.attach("", "fr", "frais_fr.docx")


class LanguageRoutingTests(PairStoreTestCase):
    """The four cases, exactly as specified.

    Expressed as an EXCLUSION of the redundant twin rather than a filter down to the
    asked language. Filtering to the language would hide every document that exists in
    one language only, which would make an English-only policy unanswerable in Arabic —
    the opposite of what routing is for.
    """

    def _pair(self, *, arabic=None, english=None):
        pair_id = ""
        for language, filename in ((ARABIC, arabic), (ENGLISH, english)):
            if filename:
                pair_id = pair_store.attach(pair_id, language, filename)["pair_id"]
        return pair_id

    def test_both_versions_and_an_english_question_drops_the_arabic_half(self):
        self._pair(arabic="fees_ar.docx", english="fees_en.docx")
        self.assertEqual(["fees_ar.docx"], pair_store.superseded_filenames(ENGLISH))

    def test_both_versions_and_an_arabic_question_drops_the_english_half(self):
        self._pair(arabic="fees_ar.docx", english="fees_en.docx")
        self.assertEqual(["fees_en.docx"], pair_store.superseded_filenames(ARABIC))

    def test_an_arabic_only_document_answers_an_arabic_question(self):
        self._pair(arabic="fees_ar.docx")
        self.assertEqual([], pair_store.superseded_filenames(ARABIC))

    def test_an_english_only_document_still_answers_an_arabic_question(self):
        """The case a language filter would break, and the reason this is an
        exclusion."""
        self._pair(english="fees_en.docx")
        self.assertEqual([], pair_store.superseded_filenames(ARABIC))

    def test_an_unpaired_document_survives_alongside_a_paired_one(self):
        self._pair(arabic="fees_ar.docx", english="fees_en.docx")
        self._pair(english="bus_en.docx")

        superseded = pair_store.superseded_filenames(ARABIC)
        self.assertIn("fees_en.docx", superseded)
        self.assertNotIn("bus_en.docx", superseded, "an unpaired document was hidden")

    def test_an_unknown_language_excludes_nothing(self):
        """A turn nobody classified searches the whole corpus rather than half of it."""
        self._pair(arabic="fees_ar.docx", english="fees_en.docx")
        for language in ("", "fr", None):
            with self.subTest(language=language):
                self.assertEqual([], pair_store.superseded_filenames(language))

    def test_unpairing_takes_effect_without_reindexing(self):
        """The property that made this a table rather than a field on every chunk."""
        self._pair(arabic="fees_ar.docx", english="fees_en.docx")
        self.assertEqual(["fees_en.docx"], pair_store.superseded_filenames(ARABIC))

        pair_store.detach("fees_en.docx")
        self.assertEqual([], pair_store.superseded_filenames(ARABIC))


class FilterExpressionTests(PairStoreTestCase):
    """What routing actually hands to Milvus."""

    def _clause(self, language):
        from backend.rag.utils import language_filter_clause

        return language_filter_clause(language)

    def test_nothing_paired_adds_no_clause(self):
        self.assertEqual("", self._clause(ARABIC))

    def test_a_paired_document_excludes_its_twin_by_filename(self):
        pair_id = pair_store.attach("", ARABIC, "fees_ar.docx")["pair_id"]
        pair_store.attach(pair_id, ENGLISH, "fees_en.docx")

        clause = self._clause(ARABIC)
        self.assertIn("filename not in", clause)
        self.assertIn("fees_en.docx", clause)
        self.assertNotIn("fees_ar.docx", clause)

    def test_filenames_are_json_encoded(self):
        """Filenames are admin-supplied. A stray quote must not produce an expression
        that is invalid, or — worse — valid and wrong."""
        pair_id = pair_store.attach("", ARABIC, "fees_ar.docx")["pair_id"]
        pair_store.attach(pair_id, ENGLISH, 'od"d.docx')

        self.assertIn(r"od\"d.docx", self._clause(ARABIC))

    def test_a_pairing_failure_does_not_fail_the_turn(self):
        """Routing is an optimisation. If the table cannot be read the right outcome is
        to search everything, never to refuse the question."""
        with patch.object(pair_store, "superseded_filenames", side_effect=RuntimeError("db down")):
            self.assertEqual("", self._clause(ARABIC))


class LanguageVerificationTests(unittest.TestCase):
    """The guard on the two-column form.

    A file in the wrong column does not fail — it silently routes every Arabic question
    to English text. That is invisible once made and cheap to catch, because the
    document has already been parsed by the time this runs.
    """

    ARABIC_TEXT = "الرسوم الدراسية للصف الرابع الابتدائي تشمل الكتب والأنشطة والنقل المدرسي"
    ENGLISH_TEXT = "Tuition fees for grade four include books, activities and school transport"

    def test_matching_content_is_accepted(self):
        self.assertTrue(language_check.verify(self.ARABIC_TEXT, ARABIC).agrees)
        self.assertTrue(language_check.verify(self.ENGLISH_TEXT, ENGLISH).agrees)

    def test_swapped_columns_are_rejected(self):
        self.assertFalse(language_check.verify(self.ENGLISH_TEXT, ARABIC).agrees)
        self.assertFalse(language_check.verify(self.ARABIC_TEXT, ENGLISH).agrees)

    def test_the_message_says_which_column_to_use(self):
        verdict = language_check.verify(self.ENGLISH_TEXT, ARABIC)
        message = language_check.describe_mismatch("fees.docx", verdict)
        self.assertIn("fees.docx", message)
        self.assertIn("English column", message)

    def test_a_mixed_document_is_accepted(self):
        """Real documents are not monolingual: an Arabic policy carries English course
        codes and Latin proper nouns. A check demanding purity would reject the corpus
        it exists to protect."""
        mixed = self.ARABIC_TEXT + " IB Diploma Programme grade 4 STEM"
        self.assertTrue(language_check.verify(mixed, ARABIC).agrees)

    def test_text_that_says_nothing_is_accepted(self):
        """A scan that parsed to no letters has a different problem, and the upload job
        reports that one on its own."""
        for text in ("", "   ", "2026 -- 45000"):
            with self.subTest(text=text):
                self.assertTrue(language_check.verify(text, ARABIC).agrees)

    def test_the_verdict_names_the_language_it_looks_like(self):
        self.assertEqual(ENGLISH, language_check.verify(self.ENGLISH_TEXT, ARABIC).looks_like)
        self.assertEqual(ARABIC, language_check.verify(self.ARABIC_TEXT, ENGLISH).looks_like)


if __name__ == "__main__":
    unittest.main()
