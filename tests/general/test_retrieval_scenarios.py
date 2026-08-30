"""Ordinary parent questions, end to end through the parts that read language.

The unit suites pin each piece on its own: `test_text_matching` covers folding and
stemming, `test_document_pairs` covers the pair table and the routing rule. This file
covers the ordinary path THROUGH them — a real question, in the language and spelling a
parent would use, arriving at a corpus that is bilingual in some places and not others.

Everything here is a normal case. There are no adversarial inputs, no malformed files
and no empty strings; those live in `test_arabic_and_adversarial.py`. What these guard
against is the failure that unit tests miss: each piece correct, and the sequence still
wrong.
"""
import re
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.chat.language import ARABIC, ENGLISH, detect_language
from backend.db.models import DocumentPair
from backend.indexing import pair_store
from backend.text_matching import search_key


def terms(text):
    """The BM25 terms a string yields, as Milvus would see them.

    `search_key` rewrites word tokens in place and leaves punctuation where it was, so
    "امتى الاجازة؟" comes back as "امت اجاز؟" — the term is right, the question mark is
    still attached to it. Milvus's `standard` tokenizer splits on that punctuation, so
    the analyzer sees "اجاز". Splitting on whitespace here instead would compare
    "اجاز؟" against "اجاز" and report a mismatch that does not exist in production.
    """
    return set(re.findall(r"\w+", search_key(text), re.UNICODE))


def _memory_sessionmaker():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    DocumentPair.__table__.create(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class ParentQuestionLanguageTests(unittest.TestCase):
    """The language a real question is answered in.

    Detection feeds pair routing, so a question read as the wrong language is answered
    from the wrong half of the corpus. These are the shapes parents actually send.
    """

    ARABIC_QUESTIONS = [
        "المصاريف كام للصف الرابع؟",
        "ابني غاب كام يوم الترم ده؟",
        "امتى تبدأ الاجازة النصف سنوية؟",
        "ممكن اعرف درجات ليلى في الرياضيات؟",
        "الباص بيعدي من منطقة المعادي؟",
        "ايه المطلوب في الزي المدرسي للبنات؟",
        "عايزة اقابل مدرس الرياضيات",
        "شكرا على المعلومات",
    ]

    ENGLISH_QUESTIONS = [
        "How much are the fees for grade 4?",
        "How many days has my son missed this term?",
        "When does the mid-year holiday start?",
        "Can I see Layla's maths grades?",
        "Does the bus cover Maadi?",
        "What is the uniform policy for girls?",
    ]

    def test_arabic_questions_are_read_as_arabic(self):
        for question in self.ARABIC_QUESTIONS:
            with self.subTest(question=question):
                self.assertEqual(ARABIC, detect_language(question))

    def test_english_questions_are_read_as_english(self):
        for question in self.ENGLISH_QUESTIONS:
            with self.subTest(question=question):
                self.assertEqual(ENGLISH, detect_language(question))

    def test_an_arabic_question_naming_an_english_thing_stays_arabic(self):
        """Arabic questions routinely carry Latin proper nouns. Reading one as English
        would answer it in the wrong language and route it to the wrong corpus half."""
        for question in [
            "ايه شروط الالتحاق بالـ IB Diploma؟",
            "المصاريف كام لسنة 2026؟",
            "عايز اعرف مواعيد الـ STEM club",
        ]:
            with self.subTest(question=question):
                self.assertEqual(ARABIC, detect_language(question))


class QuestionsReachTheirTopicTests(unittest.TestCase):
    """A parent's wording and the corpus's wording meeting on the same BM25 term.

    Each case is a question as a parent types it and a sentence as a school document
    writes it. They share a topic, so they must share at least one content term — that
    overlap is the entire contribution of the sparse half of retrieval.
    """

    CASES = [
        ("المصاريف كام للصف الرابع؟", "مصاريف الصف الرابع الابتدائي للعام الدراسي", "مصاريف"),
        ("الرسوم بتتدفع على كام قسط؟", "تُدفع الرسوم الدراسية على ثلاثة أقساط", "رسوم"),
        ("كام يوم غياب مسموح؟", "يُحتسب الغياب من بداية الفصل الدراسي", "غياب"),
        # Singular in the question, plural in the document — the commonest shape of all.
        ("امتى الاجازة؟", "تبدأ الاجازات الرسمية في شهر يناير", "اجاز"),
        ("ايه الزي المطلوب؟", "الزي المدرسي إلزامي لجميع الصفوف", "زي"),
        ("الباص بيعدي منين؟", "خطوط الباص تغطي المناطق التالية", "باص"),
        ("عايزة درجة ليلى", "تصدر الدرجات في نهاية كل فصل دراسي", "درج"),
        ("الكتب داخلة في المصاريف؟", "تشمل الرسوم الكتب المدرسية", "كتب"),
        ("امتى الامتحان؟", "تعقد الامتحانات في نهاية الفصل", "امتح"),
        ("في رحلات السنة دي؟", "تنظم المدرسة رحلة سنوية لكل صف", "رحل"),
    ]

    def test_the_question_and_the_document_share_a_term(self):
        for question, document, _ in self.CASES:
            with self.subTest(question=question):
                shared = terms(question) & terms(document)
                self.assertTrue(
                    shared,
                    f"no shared term between {question!r} and {document!r}",
                )

    def test_the_shared_term_is_the_topic_not_a_filler_word(self):
        """Overlap on «في» or «كل» would be overlap with the whole corpus. The term the
        two have in common has to be the thing being asked about."""
        for question, document, topic in self.CASES:
            with self.subTest(question=question):
                shared = terms(question) & terms(document)
                self.assertTrue(
                    any(term.startswith(topic) for term in shared),
                    f"{question!r} matched {document!r} only on {shared}",
                )

    def test_an_unrelated_document_shares_no_topic_term(self):
        """Recall must not have been bought with precision."""
        fees = terms("مصاريف الصف الرابع الابتدائي للعام الدراسي")
        uniform = terms("الزي المدرسي إلزامي لجميع الصفوف")
        self.assertEqual(set(), fees & uniform)

    def test_a_verb_does_not_reach_its_verbal_noun(self):
        """A documented limit, not an oversight.

        «غاب» (he was absent) and «الغياب» (absence) share a root but are different
        words, and only ROOT stemming merges them — the thing this module deliberately
        does not do, because the same mechanism merges طالب (a student) with طلب (an
        application). So the sparse half misses this pair and the dense half is what
        carries it: bge-m3 reads Arabic morphology, which is exactly the division of
        labour hybrid retrieval exists for.

        If this test ever fails, someone has swapped in a root stemmer.
        """
        self.assertNotEqual(search_key("غاب"), search_key("الغياب"))
        self.assertNotEqual(search_key("حضر"), search_key("الحضور"))

    def test_a_broken_plural_does_not_reach_its_singular(self):
        """The other documented limit. مادة/مواد changes its internal vowels rather
        than taking a suffix, so no suffix-stripping stemmer can connect them."""
        self.assertNotEqual(search_key("مادة"), search_key("مواد"))


class BilingualCorpusRoutingTests(unittest.TestCase):
    """A corpus that is bilingual in places, answering ordinary questions.

    The fixture is the shape a school actually ends up with: the important documents
    translated, the rest in whichever language they were written.
    """

    def setUp(self):
        patcher = patch.object(pair_store, "SessionLocal", _memory_sessionmaker())
        patcher.start()
        self.addCleanup(patcher.stop)

        # Fees and the calendar exist in both languages; bus routes are English only
        # and the uniform policy is Arabic only.
        fees = pair_store.attach("", ARABIC, "fees_ar.docx", title="Fees")["pair_id"]
        pair_store.attach(fees, ENGLISH, "fees_en.docx")
        calendar = pair_store.attach("", ENGLISH, "calendar_en.docx", title="Calendar")["pair_id"]
        pair_store.attach(calendar, ARABIC, "calendar_ar.docx")
        pair_store.attach("", ENGLISH, "bus_en.docx", title="Bus routes")
        pair_store.attach("", ARABIC, "uniform_ar.docx", title="Uniform")

    def _excluded(self, question):
        return set(pair_store.superseded_filenames(detect_language(question)))

    def test_an_arabic_question_drops_every_english_twin(self):
        excluded = self._excluded("المصاريف كام للصف الرابع؟")
        self.assertEqual({"fees_en.docx", "calendar_en.docx"}, excluded)

    def test_an_english_question_drops_every_arabic_twin(self):
        excluded = self._excluded("How much are the fees for grade 4?")
        self.assertEqual({"fees_ar.docx", "calendar_ar.docx"}, excluded)

    def test_single_language_documents_are_never_dropped(self):
        """The bus routes exist only in English and the uniform policy only in Arabic.
        Both must stay reachable from a question in either language."""
        for question in ["الباص بيعدي منين؟", "Does the bus cover Maadi?"]:
            with self.subTest(question=question):
                excluded = self._excluded(question)
                self.assertNotIn("bus_en.docx", excluded)
                self.assertNotIn("uniform_ar.docx", excluded)

    def test_an_arabic_question_about_an_english_only_document_is_answerable(self):
        """The case that makes this an exclusion rather than a language filter."""
        self.assertNotIn("bus_en.docx", self._excluded("الباص بيعدي من المعادي؟"))

    def test_exactly_one_half_of_each_pair_survives(self):
        for question in ["المصاريف كام؟", "How much are the fees?"]:
            with self.subTest(question=question):
                excluded = self._excluded(question)
                for pair in (("fees_ar.docx", "fees_en.docx"),
                             ("calendar_ar.docx", "calendar_en.docx")):
                    surviving = [name for name in pair if name not in excluded]
                    self.assertEqual(1, len(surviving), f"{pair} -> {surviving}")

    def test_translating_a_document_changes_which_half_answers(self):
        """The everyday admin action: the uniform policy gets an English version, and
        English questions start being answered from it instead of the Arabic one."""
        english = "What is the uniform policy?"
        self.assertEqual(set(), self._excluded(english) & {"uniform_ar.docx"})

        row = pair_store.find_by_filename("uniform_ar.docx")
        pair_store.attach(row["pair_id"], ENGLISH, "uniform_en.docx")

        self.assertIn("uniform_ar.docx", self._excluded(english))
        self.assertIn("uniform_en.docx", self._excluded("ايه الزي المطلوب؟"))


class RetrievalFilterShapeTests(unittest.TestCase):
    """What the routing decision looks like by the time Milvus sees it."""

    def setUp(self):
        patcher = patch.object(pair_store, "SessionLocal", _memory_sessionmaker())
        patcher.start()
        self.addCleanup(patcher.stop)

    def _clause(self, question):
        from backend.rag.utils import language_filter_clause

        return language_filter_clause(detect_language(question))

    def test_a_corpus_with_nothing_paired_adds_no_filter(self):
        """The state every deployment starts in: routing must cost nothing and change
        nothing until an admin actually pairs a document."""
        pair_store.attach("", ENGLISH, "bus_en.docx")
        pair_store.attach("", ARABIC, "uniform_ar.docx")

        self.assertEqual("", self._clause("الباص بيعدي منين؟"))
        self.assertEqual("", self._clause("Does the bus cover Maadi?"))

    def test_a_paired_corpus_filters_by_filename(self):
        pair_id = pair_store.attach("", ARABIC, "fees_ar.docx")["pair_id"]
        pair_store.attach(pair_id, ENGLISH, "fees_en.docx")

        clause = self._clause("المصاريف كام؟")
        self.assertTrue(clause.startswith(" and filename not in ["))
        self.assertIn("fees_en.docx", clause)
        self.assertNotIn("fees_ar.docx", clause)


if __name__ == "__main__":
    unittest.main()
