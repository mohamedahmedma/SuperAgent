"""Adversarial / bug-hunting suite, focused on Arabic (RTL, unicameral script).

Unlike the systematic suite, every test here was written to BREAK the pipeline.
Arabic is the highest-risk input for this code because:
  - it is UNICAMERAL (no upper/lower case), so any heuristic keyed on case
    silently misclassifies it;
  - it uses its own punctuation (؟ ، ؛ ۔) that Latin/CJK terminator sets miss;
  - PDF extraction often yields presentation forms (U+FB50-U+FEFF) and bidi
    control marks that change how text compares and displays;
  - ZWNJ/ZWJ are ORTHOGRAPHIC in Arabic-script languages, not decoration.

Tests assert the CORRECT behavior. A failure here is a defect in the pipeline,
not in the test.
"""
import unittest
from pathlib import Path
from unittest.mock import patch

import backend.indexing.document_loader as document_loader_module
from backend.indexing.document_loader import DocumentLoader, SentenceSplitter
from backend.indexing.pdf_layout import (
    build_blocks_from_pages,
    is_heading_line,
    looks_like_real_table,
    remove_page_furniture,
)
from tests.test_milvus_writer import FakeMilvusStore, load_milvus_writer_module

REPO_ROOT = Path(__file__).resolve().parents[1]

# Realistic Arabic school-domain strings.
AR_HEADING_FEES = "الرسوم الدراسية"
AR_HEADING_TRANSPORT = "النقل المدرسي"
AR_SENTENCE_1 = "تشمل الرسوم جميع الكتب المدرسية"
AR_SENTENCE_2 = "والأنشطة اللاصفية للفصل الأول."
AR_QUESTION = "ما هي الرسوم الدراسية؟"


def _line(text, top=10.0, size=10.0, bold=False):
    return {"text": text, "top": top, "bottom": top + size, "x0": 0.0, "x1": 200.0,
            "size": size, "bold": bold}


def _text_block(content, page, top):
    return {"type": "text", "content": content, "page_number": page, "top": top}


def _table_block(rows, page, top):
    return {"type": "table", "content": "", "rows": rows, "page_number": page, "top": top}


class ArabicCrossPageStitchingTests(unittest.TestCase):
    """Arabic has no letter case. Continuation detection keyed on `islower()`
    silently rejects EVERY Arabic fragment, so Arabic paragraphs split by a page
    break would never rejoin — the exact failure the stitching stage exists to
    prevent, invisible on Latin test data."""

    def setUp(self):
        self.loader = DocumentLoader()

    def test_arabic_paragraph_split_across_pages_is_stitched(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block(AR_SENTENCE_1, 0, 700.0),
            _text_block(AR_SENTENCE_2, 1, 30.0),
        ])
        self.assertEqual(1, len(blocks), "Arabic continuation was not stitched")
        self.assertIn("الكتب المدرسية والأنشطة", blocks[0]["content"])

    def test_hebrew_paragraph_split_across_pages_is_stitched(self):
        # Same unicameral property; guards the general fix, not an Arabic special case.
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("שכר הלימוד כולל", 0, 700.0),
            _text_block("את כל הספרים.", 1, 30.0),
        ])
        self.assertEqual(1, len(blocks))

    def test_arabic_paragraph_ending_with_arabic_question_mark_is_not_stitched(self):
        # ؟ (U+061F) terminates a sentence: the next page starts new content.
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block(AR_QUESTION, 0, 700.0),
            _text_block("الرسوم تختلف حسب الصف الدراسي.", 1, 30.0),
        ])
        self.assertEqual(2, len(blocks), "false stitch across an Arabic question mark")

    def test_arabic_paragraph_ending_with_arabic_semicolon_is_not_stitched(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("الرسوم تشمل الكتب؛", 0, 700.0),
            _text_block("النقل يحسب بشكل منفصل.", 1, 30.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_arabic_paragraph_ending_with_urdu_full_stop_is_not_stitched(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("فیس میں کتابیں شامل ہیں۔", 0, 700.0),
            _text_block("بس سروس الگ ہے۔", 1, 30.0),
        ])
        self.assertEqual(2, len(blocks))


class ArabicSentenceSplitterTests(unittest.TestCase):
    """CHUNK_STRATEGY=sentence on an Arabic corpus. Arabic sentences end with ؟
    and ۔, never with the Latin/CJK set alone."""

    def test_arabic_question_mark_splits_sentences(self):
        splitter = SentenceSplitter(max_chars=30, overlap_sentences=0)
        chunks = splitter.split_text("ما هي الرسوم؟ الرسوم تختلف حسب الصف.")
        self.assertEqual(2, len(chunks), "Arabic ؟ did not terminate a sentence")
        self.assertTrue(chunks[0].endswith("؟"))

    def test_urdu_full_stop_splits_sentences(self):
        splitter = SentenceSplitter(max_chars=30, overlap_sentences=0)
        chunks = splitter.split_text("فیس میں کتابیں شامل ہیں۔ بس سروس الگ ہے۔")
        self.assertEqual(2, len(chunks))

    def test_arabic_comma_does_not_split_sentences(self):
        # ، is a comma, not a terminator: one sentence must stay one chunk.
        splitter = SentenceSplitter(max_chars=200, overlap_sentences=0)
        chunks = splitter.split_text("تشمل الرسوم الكتب، والأنشطة، والزي المدرسي.")
        self.assertEqual(1, len(chunks))

    def test_arabic_content_is_never_lost_by_splitting(self):
        splitter = SentenceSplitter(max_chars=25, overlap_sentences=0)
        text = "الرسوم تشمل الكتب؟ النقل منفصل؟ الزي مطلوب."
        joined = "".join(splitter.split_text(text))
        for token in ("الكتب", "النقل", "الزي"):
            self.assertIn(token, joined)


class ArabicHeadingDetectionTests(unittest.TestCase):
    """Heading heuristics on a unicameral script: the all-caps rule must never
    fire, size/bold must still work, and Arabic punctuation must disqualify."""

    def test_arabic_heading_detected_by_font_size(self):
        self.assertTrue(is_heading_line(_line(AR_HEADING_FEES, size=18.0), body_size=10.0))

    def test_arabic_heading_detected_by_bold(self):
        self.assertTrue(is_heading_line(_line(AR_HEADING_TRANSPORT, bold=True), body_size=10.0))

    def test_arabic_line_ending_with_arabic_comma_is_not_a_heading(self):
        line = _line("الرسوم والنقل،", size=18.0, bold=True)
        self.assertFalse(is_heading_line(line, body_size=10.0),
                         "Arabic comma ، did not disqualify a heading")

    def test_arabic_line_ending_with_arabic_question_mark_is_not_a_heading(self):
        line = _line(AR_QUESTION, size=18.0, bold=True)
        self.assertFalse(is_heading_line(line, body_size=10.0))

    def test_plain_arabic_body_line_is_not_a_heading(self):
        # No size/bold signal, no case signal available -> must stay body text.
        line = _line("تشمل الرسوم الدراسية جميع الكتب والأنشطة المدرسية")
        self.assertFalse(is_heading_line(line, body_size=10.0))

    def test_arabic_indic_numbered_heading_is_detected(self):
        # ٢.١ uses Arabic-Indic digits; \d is Unicode-aware in Python.
        self.assertTrue(is_heading_line(_line("٢.١ الرسوم"), body_size=10.0))


class ArabicNormalizationTests(unittest.TestCase):
    """sanitize_text on Arabic: bidi marks are invisible formatting and should go,
    but ZWNJ/ZWJ are ORTHOGRAPHIC in Arabic-script languages and must survive —
    stripping ZWNJ changes Persian word identity (می‌رود -> میرود)."""

    def setUp(self):
        self.loader = DocumentLoader()

    def _load(self, blocks):
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            return self.loader.load_document("ar.pdf", "ar.pdf")

    def test_zwnj_is_preserved_in_persian_text(self):
        docs = self._load([_text_block("برنامه‌ریزی سالانه مدرسه انجام می‌شود.", 0, 10.0)])
        joined = " ".join(d["text"] for d in docs)
        self.assertIn("‌", joined, "ZWNJ stripped — Persian word identity destroyed")

    def test_bidi_control_marks_are_removed(self):
        docs = self._load([_text_block("‏الرسوم ‫الدراسية‬ مطلوبة.", 0, 10.0)])
        for doc in docs:
            for mark in ("‏", "‫", "‬"):
                self.assertNotIn(mark, doc["text"])

    def test_arabic_diacritics_are_preserved(self):
        # Harakat carry meaning; NFC must not drop them.
        docs = self._load([_text_block("الرَّسْمُ الدِّرَاسِيُّ مَطْلُوبٌ لِلتَّسْجِيلِ.", 0, 10.0)])
        joined = " ".join(d["text"] for d in docs)
        self.assertIn("ّ", joined)  # shadda survived

    def test_arabic_text_round_trips_without_mojibake(self):
        original = "التسجيل مفتوح للعام الدراسي ٢٠٢٦-٢٠٢٧."
        docs = self._load([_text_block(original, 0, 10.0)])
        joined = " ".join(d["text"] for d in docs)
        self.assertIn("٢٠٢٦", joined)
        self.assertNotIn("?", joined)
        self.assertNotIn("�", joined)


class ArabicTableTests(unittest.TestCase):
    """RTL tables: the pipe-rendered rows, the real-table heuristic on Arabic
    cells, and section prefixing of an Arabic table."""

    def setUp(self):
        self.loader = DocumentLoader()

    def _load(self, blocks):
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            return self.loader.load_document("ar.pdf", "ar.pdf")

    def test_arabic_grid_is_recognized_as_a_real_table(self):
        rows = [["الصف", "الرسوم"], ["الأول", "١٠٠٠٠"], ["الثاني", "١١٠٠٠"]]
        self.assertTrue(looks_like_real_table(rows))

    def test_arabic_prose_grid_is_demoted(self):
        rows = [
            ["هذه جملة طويلة جدا تصف سياسة القبول في المدرسة بالتفصيل الكامل"],
            ["وهذه جملة أخرى طويلة تكمل نفس الفقرة بنفس الأسلوب السردي الطويل"],
        ]
        self.assertFalse(looks_like_real_table(rows))

    def test_arabic_table_leaf_is_prefixed_with_arabic_section(self):
        blocks = [
            {"type": "heading", "content": AR_HEADING_FEES, "level": 1, "page_number": 0, "top": 5.0},
            _table_block([["الصف", "الرسوم"], ["الأول", "١٠٠٠٠"]], 0, 20.0),
        ]
        docs = self._load(blocks)
        leaf = next(d for d in docs if d["chunk_level"] == 3 and "الصف" in d["text"])
        self.assertTrue(leaf["text"].startswith(AR_HEADING_FEES))
        self.assertIn("الأول | ١٠٠٠٠", leaf["text"])

    def test_arabic_table_split_across_pages_stitches(self):
        blocks = self.loader._stitch_cross_page_blocks([
            _table_block([["الصف", "الرسوم"], ["الأول", "١٠٠٠٠"]], 0, 700.0),
            _table_block([["الصف", "الرسوم"], ["الثاني", "١١٠٠٠"]], 1, 30.0),
        ])
        self.assertEqual(1, len(blocks))
        self.assertEqual(3, len(blocks[0]["rows"]))


class ArabicFurnitureTests(unittest.TestCase):
    """Arabic-Indic digits in repeated footers must be masked like ASCII digits,
    otherwise every page's footer looks unique and none are removed."""

    @staticmethod
    def _page(page_number, footer):
        return {
            "page_number": page_number,
            "height": 800.0,
            "tables": [],
            "lines": [
                _line(f"محتوى الصفحة رقم {page_number} مع كلمات مختلفة.", top=400.0),
                _line(footer, top=780.0),
            ],
        }

    def test_arabic_indic_page_numbers_are_masked_as_furniture(self):
        digits = "٠١٢٣٤"
        pages = [self._page(i, f"صفحة {digits[i]} من ٤") for i in range(4)]
        cleaned = remove_page_furniture(pages)
        for page in cleaned:
            self.assertEqual(1, len(page["lines"]), "Arabic-Indic footer not removed")
            self.assertIn("محتوى الصفحة", page["lines"][0]["text"])

    def test_arabic_footer_is_not_removed_from_short_documents(self):
        pages = [self._page(i, "صفحة ١") for i in range(2)]
        self.assertEqual(pages, remove_page_furniture(pages))


class ArabicPipelineIntegrationTests(unittest.TestCase):
    """Full-document Arabic run: section stack, merging, and hierarchy integrity
    on a realistic bilingual school handbook."""

    BLOCKS = [
        {"type": "heading", "content": AR_HEADING_FEES, "level": 1, "page_number": 0, "top": 5.0},
        _text_block("تشمل الرسوم الكتب المدرسية.", 0, 20.0),
        _text_block("تدفع الرسوم على ثلاث دفعات.", 0, 35.0),
        _table_block([["الصف", "الرسوم"], ["الأول", "١٠٠٠٠"]], 0, 50.0),
        {"type": "heading", "content": AR_HEADING_TRANSPORT, "level": 1, "page_number": 1, "top": 5.0},
        _text_block("خدمة الحافلات اختيارية لجميع الصفوف.", 1, 20.0),
    ]

    @classmethod
    def setUpClass(cls):
        loader = DocumentLoader()
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=cls.BLOCKS):
            cls.docs = loader.load_document("handbook_ar.pdf", "handbook_ar.pdf")

    def test_hierarchy_integrity_holds_for_arabic(self):
        by_id = {d["chunk_id"]: d for d in self.docs}
        self.assertEqual({1, 2, 3}, {d["chunk_level"] for d in self.docs})
        for doc in self.docs:
            if doc["chunk_level"] == 1:
                continue
            self.assertIn(doc["parent_chunk_id"], by_id)
            self.assertIn(doc["root_chunk_id"], by_id)

    def test_arabic_sections_do_not_bleed_into_each_other(self):
        leaves = [d for d in self.docs if d["chunk_level"] == 3]
        fees_leaves = [d for d in leaves if d["text"].startswith(AR_HEADING_FEES)]
        transport_leaves = [d for d in leaves if d["text"].startswith(AR_HEADING_TRANSPORT)]
        self.assertTrue(fees_leaves)
        self.assertTrue(transport_leaves)
        for leaf in fees_leaves:
            self.assertNotIn("الحافلات", leaf["text"])
        for leaf in transport_leaves:
            self.assertNotIn("الكتب", leaf["text"])

    def test_small_arabic_fragments_merge_into_one_leaf(self):
        fees_leaves = [
            d for d in self.docs
            if d["chunk_level"] == 3 and "تشمل الرسوم" in d["text"]
        ]
        self.assertEqual(1, len(fees_leaves))
        self.assertIn("ثلاث دفعات", fees_leaves[0]["text"])

    def test_arabic_chunk_ids_are_ascii_safe_and_unique(self):
        ids = [d["chunk_id"] for d in self.docs]
        self.assertEqual(len(ids), len(set(ids)))
        for chunk_id in ids:
            self.assertLessEqual(len(chunk_id.encode("utf-8")), 512)


class MixedDirectionTests(unittest.TestCase):
    """Bilingual documents (Arabic headings + English tables) are the norm for
    this domain and mix both code paths in one stream."""

    def setUp(self):
        self.loader = DocumentLoader()

    def _load(self, blocks):
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            return self.loader.load_document("mixed.pdf", "mixed.pdf")

    def test_arabic_heading_over_english_table(self):
        blocks = [
            {"type": "heading", "content": AR_HEADING_FEES, "level": 1, "page_number": 0, "top": 5.0},
            _table_block([["Grade", "Fee"], ["1", "10000"]], 0, 20.0),
        ]
        docs = self._load(blocks)
        leaf = next(d for d in docs if d["chunk_level"] == 3 and "Grade | Fee" in d["text"])
        self.assertTrue(leaf["text"].startswith(AR_HEADING_FEES))

    def test_english_sentence_then_arabic_fragment_does_not_false_stitch(self):
        # English sentence properly terminated -> no stitch even though the Arabic
        # fragment would qualify as a continuation on its own.
        blocks = self.loader._stitch_cross_page_blocks([
            _text_block("Fees are due in September.", 0, 700.0),
            _text_block("الرسوم تشمل الكتب.", 1, 30.0),
        ])
        self.assertEqual(2, len(blocks))

    def test_mixed_arabic_english_line_heading_detection(self):
        line = _line("الرسوم / Fees", size=18.0, bold=True)
        self.assertTrue(is_heading_line(line, body_size=10.0))


class ArabicDedupTests(unittest.TestCase):
    """Exact-hash dedup on Arabic: whitespace/case normalization is Latin-centric;
    confirm Arabic duplicates are still caught and near-variants are NOT."""

    @staticmethod
    def _doc(idx, text):
        return {"text": text, "filename": "ar.pdf", "file_type": "PDF", "chunk_id": f"c{idx}"}

    def _run(self, texts):
        module = load_milvus_writer_module()
        events = []

        class _Service:
            def get_embeddings(self, batch):
                return [[1.0] for _ in batch]

        writer = module.MilvusWriter(embedding_service=_Service(), milvus_manager=FakeMilvusStore(events))
        writer.write_documents([self._doc(i, t) for i, t in enumerate(texts)])
        return [cid for event in events if event[0] == "insert" for cid in event[1]]

    def test_identical_arabic_text_with_extra_whitespace_is_deduped(self):
        kept = self._run(["الرسوم  الدراسية مطلوبة", "الرسوم الدراسية مطلوبة"])
        self.assertEqual(["c0"], kept)

    def test_arabic_texts_differing_by_digits_are_both_kept(self):
        # Fee rows differing only in the amount must never be collapsed.
        kept = self._run(["الرسوم ١٠٠٠٠ ريال", "الرسوم ١١٠٠٠ ريال"])
        self.assertEqual(["c0", "c1"], kept)


class PathologicalInputTests(unittest.TestCase):
    """Error guessing beyond Arabic: inputs a real corpus eventually produces."""

    def setUp(self):
        self.loader = DocumentLoader()

    def _load(self, blocks):
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            return self.loader.load_document("bad.pdf", "bad.pdf")

    def test_whitespace_only_blocks_produce_no_chunks_and_no_crash(self):
        class _StubPage:
            page_content = "fallback"
            metadata = {"page": 0}

        class _StubPyPDFLoader:
            def __init__(self, path):
                pass

            def load(self):
                return [_StubPage()]

        with patch.object(document_loader_module, "parse_pdf_blocks",
                          return_value=[_text_block("   \n\t  ", 0, 10.0)]), \
             patch.object(document_loader_module, "PyPDFLoader", _StubPyPDFLoader):
            docs = self.loader.load_document("blank.pdf", "blank.pdf")
        self.assertTrue(any("fallback" in d["text"] for d in docs))

    def test_table_with_ragged_row_lengths_does_not_crash(self):
        blocks = [_table_block([["a", "b", "c"], ["d"], ["e", "f"]], 0, 10.0)]
        docs = self._load(blocks)
        self.assertTrue(docs)

    def test_heading_with_only_punctuation_is_handled(self):
        blocks = [
            {"type": "heading", "content": "؟؟؟", "level": 1, "page_number": 0, "top": 5.0},
            _text_block("محتوى تحت عنوان غريب.", 0, 20.0),
        ]
        docs = self._load(blocks)
        self.assertTrue(docs)

    def test_duplicate_heading_titles_do_not_collide_chunk_ids(self):
        blocks = []
        for page in range(3):
            blocks.append({"type": "heading", "content": AR_HEADING_FEES, "level": 1,
                           "page_number": page, "top": 5.0})
            blocks.append(_text_block(f"محتوى الصفحة {page} مع تفاصيل الرسوم الدراسية." * 20, page, 20.0))
        docs = self._load(blocks)
        ids = [d["chunk_id"] for d in docs]
        self.assertEqual(len(ids), len(set(ids)), "chunk_id collision across repeated headings")

    def test_very_deep_heading_nesting_does_not_explode_prefix(self):
        blocks = [
            {"type": "heading", "content": f"عنوان مستوى {level}", "level": level,
             "page_number": 0, "top": float(level)}
            for level in range(1, 7)
        ]
        blocks.append(_text_block("المحتوى النهائي تحت التسلسل العميق.", 0, 10.0))
        docs = self._load(blocks)
        for doc in docs:
            first_line = doc["text"].splitlines()[0]
            self.assertLessEqual(len(first_line), 200)


class Utf8ByteLimitTests(unittest.TestCase):
    """Milvus VARCHAR limits are BYTE limits, but chunk budgets are counted in
    characters. Arabic costs 2 bytes/char (CJK 3), so a chunk that looks within
    budget can blow the column limit and fail the whole insert batch AFTER all
    embedding work is paid for. Latin-only test data can never reveal this."""

    # Mirrors milvus_client.ensure_collection.
    TEXT_LIMIT = 65535
    FILENAME_LIMIT = 255
    CHUNK_ID_LIMIT = 512

    def setUp(self):
        self.loader = DocumentLoader()

    def _load(self, blocks, filename="ar.pdf"):
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            return self.loader.load_document(filename, filename)

    def _leaves(self, docs):
        return [d for d in docs if d["chunk_level"] == 3]

    def test_huge_arabic_table_leaf_fits_milvus_text_bytes(self):
        rows = [["ق" * 40000, "ب" * 40000]]
        docs = self._load([_table_block(rows, 0, 10.0)])
        leaves = self._leaves(docs)
        self.assertTrue(leaves)
        for leaf in leaves:
            self.assertLessEqual(
                len(leaf["text"].encode("utf-8")), self.TEXT_LIMIT,
                "Arabic leaf exceeds the Milvus VARCHAR byte limit",
            )

    def test_huge_cjk_table_leaf_fits_milvus_text_bytes(self):
        # 3 bytes/char — the worst case for a character-based cap.
        rows = [["学" * 40000, "费" * 40000]]
        docs = self._load([_table_block(rows, 0, 10.0)])
        for leaf in self._leaves(docs):
            self.assertLessEqual(len(leaf["text"].encode("utf-8")), self.TEXT_LIMIT)

    def test_byte_truncation_never_splits_a_character(self):
        rows = [["ق" * 40000, "ب" * 40000]]
        docs = self._load([_table_block(rows, 0, 10.0)])
        for leaf in self._leaves(docs):
            # Round-trips cleanly: no lone surrogate / partial UTF-8 sequence.
            self.assertEqual(leaf["text"], leaf["text"].encode("utf-8").decode("utf-8"))
            self.assertNotIn("�", leaf["text"])

    def test_long_arabic_filename_fails_fast_with_actionable_message(self):
        long_name = "دليل_الرسوم_الدراسية_للعام_الدراسي_الجديد_نسخة_محدثة" * 3 + ".pdf"
        self.assertGreater(len(long_name.encode("utf-8")), self.FILENAME_LIMIT)
        with self.assertRaises(ValueError) as ctx:
            self._load([_text_block("محتوى", 0, 10.0)], filename=long_name)
        message = str(ctx.exception)
        self.assertIn("255", message)
        self.assertIn("BYTES", message)

    def test_arabic_filename_within_limit_is_accepted(self):
        name = "دليل_الرسوم.pdf"
        docs = self._load([_text_block("تشمل الرسوم الكتب المدرسية.", 0, 10.0)], filename=name)
        self.assertTrue(docs)
        for doc in docs:
            self.assertLessEqual(len(doc["filename"].encode("utf-8")), self.FILENAME_LIMIT)
            self.assertLessEqual(len(doc["chunk_id"].encode("utf-8")), self.CHUNK_ID_LIMIT)


class ArabicPresentationFormTests(unittest.TestCase):
    """Many PDF producers emit Arabic as presentation-form glyphs (U+FB50-U+FEFF)
    rather than standard letters. NFC leaves them untouched, so such chunks embed
    and index as a different script from the same words typed normally — the
    query never matches the document."""

    def setUp(self):
        self.loader = DocumentLoader()

    def _load(self, blocks):
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            return self.loader.load_document("pres.pdf", "pres.pdf")

    def test_presentation_forms_normalize_to_standard_arabic(self):
        from backend.indexing.document_loader import sanitize_text

        # ﺍﻟﺮﺳﻮﻡ written with presentation-form glyphs.
        presentation = "ﺍﻟﺮﺳﻮﻡ"
        normalized = sanitize_text(presentation)
        self.assertEqual("الرسوم", normalized)

    def test_lam_alef_ligature_is_decomposed(self):
        from backend.indexing.document_loader import sanitize_text

        self.assertEqual("لا", sanitize_text("ﻻ"))

    def test_presentation_form_document_is_searchable_as_standard_arabic(self):
        presentation = "ﺍﻟﺮﺳﻮﻡ"  # الرسوم
        docs = self._load([_text_block(f"{presentation} مطلوبة للتسجيل.", 0, 10.0)])
        joined = " ".join(d["text"] for d in docs)
        self.assertIn("الرسوم", joined)

    def test_standard_arabic_is_left_unchanged(self):
        from backend.indexing.document_loader import sanitize_text

        original = "الرسوم الدراسية مطلوبة"
        self.assertEqual(original, sanitize_text(original))

    def test_latin_ligatures_and_fullwidth_are_not_touched(self):
        # Targeted NFKC must not leak into non-Arabic text.
        from backend.indexing.document_loader import sanitize_text

        self.assertEqual("ﬁle", sanitize_text("ﬁle"))   # ﬁle stays ﬁle
        self.assertEqual("ＡＢ", sanitize_text("ＡＢ"))  # ＡＢ unchanged


class StorageRoutingInvariantTests(unittest.TestCase):
    """The byte cap is applied to LEAVES only, because only leaves (level 3) are
    written to Milvus; levels 1-2 go to ParentChunk.text, a Postgres TEXT column
    with no length bound. This test pins that routing assumption: if a refactor
    ever sends parents to Milvus, the oversized-parent case must be revisited."""

    def test_only_leaves_are_byte_capped_and_parents_may_exceed(self):
        loader = DocumentLoader()
        rows = [["ق" * 40000, "ب" * 40000]]
        with patch.object(document_loader_module, "parse_pdf_blocks",
                          return_value=[_table_block(rows, 0, 10.0)]):
            docs = loader.load_document("big.pdf", "big.pdf")

        leaves = [d for d in docs if d["chunk_level"] == 3]
        parents = [d for d in docs if d["chunk_level"] in (1, 2)]
        self.assertTrue(leaves)
        self.assertTrue(parents)
        for leaf in leaves:
            self.assertLessEqual(len(leaf["text"].encode("utf-8")), 65535)
        # Parents intentionally keep full context (Postgres TEXT, unbounded).
        self.assertTrue(any(len(p["text"].encode("utf-8")) > 65535 for p in parents))

    def test_upload_route_partitioning_matches_this_assumption(self):
        # Mirrors backend/api/routes/documents.py: level 1-2 -> parent store,
        # level 3 -> Milvus. Guards against a silent change in either place.
        loader = DocumentLoader()
        with patch.object(document_loader_module, "parse_pdf_blocks",
                          return_value=[_text_block("محتوى تجريبي للاختبار.", 0, 10.0)]):
            docs = loader.load_document("r.pdf", "r.pdf")
        parent_docs = [d for d in docs if int(d.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [d for d in docs if int(d.get("chunk_level", 0) or 0) == 3]
        self.assertEqual(len(docs), len(parent_docs) + len(leaf_docs),
                         "a chunk level exists that no storage path claims")


class FitUtf8BytesTests(unittest.TestCase):
    """BVA on the byte-truncation helper itself."""

    def test_exact_fit_is_unchanged(self):
        from backend.indexing.document_loader import fit_utf8_bytes

        text = "قق"  # 4 bytes
        self.assertEqual(text, fit_utf8_bytes(text, 4))

    def test_one_byte_short_drops_the_last_whole_character(self):
        from backend.indexing.document_loader import fit_utf8_bytes

        self.assertEqual("ق", fit_utf8_bytes("قق", 3))

    def test_budget_smaller_than_one_character_yields_empty(self):
        from backend.indexing.document_loader import fit_utf8_bytes

        self.assertEqual("", fit_utf8_bytes("ق", 1))

    def test_empty_input_is_safe(self):
        from backend.indexing.document_loader import fit_utf8_bytes

        self.assertEqual("", fit_utf8_bytes("", 10))


class ArabicAcronymHeadingTests(unittest.TestCase):
    """`str.isupper()` is True for caseless text containing one Latin acronym.
    In a school corpus Arabic sentences are full of IGCSE / IB / SAT / IELTS /
    KG, so the all-caps heading rule would promote ordinary body sentences to
    section headings and corrupt the whole section stack."""

    def test_arabic_sentence_with_latin_acronym_is_not_a_heading(self):
        for text in (
            "يجب تقديم شهادة IELTS للتسجيل",
            "منهج IGCSE متاح لجميع الصفوف",
            "نتائج اختبار SAT مطلوبة",
        ):
            with self.subTest(text=text):
                self.assertTrue(text.isupper(), "precondition: isupper() is the trap")
                self.assertFalse(
                    is_heading_line(_line(text), body_size=10.0),
                    "Arabic body sentence promoted to heading by the all-caps rule",
                )

    def test_genuine_latin_all_caps_heading_still_detected(self):
        self.assertTrue(is_heading_line(_line("ADMISSION FEES"), body_size=10.0))

    def test_arabic_acronym_line_still_detected_when_bold(self):
        # A real Arabic heading mentioning an acronym must still work via bold/size.
        self.assertTrue(is_heading_line(_line("شهادة IGCSE", bold=True), body_size=10.0))

    def test_arabic_acronym_sentence_stays_body_text_end_to_end(self):
        loader = DocumentLoader()
        blocks = [
            {"type": "heading", "content": AR_HEADING_FEES, "level": 1, "page_number": 0, "top": 5.0},
            _text_block("يجب تقديم شهادة IELTS للتسجيل", 0, 20.0),
        ]
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks):
            docs = loader.load_document("ar.pdf", "ar.pdf")
        for doc in docs:
            self.assertTrue(doc["text"].startswith(AR_HEADING_FEES),
                            "acronym line became its own section")


class ArabicTatweelTests(unittest.TestCase):
    """Tatweel/kashida (U+0640) is decorative elongation with no meaning, and
    justified Arabic PDFs are full of it. Left in, "الرســوم" and "الرسوم" are
    different BM25 tokens and keyword retrieval silently misses."""

    def test_tatweel_is_stripped_from_indexed_text(self):
        from backend.text_normalization import sanitize_text

        self.assertEqual("الرسوم", sanitize_text("الرســـوم"))

    def test_tatweel_variants_converge_to_one_dedup_key(self):
        module = load_milvus_writer_module()
        events = []

        class _Service:
            def get_embeddings(self, texts):
                return [[1.0] for _ in texts]

        writer = module.MilvusWriter(embedding_service=_Service(), milvus_manager=FakeMilvusStore(events))
        loader = DocumentLoader()
        blocks_a = [_text_block("الرســـوم الدراسية مطلوبة للتسجيل.", 0, 10.0)]
        blocks_b = [_text_block("الرسوم الدراسية مطلوبة للتسجيل.", 0, 10.0)]
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks_a):
            docs_a = loader.load_document("a.pdf", "a.pdf")
        with patch.object(document_loader_module, "parse_pdf_blocks", return_value=blocks_b):
            docs_b = loader.load_document("b.pdf", "b.pdf")
        leaf_a = next(d for d in docs_a if d["chunk_level"] == 3)
        leaf_b = next(d for d in docs_b if d["chunk_level"] == 3)
        self.assertEqual(leaf_a["text"], leaf_b["text"])

    def test_tatweel_in_query_matches_indexed_form(self):
        from backend.text_normalization import normalize_query, sanitize_text

        self.assertEqual(sanitize_text("الرسوم"), normalize_query("الرســـوم"))


class QueryIndexSymmetryTests(unittest.TestCase):
    """Documents were normalized at index time but queries were not, so Arabic
    pasted from a PDF (presentation forms / tatweel / zero-width marks) could
    never match the chunk it came from."""

    def test_retrieve_documents_normalizes_the_query(self):
        import importlib.util
        import sys
        import types

        captured = {}

        fake_indexing = types.ModuleType("backend.indexing")
        fake_indexing.__path__ = []

        fake_milvus = types.ModuleType("backend.indexing.milvus_client")

        class _Store:
            def hybrid_retrieve(self, dense_embedding, query, top_k, filter_expr):
                captured["bm25_query"] = query
                return []

        fake_milvus.get_milvus_store = lambda: _Store()

        fake_embedding = types.ModuleType("backend.indexing.embedding")

        class _Embed:
            def get_embeddings(self, texts):
                captured["embedded"] = texts[0]
                return [[0.1, 0.2]]

        fake_embedding.embedding_service = _Embed()

        fake_parent = types.ModuleType("backend.indexing.parent_chunk_store")

        class ParentChunkStore:
            def get_documents_by_ids(self, ids):
                return []

        fake_parent.ParentChunkStore = ParentChunkStore

        spec = importlib.util.spec_from_file_location(
            "rag_utils_symmetry",
            REPO_ROOT / "backend" / "rag" / "utils.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "backend.indexing": fake_indexing,
            "backend.indexing.milvus_client": fake_milvus,
            "backend.indexing.embedding": fake_embedding,
            "backend.indexing.parent_chunk_store": fake_parent,
        }):
            spec.loader.exec_module(module)

        # Query as a user would paste it out of an Arabic PDF.
        module.retrieve_documents("ﺍﻟﺮﺳﻮﻡ الدراســية", top_k=3)

        self.assertEqual("الرسوم الدراسية", captured["embedded"],
                         "query not normalized before embedding")
        self.assertEqual("الرسوم الدراسية", captured["bm25_query"],
                         "query not normalized before BM25 search")

    def test_normalize_query_matches_document_sanitization(self):
        from backend.text_normalization import normalize_query, sanitize_text

        raw = "ﺍﻟﺮﺳﻮﻡ​الدراســية"
        self.assertEqual(sanitize_text(raw).strip(), normalize_query(raw))


class ArabicComplexityRoutingTests(unittest.TestCase):
    """The complexity fast-path had only Chinese/English markers. A short Arabic
    comparison question was fast-pathed as SIMPLE, skipping sub-question
    decomposition entirely — the retrieval quality feature silently did not
    exist for the agent's primary language."""

    def _reason(self, question):
        import backend.rag.pipeline as pipeline

        return pipeline._simple_question_fast_path_reason(question)

    def test_arabic_comparison_questions_are_not_fast_pathed(self):
        for question in (
            "قارن الرسوم",
            "ما الفرق بين الصفين؟",
            "مقارنة المناهج",
            "مميزات وعيوب النقل",
            "لماذا الرسوم؟",
            "كيف أسجل؟",
            "اشرح الشروط",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    self._reason(question),
                    "Arabic multi-part question was fast-pathed as simple",
                )

    def test_arabic_single_fact_questions_still_fast_path(self):
        for question in ("ما هي الرسوم؟", "كم الرسوم؟", "متى التسجيل؟", "أين المدرسة؟"):
            with self.subTest(question=question):
                self.assertIsNotNone(self._reason(question))

    def test_english_behavior_is_unchanged(self):
        self.assertIsNone(self._reason("compare fees"))
        self.assertIsNotNone(self._reason("What is the fee?"))

    def test_arabic_question_mark_is_stripped_by_the_length_rule(self):
        # Exactly at the boundary: 18 characters + "؟". If ؟ is not stripped the
        # question measures 19 and loses the fast path purely for its punctuation.
        question = "مواعيد تسجيل الطلب؟"
        self.assertEqual(18, len(question.rstrip("؟")))
        self.assertEqual(
            self._reason(question.rstrip("؟")),
            self._reason(question),
            "Arabic ؟ changed the fast-path verdict",
        )
        self.assertIsNotNone(self._reason(question))


class MilvusAnalyzerTests(unittest.TestCase):
    """The BM25 analyzer drives the sparse half of hybrid retrieval. A Chinese
    (jieba) analyzer tokenizes Arabic poorly; "standard" uses Unicode word
    boundaries and is correct for Arabic and English alike."""

    def test_default_analyzer_is_not_chinese(self):
        import importlib

        import backend.indexing.milvus_client as milvus_client

        importlib.reload(milvus_client)
        self.assertEqual("standard", milvus_client.TEXT_ANALYZER_TYPE)

    def test_analyzer_is_configurable(self):
        import importlib
        import os as _os

        import backend.indexing.milvus_client as milvus_client

        with patch.dict(_os.environ, {"MILVUS_TEXT_ANALYZER": "chinese"}):
            importlib.reload(milvus_client)
            self.assertEqual("chinese", milvus_client.TEXT_ANALYZER_TYPE)
        importlib.reload(milvus_client)

    def test_schema_uses_the_configured_analyzer(self):
        import inspect

        import backend.indexing.milvus_client as milvus_client

        source = inspect.getsource(milvus_client.MilvusStore.ensure_collection)
        self.assertIn('analyzer_params={"type": TEXT_ANALYZER_TYPE}', source)
        self.assertNotIn('"type": "chinese"', source)


if __name__ == "__main__":
    unittest.main()
