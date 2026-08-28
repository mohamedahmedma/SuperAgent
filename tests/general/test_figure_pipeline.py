"""Phase 3: triage, extraction, and the path from a document image to a chunk.

The end-to-end assertion that matters is the last class: an image block entering the
layout pipeline must come out as a leaf chunk whose text is searchable and which
still points at the image that produced it.
"""
import base64
import io
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.assets.blobs import LocalBlobStore
from backend.assets.dossier import AssetRole, AssetTier, ExtractionStatus, compute_sha256
from backend.assets.extractors import (
    CAPTION_PREFIX_RE,
    ExtractionRequest,
    FigureExtraction,
    HeuristicExtractor,
    VisionExtractor,
    build_extractor,
    downscale_image,
)
from backend.assets.pipeline import FigurePipeline, ImageInput
from backend.assets.store import AssetStore
from backend.assets.triage import ImageFacts, count_digest_pages, probe_dimensions, triage_image
from backend.indexing.asset_enrichment import enrich_image_blocks
from backend.profiles.registry import load_profile


def make_png(width=200, height=200, seed=1) -> bytes:
    """A real PNG of deterministic pseudo-random noise.

    Noise rather than a flat colour on purpose: a solid-colour PNG compresses to a
    few hundred bytes and is dropped by triage's min_byte_size rule, which made an
    earlier version of these tests silently exercise nothing at all. Seeded so the
    same arguments always yield byte-identical output, which the dedup and
    page-furniture tests depend on.
    """
    from PIL import Image

    data = random.Random(seed).randbytes(width * height * 3)
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), data).save(buffer, format="PNG")
    return buffer.getvalue()


def triage_config(**overrides):
    config = load_profile("base").assets.triage
    return config.model_copy(update=overrides) if overrides else config


def figures_config(**overrides):
    config = load_profile("base").assets.figures
    return config.model_copy(update=overrides) if overrides else config


class TriageTests(unittest.TestCase):
    def _facts(self, **overrides):
        base = dict(sha256="a" * 64, byte_size=50_000, width=400, height=300)
        base.update(overrides)
        return ImageFacts(**base)

    def test_a_normal_figure_survives(self):
        result = triage_image(self._facts(), triage_config())
        self.assertFalse(result.is_dropped)
        self.assertEqual(AssetRole.FIGURE, result.role)
        self.assertEqual(AssetTier.SIMPLE, result.tier)

    def test_tiny_images_are_dropped(self):
        result = triage_image(self._facts(width=16, height=16), triage_config())
        self.assertTrue(result.is_dropped)
        self.assertEqual(AssetRole.DECORATIVE, result.role)
        self.assertIn("smaller_than", result.reason)

    def test_small_area_is_dropped(self):
        result = triage_image(self._facts(width=80, height=80), triage_config())
        self.assertTrue(result.is_dropped)
        self.assertIn("area<", result.reason)

    def test_hairline_rules_are_dropped_by_aspect_ratio(self):
        result = triage_image(self._facts(width=2000, height=70), triage_config())
        self.assertTrue(result.is_dropped)
        self.assertIn("aspect_ratio", result.reason)

    def test_tiny_byte_payloads_are_dropped_first(self):
        result = triage_image(self._facts(byte_size=100), triage_config())
        self.assertTrue(result.is_dropped)
        self.assertIn("byte_size", result.reason)

    def test_html_declared_decorative_is_honoured(self):
        result = triage_image(self._facts(declared_decorative=True), triage_config())
        self.assertTrue(result.is_dropped)
        self.assertEqual("declared_decorative", result.reason)

    def test_page_furniture_is_dropped_however_large(self):
        """A letterhead is furniture at any size — the rule runs before the size rules."""
        result = triage_image(
            self._facts(width=1200, height=900, pages_with_digest=8, total_pages=10),
            triage_config(),
        )
        self.assertTrue(result.is_dropped)
        self.assertIn("page_furniture", result.reason)

    def test_an_image_on_a_minority_of_pages_is_not_furniture(self):
        result = triage_image(
            self._facts(pages_with_digest=2, total_pages=10), triage_config()
        )
        self.assertFalse(result.is_dropped)

    def test_a_single_page_document_cannot_establish_repetition(self):
        result = triage_image(
            self._facts(pages_with_digest=1, total_pages=1), triage_config()
        )
        self.assertFalse(result.is_dropped)

    def test_unknown_dimensions_do_not_drop_the_image(self):
        """An unknown size must not be read as zero — that would silently drop every
        image from a format that does not expose dimensions cheaply."""
        result = triage_image(self._facts(width=0, height=0), triage_config())
        self.assertFalse(result.is_dropped)
        self.assertEqual(AssetTier.SIMPLE, result.tier)

    def test_large_images_are_tiered_complex(self):
        result = triage_image(self._facts(width=1200, height=900), triage_config())
        self.assertEqual(AssetTier.COMPLEX, result.tier)
        self.assertFalse(result.is_dropped)

    def test_thresholds_are_profile_driven(self):
        strict = triage_config(min_width=1000, min_height=1000)
        self.assertTrue(triage_image(self._facts(), strict).is_dropped)

    def test_digest_page_counting(self):
        counts, pages = count_digest_pages([
            {"sha256": "a", "page_number": 0},
            {"sha256": "a", "page_number": 1},
            {"sha256": "a", "page_number": 1},
            {"sha256": "b", "page_number": 2},
        ])
        self.assertEqual({"a": 2, "b": 1}, counts)
        self.assertEqual(3, pages)

    def test_digest_page_counting_on_an_empty_document(self):
        self.assertEqual(({}, 1), count_digest_pages([]))

    def test_probe_dimensions_reads_a_real_png_and_rejects_junk(self):
        self.assertEqual((320, 240), probe_dimensions(make_png(320, 240)))
        self.assertIsNone(probe_dimensions(b"not an image"))


class CaptionHeuristicTests(unittest.TestCase):
    def test_caption_markers_are_recognised_across_scripts(self):
        for text in ["Figure 3: Tuition by grade", "Fig. 2 — Enrolment",
                     "Table 4. Fees", "الشكل ٢: الرسوم الدراسية", "جدول 1 - المواعيد"]:
            with self.subTest(text=text):
                self.assertIsNotNone(CAPTION_PREFIX_RE.match(text))

    def test_ordinary_prose_is_not_a_caption(self):
        self.assertIsNone(CAPTION_PREFIX_RE.match("The following section explains fees."))

    def test_a_marked_caption_after_the_image_wins_over_the_text_before(self):
        request = ExtractionRequest(
            data=b"x",
            text_before="Some unrelated paragraph.",
            text_after="Figure 3: Tuition by grade",
        )
        self.assertEqual("Figure 3: Tuition by grade", request.caption_candidate())

    def test_alt_text_is_preferred_when_present(self):
        request = ExtractionRequest(data=b"x", alt_text="Org chart", text_after="Unrelated prose")
        self.assertEqual("Org chart", request.caption_candidate())

    def test_no_signal_yields_no_caption(self):
        request = ExtractionRequest(data=b"x", text_after="x" * 500)
        self.assertEqual("", request.caption_candidate())


class HeuristicExtractorTests(unittest.TestCase):
    def test_caption_marker_is_stripped_but_the_words_are_kept(self):
        """'Figure 3' is not a retrieval signal; the rest of the line is."""
        payload = HeuristicExtractor(figures_config()).extract(
            ExtractionRequest(data=b"x", text_after="Figure 3: Tuition by grade")
        )
        self.assertEqual("Tuition by grade", payload.text.caption)

    def test_section_path_becomes_tags(self):
        payload = HeuristicExtractor(figures_config()).extract(
            ExtractionRequest(data=b"x", alt_text="Fees", section_path=["Handbook", "Admissions", "Fees"])
        )
        self.assertEqual(["Admissions", "Fees"], payload.text.tags)

    def test_nothing_is_invented_when_there_is_no_signal(self):
        """An empty surface is honest and correctly non-indexable; a fabricated one
        would pollute recall permanently."""
        payload = HeuristicExtractor(figures_config()).extract(ExtractionRequest(data=b"x"))
        self.assertTrue(payload.text.is_empty())
        self.assertTrue(payload.provenance.needs_review)
        self.assertEqual(0.0, payload.provenance.confidence)

    def test_confidence_stays_low_so_assets_are_re_extraction_candidates(self):
        payload = HeuristicExtractor(figures_config()).extract(
            ExtractionRequest(data=b"x", alt_text="Org chart")
        )
        self.assertLess(payload.provenance.confidence, 0.5)
        self.assertEqual("heuristic", payload.provenance.model_used)


class VisionExtractorTests(unittest.TestCase):
    def _extractor(self, result, **config_overrides):
        extractor = VisionExtractor(
            figures_config(**config_overrides), model_id="vl-test", api_key="k", base_url="u"
        )
        structured = Mock()
        structured.invoke.return_value = result
        model = Mock()
        model.with_structured_output.return_value = structured
        extractor._model = model
        return extractor, structured

    def test_structured_output_maps_onto_the_text_surface(self):
        extractor, _ = self._extractor(FigureExtraction(
            caption="Tuition by grade",
            description="Bar chart of fees.",
            transcription="| Grade | Fee |\n| 5 | 42000 |",
            tags=["fees"],
            image_type="chart",
            answerable_questions=["How much is grade 5?"],
            confidence=0.9,
        ))
        payload = extractor.extract(ExtractionRequest(data=make_png(), tier=AssetTier.COMPLEX))

        self.assertEqual("Tuition by grade", payload.text.caption)
        self.assertIn("42000", payload.text.transcription)
        self.assertIn("chart", payload.text.tags)
        self.assertEqual(["How much is grade 5?"], payload.answerable_questions)
        self.assertEqual("chart", payload.structured.attributes["image_type"])
        self.assertFalse(payload.provenance.needs_review)

    def test_low_confidence_flags_the_asset_for_review(self):
        extractor, _ = self._extractor(FigureExtraction(caption="blurry", confidence=0.1))
        payload = extractor.extract(ExtractionRequest(data=make_png()))
        self.assertTrue(payload.provenance.needs_review)

    def test_tags_and_questions_are_capped_by_the_profile(self):
        extractor, _ = self._extractor(
            FigureExtraction(
                caption="c",
                tags=[f"t{i}" for i in range(20)],
                answerable_questions=[f"q{i}" for i in range(20)],
                confidence=0.9,
            ),
            max_tags=3,
            max_answerable_questions=2,
        )
        payload = extractor.extract(ExtractionRequest(data=make_png()))
        self.assertEqual(4, len(payload.text.tags))  # 3 capped + the image_type tag
        self.assertEqual(2, len(payload.answerable_questions))

    def test_the_image_is_sent_as_a_data_uri_alongside_the_prompt(self):
        extractor, structured = self._extractor(FigureExtraction(caption="c", confidence=0.8))
        extractor.extract(ExtractionRequest(data=make_png(), content_type="image/png"))

        content = structured.invoke.call_args[0][0][0]["content"]
        self.assertEqual("text", content[0]["type"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_context_is_woven_into_the_prompt(self):
        extractor, structured = self._extractor(FigureExtraction(caption="c", confidence=0.8))
        extractor.extract(ExtractionRequest(
            data=make_png(),
            section_path=["Handbook", "Fees"],
            text_after="Figure 3: Tuition",
            filename="doc.pdf",
            page_number=4,
        ))
        prompt = structured.invoke.call_args[0][0][0]["content"][0]["text"]
        self.assertIn("Handbook > Fees", prompt)
        self.assertIn("Figure 3: Tuition", prompt)
        self.assertIn("doc.pdf page 4", prompt)


class DownscaleTests(unittest.TestCase):
    def test_large_images_are_shrunk_to_the_max_edge(self):
        shrunk = downscale_image(make_png(1000, 500), max_edge=250)
        self.assertEqual((250, 125), probe_dimensions(shrunk))

    def test_small_images_are_returned_untouched(self):
        original = make_png(100, 100)
        self.assertIs(original, downscale_image(original, max_edge=500))

    def test_undecodable_bytes_are_returned_untouched(self):
        self.assertEqual(b"junk", downscale_image(b"junk", max_edge=500))


class BuildExtractorTests(unittest.TestCase):
    def test_vision_disabled_yields_the_heuristic_extractor(self):
        self.assertIsInstance(build_extractor(figures_config(vision_enabled=False)), HeuristicExtractor)

    def test_missing_credentials_fall_back_to_heuristic(self):
        with patch.dict("os.environ", {"VISION_MODEL": "", "MODEL": "", "ARK_API_KEY": ""}):
            extractor = build_extractor(figures_config(vision_enabled=True))
        self.assertIsInstance(extractor, HeuristicExtractor)

    def test_configured_credentials_yield_the_vision_extractor(self):
        with patch.dict("os.environ", {"VISION_MODEL": "vl-1", "ARK_API_KEY": "k", "BASE_URL": "u"}):
            extractor = build_extractor(figures_config(vision_enabled=True))
        self.assertIsInstance(extractor, VisionExtractor)


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        from backend.db.models import AssetExtraction, DocumentAsset

        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        DocumentAsset.__table__.create(self.engine)
        AssetExtraction.__table__.create(self.engine)
        session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

        self._tmp = TemporaryDirectory()
        self.blobs = LocalBlobStore(Path(self._tmp.name))
        self.store = AssetStore(session_factory=session_factory, blob_store=self.blobs, cache_enabled=False)
        self.profile = load_profile("base")

    def tearDown(self):
        self._tmp.cleanup()
        self.engine.dispose()

    def pipeline(self, extractor=None, fallback=None):
        return FigurePipeline(
            profile=self.profile,
            store=self.store,
            blob_store=self.blobs,
            extractor=extractor or HeuristicExtractor(self.profile.assets.figures),
            fallback_extractor=fallback or HeuristicExtractor(self.profile.assets.figures),
        )

    @staticmethod
    def image(index=0, page=0, width=400, height=300, **kwargs):
        return ImageInput(
            data=make_png(width, height, seed=index),
            page_number=page,
            index=index,
            **kwargs,
        )


class FigurePipelineTests(PipelineTestCase):
    def test_a_surviving_figure_is_stored_extracted_and_indexable(self):
        dossiers, report = self.pipeline().process(
            [self.image(text_after="Figure 1: Fee schedule")], filename="doc.pdf"
        )
        dossier = dossiers[0]
        self.assertEqual(ExtractionStatus.EXTRACTED, dossier.status)
        self.assertTrue(dossier.is_indexable)
        self.assertTrue(dossier.blob.uri)
        self.assertEqual(1, report.extracted)
        self.assertEqual(1, report.indexable)
        self.assertIsNotNone(self.store.get(dossier.asset_id))

    def test_triaged_out_images_are_recorded_with_a_reason_but_never_stored(self):
        """Recording the rejection is what makes a threshold tunable after the fact;
        not storing the blob is what stops a logo earning disk space."""
        dossiers, report = self.pipeline().process([self.image(width=40, height=40)], filename="doc.pdf")
        dossier = dossiers[0]
        self.assertEqual(ExtractionStatus.SKIPPED, dossier.status)
        self.assertEqual(AssetRole.DECORATIVE, dossier.role)
        self.assertFalse(dossier.is_indexable)
        self.assertEqual("", dossier.blob.uri)
        self.assertIn("smaller_than", dossier.extraction.provenance.error)
        self.assertEqual(1, report.dropped)
        self.assertEqual([], list(Path(self._tmp.name).rglob("*.png")))

    def test_a_repeated_logo_is_triaged_as_furniture_across_pages(self):
        logo = make_png(300, 300, seed=42)
        images = [ImageInput(data=logo, page_number=page, index=page) for page in range(5)]
        _, report = self.pipeline().process(images, filename="doc.pdf")
        self.assertEqual(5, report.dropped)
        self.assertEqual(0, report.indexable)

    def test_a_repeat_image_is_served_from_the_extraction_cache(self):
        """The second occurrence of identical bytes must not reach an extractor."""
        shared = make_png(400, 300, seed=7)
        counting = Mock(wraps=HeuristicExtractor(self.profile.assets.figures))
        # The pipeline reads .name to decide whether a cached extraction was produced
        # by a weaker extractor; a bare Mock would look vision-capable.
        counting.name = "heuristic"
        counting.extract.side_effect = HeuristicExtractor(self.profile.assets.figures).extract

        pipeline = self.pipeline(extractor=counting)
        pipeline.process(
            [ImageInput(data=shared, page_number=0, index=0, alt_text="Org chart")],
            filename="a.pdf",
        )
        _, report = pipeline.process(
            [ImageInput(data=shared, page_number=0, index=0, alt_text="Org chart")],
            filename="b.pdf",
        )

        self.assertEqual(1, counting.extract.call_count)
        self.assertEqual(1, report.cached)
        self.assertEqual(0, report.extracted)

    def test_a_heuristic_cache_entry_does_not_block_a_vision_upgrade(self):
        """Turning vision on must actually take effect. A cached heuristic extraction
        is a valid entry while heuristic is all there is, but once a vision extractor
        is configured it is strictly worse for the same bytes."""
        from backend.assets.dossier import ExtractionPayload, Provenance, TextSurface

        shared = make_png(400, 300, seed=11)
        heuristic = self.pipeline()
        heuristic.process([ImageInput(data=shared, index=0, alt_text="Timetable")], filename="a.pdf")

        vision = Mock()
        vision.name = "vision"
        vision.extract.return_value = ExtractionPayload(
            text=TextSurface(caption="Weekly timetable", transcription="Mon 08:00 Maths"),
            provenance=Provenance(model_used="vl-test", confidence=0.9),
        )
        upgraded = self.pipeline(extractor=vision)
        dossiers, report = upgraded.process(
            [ImageInput(data=shared, index=0, alt_text="Timetable")], filename="b.pdf"
        )

        vision.extract.assert_called_once()
        self.assertEqual(0, report.cached)
        self.assertEqual(1, report.extracted)
        self.assertIn("Mon 08:00 Maths", dossiers[0].extraction.text.transcription)

    def test_a_vision_cache_entry_is_reused_by_a_vision_extractor(self):
        """Only an UPGRADE bypasses the cache; same-tier repeats still hit it."""
        from backend.assets.dossier import ExtractionPayload, Provenance, TextSurface

        shared = make_png(400, 300, seed=12)
        vision = Mock()
        vision.name = "vision"
        vision.extract.return_value = ExtractionPayload(
            text=TextSurface(caption="Weekly timetable"),
            provenance=Provenance(model_used="vl-test", confidence=0.9),
        )
        pipeline = self.pipeline(extractor=vision)
        pipeline.process([ImageInput(data=shared, index=0)], filename="a.pdf")
        _, report = pipeline.process([ImageInput(data=shared, index=0)], filename="b.pdf")

        self.assertEqual(1, vision.extract.call_count)
        self.assertEqual(1, report.cached)

    def test_extractor_failure_falls_back_instead_of_losing_the_asset(self):
        failing = Mock()
        failing.extract.side_effect = RuntimeError("vision endpoint down")
        dossiers, report = self.pipeline(extractor=failing).process(
            [self.image(alt_text="Org chart")], filename="doc.pdf"
        )
        self.assertEqual(ExtractionStatus.EXTRACTED, dossiers[0].status)
        self.assertEqual("Org chart", dossiers[0].extraction.text.caption)
        self.assertEqual(1, report.extracted)

    def test_both_extractors_failing_records_the_asset_as_failed(self):
        failing = Mock()
        failing.extract.side_effect = RuntimeError("down")
        broken_fallback = Mock()
        broken_fallback.extract.side_effect = RuntimeError("also down")

        dossiers, report = self.pipeline(extractor=failing, fallback=broken_fallback).process(
            [self.image()], filename="doc.pdf"
        )
        self.assertEqual(ExtractionStatus.FAILED, dossiers[0].status)
        self.assertFalse(dossiers[0].is_indexable)
        self.assertEqual(1, report.failed)

    def test_documents_are_capped_at_the_profile_image_limit(self):
        self.profile = self.profile.model_copy(deep=True)
        self.profile.assets.max_images_per_document = 3
        images = [self.image(index=i, page=i) for i in range(10)]
        dossiers, report = self.pipeline().process(images, filename="doc.pdf")
        self.assertEqual(3, len(dossiers))
        self.assertEqual(7, report.skipped_over_limit)

    def test_an_empty_document_is_a_noop(self):
        dossiers, report = self.pipeline().process([], filename="doc.pdf")
        self.assertEqual([], dossiers)
        self.assertEqual(0, report.total)

    def test_source_reference_is_recorded_for_citations(self):
        dossiers, _ = self.pipeline().process(
            [self.image(page=4, bbox=[1.0, 2.0, 3.0, 4.0], alt_text="Chart")],
            filename="doc.pdf",
            file_path="/data/doc.pdf",
        )
        source = dossiers[0].source
        self.assertEqual(("doc.pdf", "/data/doc.pdf", 4), (source.filename, source.file_path, source.page_number))
        self.assertEqual([1.0, 2.0, 3.0, 4.0], source.bbox)


class EnrichmentStageTests(PipelineTestCase):
    def _blocks(self):
        return [
            {"type": "heading", "content": "Admissions", "level": 1, "page_number": 0, "top": 0.0},
            {"type": "text", "content": "Fees are set annually.", "page_number": 0, "top": 1.0},
            {"type": "image", "data": make_png(), "content_type": "image/png",
             "page_number": 0, "top": 2.0},
            {"type": "text", "content": "Figure 1: Tuition by grade", "page_number": 0, "top": 3.0},
        ]

    def test_an_image_block_becomes_a_text_block_carrying_its_asset_id(self):
        enriched, report = enrich_image_blocks(
            self._blocks(), filename="doc.pdf", pipeline=self.pipeline()
        )
        figure_blocks = [b for b in enriched if b.get("asset_ids")]
        self.assertEqual(1, len(figure_blocks))
        self.assertEqual("text", figure_blocks[0]["type"])
        self.assertIn("Tuition by grade", figure_blocks[0]["content"])
        self.assertEqual(1, report.indexable)

    def test_the_caption_line_after_the_image_is_used_as_context(self):
        enriched, _ = enrich_image_blocks(
            self._blocks(), filename="doc.pdf", pipeline=self.pipeline()
        )
        surrogate = next(b["content"] for b in enriched if b.get("asset_ids"))
        # No section prefix here: the surrogate is a BLOCK, and the chunking pipeline
        # applies the section path itself. Adding it at both layers would duplicate it.
        self.assertTrue(surrogate.startswith("[Figure] Tuition by grade"))

    def test_documents_without_images_pass_through_untouched(self):
        blocks = [{"type": "text", "content": "Only prose", "page_number": 0, "top": 0.0}]
        enriched, report = enrich_image_blocks(blocks, filename="doc.pdf", pipeline=self.pipeline())
        self.assertIs(blocks, enriched)
        self.assertEqual(0, report.total)

    def test_non_indexable_images_are_removed_from_the_stream(self):
        blocks = [{"type": "image", "data": make_png(40, 40), "page_number": 0, "top": 0.0}]
        enriched, report = enrich_image_blocks(blocks, filename="doc.pdf", pipeline=self.pipeline())
        self.assertEqual([], enriched)
        self.assertEqual(1, report.dropped)

    def test_a_pipeline_failure_still_indexes_the_document_text(self):
        """Enrichment is additive: losing figures beats losing the document."""
        exploding = Mock()
        exploding.process.side_effect = RuntimeError("pipeline down")
        enriched, report = enrich_image_blocks(self._blocks(), filename="doc.pdf", pipeline=exploding)
        self.assertEqual(3, len(enriched))
        self.assertTrue(all(block["type"] != "image" for block in enriched))
        self.assertEqual(0, report.total)

    def test_section_path_is_tracked_across_headings(self):
        blocks = [
            {"type": "heading", "content": "Handbook", "level": 1, "page_number": 0, "top": 0.0},
            {"type": "heading", "content": "Fees", "level": 2, "page_number": 0, "top": 1.0},
            {"type": "image", "data": make_png(), "page_number": 0, "top": 2.0},
            {"type": "text", "content": "Figure 1: Fee table", "page_number": 0, "top": 3.0},
        ]
        enriched, _ = enrich_image_blocks(blocks, filename="doc.pdf", pipeline=self.pipeline())
        surrogate = next(b["content"] for b in enriched if b.get("asset_ids"))
        # The section path reaches the EXTRACTOR as context; the heuristic extractor
        # surfaces it as tags, which is what makes it retrievable.
        self.assertIn("Tags: Handbook, Fees", surrogate)

    def test_image_blocks_without_bytes_are_dropped(self):
        blocks = [{"type": "image", "page_number": 0, "top": 0.0}]
        enriched, _ = enrich_image_blocks(blocks, filename="doc.pdf", pipeline=self.pipeline())
        self.assertEqual([], enriched)


class ChunkIntegrationTests(PipelineTestCase):
    """The end-to-end claim: a document image becomes a searchable leaf chunk that
    still points at its image."""

    def _chunks(self):
        from backend.indexing.document_loader import DocumentLoader

        blocks = [
            {"type": "heading", "content": "Admissions", "level": 1, "page_number": 0, "top": 0.0},
            {"type": "text", "content": "Tuition is reviewed each year. " * 20, "page_number": 0, "top": 1.0},
            {"type": "image", "data": make_png(), "content_type": "image/png",
             "page_number": 0, "top": 2.0},
            {"type": "text", "content": "Figure 1: Tuition by grade", "page_number": 0, "top": 3.0},
        ]
        loader = DocumentLoader()
        with patch.object(DocumentLoader, "_enrich_assets", staticmethod(
            lambda blocks, filename, file_path: enrich_image_blocks(
                blocks, filename=filename, file_path=file_path, pipeline=self.pipeline()
            )[0]
        )):
            return loader._load_blocks_with_layout(blocks, "/data/doc.pdf", "doc.pdf", "PDF")

    def test_a_figure_produces_a_leaf_chunk_tagged_and_linked(self):
        chunks = self._chunks()
        leaves = [c for c in chunks if c["chunk_level"] == 3]
        figures = [c for c in leaves if c.get("modality") == "figure"]

        self.assertEqual(1, len(figures))
        self.assertIn("Tuition by grade", figures[0]["text"])
        self.assertEqual(1, len(figures[0]["asset_ids"]))
        self.assertTrue(figures[0]["asset_ids"][0].startswith("doc.pdf::p0::img"))

    def test_a_figure_leaf_is_isolated_from_surrounding_prose(self):
        """Leaf isolation keeps the figure's embedding about the figure."""
        figures = [c for c in self._chunks() if c.get("modality") == "figure" and c["chunk_level"] == 3]
        self.assertNotIn("reviewed each year", figures[0]["text"])

    def test_parents_inherit_the_asset_reference(self):
        """Auto-merging a figure hit to its parent must not lose the image."""
        chunks = self._chunks()
        parents = [c for c in chunks if c["chunk_level"] in (1, 2) and c.get("asset_ids")]
        self.assertTrue(parents)
        self.assertEqual("figure", parents[0]["modality"])

    def test_the_section_topic_appears_exactly_once_in_a_figure_chunk(self):
        """Guards the layering. The surrogate omits the section path because the
        chunker applies it — and _apply_section_prefix additionally skips when the
        topic is already near the top of the text (the heuristic extractor puts it in
        tags). Either route is fine; two copies would not be."""
        figures = [c for c in self._chunks() if c.get("modality") == "figure" and c["chunk_level"] == 3]
        self.assertEqual(1, figures[0]["text"].count("Admissions"))

    def test_plain_text_chunks_stay_text_modality_with_no_assets(self):
        prose = [c for c in self._chunks() if c["chunk_level"] == 3 and c.get("modality") == "text"]
        self.assertTrue(prose)
        self.assertEqual([], prose[0]["asset_ids"])


class HtmlImageParsingTests(unittest.TestCase):
    def _parse(self, html: str, extra_files: dict | None = None):
        from backend.indexing.html_layout import parse_html_blocks

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name, payload in (extra_files or {}).items():
                (path / name).write_bytes(payload)
            page = path / "page.html"
            page.write_text(html, encoding="utf-8")
            return parse_html_blocks(str(page))

    def test_a_data_uri_image_becomes_an_image_block(self):
        encoded = base64.b64encode(make_png()).decode()
        blocks = self._parse(f'<p>Intro</p><img src="data:image/png;base64,{encoded}" alt="Org chart">')
        images = [b for b in blocks if b["type"] == "image"]
        self.assertEqual(1, len(images))
        self.assertEqual("Org chart", images[0]["alt_text"])
        self.assertEqual(compute_sha256(make_png()), compute_sha256(images[0]["data"]))

    def test_a_local_relative_image_is_loaded(self):
        blocks = self._parse('<img src="chart.png" alt="Chart">', {"chart.png": make_png()})
        self.assertEqual(1, len([b for b in blocks if b["type"] == "image"]))

    def test_a_remote_image_is_never_fetched_but_keeps_its_alt_text(self):
        """Dereferencing a URL from an uploaded document would make ingest a
        server-side request forge."""
        blocks = self._parse('<img src="https://example.com/x.png" alt="Remote chart">')
        self.assertEqual([], [b for b in blocks if b["type"] == "image"])
        self.assertIn("Remote chart", [b.get("content") for b in blocks])

    def test_a_path_traversal_src_is_refused(self):
        blocks = self._parse('<img src="../../../etc/passwd" alt="evil">')
        self.assertEqual([], [b for b in blocks if b["type"] == "image"])

    def test_an_empty_alt_marks_the_image_decorative(self):
        encoded = base64.b64encode(make_png()).decode()
        blocks = self._parse(f'<img src="data:image/png;base64,{encoded}" alt="">')
        images = [b for b in blocks if b["type"] == "image"]
        self.assertTrue(images[0]["declared_decorative"])

    def test_a_missing_alt_is_not_declared_decorative(self):
        encoded = base64.b64encode(make_png()).decode()
        blocks = self._parse(f'<img src="data:image/png;base64,{encoded}">')
        self.assertFalse([b for b in blocks if b["type"] == "image"][0]["declared_decorative"])

    def test_malformed_data_uris_are_ignored(self):
        blocks = self._parse('<img src="data:image/png;base64,!!!not-base64!!!" alt="broken">')
        self.assertEqual([], [b for b in blocks if b["type"] == "image"])

    def test_figure_wrappers_do_not_duplicate_their_caption(self):
        encoded = base64.b64encode(make_png()).decode()
        blocks = self._parse(
            f'<figure><img src="data:image/png;base64,{encoded}" alt="Chart">'
            f"<figcaption>Figure 1: Fees</figcaption></figure>"
        )
        self.assertEqual(1, len([b for b in blocks if b["type"] == "image"]))


if __name__ == "__main__":
    unittest.main()
