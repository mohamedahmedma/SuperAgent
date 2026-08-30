"""Phase 5: attribute schema, entity extraction, the attribute index, and
context-first retrieval.

The heaviest class is `DynamicSchemaTests`. The requirement is that a new domain is a
YAML block rather than a code change, and the way that is enforced is by building the
extraction schema, the tool signature, filter validation, and the index from one
declaration — then asserting all four track a vocabulary this file invents at runtime.
"""
import io
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.assets.attributes import (
    AttributeSchema,
    AttributeSpec,
    AttributeType,
    NumberRange,
    build_attribute_schema,
)
from backend.assets.blobs import LocalBlobStore
from backend.assets.dossier import AssetRole, ExtractionStatus
from backend.assets.entity_extractor import (
    HeuristicEntityExtractor,
    VisionEntityExtractor,
    build_entity_extractor,
    build_entity_model,
    resolve_role,
)
from backend.assets.entity_store import EntityAttributeIndex
from backend.assets.extractors import ExtractionRequest
from backend.assets.pipeline import FigurePipeline, ImageInput
from backend.assets.store import AssetStore
from backend.profiles.registry import load_profile
from backend.rag.entity_retrieval import EntityRetriever


def make_png(width=400, height=300, seed=1) -> bytes:
    from PIL import Image

    data = random.Random(seed).randbytes(width * height * 3)
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), data).save(buffer, format="PNG")
    return buffer.getvalue()


def shop_schema() -> AttributeSchema:
    return build_attribute_schema(load_profile("ecommerce").assets.entities)


def custom_schema(*specs) -> AttributeSchema:
    return AttributeSchema([AttributeSpec(**spec) for spec in specs])


class AttributeSpecTests(unittest.TestCase):
    def test_names_must_be_safe_identifiers(self):
        for bad in ("Color", "shoe size", "1st", "colour-name", ""):
            with self.subTest(name=bad):
                with self.assertRaises(Exception):
                    AttributeSpec(name=bad)

    def test_prompt_lines_carry_type_vocabulary_and_unit(self):
        spec = AttributeSpec(name="price", type="number", unit="EGP", description="Listed price")
        line = spec.prompt_line()
        self.assertIn("price", line)
        self.assertIn("number", line)
        self.assertIn("EGP", line)
        self.assertIn("Listed price", line)

        multi = AttributeSpec(name="color", multi=True, values=["red", "blue"])
        self.assertIn("one or more", multi.prompt_line())
        self.assertIn("red, blue", multi.prompt_line())

    def test_a_duplicate_attribute_is_rejected(self):
        with self.assertRaises(ValueError):
            custom_schema({"name": "color"}, {"name": "color"})


class DynamicSchemaTests(unittest.TestCase):
    """One declaration, four derived artifacts — proven against a vocabulary invented
    here, so nothing can be passing by virtue of hardcoded shop knowledge."""

    def setUp(self):
        self.schema = custom_schema(
            {"name": "voltage", "type": "number", "unit": "V"},
            {"name": "connector", "type": "string", "multi": True,
             "values": ["usb_c", "xlr", "jack"]},
            {"name": "certified", "type": "boolean"},
            {"name": "serial", "type": "string", "filterable": False},
        )

    def test_the_extraction_model_mirrors_the_declaration(self):
        model = self.schema.build_extraction_model()
        self.assertEqual({"voltage", "connector", "certified", "serial"}, set(model.model_fields))
        # Every field optional: an extractor that cannot see a value must omit it.
        instance = model()
        self.assertIsNone(instance.voltage)
        self.assertEqual([], instance.connector)

    def test_the_filter_model_exposes_only_filterable_attributes(self):
        model = self.schema.build_filter_model()
        self.assertIn("voltage", model.model_fields)
        self.assertNotIn("serial", model.model_fields)

    def test_filter_types_follow_attribute_types(self):
        model = self.schema.build_filter_model()
        instance = model(voltage={"min": 5, "max": 12}, connector=["usb_c"], certified=True)
        self.assertIsInstance(instance.voltage, NumberRange)
        self.assertEqual(["usb_c"], instance.connector)
        self.assertIs(True, instance.certified)

    def test_the_filter_model_rejects_an_invented_facet(self):
        """The agent cannot name a filter the index does not have."""
        model = self.schema.build_filter_model()
        with self.assertRaises(Exception):
            model(bluetooth=True)

    def test_closed_vocabularies_reach_the_tool_description(self):
        schema_json = self.schema.build_filter_model().model_json_schema()
        self.assertIn("usb_c", schema_json["properties"]["connector"]["description"])

    def test_the_prompt_describes_every_attribute(self):
        described = self.schema.describe()
        for name in ("voltage", "connector", "certified", "serial"):
            with self.subTest(name=name):
                self.assertIn(name, described)

    def test_an_empty_vocabulary_produces_empty_but_valid_models(self):
        empty = AttributeSchema([])
        self.assertFalse(empty)
        self.assertEqual({}, empty.build_extraction_model()().model_dump())
        self.assertEqual({}, empty.build_filter_model()().model_dump())


class NormalisationTests(unittest.TestCase):
    def setUp(self):
        self.schema = shop_schema()

    def test_closed_vocabulary_values_are_canonicalised(self):
        self.assertEqual({"color": ["red"]}, self.schema.normalize({"color": ["Red"]}))

    def test_values_outside_the_vocabulary_are_dropped(self):
        """Storing 'burgundy' would create a facet that looks real and matches nothing."""
        self.assertEqual({}, self.schema.normalize({"color": ["burgundy"]}))
        self.assertEqual({"color": ["red"]}, self.schema.normalize({"color": ["red", "burgundy"]}))

    def test_unknown_attributes_are_dropped(self):
        self.assertEqual({}, self.schema.normalize({"shoe_size": 42}))

    def test_numbers_are_coerced_from_strings(self):
        self.assertEqual({"price": 850.0}, self.schema.normalize({"price": "850"}))
        self.assertEqual({}, self.schema.normalize({"price": "not a number"}))

    def test_booleans_accept_natural_phrasing(self):
        for raw, expected in [("yes", True), ("in stock", True), (True, True),
                              ("no", False), ("out of stock", False), (False, False)]:
            with self.subTest(raw=raw):
                self.assertEqual({"in_stock": expected}, self.schema.normalize({"in_stock": raw}))
        self.assertEqual({}, self.schema.normalize({"in_stock": "maybe"}))

    def test_single_values_are_wrapped_for_multi_attributes(self):
        self.assertEqual({"color": ["red"]}, self.schema.normalize({"color": "red"}))

    def test_lists_are_unwrapped_for_single_attributes(self):
        self.assertEqual({"category": "shoe"}, self.schema.normalize({"category": ["shoe"]}))

    def test_nulls_and_empties_are_dropped(self):
        self.assertEqual({}, self.schema.normalize({"color": None, "category": "", "price": None}))
        self.assertEqual({}, self.schema.normalize(None))
        self.assertEqual({}, self.schema.normalize("not a dict"))


class FilterValidationTests(unittest.TestCase):
    def setUp(self):
        self.schema = shop_schema()

    def test_valid_filters_pass_through_canonicalised(self):
        usable, problems = self.schema.validate_filters({"color": ["Red"], "category": "shoe"})
        self.assertEqual({"color": ["red"], "category": ["shoe"]}, usable)
        self.assertEqual([], problems)

    def test_an_unknown_attribute_is_reported_with_the_known_ones(self):
        usable, problems = self.schema.validate_filters({"shoe_size": [42]})
        self.assertEqual({}, usable)
        self.assertIn("shoe_size", problems[0])
        self.assertIn("category", problems[0])

    def test_a_value_outside_the_vocabulary_is_reported_not_silently_dropped(self):
        """Reported so the agent can retry, rather than silently returning the wrong set."""
        usable, problems = self.schema.validate_filters({"color": ["red", "burgundy"]})
        self.assertEqual({"color": ["red"]}, usable)
        self.assertIn("burgundy", problems[0])

    def test_number_ranges_are_parsed_from_plain_dicts(self):
        usable, problems = self.schema.validate_filters({"price": {"max": 100}})
        self.assertEqual([], problems)
        self.assertEqual(100.0, usable["price"].max)
        self.assertIsNone(usable["price"].min)

    def test_a_malformed_range_is_reported(self):
        _, problems = self.schema.validate_filters({"price": "cheap"})
        self.assertIn("range", problems[0])

    def test_an_empty_range_is_dropped_as_a_no_op(self):
        usable, problems = self.schema.validate_filters({"price": {}})
        self.assertEqual({}, usable)
        self.assertEqual([], problems)

    def test_none_values_are_ignored(self):
        self.assertEqual(({}, []), self.schema.validate_filters({"color": None}))
        self.assertEqual(({}, []), self.schema.validate_filters(None))


class InMemoryMatchTests(unittest.TestCase):
    def setUp(self):
        self.schema = shop_schema()
        self.product = {"category": "shoe", "color": ["red", "white"], "price": 85.0, "in_stock": True}

    def test_no_filters_matches_everything(self):
        self.assertTrue(self.schema.matches(self.product, {}))
        self.assertTrue(self.schema.matches(self.product, None))

    def test_multi_valued_attributes_match_on_any_value(self):
        self.assertTrue(self.schema.matches(self.product, {"color": ["red"]}))
        self.assertTrue(self.schema.matches(self.product, {"color": ["white"]}))
        self.assertFalse(self.schema.matches(self.product, {"color": ["blue"]}))

    def test_ranges_are_inclusive_and_open_ended(self):
        self.assertTrue(self.schema.matches(self.product, {"price": NumberRange(max=100)}))
        self.assertTrue(self.schema.matches(self.product, {"price": NumberRange(min=85)}))
        self.assertFalse(self.schema.matches(self.product, {"price": NumberRange(min=100)}))

    def test_filters_combine_with_and(self):
        self.assertTrue(self.schema.matches(self.product, {"color": ["red"], "category": ["shoe"]}))
        self.assertFalse(self.schema.matches(self.product, {"color": ["red"], "category": ["bag"]}))

    def test_a_missing_attribute_is_a_non_match(self):
        """A shopper asking for red shoes should not be shown items of unknown colour."""
        self.assertFalse(self.schema.matches({"category": "shoe"}, {"color": ["red"]}))
        self.assertFalse(self.schema.matches({}, {"color": ["red"]}))


class RoleStrategyTests(unittest.TestCase):
    def test_the_strategy_decides_the_default_role(self):
        self.assertEqual(AssetRole.FIGURE, resolve_role("figure"))
        self.assertEqual(AssetRole.ENTITY, resolve_role("entity"))

    def test_an_explicit_declaration_always_wins(self):
        self.assertEqual(AssetRole.ENTITY, resolve_role("figure", AssetRole.ENTITY))
        self.assertEqual(AssetRole.FIGURE, resolve_role("entity", AssetRole.FIGURE))


class EntityExtractorTests(unittest.TestCase):
    def setUp(self):
        self.config = load_profile("ecommerce").assets.entities
        self.schema = shop_schema()

    def test_the_heuristic_extractor_invents_no_attributes(self):
        payload = HeuristicEntityExtractor(self.config, self.schema).extract(
            ExtractionRequest(data=b"x", alt_text="Red running shoes")
        )
        self.assertEqual("Red running shoes", payload.text.caption)
        self.assertEqual({}, payload.structured.attributes)
        self.assertTrue(payload.provenance.needs_review)

    def test_the_composed_output_model_carries_core_fields_and_attributes(self):
        model = build_entity_model(self.schema)
        self.assertEqual({"title", "summary", "confidence", "attributes"}, set(model.model_fields))
        instance = model(title="Shoe", attributes={"color": ["red"]})
        self.assertEqual(["red"], instance.attributes.color)

    def _vision(self, result):
        extractor = VisionEntityExtractor(
            self.config, self.schema, model_id="vl", api_key="k", base_url="u",
            figures_config=load_profile("ecommerce").assets.figures,
        )
        structured = Mock()
        structured.invoke.return_value = result
        model = Mock()
        model.with_structured_output.return_value = structured
        extractor._model = model
        return extractor, structured

    def test_vision_extraction_produces_normalised_attributes(self):
        output_model = build_entity_model(self.schema)
        extractor, _ = self._vision(output_model(
            title="RS-200 Running Shoe",
            summary="A lightweight red running shoe.",
            confidence=0.9,
            attributes={"category": "shoe", "color": ["Red", "White"], "price": 850},
        ))
        payload = extractor.extract(ExtractionRequest(data=make_png()))

        self.assertEqual("RS-200 Running Shoe", payload.text.caption)
        self.assertEqual("shoe", payload.structured.attributes["category"])
        self.assertEqual(["red", "white"], payload.structured.attributes["color"])
        self.assertEqual(850.0, payload.structured.attributes["price"])
        self.assertFalse(payload.provenance.needs_review)

    def test_attributes_also_join_the_text_surface_so_semantic_recall_finds_them(self):
        """Filters narrow; they do not do the finding. The attribute text is what makes
        'red shoes' hit before any filter runs."""
        output_model = build_entity_model(self.schema)
        extractor, _ = self._vision(output_model(
            title="Shoe", confidence=0.9, attributes={"color": ["red"], "category": "shoe"}
        ))
        payload = extractor.extract(ExtractionRequest(data=make_png()))
        self.assertIn("red", payload.text.transcription)
        self.assertIn("red", payload.text.tags)

    def test_invented_values_are_stripped_after_the_model_returns(self):
        """A model told to pick from a list still occasionally invents one."""
        output_model = build_entity_model(self.schema)
        extractor, _ = self._vision(output_model(
            title="Shoe", confidence=0.9, attributes={"color": ["burgundy"]}
        ))
        payload = extractor.extract(ExtractionRequest(data=make_png()))
        self.assertEqual({}, payload.structured.attributes)
        self.assertTrue(payload.provenance.needs_review)

    def test_the_vocabulary_reaches_the_prompt(self):
        output_model = build_entity_model(self.schema)
        extractor, structured = self._vision(output_model(title="x", confidence=0.5))
        extractor.extract(ExtractionRequest(data=make_png(), alt_text="A shoe"))
        prompt = structured.invoke.call_args[0][0][0]["content"][0]["text"]
        self.assertIn("category", prompt)
        self.assertIn("leather", prompt)
        self.assertIn("A shoe", prompt)

    def test_building_falls_back_to_heuristic_without_vision(self):
        self.assertIsInstance(
            build_entity_extractor(self.config, self.schema), HeuristicEntityExtractor
        )

    def test_building_falls_back_when_the_vocabulary_is_empty(self):
        """Vision has nothing to fill in without a vocabulary."""
        config = self.config.model_copy(update={"vision_enabled": True})
        with patch.dict("os.environ", {"VISION_MODEL": "vl", "ARK_API_KEY": "k"}):
            extractor = build_entity_extractor(config, AttributeSchema([]))
        self.assertIsInstance(extractor, HeuristicEntityExtractor)

    def test_building_uses_vision_when_configured(self):
        config = self.config.model_copy(update={"vision_enabled": True})
        with patch.dict("os.environ", {"VISION_MODEL": "vl", "ARK_API_KEY": "k"}):
            extractor = build_entity_extractor(config, self.schema)
        self.assertIsInstance(extractor, VisionEntityExtractor)


class IndexTestCase(unittest.TestCase):
    def setUp(self):
        from backend.db.models import AssetExtraction, DocumentAsset, EntityAttribute

        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        for table in (DocumentAsset, AssetExtraction, EntityAttribute):
            table.__table__.create(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.index = EntityAttributeIndex(session_factory=self.session_factory)
        self.schema = shop_schema()

    def tearDown(self):
        self.engine.dispose()

    def _seed(self):
        catalogue = {
            "p1": {"category": "shoe", "color": ["red", "white"], "price": 85.0, "in_stock": True},
            "p2": {"category": "shoe", "color": ["blue"], "price": 150.0, "in_stock": True},
            "p3": {"category": "bag", "color": ["red"], "price": 60.0, "in_stock": False},
            "p4": {"category": "shoe", "color": ["red"], "price": 220.0, "in_stock": True},
        }
        for asset_id, attributes in catalogue.items():
            self.index.index_asset(asset_id, "ecommerce", attributes, self.schema)
        return catalogue


class EntityIndexTests(IndexTestCase):
    def test_multi_valued_attributes_produce_one_row_per_value(self):
        written = self.index.index_asset(
            "p1", "ecommerce", {"color": ["red", "white"], "category": "shoe"}, self.schema
        )
        self.assertEqual(3, written)

    def test_reindexing_replaces_rather_than_accumulates(self):
        """An attribute that disappears on re-extraction must leave the index too."""
        self.index.index_asset("p1", "ecommerce", {"color": ["red", "blue"]}, self.schema)
        self.index.index_asset("p1", "ecommerce", {"color": ["green"]}, self.schema)
        self.assertEqual({"p1"}, self.index.find({"color": ["green"]}, self.schema))
        self.assertEqual(set(), self.index.find({"color": ["red"]}, self.schema))

    def test_unknown_attributes_are_not_indexed(self):
        self.assertEqual(0, self.index.index_asset("p1", "ecommerce", {"bogus": 1}, self.schema))

    def test_find_with_no_filters_returns_none_not_empty(self):
        """None means 'no constraint'; an empty set means 'nothing qualifies'."""
        self._seed()
        self.assertIsNone(self.index.find({}, self.schema))
        self.assertIsNone(self.index.find({"color": None}, self.schema))

    def test_string_filters_match_any_listed_value(self):
        self._seed()
        self.assertEqual({"p1", "p3", "p4"}, self.index.find({"color": ["red"]}, self.schema))
        self.assertEqual({"p1", "p2", "p3", "p4"},
                         self.index.find({"color": ["red", "blue"]}, self.schema))

    def test_matching_is_case_insensitive(self):
        self._seed()
        self.assertEqual({"p1", "p3", "p4"}, self.index.find({"color": ["RED"]}, self.schema))

    def test_range_filters_work_on_numbers(self):
        self._seed()
        self.assertEqual({"p1", "p3"}, self.index.find({"price": NumberRange(max=100)}, self.schema))
        self.assertEqual({"p2", "p4"}, self.index.find({"price": NumberRange(min=100)}, self.schema))
        self.assertEqual({"p1", "p2"},
                         self.index.find({"price": NumberRange(min=80, max=150)}, self.schema))

    def test_boolean_filters(self):
        self._seed()
        self.assertEqual({"p1", "p2", "p4"}, self.index.find({"in_stock": True}, self.schema))
        self.assertEqual({"p3"}, self.index.find({"in_stock": False}, self.schema))

    def test_multiple_filters_intersect(self):
        """The 'red shoes under 100' query, as a pure database operation."""
        self._seed()
        self.assertEqual(
            {"p1"},
            self.index.find(
                {"color": ["red"], "category": ["shoe"], "price": NumberRange(max=100)},
                self.schema,
            ),
        )

    def test_an_impossible_combination_yields_an_empty_set(self):
        self._seed()
        self.assertEqual(set(), self.index.find({"color": ["green"]}, self.schema))

    def test_profile_scoping_isolates_catalogues(self):
        self.index.index_asset("p1", "ecommerce", {"color": ["red"]}, self.schema)
        self.index.index_asset("p9", "other_shop", {"color": ["red"]}, self.schema)
        self.assertEqual({"p1"}, self.index.find({"color": ["red"]}, self.schema, profile="ecommerce"))
        self.assertEqual({"p1", "p9"}, self.index.find({"color": ["red"]}, self.schema))

    def test_narrow_preserves_retrieval_rank(self):
        self._seed()
        ranked = ["p4", "p3", "p1", "p2"]
        self.assertEqual(["p4", "p3", "p1"],
                         self.index.narrow(ranked, {"color": ["red"]}, self.schema))

    def test_narrow_without_filters_is_a_passthrough(self):
        self._seed()
        self.assertEqual(["p2", "p1"], self.index.narrow(["p2", "p1"], {}, self.schema))

    def test_narrow_on_an_empty_candidate_set(self):
        self.assertEqual([], self.index.narrow([], {"color": ["red"]}, self.schema))

    def test_deleting_assets_clears_their_rows(self):
        self._seed()
        # p1 contributes 5 rows: category, two colours, price, in_stock.
        self.assertEqual(5, self.index.delete_assets(["p1"]))
        self.assertEqual({"p3", "p4"}, self.index.find({"color": ["red"]}, self.schema))

    def test_facets_count_values_for_a_ui(self):
        self._seed()
        facets = dict(self.index.facets("color", self.schema))
        self.assertEqual(3, facets["red"])
        self.assertEqual(1, facets["blue"])

    def test_facets_can_be_scoped_to_a_candidate_set(self):
        self._seed()
        facets = dict(self.index.facets("color", self.schema, restrict_to=["p1"]))
        self.assertEqual({"red": 1, "white": 1}, facets)

    def test_stats_report_index_coverage(self):
        self._seed()
        stats = self.index.stats()
        self.assertEqual(4, stats["indexed_assets"])
        self.assertEqual(5, stats["by_attribute"]["color"])


class EntityIngestTests(IndexTestCase):
    """The pipeline path: an entity-strategy profile indexes attributes at ingest."""

    def setUp(self):
        super().setUp()
        self._tmp = TemporaryDirectory()
        self.blobs = LocalBlobStore(Path(self._tmp.name))
        self.store = AssetStore(session_factory=self.session_factory, blob_store=self.blobs,
                                cache_enabled=False)
        self.profile = load_profile("ecommerce")

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def _pipeline(self, attributes=None):
        extractor = Mock()
        if attributes is not None:
            from backend.assets.dossier import ExtractionPayload, Provenance, StructuredSurface, TextSurface

            extractor.extract.return_value = ExtractionPayload(
                text=TextSurface(caption="Red running shoes"),
                structured=StructuredSurface(attributes=attributes),
                provenance=Provenance(model_used="test", confidence=0.9),
            )
        return FigurePipeline(
            profile=self.profile, store=self.store, blob_store=self.blobs,
            entity_extractor=extractor, entity_index=self.index,
            extractor=Mock(), fallback_extractor=Mock(),
        ), extractor

    def test_an_entity_profile_produces_entity_assets(self):
        pipeline, _ = self._pipeline({"category": "shoe", "color": ["red"]})
        dossiers, report = pipeline.process(
            [ImageInput(data=make_png(), index=0)], filename="catalogue.pdf"
        )
        self.assertEqual(AssetRole.ENTITY, dossiers[0].role)
        self.assertEqual(ExtractionStatus.EXTRACTED, dossiers[0].status)
        self.assertEqual(1, report.entities)
        self.assertEqual(2, report.attributes_indexed)

    def test_the_attributes_become_queryable(self):
        pipeline, _ = self._pipeline({"category": "shoe", "color": ["red"], "price": 85.0})
        dossiers, _ = pipeline.process([ImageInput(data=make_png(), index=0)], filename="c.pdf")
        matched = self.index.find(
            {"color": ["red"], "price": NumberRange(max=100)}, self.schema, profile="ecommerce"
        )
        self.assertEqual({dossiers[0].asset_id}, matched)

    def test_the_entity_extractor_is_used_not_the_figure_one(self):
        pipeline, entity_extractor = self._pipeline({"color": ["red"]})
        pipeline.process([ImageInput(data=make_png(), index=0)], filename="c.pdf")
        entity_extractor.extract.assert_called_once()
        pipeline.extractor.extract.assert_not_called()

    def test_a_declared_role_overrides_the_profile_strategy(self):
        pipeline, _ = self._pipeline({"color": ["red"]})
        dossiers, report = pipeline.process(
            [ImageInput(data=make_png(), index=0, declared_role=AssetRole.FIGURE)],
            filename="c.pdf",
        )
        self.assertEqual(AssetRole.FIGURE, dossiers[0].role)
        self.assertEqual(0, report.entities)

    def test_triaged_out_images_stay_decorative_under_an_entity_profile(self):
        """Declaring a catalogue import must not resurrect its spacers and logos."""
        pipeline, _ = self._pipeline({"color": ["red"]})
        dossiers, report = pipeline.process(
            [ImageInput(data=make_png(30, 30), index=0)], filename="c.pdf"
        )
        self.assertEqual(AssetRole.DECORATIVE, dossiers[0].role)
        self.assertEqual(0, report.entities)

    def test_an_entity_with_no_attributes_is_still_stored(self):
        pipeline, _ = self._pipeline({})
        dossiers, report = pipeline.process([ImageInput(data=make_png(), index=0)], filename="c.pdf")
        self.assertEqual(1, report.entities)
        self.assertEqual(0, report.attributes_indexed)
        self.assertIsNotNone(self.store.get(dossiers[0].asset_id))

    def test_an_index_failure_does_not_fail_the_upload(self):
        pipeline, _ = self._pipeline({"color": ["red"]})
        pipeline._entity_index = Mock()
        pipeline._entity_index.index_asset.side_effect = RuntimeError("db down")
        dossiers, report = pipeline.process([ImageInput(data=make_png(), index=0)], filename="c.pdf")
        self.assertEqual(ExtractionStatus.EXTRACTED, dossiers[0].status)
        self.assertEqual(0, report.attributes_indexed)

    def test_a_cached_extraction_is_still_indexed_for_the_new_occurrence(self):
        """The extraction is shared by digest, but the index is per occurrence."""
        shared = make_png(seed=9)
        pipeline, extractor = self._pipeline({"color": ["red"]})
        pipeline.process([ImageInput(data=shared, index=0)], filename="a.pdf")
        _, report = pipeline.process([ImageInput(data=shared, index=0)], filename="b.pdf")

        self.assertEqual(1, extractor.extract.call_count)  # cache hit
        self.assertEqual(1, report.cached)
        self.assertEqual(1, report.attributes_indexed)
        self.assertEqual(2, len(self.index.find({"color": ["red"]}, self.schema)))


class EntityRetrievalTests(IndexTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = TemporaryDirectory()
        self.store = AssetStore(session_factory=self.session_factory,
                                blob_store=LocalBlobStore(Path(self._tmp.name)),
                                cache_enabled=False)
        self.profile = load_profile("ecommerce")
        self.catalogue = self._seed()
        for asset_id, attributes in self.catalogue.items():
            self.store.record(self._dossier(asset_id, attributes))

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    @staticmethod
    def _dossier(asset_id, attributes):
        from backend.assets.dossier import (
            AssetDossier, ExtractionPayload, SourceRef, StructuredSurface, TextSurface,
        )

        return AssetDossier(
            asset_id=asset_id, sha256="a" * 64, profile="ecommerce",
            role=AssetRole.ENTITY, status=ExtractionStatus.EXTRACTED,
            source=SourceRef(filename="catalogue.pdf", page_number=1),
            extraction=ExtractionPayload(
                text=TextSurface(caption=f"Product {asset_id}", description="A product."),
                structured=StructuredSurface(attributes=attributes),
            ),
        )

    def _retriever(self, recalled):
        return EntityRetriever(
            profile=self.profile, asset_store=self.store, entity_index=self.index,
            recall=lambda query, top_k, language="": [
                {"asset_ids": [asset_id], "score": score} for asset_id, score in recalled
            ],
        )

    def test_recall_finds_and_filters_narrow(self):
        """The whole flow: context first, then attributes."""
        retriever = self._retriever([("p1", 0.9), ("p2", 0.8), ("p3", 0.7), ("p4", 0.6)])
        result = retriever.search("red shoes", {"color": ["red"], "category": ["shoe"]})

        self.assertEqual(4, result.recalled)
        self.assertEqual(2, result.after_filter)
        self.assertEqual(["p1", "p4"], [hit.asset_id for hit in result.hits])
        self.assertFalse(result.filtered_to_empty)

    def test_retrieval_rank_survives_filtering(self):
        retriever = self._retriever([("p4", 0.9), ("p1", 0.5)])
        result = retriever.search("red shoes", {"color": ["red"]})
        self.assertEqual(["p4", "p1"], [hit.asset_id for hit in result.hits])
        self.assertEqual(0.9, result.hits[0].score)

    def test_a_range_filter_narrows_further(self):
        retriever = self._retriever([("p1", 0.9), ("p4", 0.8)])
        result = retriever.search("red shoes", {"color": ["red"], "price": {"max": 100}})
        self.assertEqual(["p1"], [hit.asset_id for hit in result.hits])

    def test_hits_carry_the_attributes_and_source(self):
        retriever = self._retriever([("p1", 0.9)])
        hit = retriever.search("shoes", {}).hits[0]
        self.assertEqual("shoe", hit.attributes["category"])
        self.assertEqual("catalogue.pdf", hit.filename)
        self.assertEqual("Product p1", hit.caption)

    def test_filtering_to_empty_is_distinguished_from_no_recall(self):
        """Different situations: one should offer to relax a filter, the other cannot."""
        retriever = self._retriever([("p2", 0.9)])
        filtered = retriever.search("shoes", {"color": ["red"]})
        self.assertTrue(filtered.filtered_to_empty)
        self.assertEqual(1, filtered.recalled)

        empty = self._retriever([]).search("spaceships", {"color": ["red"]})
        self.assertFalse(empty.filtered_to_empty)
        self.assertEqual(0, empty.recalled)

    def test_rejected_filters_are_reported_but_do_not_block_the_search(self):
        retriever = self._retriever([("p1", 0.9)])
        result = retriever.search("shoes", {"color": ["red"], "shoe_size": [42]})
        self.assertEqual(["p1"], [hit.asset_id for hit in result.hits])
        self.assertIn("shoe_size", result.rejected_filters[0])

    def test_no_filters_returns_the_recalled_set(self):
        retriever = self._retriever([("p1", 0.9), ("p2", 0.8)])
        result = retriever.search("shoes")
        self.assertEqual(2, len(result.hits))

    def test_results_are_capped_by_the_profile(self):
        profile = self.profile.model_copy(deep=True)
        profile.assets.entities.max_results = 2
        retriever = EntityRetriever(
            profile=profile, asset_store=self.store, entity_index=self.index,
            recall=lambda q, k: [{"asset_ids": [a], "score": 1.0} for a in self.catalogue],
        )
        self.assertEqual(2, len(retriever.search("anything").hits))

    def test_an_index_failure_returns_nothing_rather_than_an_unfiltered_set(self):
        """Silently widening would look filtered while not being — worse than empty."""
        broken = Mock()
        broken.narrow.side_effect = RuntimeError("db down")
        retriever = EntityRetriever(
            profile=self.profile, asset_store=self.store, entity_index=broken,
            recall=lambda q, k: [{"asset_ids": ["p1"], "score": 1.0}],
        )
        result = retriever.search("red shoes", {"color": ["red"]})
        self.assertEqual([], result.hits)

    def test_the_result_serialises_for_a_trace(self):
        payload = self._retriever([("p1", 0.9)]).search("shoes", {"color": ["red"]}).as_dict()
        self.assertEqual(["hits", "recalled", "after_filter", "filters_applied",
                          "rejected_filters", "filtered_to_empty"], list(payload))


class SearchProductsToolTests(unittest.TestCase):
    def setUp(self):
        from backend.profiles.registry import set_profile

        set_profile(load_profile("ecommerce"))

    def tearDown(self):
        from backend.profiles.registry import set_profile
        from backend.rag.entity_retrieval import set_entity_retriever

        set_profile(None)
        set_entity_retriever(None)

    def _tool(self, ctx=None):
        from backend.chat.request_context import ChatRequestContext
        from backend.tools.products import make_search_products

        return make_search_products(ctx or ChatRequestContext.for_sync(user_id="u", session_id="s"))

    def test_the_tool_signature_is_generated_from_the_profile(self):
        """This is what makes filter slot-filling free: the vocabulary is in the schema
        the model already receives."""
        schema = self._tool().args_schema.model_json_schema()
        self.assertEqual({"query", "filters"}, set(schema["properties"]))
        rendered = str(schema)
        for name in ("category", "color", "price", "in_stock"):
            with self.subTest(name=name):
                self.assertIn(name, rendered)

    def test_the_tool_is_registered(self):
        from backend.tools import TOOL_BUILDERS

        self.assertIn("search_products", TOOL_BUILDERS)

    def test_results_are_formatted_with_asset_ids_for_display(self):
        from backend.chat.request_context import ChatRequestContext
        from backend.rag.entity_retrieval import EntityHit, EntitySearchResult, set_entity_retriever

        retriever = Mock()
        retriever.search.return_value = EntitySearchResult(
            hits=[EntityHit(asset_id="p1", caption="RS-200", summary="A red shoe.",
                            attributes={"color": ["red"], "price": 85.0})],
            recalled=4, after_filter=1,
        )
        set_entity_retriever(retriever)

        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        output = self._tool(ctx).invoke({"query": "red shoes", "filters": {"color": ["red"]}})

        self.assertIn("RS-200", output)
        self.assertIn("color=red", output)
        self.assertIn("asset_id: p1", output)
        # Surfacing the asset is what lets the response carry the picture.
        self.assertEqual(["p1"], ctx.surfaced_asset_ids())

    def test_filtering_to_empty_suggests_relaxing_a_constraint(self):
        from backend.rag.entity_retrieval import EntitySearchResult, set_entity_retriever

        retriever = Mock()
        retriever.search.return_value = EntitySearchResult(
            recalled=6, after_filter=0, filtered_to_empty=True, filters_applied={"color": ["green"]}
        )
        set_entity_retriever(retriever)
        output = self._tool().invoke({"query": "shoes", "filters": {"color": ["green"]}})
        self.assertIn("NO_PRODUCTS_FOUND", output)
        self.assertIn("relaxing", output)

    def test_a_profile_without_a_catalogue_says_so(self):
        from backend.profiles.registry import set_profile

        set_profile(load_profile("base"))
        self.assertIn("PRODUCT_SEARCH_UNAVAILABLE", self._tool().invoke({"query": "shoes"}))

    def test_a_retrieval_failure_is_reported_not_raised(self):
        from backend.rag.entity_retrieval import set_entity_retriever

        retriever = Mock()
        retriever.search.side_effect = RuntimeError("milvus down")
        set_entity_retriever(retriever)
        self.assertIn("PRODUCT_SEARCH_ERROR", self._tool().invoke({"query": "shoes"}))

    def test_the_turn_budget_is_enforced(self):
        from backend.chat.request_context import ChatRequestContext
        from backend.rag.entity_retrieval import EntitySearchResult, set_entity_retriever

        retriever = Mock()
        retriever.search.return_value = EntitySearchResult()
        set_entity_retriever(retriever)

        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        tool = self._tool(ctx)
        tool.invoke({"query": "shoes"})
        self.assertIn("TOOL_CALL_LIMIT_REACHED", tool.invoke({"query": "boots"}))

    def test_an_invented_filter_key_is_refused_by_the_generated_schema(self):
        with self.assertRaises(Exception):
            self._tool().invoke({"query": "shoes", "filters": {"bluetooth": True}})


class ProfileIntegrationTests(unittest.TestCase):
    def test_the_ecommerce_profile_declares_a_real_vocabulary(self):
        schema = shop_schema()
        self.assertGreater(len(schema), 5)
        self.assertIn("color", schema.names())
        self.assertIn("price", schema.names())

    def test_base_ships_with_entities_disabled_and_no_vocabulary(self):
        entities = load_profile("base").assets.entities
        self.assertFalse(entities.enabled)
        self.assertEqual("figure", entities.role_strategy)
        self.assertEqual([], entities.attributes)

    def test_every_shipped_profile_builds_a_valid_schema(self):
        from backend.profiles.registry import available_profiles

        for name in available_profiles():
            with self.subTest(profile=name):
                schema = build_attribute_schema(load_profile(name).assets.entities)
                schema.build_extraction_model()
                schema.build_filter_model()

    def test_attribute_types_are_constrained_by_the_profile_schema(self):
        with self.assertRaises(Exception):
            AttributeSpec(name="color", type="colour")


if __name__ == "__main__":
    unittest.main()
