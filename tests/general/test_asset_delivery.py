"""Asset delivery: renditions, storage round-trips, and which figure an answer shows.

The heaviest class here is `UiIndependenceTests`. The requirement is that replacing
the frontend — with React, a Slack bot, a CLI, or another service — needs no backend
change, and the way that is enforced is by asserting the backend never emits
presentation: no HTML, no markdown image syntax, no field named after a component.
What varies between clients is capability, declared per request.
"""
import importlib
import io
import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.assets.blobs import LocalBlobStore
from backend.assets.delivery import (
    AssetPresenter,
    AssetReference,
    AssetRenditionMode,
    ClientCapabilities,
    asset_url_path,
    collect_asset_ids,
)
from backend.assets.dossier import (
    AssetDossier,
    AssetRole,
    AssetTier,
    BlobRef,
    ExtractionPayload,
    ExtractionStatus,
    Provenance,
    SourceRef,
    TextSurface,
    compute_sha256,
)
from backend.assets.store import AssetStore
from backend.chat.assets_bridge import (
    asset_ids_for_turn,
    attach_assets_to_trace,
    build_asset_references,
    effective_capabilities,
    restore_session_assets,
    trace_for_storage,
)
from backend.chat.request_context import ChatRequestContext
from backend.profiles.registry import load_profile


def make_png(width=200, height=200, seed=1) -> bytes:
    from PIL import Image

    data = random.Random(seed).randbytes(width * height * 3)
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), data).save(buffer, format="PNG")
    return buffer.getvalue()


def make_dossier(asset_id="doc.pdf::p1::img0", uri="file://ab/cd/x.png", caption="Fee schedule",
                 byte_size=4096, sha256=None, extraction=True, role=AssetRole.FIGURE):
    payload = None
    if extraction:
        payload = ExtractionPayload(
            text=TextSurface(caption=caption, description="A table of tuition fees.",
                             transcription="Grade | Fee\n5 | 42000", tags=["fees"]),
            answerable_questions=["How much is grade 5?"],
            provenance=Provenance(tier=AssetTier.SIMPLE, model_used="test"),
        )
    return AssetDossier(
        asset_id=asset_id,
        sha256=sha256 or compute_sha256(asset_id.encode()),
        role=role,
        status=ExtractionStatus.EXTRACTED if extraction else ExtractionStatus.PENDING,
        source=SourceRef(filename="doc.pdf", page_number=1),
        blob=BlobRef(uri=uri, content_type="image/png", byte_size=byte_size, width=200, height=150),
        extraction=payload,
    )


class RenditionTests(unittest.TestCase):
    def setUp(self):
        self.presenter = AssetPresenter(delivery_config=load_profile("base").assets.delivery)

    def test_a_browser_client_gets_a_url(self):
        reference = self.presenter.present(make_dossier(), ClientCapabilities())
        self.assertEqual(AssetRenditionMode.REFERENCE, reference.mode)
        self.assertEqual("/media/doc.pdf%3A%3Ap1%3A%3Aimg0", reference.url)
        self.assertIsNone(reference.inline_data)

    def test_a_client_that_cannot_fetch_gets_bytes_inline(self):
        data = make_png(40, 40)
        with TemporaryDirectory() as tmp:
            blobs = LocalBlobStore(Path(tmp))
            digest = compute_sha256(data)
            uri = blobs.put(digest, data, "image/png")
            presenter = AssetPresenter(
                delivery_config=load_profile("base").assets.delivery, blob_store=blobs
            )
            reference = presenter.present(
                make_dossier(uri=uri, byte_size=len(data), sha256=digest),
                ClientCapabilities(prefers_inline=True, max_inline_bytes=10_000_000),
            )
        self.assertEqual(AssetRenditionMode.INLINE, reference.mode)
        self.assertTrue(reference.inline_data.startswith("data:image/png;base64,"))
        self.assertIsNone(reference.url)

    def test_a_text_only_client_gets_metadata(self):
        reference = self.presenter.present(
            make_dossier(), ClientCapabilities(accepts_images=False)
        )
        self.assertEqual(AssetRenditionMode.METADATA, reference.mode)
        self.assertIsNone(reference.url)
        self.assertIsNone(reference.inline_data)
        # Still says something useful — a caption is not an image.
        self.assertEqual("Fee schedule", reference.caption)

    def test_an_oversized_image_falls_back_to_a_url_rather_than_inlining(self):
        reference = self.presenter.present(
            make_dossier(byte_size=5_000_000),
            ClientCapabilities(prefers_inline=True, max_inline_bytes=1000),
        )
        self.assertEqual(AssetRenditionMode.REFERENCE, reference.mode)

    def test_an_unsupported_content_type_degrades_to_metadata(self):
        reference = self.presenter.present(
            make_dossier(), ClientCapabilities(accepted_content_types=["image/avif"])
        )
        self.assertEqual(AssetRenditionMode.METADATA, reference.mode)

    def test_an_asset_with_no_stored_bytes_is_always_metadata(self):
        """Triaged-out assets have no blob; no capability can conjure one."""
        reference = self.presenter.present(make_dossier(uri=""), ClientCapabilities())
        self.assertEqual(AssetRenditionMode.METADATA, reference.mode)

    def test_unreadable_bytes_fall_back_to_a_url_instead_of_dropping_the_asset(self):
        broken = Mock()
        broken.get.side_effect = OSError("gone")
        presenter = AssetPresenter(
            delivery_config=load_profile("base").assets.delivery, blob_store=broken
        )
        reference = presenter.present(
            make_dossier(), ClientCapabilities(prefers_inline=True, max_inline_bytes=10_000_000)
        )
        self.assertEqual(AssetRenditionMode.REFERENCE, reference.mode)
        self.assertIsNotNone(reference.url)

    def test_present_many_respects_the_client_asset_cap(self):
        dossiers = [make_dossier(asset_id=f"doc.pdf::p1::img{i}") for i in range(10)]
        references = self.presenter.present_many(dossiers, ClientCapabilities(max_assets=3))
        self.assertEqual(3, len(references))

    def test_caption_and_source_travel_in_every_mode(self):
        for capabilities in (
            ClientCapabilities(),
            ClientCapabilities(accepts_images=False),
            ClientCapabilities(prefers_inline=True),
        ):
            with self.subTest(mode=capabilities.accepts_images):
                reference = self.presenter.present(make_dossier(), capabilities)
                self.assertEqual("Fee schedule", reference.caption)
                self.assertEqual("doc.pdf", reference.source.filename)
                self.assertEqual(1, reference.source.page_number)


class UrlEncodingTests(unittest.TestCase):
    def test_asset_ids_are_fully_percent_encoded(self):
        """The id embeds a filename, so every separator must be escaped or the URL
        would fork into extra path segments."""
        self.assertEqual(
            "/media/a%20b%2Fc.pdf%3A%3Ap0%3A%3Aimg0",
            asset_url_path("a b/c.pdf::p0::img0"),
        )

    def test_non_latin_filenames_survive(self):
        url = asset_url_path("دليل.pdf::p2::img1")
        self.assertNotIn(" ", url)
        self.assertTrue(url.startswith("/media/"))

    def test_base_path_is_configurable_for_proxied_deployments(self):
        self.assertTrue(asset_url_path("x::p0::img0", base_path="/api/v2/media").startswith("/api/v2/media/"))


class UiIndependenceTests(unittest.TestCase):
    """The backend must never encode presentation. These are the tests that make
    replacing the frontend a no-op on this side."""

    def setUp(self):
        self.presenter = AssetPresenter(delivery_config=load_profile("base").assets.delivery)

    def test_no_markup_is_ever_emitted(self):
        payload = json.dumps(
            self.presenter.present(make_dossier(), ClientCapabilities()).model_dump(mode="json")
        )
        for markup in ("<img", "<div", "![", "</", "<a ", "style=", "class="):
            with self.subTest(markup=markup):
                self.assertNotIn(markup, payload)

    def test_the_reference_schema_carries_no_ui_vocabulary(self):
        forbidden = {"html", "markdown", "component", "css", "element", "vue", "react", "widget"}
        for field in AssetReference.model_fields:
            with self.subTest(field=field):
                self.assertFalse(
                    forbidden & set(field.lower().split("_")),
                    f"{field} names a UI concept",
                )

    def test_the_contract_is_json_serialisable_and_reconstructible(self):
        """Any consumer in any language must be able to round-trip it."""
        original = self.presenter.present(make_dossier(), ClientCapabilities())
        restored = AssetReference.model_validate(json.loads(json.dumps(original.model_dump(mode="json"))))
        self.assertEqual(original, restored)

    def test_a_text_only_channel_can_render_without_images(self):
        reference = self.presenter.present(make_dossier(), ClientCapabilities(accepts_images=False))
        self.assertEqual("[image: Fee schedule] (doc.pdf p1)", reference.describe())

    def test_capabilities_are_declared_not_inferred(self):
        """Same asset, three clients, three renditions — driven only by what each
        client said about itself."""
        dossier = make_dossier()
        modes = {
            self.presenter.present(dossier, capabilities).mode
            for capabilities in (
                ClientCapabilities(),
                ClientCapabilities(accepts_images=False),
                ClientCapabilities(prefers_inline=True, max_inline_bytes=1),
            )
        }
        self.assertIn(AssetRenditionMode.REFERENCE, modes)
        self.assertIn(AssetRenditionMode.METADATA, modes)

    def test_an_unknown_capability_key_is_rejected(self):
        """extra='forbid' stops a client silently relying on a field the server never
        implemented."""
        with self.assertRaises(Exception):
            ClientCapabilities.model_validate({"accepts_images": True, "supports_webgl": True})


class CapabilityPolicyTests(unittest.TestCase):
    def setUp(self):
        self.delivery = load_profile("base").assets.delivery

    def test_a_client_may_ask_for_less_than_policy_allows(self):
        capabilities = effective_capabilities(
            ClientCapabilities(max_assets=2, max_inline_bytes=1000), self.delivery
        )
        self.assertEqual(2, capabilities.max_assets)
        self.assertEqual(1000, capabilities.max_inline_bytes)

    def test_a_client_may_never_ask_for_more(self):
        """Otherwise an integrator sets the server's egress budget for it."""
        capabilities = effective_capabilities(
            ClientCapabilities(max_assets=9999, max_inline_bytes=40_000_000), self.delivery
        )
        self.assertEqual(self.delivery.max_assets_per_response, capabilities.max_assets)
        self.assertEqual(self.delivery.max_inline_bytes, capabilities.max_inline_bytes)

    def test_no_declaration_yields_browser_defaults(self):
        capabilities = effective_capabilities(None, self.delivery)
        self.assertTrue(capabilities.accepts_images)
        self.assertFalse(capabilities.prefers_inline)

    def test_clamping_does_not_mutate_the_caller_object(self):
        requested = ClientCapabilities(max_assets=9999)
        effective_capabilities(requested, self.delivery)
        self.assertEqual(9999, requested.max_assets)


class CollectAssetIdsTests(unittest.TestCase):
    def test_ids_keep_retrieval_rank_order_and_deduplicate(self):
        chunks = [
            {"asset_ids": ["a", "b"]},
            {"asset_ids": ["b", "c"]},
            {"asset_ids": []},
            {},
        ]
        self.assertEqual(["a", "b", "c"], collect_asset_ids(chunks))

    def test_the_limit_is_honoured(self):
        chunks = [{"asset_ids": [f"a{i}"]} for i in range(50)]
        self.assertEqual(5, len(collect_asset_ids(chunks, limit=5)))

    def test_empty_and_malformed_chunks_are_tolerated(self):
        self.assertEqual([], collect_asset_ids([]))
        self.assertEqual([], collect_asset_ids([{"asset_ids": None}, {"asset_ids": [""]}]))


class RequestContextAssetTests(unittest.TestCase):
    def test_surfaced_assets_are_pinned_in_retrieval_order(self):
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        ctx.note_surfaced_assets(["a", "b"])
        ctx.note_surfaced_assets(["b", "c"])
        self.assertEqual(["a", "b", "c"], ctx.surfaced_asset_ids())

    def test_resetting_restores_the_knowledge_budget(self):
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        self.assertTrue(ctx.acquire_knowledge_tool_slot())
        self.assertFalse(ctx.acquire_knowledge_tool_slot())
        ctx.reset_knowledge_tool_budget()
        self.assertTrue(ctx.acquire_knowledge_tool_slot())

    def test_a_closed_context_stops_accepting_assets(self):
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        ctx.close()
        ctx.note_surfaced_assets(["a"])
        self.assertEqual([], ctx.surfaced_asset_ids())


class AssetStoreTestCase(unittest.TestCase):
    """A real AssetStore over SQLite with one dossier and its blob in it, so the tests
    below exercise the actual lookup rather than a stub."""

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
        self.store = AssetStore(session_factory=session_factory, blob_store=self.blobs,
                                cache_enabled=False)

        data = make_png(60, 60)
        digest = compute_sha256(data)
        uri = self.blobs.put(digest, data, "image/png")
        self.dossier = make_dossier(uri=uri, sha256=digest, byte_size=len(data))
        self.store.record(self.dossier)

        from backend.assets.store import set_asset_store
        from backend.assets.blobs import set_blob_store

        set_asset_store(self.store)
        set_blob_store(self.blobs)

    def tearDown(self):
        from backend.assets.store import set_asset_store
        from backend.assets.blobs import set_blob_store

        set_asset_store(None)
        set_blob_store(None)
        self._tmp.cleanup()
        self.engine.dispose()


class AssetsBridgeTests(unittest.TestCase):
    def setUp(self):
        self.delivery = load_profile("base").assets.delivery

    def test_context_ids_take_priority_over_the_trace(self):
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        ctx.note_surfaced_assets(["from-ctx"])
        trace = {"retrieved_chunks": [{"asset_ids": ["from-trace"]}]}
        self.assertEqual(["from-ctx"], asset_ids_for_turn(ctx, trace))

    def test_the_trace_is_the_fallback_for_paths_that_skip_the_tool(self):
        """The HITL resume flow answers from docs without calling the knowledge tool."""
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        trace = {"retrieved_chunks": [{"asset_ids": ["from-trace"]}]}
        self.assertEqual(["from-trace"], asset_ids_for_turn(ctx, trace))

    def test_no_assets_anywhere_yields_an_empty_list(self):
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        self.assertEqual([], asset_ids_for_turn(ctx, None))
        self.assertEqual([], asset_ids_for_turn(None, None))

    def test_attach_is_a_noop_without_assets_or_trace(self):
        self.assertIsNone(attach_assets_to_trace(None, []))
        self.assertEqual({"a": 1}, attach_assets_to_trace({"a": 1}, []))

    def test_assets_without_a_trace_still_get_somewhere_to_live(self):
        """The trace is where a stored message keeps its images. Returning None here
        would show the pictures live and lose them on the next reload."""
        enriched = attach_assets_to_trace(None, [AssetReference(asset_id="x")])
        self.assertEqual(["x"], [asset["asset_id"] for asset in enriched["assets"]])

    def test_what_is_stored_is_the_id_and_not_the_rendition(self):
        wire = attach_assets_to_trace(
            {"route": "answer"},
            [AssetReference(asset_id="x", inline_data="data:image/png;base64,AAAA")],
        )
        stored = trace_for_storage(wire)

        self.assertEqual(["x"], stored["asset_ids"])
        self.assertNotIn("assets", stored)
        self.assertEqual("answer", stored["route"], "the rest of the trace is untouched")

    def test_storing_a_trace_leaves_the_wire_copy_alone(self):
        """The client is still holding the renditions it was sent."""
        wire = attach_assets_to_trace({"route": "answer"}, [AssetReference(asset_id="x")])
        trace_for_storage(wire)
        self.assertIn("assets", wire)

    def test_a_trace_without_assets_passes_straight_through(self):
        self.assertIsNone(trace_for_storage(None))
        self.assertEqual({"route": "answer"}, trace_for_storage({"route": "answer"}))

    def test_the_service_stores_every_turns_trace_by_id(self):
        """The wiring, not just the function: a save path that skipped it would put
        renditions back in the database without any test noticing."""
        service = importlib.import_module("backend.chat.service")
        wire = attach_assets_to_trace({"route": "answer"}, [AssetReference(asset_id="x")])

        extra = service._message_data_for_save([1, 2], wire)

        self.assertEqual([None], extra[:1])
        self.assertEqual(["x"], extra[-1]["rag_trace"]["asset_ids"])
        self.assertNotIn("assets", extra[-1]["rag_trace"])

    def test_attach_does_not_mutate_the_original_trace(self):
        trace = {"route": "answer"}
        reference = AssetReference(asset_id="x")
        attach_assets_to_trace(trace, [reference])
        self.assertNotIn("assets", trace)

    def test_disabling_attachment_returns_nothing(self):
        config = self.delivery.model_copy(update={"attach_to_response": False})
        self.assertEqual([], build_asset_references(["a"], ClientCapabilities(), config))

    def test_a_store_failure_costs_pictures_not_the_answer(self):
        with patch("backend.assets.store.get_asset_store", side_effect=RuntimeError("db down")):
            self.assertEqual(
                [], build_asset_references(["a"], ClientCapabilities(), self.delivery)
            )


class DictCache:
    """Stands in for Redis. Backed by JSON so the round trip is the real one."""

    def __init__(self):
        self.store = {}

    def get_json(self, key):
        value = self.store.get(key)
        return json.loads(value) if value is not None else None

    def set_json(self, key, value, ttl=None):
        self.store[key] = json.dumps(value, ensure_ascii=False)

    def delete(self, key):
        self.store.pop(key, None)

    def clear(self):
        self.store.clear()


class StoredConversationAssetTests(unittest.TestCase):
    """An answer's images have to survive the next turn, not just the current one.

    Images ride on the message's trace, and a save that rewrote the whole conversation
    while the caller supplied a trace only for the turn it just finished dropped every
    other one. An answer's pictures then lived exactly until the user's next message:
    visible in the tab that received them, gone on reload.

    Saving is an append now, and what it stores is ids rather than renditions — so these
    also pin the two properties that come with that: rows are not churned, and a
    conversation does not carry copies of the pictures in it.
    """

    def setUp(self):
        import backend.chat.storage as storage_module

        from backend.db.models import ChatMessage, ChatSession, User

        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        for model in (User, ChatSession, ChatMessage):
            model.__table__.create(self.engine)
        factory = sessionmaker(bind=self.engine, expire_on_commit=False)

        db = factory()
        db.add(User(username="u", password_hash="x"))
        db.commit()
        db.close()

        self.factory = factory
        self.cache = DictCache()
        self.storage = storage_module.ConversationStorage()
        self._patches = [
            patch.object(storage_module, "SessionLocal", factory),
            patch.object(storage_module, "cache", self.cache),
        ]
        for item in self._patches:
            item.start()

    def tearDown(self):
        for item in self._patches:
            item.stop()
        self.engine.dispose()

    def _trace(self, asset_id, inline=False):
        """A trace as it goes on the WIRE: renditions in full, inline bytes and all."""
        reference = AssetReference(
            asset_id=asset_id,
            caption="Chart",
            mode=AssetRenditionMode.INLINE if inline else AssetRenditionMode.REFERENCE,
            inline_data="data:image/png;base64,AAAA" * 64 if inline else None,
            url=None if inline else asset_url_path(asset_id),
        )
        return {"tool_used": True, "assets": [reference.model_dump(mode="json", exclude_none=True)]}

    def _turn(self, question, answer, trace=None):
        """One turn, saved the way the chat service saves it: once when the question
        arrives, once when the answer is complete, with the trace stored by id."""
        from langchain_core.messages import AIMessage, HumanMessage

        messages = self.storage.load("u", "s")
        messages.append(HumanMessage(content=question))
        self.storage.save("u", "s", messages)
        messages.append(AIMessage(content=answer))
        self.storage.save(
            "u", "s", messages,
            metadata={},
            extra_message_data=[None] * (len(messages) - 1)
            + [{"rag_trace": trace_for_storage(trace)}],
        )

    def _stored_ids(self):
        return [
            list((message["rag_trace"] or {}).get("asset_ids") or [])
            for message in self.storage.get_session_messages("u", "s")
        ]

    def _rows(self):
        from backend.db.models import ChatMessage

        db = self.factory()
        try:
            return db.query(ChatMessage).order_by(ChatMessage.id.asc()).all()
        finally:
            db.close()

    def test_an_answers_images_survive_the_following_turns(self):
        self._turn("what is the uniform?", "Navy, see the chart. [1]", self._trace("a1"))
        self._turn("and the bus times?", "Buses run at 07:30.", {"tool_used": True})
        self._turn("thanks", "Any time.")

        self.assertEqual(
            [[], ["a1"], [], [], [], []],
            self._stored_ids(),
            "the first answer lost its images once the conversation continued",
        )

    def test_images_survive_a_restart_that_empties_the_cache(self):
        """The reported bug. A restart drops Redis, so history is rebuilt from
        Postgres — which is exactly where the traces were being erased."""
        self._turn("what is the uniform?", "Navy, see the chart. [1]", self._trace("a1"))
        self._turn("and the bus times?", "Buses run at 07:30.", {"tool_used": True})

        self.cache.clear()

        self.assertEqual([[], ["a1"], [], []], self._stored_ids())

    def test_a_supplied_trace_replaces_the_stored_one(self):
        """Keeping what is stored must not outrank what this save was actually given."""
        from langchain_core.messages import AIMessage, HumanMessage

        self._turn("what is the uniform?", "Navy. [1]", self._trace("a1"))
        messages = [HumanMessage(content="what is the uniform?"), AIMessage(content="Navy. [1]")]
        self.storage.save(
            "u", "s", messages, metadata={},
            extra_message_data=[None, {"rag_trace": trace_for_storage(self._trace("a2"))}],
        )

        self.assertEqual([[], ["a2"]], self._stored_ids())

    def test_a_rewritten_conversation_is_replaced_rather_than_appended_to(self):
        """Position alone is not identity. Roles that disagree mean the conversation was
        rewritten, and appending to it would splice two conversations together."""
        from langchain_core.messages import AIMessage, HumanMessage

        self._turn("what is the uniform?", "Navy. [1]", self._trace("a1"))
        self.storage.save("u", "s", [AIMessage(content="Navy. [1]"), HumanMessage(content="ok")])

        self.assertEqual([[], []], self._stored_ids())
        self.assertEqual(["ai", "human"], [row.message_type for row in self._rows()])

    def test_a_shorter_conversation_replaces_what_was_stored(self):
        from langchain_core.messages import HumanMessage

        self._turn("what is the uniform?", "Navy. [1]", self._trace("a1"))
        self.storage.save("u", "s", [HumanMessage(content="starting over")])

        self.assertEqual(1, len(self._rows()))

    def test_the_cached_copy_and_the_database_agree(self):
        """`get_session_messages` prefers the cache, so a fix that only reached the
        database would still hand a live client the wrong history."""
        self._turn("what is the uniform?", "Navy. [1]", self._trace("a1"))
        self._turn("and the bus times?", "Buses run at 07:30.", {"tool_used": True})

        cached = self._stored_ids()
        self.cache.clear()
        self.assertEqual(cached, self._stored_ids())

    def test_a_turn_appends_rather_than_rewriting_the_conversation(self):
        """Three turns used to cost three deletes and twelve inserts. The rows that were
        already right are left alone now, which is what keeps their ids and timestamps."""
        self._turn("what is the uniform?", "Navy. [1]", self._trace("a1"))
        first_pass = [(row.id, row.timestamp) for row in self._rows()]

        self._turn("and the bus times?", "Buses run at 07:30.")
        self._turn("thanks", "Any time.")
        second_pass = [(row.id, row.timestamp) for row in self._rows()]

        self.assertEqual(first_pass, second_pass[:2], "existing rows were rewritten")
        self.assertEqual(6, len(second_pass))

    def _page(self, limit=None, before_id=None, cold=False):
        if cold:
            self.cache.clear()
        return self.storage.get_session_page("u", "s", limit=limit, before_id=before_id)

    def _long_conversation(self, turns=10):
        for index in range(turns):
            self._turn(f"question {index}", f"answer {index}")

    def test_opening_a_conversation_reads_a_batch_not_all_of_it(self):
        self._long_conversation(turns=10)  # 20 messages

        for cold in (False, True):
            with self.subTest(cache="cold" if cold else "warm"):
                page = self._page(limit=6, cold=cold)
                self.assertEqual(
                    ["question 7", "answer 7", "question 8", "answer 8", "question 9", "answer 9"],
                    [message["content"] for message in page["messages"]],
                    "a page must be the NEWEST messages, in reading order",
                )
                self.assertTrue(page["has_more"])

    def test_scrolling_back_walks_the_conversation_without_gaps_or_repeats(self):
        self._long_conversation(turns=10)

        for cold in (False, True):
            with self.subTest(cache="cold" if cold else "warm"):
                if cold:
                    self.cache.clear()
                seen, cursor, pages = [], None, 0
                while True:
                    page = self.storage.get_session_page("u", "s", limit=6, before_id=cursor)
                    seen = [message["content"] for message in page["messages"]] + seen
                    pages += 1
                    if not page["has_more"]:
                        break
                    cursor = page["messages"][0]["id"]

                expected = [
                    text
                    for index in range(10)
                    for text in (f"question {index}", f"answer {index}")
                ]
                self.assertEqual(expected, seen)
                self.assertEqual(4, pages, "20 messages in batches of 6")

    def test_the_last_batch_reports_that_there_is_nothing_older(self):
        self._long_conversation(turns=2)  # 4 messages

        page = self._page(limit=10)
        self.assertFalse(page["has_more"])
        self.assertEqual(4, len(page["messages"]))

    def test_a_page_carries_the_cursor_its_own_scroll_back_needs(self):
        """Ids come from the cache as well as the database, or the first scroll-back
        after a save would have nothing to page from."""
        self._long_conversation(turns=3)

        warm = self._page(limit=2)
        cold = self._page(limit=2, cold=True)

        self.assertEqual(
            [message["id"] for message in cold["messages"]],
            [message["id"] for message in warm["messages"]],
            "the cached page and the database page must agree on the cursor",
        )
        self.assertTrue(all(isinstance(message["id"], int) for message in warm["messages"]))

    def test_a_page_never_pollutes_the_whole_conversation_cache(self):
        """`load` reads that key for the agent's history. A slice left under it would
        quietly truncate the conversation the model can see."""
        self._long_conversation(turns=5)
        self.cache.clear()

        self.storage.get_session_page("u", "s", limit=2)

        self.assertEqual(10, len(self.storage.load("u", "s")))

    def test_an_absent_session_pages_as_empty_rather_than_failing(self):
        page = self.storage.get_session_page("u", "does-not-exist", limit=5)
        self.assertEqual({"messages": [], "has_more": False}, page)

    def test_a_page_size_cannot_be_used_to_pull_the_whole_conversation(self):
        self._long_conversation(turns=3)
        self.assertLessEqual(
            len(self.storage.get_session_page("u", "s", limit=10_000)["messages"]),
            self.storage.MAX_PAGE_SIZE,
        )

    def test_an_image_on_an_older_message_is_paged_back_with_it(self):
        """The images have to survive the scroll-back too, not just the first batch."""
        self._turn("what is the uniform?", "Navy. [1]", self._trace("a1"))
        self._long_conversation(turns=8)

        page = self._page(limit=4)
        self.assertEqual([[], [], [], []], [
            list((m["rag_trace"] or {}).get("asset_ids") or []) for m in page["messages"]
        ])

        oldest = self.storage.get_session_page("u", "s", limit=4, before_id=page["messages"][0]["id"])
        while oldest["has_more"]:
            oldest = self.storage.get_session_page(
                "u", "s", limit=4, before_id=oldest["messages"][0]["id"]
            )
        self.assertEqual(["a1"], oldest["messages"][1]["rag_trace"]["asset_ids"])

    def test_the_stored_conversation_does_not_carry_a_copy_of_the_image(self):
        """The point of storing ids. An inline rendition is the picture itself, base64'd
        — persisting it would grow the conversation with the images in it, once per
        message that showed them."""
        self._turn("show me the chart", "Here. [1]", self._trace("a1", inline=True))

        stored = json.dumps([row.rag_trace for row in self._rows()])
        self.assertNotIn("base64", stored)
        self.assertIn("a1", stored)
        self.assertLess(len(stored), 200)


class StoredPointerRoundTripTests(AssetStoreTestCase):
    """Storing a pointer is only worth it if loading resolves it — by key, in one call.

    The setUp inherited here registers a real asset store over SQLite with one dossier
    in it, so this exercises the actual lookup rather than a stub.
    """

    def _record(self, asset_ids, message_type="ai"):
        return {
            "type": message_type,
            "content": "Here it is. [1]",
            "timestamp": "2026-01-01T00:00:00",
            "rag_trace": {"tool_used": True, "asset_ids": list(asset_ids)},
        }

    def test_an_id_is_resolved_into_something_the_client_can_display(self):
        restored = restore_session_assets([self._record([self.dossier.asset_id])])
        asset = restored[0]["rag_trace"]["assets"][0]

        self.assertEqual(self.dossier.asset_id, asset["asset_id"])
        self.assertEqual(asset_url_path(self.dossier.asset_id), asset["url"])
        self.assertEqual("reference", asset["mode"])

    def test_a_whole_session_costs_one_lookup(self):
        """Resolving per message would be a query per message. Ten messages showing the
        same figure is one `IN` query, and it stays one as the conversation grows."""
        records = [self._record([self.dossier.asset_id]) for _ in range(10)]

        with patch.object(
            self.store, "get_many", wraps=self.store.get_many
        ) as get_many:
            restored = restore_session_assets(records)

        self.assertEqual(1, get_many.call_count)
        self.assertTrue(all(item["rag_trace"]["assets"] for item in restored))

    def test_an_id_that_no_longer_resolves_yields_no_image_rather_than_a_broken_one(self):
        """Its document was deleted or re-ingested under new ids. A card pointing at
        nothing is worse than no card."""
        restored = restore_session_assets([self._record(["gone::p1::img0"])])
        self.assertEqual([], restored[0]["rag_trace"]["assets"])

    def test_messages_stored_before_ids_keep_their_renditions(self):
        """Conversations saved by the previous shape carry `assets` and no ids. They are
        left exactly as they are rather than being emptied."""
        legacy = {
            "type": "ai",
            "content": "Here it is.",
            "timestamp": "2026-01-01T00:00:00",
            "rag_trace": {"assets": [{"asset_id": "old", "mode": "reference", "url": "/media/old"}]},
        }
        restored = restore_session_assets([legacy, self._record([self.dossier.asset_id])])

        self.assertEqual("old", restored[0]["rag_trace"]["assets"][0]["asset_id"])
        self.assertEqual(self.dossier.asset_id, restored[1]["rag_trace"]["assets"][0]["asset_id"])

    def test_a_conversation_without_images_is_handed_back_untouched(self):
        records = [{"type": "human", "content": "hello", "timestamp": "t", "rag_trace": None}]
        self.assertIs(records, restore_session_assets(records))

    def test_a_store_failure_costs_the_pictures_not_the_history(self):
        with patch("backend.assets.store.get_asset_store", side_effect=RuntimeError("db down")):
            restored = restore_session_assets([self._record([self.dossier.asset_id])])
        self.assertEqual("Here it is. [1]", restored[0]["content"])


class SchemaContractTests(unittest.TestCase):
    def test_retrieved_chunks_carry_asset_ids_through_normalisation(self):
        from backend.schemas.chat import normalize_rag_trace

        trace = normalize_rag_trace({
            "retrieved_chunks": [{
                "filename": "doc.pdf", "text": "x", "chunk_id": "c1",
                "modality": "figure", "asset_ids": ["doc.pdf::p1::img0"],
            }],
        })
        chunk = trace["retrieved_chunks"][0]
        self.assertEqual(["doc.pdf::p1::img0"], chunk["asset_ids"])
        self.assertEqual("figure", chunk["modality"])

    def test_the_chat_response_exposes_assets_independently_of_the_trace(self):
        """A client must be able to show images without depending on the trace shape."""
        from backend.schemas.chat import ChatResponse

        response = ChatResponse(response="hi", assets=[AssetReference(asset_id="a")])
        payload = response.model_dump(mode="json")
        self.assertEqual("a", payload["assets"][0]["asset_id"])
        self.assertIsNone(payload["rag_trace"])

    def test_the_chat_request_accepts_declared_capabilities(self):
        from backend.schemas.chat import ChatRequest

        request = ChatRequest(
            message="hi", client_capabilities={"accepts_images": False, "max_assets": 2}
        )
        self.assertFalse(request.client_capabilities.accepts_images)

    def test_omitting_capabilities_is_valid(self):
        from backend.schemas.chat import ChatRequest

        self.assertIsNone(ChatRequest(message="hi").client_capabilities)


class AssetRouteTests(unittest.TestCase):
    """The HTTP surface, exercised through FastAPI with auth overridden."""

    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from backend.api.routes.assets import router
        from backend.assets.blobs import set_blob_store
        from backend.assets.store import set_asset_store
        from backend.db.models import AssetExtraction, DocumentAsset, User
        from backend.infra.auth import get_current_user

        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        DocumentAsset.__table__.create(self.engine)
        AssetExtraction.__table__.create(self.engine)
        session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

        self._tmp = TemporaryDirectory()
        self.blobs = LocalBlobStore(Path(self._tmp.name))
        self.store = AssetStore(session_factory=session_factory, blob_store=self.blobs,
                                cache_enabled=False)
        set_asset_store(self.store)
        set_blob_store(self.blobs)

        self.data = make_png(50, 50)
        digest = compute_sha256(self.data)
        uri = self.blobs.put(digest, self.data, "image/png")
        self.dossier = make_dossier(uri=uri, sha256=digest, byte_size=len(self.data))
        self.store.record(self.dossier)

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: User(id=1, username="u", role="user")
        self.client = TestClient(app)

    def tearDown(self):
        from backend.assets.blobs import set_blob_store
        from backend.assets.store import set_asset_store

        set_asset_store(None)
        set_blob_store(None)
        self._tmp.cleanup()
        self.engine.dispose()

    def _url(self, suffix=""):
        return asset_url_path(self.dossier.asset_id) + suffix

    def test_bytes_are_served_with_the_right_content_type(self):
        response = self.client.get(self._url())
        self.assertEqual(200, response.status_code)
        self.assertEqual("image/png", response.headers["content-type"])
        self.assertEqual(self.data, response.content)

    def test_content_addressing_gives_a_strong_etag_and_a_304(self):
        first = self.client.get(self._url())
        etag = first.headers["etag"]
        self.assertEqual(f'"{self.dossier.sha256}"', etag)
        second = self.client.get(self._url(), headers={"If-None-Match": etag})
        self.assertEqual(304, second.status_code)
        self.assertEqual(b"", second.content)

    def test_immutable_caching_is_advertised(self):
        headers = self.client.get(self._url()).headers
        self.assertIn("immutable", headers["cache-control"])
        self.assertEqual("nosniff", headers["x-content-type-options"])

    def test_an_unknown_asset_is_a_404(self):
        self.assertEqual(404, self.client.get("/media/nope").status_code)

    def test_metadata_returns_a_reference_without_bytes(self):
        response = self.client.get(self._url("/metadata"))
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("Fee schedule", body["caption"])
        self.assertIsNone(body.get("inline_data"))

    def test_resolve_returns_references_and_reports_the_missing(self):
        response = self.client.post(
            "/media/resolve",
            json={"asset_ids": [self.dossier.asset_id, "ghost"]},
        )
        body = response.json()
        self.assertEqual([self.dossier.asset_id], [a["asset_id"] for a in body["assets"]])
        self.assertEqual(["ghost"], body["missing"])

    def test_resolve_honours_declared_capabilities(self):
        """The integration path: a bot asks for inline bytes and gets them."""
        response = self.client.post("/media/resolve", json={
            "asset_ids": [self.dossier.asset_id],
            "capabilities": {"prefers_inline": True, "max_inline_bytes": 10_000_000},
        })
        asset = response.json()["assets"][0]
        self.assertEqual("inline", asset["mode"])
        self.assertTrue(asset["inline_data"].startswith("data:image/png;base64,"))

    def test_resolve_with_no_ids_is_an_empty_success(self):
        body = self.client.post("/media/resolve", json={"asset_ids": []}).json()
        self.assertEqual({"assets": [], "missing": []}, body)

    def test_a_missing_blob_reports_404_not_500(self):
        self.blobs.delete(self.dossier.blob.uri)
        self.assertEqual(404, self.client.get(self._url()).status_code)

    def test_asset_support_can_be_disabled_by_profile(self):
        from backend.profiles.registry import load_profile as load, set_profile

        profile = load("base").model_copy(deep=True)
        profile.assets.enabled = False
        set_profile(profile)
        try:
            self.assertEqual(404, self.client.get(self._url()).status_code)
        finally:
            set_profile(None)


class CitationFilteringTests(unittest.TestCase):
    """Retrieval surfaces every nearby figure; the answer rests on one.

    Which one is read out of the `[n]` markers the agent emits anyway — no second model
    call, and the chunk header's [FIGURE] marker is what makes the choice deliberate.
    When the markers cannot select, the best-ranked figure is shown and ONLY that one:
    showing three pictures to someone who asked about the PE kit is the failure this
    class exists to prevent, and showing none is the other one."""

    def setUp(self):
        self.delivery = load_profile("base").assets.delivery
        self.trace = {
            "retrieved_chunks": [
                {"filename": "kb.docx", "asset_ids": ["a1", "a2"]},   # [1]
                {"filename": "kb.docx", "asset_ids": []},             # [2] text only
                {"filename": "kb.docx", "asset_ids": ["a3"]},         # [3]
                {"filename": "kb.docx", "asset_ids": ["a4"]},         # [4]
            ]
        }
        self.ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        self.ctx.note_surfaced_assets(["a1", "a2", "a3", "a4"])

    def _ids(self, answer, config=None):
        from backend.chat.assets_bridge import asset_ids_for_answer

        return asset_ids_for_answer(answer, self.ctx, self.trace, config or self.delivery)

    def test_citation_markers_are_parsed_in_all_the_shapes_a_model_emits(self):
        from backend.chat.assets_bridge import cited_chunk_indices

        self.assertEqual([1], cited_chunk_indices("The uniform is navy [1]."))
        self.assertEqual([2, 3], cited_chunk_indices("Both apply [2][3]."))
        self.assertEqual([1, 4], cited_chunk_indices("See [1, 4] for details."))
        self.assertEqual([3, 1], cited_chunk_indices("First [3], then [1], again [3]."))
        self.assertEqual([], cited_chunk_indices("No citations at all."))
        self.assertEqual([], cited_chunk_indices(""))

    def test_only_the_cited_chunks_assets_are_attached(self):
        """The behaviour asked for: one question, one figure — not every figure the
        retriever happened to touch."""
        self.assertEqual(["a1", "a2"], self._ids("Sports kit is navy blue. [1]"))
        self.assertEqual(["a3"], self._ids("EYFS wear white polo shirts. [3]"))

    def test_several_citations_attach_all_of_their_assets_in_cited_order(self):
        self.assertEqual(["a3", "a1", "a2"], self._ids("Compare [3] with [1]."))

    def test_citing_only_text_chunks_falls_back_instead_of_showing_nothing(self):
        """The regression this class exists for now.

        This used to return []: "the answer cited no figure, so it used no figure".
        That reading is wrong about how a figure reaches the model. It arrives as a
        text surrogate — caption, description, transcription — and by the time the
        model picks a marker it looks exactly like a paragraph, so an answer written
        entirely FROM a caption routinely cites the prose beside it.

        The cost was paid by the question that most wants a picture: a parent asking
        "فين صورة اللبس؟" got four bullets read off the figure, one [2] pointing at the
        uniform policy text, and no image at all.

        One image, not four: the citation told us nothing, so retrieval's own ranking
        decides, and it ranked a1's chunk first.
        """
        self.assertEqual(["a1"], self._ids("The policy says so. [2]"))

    def test_the_answer_that_lost_its_figure_keeps_it(self):
        """The deployed shape, verbatim: prose off the figure, one marker on the text
        chunk, and an image link the model invented because nothing had rendered one."""
        answer = """الملابس عبارة عن زي رياضي أزرق داكن مع تفاصيل ذهبية.
تقدر تشوف الصورة هنا:
![Sports Wear: All Grades - Unisex](/media/kb.docx::p0::img5) [2]"""
        self.assertEqual(["a1"], self._ids(answer))

    def test_a_figure_citation_beside_a_text_one_still_narrows(self):
        """The guard on the fallback: it must not swallow the feature it backs up.

        [1] carries figures and [2] does not. The answer named a figure, so the
        narrowing is real information and the other document's images stay out.
        """
        self.assertEqual(["a1", "a2"], self._ids("Both the kit [1] and the rule [2]."))

    def test_the_fallback_shows_one_figure_and_it_is_the_best_ranked(self):
        """Retrieval's rank is the only signal left once the citation has failed to
        select, and it is a good one — the chunk it put first is the chunk that matched
        the question best. Taking all of them instead is what put two pictures of day
        wear under an answer about the PE kit."""
        for answer in ("No markers here.", "Text only. [2]", "See [42]."):
            with self.subTest(answer=answer):
                self.assertEqual(["a1"], self._ids(answer))

    def test_an_uncited_answer_still_shows_the_picture_it_probably_used(self):
        """A model that forgets its markers has not said "no image" — it has said
        nothing, and the turn still retrieved a figure worth showing."""
        self.assertEqual(["a1"], self._ids("The uniform is navy blue."))

    def test_out_of_range_citations_are_ignored(self):
        """Models occasionally cite a chunk number that was never returned."""
        self.assertEqual(["a1", "a2"], self._ids("See [1] and [99]."))

    def test_citations_that_are_all_out_of_range_fall_back(self):
        """A marker pointing at nothing is a model mistake, not a statement that the
        turn had no figures — the same reason a text-only citation falls back."""
        self.assertEqual(["a1"], self._ids("See [42]."))

    def test_the_feature_can_be_switched_off_by_profile(self):
        config = self.delivery.model_copy(update={"attach_only_cited": False})
        self.assertEqual(["a1", "a2", "a3", "a4"], self._ids("Sports kit. [1]", config))

    def test_no_surfaced_assets_short_circuits(self):
        empty_ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        from backend.chat.assets_bridge import asset_ids_for_answer

        self.assertEqual([], asset_ids_for_answer("Answer [1].", empty_ctx, {}, self.delivery))

    def test_a_trace_without_chunks_falls_back_rather_than_dropping_everything(self):
        """Nothing to map the markers against, so the same one-figure fallback applies."""
        from backend.chat.assets_bridge import asset_ids_for_answer

        self.assertEqual(
            ["a1"],
            asset_ids_for_answer("Answer [1].", self.ctx, {}, self.delivery),
        )

    def test_duplicate_assets_across_cited_chunks_appear_once(self):
        trace = {"retrieved_chunks": [{"asset_ids": ["a1"]}, {"asset_ids": ["a1", "a2"]}]}
        from backend.chat.assets_bridge import asset_ids_for_answer

        self.assertEqual(["a1", "a2"], asset_ids_for_answer("[1][2]", self.ctx, trace, self.delivery))


class AttachmentEdgeCaseTests(unittest.TestCase):
    """The paths that reach `asset_ids_for_answer` without going through a normal
    answered turn. The fallback shows a figure on more turns than the citation alone
    would, so the cases where attaching NOTHING is still right are pinned here."""

    def setUp(self):
        self.delivery = load_profile("base").assets.delivery
        self.empty_ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")

    def _ids(self, answer, ctx, trace, config=None):
        from backend.chat.assets_bridge import asset_ids_for_answer

        return asset_ids_for_answer(answer, ctx, trace, config or self.delivery)

    def test_a_refusal_turn_attaches_nothing_even_though_retrieval_saw_figures(self):
        """`decide_route` empties `retrieved_chunks` on no_knowledge, clarify, scope_select
        and retrieval_error precisely so attachment finds nothing there, and keeps the full
        set in `initial_retrieved_chunks` for the trace panel.

        That contract now carries more weight than it did: the fallback fires on far more
        turns, so "the knowledge base has nothing reliable on this" arriving WITH the
        picture that answers the question is a live failure mode rather than a theoretical
        one. The knowledge tool returns before it pins anything on these routes, so the
        context is empty too — both halves are asserted.
        """
        trace = {
            "route": "no_knowledge",
            "retrieval_status": "no_knowledge",
            "retrieved_chunks": [],
            "initial_retrieved_chunks": [{"asset_ids": ["a1"]}, {"asset_ids": ["a2"]}],
        }
        self.assertEqual([], self._ids("I don't have that.", self.empty_ctx, trace))

        from backend.chat.assets_bridge import asset_ids_for_turn

        self.assertEqual([], asset_ids_for_turn(self.empty_ctx, trace))

    def test_the_hitl_resume_path_attaches_from_the_trace(self):
        """Resuming a clarification answers from `docs` without calling the knowledge
        tool, so nothing is ever pinned on the context. The trace is the only record
        that the turn had figures at all."""
        trace = {"retrieved_chunks": [{"asset_ids": ["a1"]}, {"asset_ids": ["a2"]}]}
        self.assertEqual(["a1"], self._ids("Year 4 it is. [1]", self.empty_ctx, trace))
        self.assertEqual(["a2"], self._ids("Year 4 it is. [2]", self.empty_ctx, trace))
        self.assertEqual(["a1"], self._ids("Year 4 it is.", self.empty_ctx, trace))

    def test_a_turn_with_no_figures_anywhere_attaches_nothing(self):
        trace = {"retrieved_chunks": [{"asset_ids": []}, {}]}
        self.assertEqual([], self._ids("Fees are 45,000. [1]", self.empty_ctx, trace))

    def test_chunks_that_never_carried_the_field_are_read_as_no_figure(self):
        """A chunk stored before `asset_ids` existed, or one restored from a resume
        snapshot that dropped it, must not raise into a chat turn."""
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        ctx.note_surfaced_assets(["a1"])
        for chunks in ([{"filename": "kb.docx"}], [{"asset_ids": None}]):
            with self.subTest(chunks=chunks):
                trace = {"retrieved_chunks": chunks}
                self.assertEqual(["a1"], self._ids("Text. [1]", ctx, trace))

    def test_blank_ids_inside_a_cited_chunk_are_skipped(self):
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        ctx.note_surfaced_assets(["a1"])
        trace = {"retrieved_chunks": [{"asset_ids": ["", "a1", None]}]}
        self.assertEqual(["a1"], self._ids("The kit. [1]", ctx, trace))

    def test_switching_the_feature_off_ignores_citations_entirely(self):
        """`attach_only_cited: false` is the deployment that resolves assets itself.
        Neither the narrowing nor the fallback may change what it gets."""
        config = self.delivery.model_copy(update={"attach_only_cited": False})
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        ctx.note_surfaced_assets(["a1", "a2"])
        trace = {"retrieved_chunks": [{"asset_ids": ["a1"]}, {"asset_ids": ["a2"]}]}
        for answer in ("Cited [1].", "Cited [2].", "Uncited."):
            with self.subTest(answer=answer):
                self.assertEqual(["a1", "a2"], self._ids(answer, ctx, trace, config))


class FigureMarkerTests(unittest.TestCase):
    """What the model is told about a figure, and what it is deliberately NOT told.

    It gets a bare [FIGURE] marker: enough to know which chunks carry a picture, so its
    `[n]` is a deliberate choice of which one to show, and nothing it could try to
    render itself. The header used to name the asset_id — the argument view_figure
    needed — and an id in the prompt became an id in the answer: shown one and told
    markdown is supported, a 20B model wrote it straight back as an image link that no
    browser could load.
    """

    def _run_tool(self, docs):
        import sys
        import types

        from backend.tools.knowledge import make_search_knowledge_base

        fake_pipeline = types.ModuleType("backend.rag.pipeline")
        fake_pipeline.run_rag_graph = lambda query, ctx: {
            "docs": docs,
            "rag_trace": {"retrieval_status": "answer", "route": "answer"},
        }
        ctx = ChatRequestContext.for_sync(user_id="u", session_id="s")
        try:
            tool = make_search_knowledge_base(ctx)
            with patch.dict(sys.modules, {"backend.rag.pipeline": fake_pipeline}):
                return tool.invoke({"query": "where is the uniform picture"}), ctx
        finally:
            ctx.close()

    @staticmethod
    def _doc(text, asset_ids=()):
        return {
            "filename": "kb.docx",
            "page_number": 0,
            "text": text,
            "chunk_id": "c1",
            "asset_ids": list(asset_ids),
        }

    def test_a_figure_chunk_is_marked_and_its_id_is_never_shown(self):
        message, ctx = self._run_tool([self._doc("[Figure] Sports Wear", ["kb.docx::p0::img5"])])
        self.assertIn("[FIGURE]", message)
        self.assertNotIn("kb.docx::p0::img5", message)
        self.assertNotIn("asset_id", message)
        # Surfaced on the context all the same — that is what a citation resolves
        # against when the turn decides which picture to attach.
        self.assertEqual(["kb.docx::p0::img5"], ctx.surfaced_asset_ids())

    def test_the_marker_is_explained_as_a_selector_not_a_label(self):
        """Without this the model cites the prose beside a figure as readily as the
        figure itself, and the answer describes a picture nobody attached."""
        message, _ = self._run_tool([self._doc("[Figure] Sports Wear", ["kb.docx::p0::img5"])])
        self.assertIn("shown to the user whenever you cite that chunk", message)
        self.assertIn("Cite the [FIGURE] chunk you actually described", message)

    def test_a_text_chunk_carries_no_marker(self):
        message, _ = self._run_tool([self._doc("Fees are 45,000 EGP.")])
        self.assertNotIn("[FIGURE]", message)

    def test_modality_alone_marks_a_chunk_whose_ids_did_not_survive(self):
        """A chunk stored before `asset_ids` existed still says what it is."""
        doc = self._doc("[Figure] Sports Wear")
        doc["modality"] = "figure"
        message, _ = self._run_tool([doc])
        self.assertIn("[FIGURE]", message)

    def test_a_text_only_turn_does_not_pay_for_the_rule(self):
        """Rung 3 of the composition ladder: an instruction that is only true when
        retrieval returned a figure is billed only on those turns."""
        with_figure, _ = self._run_tool([self._doc("[Figure] Sports Wear", ["kb.docx::p0::img5"])])
        without, ctx = self._run_tool([self._doc("Fees are 45,000 EGP.")])
        self.assertNotIn("marked [FIGURE]", without)
        self.assertLess(len(without), len(with_figure))
        self.assertEqual([], ctx.surfaced_asset_ids())

    def test_one_figure_among_text_chunks_still_triggers_the_rule(self):
        message, _ = self._run_tool([
            self._doc("Fees are 45,000 EGP."),
            self._doc("[Figure] Sports Wear", ["kb.docx::p0::img5"]),
        ])
        self.assertIn("marked [FIGURE]", message)


if __name__ == "__main__":
    unittest.main()
