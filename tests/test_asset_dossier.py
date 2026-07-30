"""AssetDossier schema, content-addressed blob storage, and the asset store.

The store is exercised against in-memory SQLite through an injected session factory,
so the whole persistence layer — including delete cascades, the extraction cache, and
the version backfill — is covered without a Postgres instance.
"""
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.assets.dossier as dossier_module
from backend.assets.blobs import LocalBlobStore, extension_for
from backend.assets.dossier import (
    DOSSIER_VERSION,
    AssetDossier,
    AssetRole,
    AssetTier,
    ExtractionPayload,
    ExtractionStatus,
    Migration,
    Provenance,
    SourceRef,
    TextSurface,
    build_asset_id,
    compute_sha256,
    migrate_payload,
)
from backend.assets.store import AssetStore


def make_dossier(
    asset_id="doc.pdf::p1::img0",
    sha256=None,
    filename="doc.pdf",
    page=1,
    role=AssetRole.FIGURE,
    tier=AssetTier.SIMPLE,
    status=ExtractionStatus.EXTRACTED,
    caption="Grade 5 fee schedule",
    uri="file://ab/cd/abcd.png",
    version=DOSSIER_VERSION,
    extraction=True,
):
    payload = None
    if extraction:
        payload = ExtractionPayload(
            text=TextSurface(
                caption=caption,
                description="A table of tuition fees by grade.",
                transcription="Grade | Fee\n5 | 42000",
                tags=["fees", "tuition"],
            ),
            answerable_questions=["How much is grade 5 tuition?"],
            provenance=Provenance(
                tier=tier,
                model_used="test-vlm",
                confidence=0.9,
                extracted_at=datetime.now(UTC),
            ),
        )
    return AssetDossier(
        asset_id=asset_id,
        sha256=sha256 or compute_sha256(asset_id.encode()),
        profile="base",
        dossier_version=version,
        role=role,
        tier=tier,
        status=status,
        source=SourceRef(filename=filename, page_number=page),
        blob={"uri": uri, "content_type": "image/png", "byte_size": 1234},
        extraction=payload,
    )


class DossierModelTests(unittest.TestCase):
    def test_asset_id_is_deterministic_and_mirrors_chunk_id_shape(self):
        self.assertEqual("doc.pdf::p3::img2", build_asset_id("doc.pdf", 3, 2))
        self.assertEqual(build_asset_id("doc.pdf", 3, 2), build_asset_id("doc.pdf", 3, 2))
        # Whitespace in a filename must not produce two ids for one asset.
        self.assertEqual("a b.pdf::p0::img0", build_asset_id("a  \n b.pdf", 0, 0))

    def test_render_surrogate_orders_most_specific_first(self):
        text = make_dossier().render_surrogate()
        self.assertTrue(text.startswith("[Figure] Grade 5 fee schedule"))
        self.assertLess(text.index("A table of tuition"), text.index("Grade | Fee"))
        self.assertLess(text.index("Grade | Fee"), text.index("Tags:"))
        self.assertIn("Answers: How much is grade 5 tuition?", text)

    def test_render_surrogate_includes_section_path(self):
        text = make_dossier().render_surrogate(section_path=["Handbook", "Admissions", "Fees"])
        self.assertTrue(text.startswith("Handbook > Admissions > Fees"))

    def test_decorative_and_dropped_assets_are_never_indexable(self):
        self.assertFalse(make_dossier(role=AssetRole.DECORATIVE).is_indexable)
        self.assertFalse(make_dossier(tier=AssetTier.DROP).is_indexable)

    def test_unextracted_or_empty_assets_are_not_indexable(self):
        """An empty text surface must not reach the index: it is unfindable noise that
        still costs an embedding."""
        self.assertFalse(make_dossier(status=ExtractionStatus.PENDING, extraction=False).is_indexable)
        self.assertFalse(make_dossier(status=ExtractionStatus.FAILED).is_indexable)
        empty = make_dossier()
        empty.extraction.text = TextSurface()
        self.assertFalse(empty.is_indexable)
        self.assertEqual("", empty.render_surrogate())

    def test_round_trips_through_json(self):
        original = make_dossier()
        restored = AssetDossier.model_validate(original.model_dump(mode="json"))
        self.assertEqual(original.asset_id, restored.asset_id)
        self.assertEqual(original.extraction.text.caption, restored.extraction.text.caption)
        self.assertEqual(original.role, restored.role)

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(Exception):
            AssetDossier.model_validate({"asset_id": "a", "sha256": "b", "nonsense": 1})


class MigrationTests(unittest.TestCase):
    def test_shipped_migration_chain_is_contiguous(self):
        dossier_module.validate_migration_chain()

    def test_pure_migration_is_applied_in_place(self):
        migration = Migration(
            from_version=1,
            description="add a derived tag",
            transform=lambda p: {**p, "tier": "complex"},
        )
        with patch.dict(dossier_module.MIGRATIONS, {1: migration}, clear=True):
            payload, needs_reextraction = migrate_payload(
                make_dossier(version=1).model_dump(mode="json"), target_version=2
            )
        self.assertFalse(needs_reextraction)
        self.assertEqual(2, payload["dossier_version"])
        self.assertEqual("complex", payload["tier"])

    def test_reextraction_migration_is_flagged_not_faked(self):
        """A migration needing new model output must never silently invent it."""
        migration = Migration(from_version=1, description="new prompt", requires_reextraction=True)
        with patch.dict(dossier_module.MIGRATIONS, {1: migration}, clear=True):
            payload, needs_reextraction = migrate_payload(
                make_dossier(version=1).model_dump(mode="json"), target_version=2
            )
        self.assertTrue(needs_reextraction)
        self.assertEqual(2, payload["dossier_version"])

    def test_reextraction_flag_is_sticky_across_a_chain(self):
        chain = {
            1: Migration(1, "needs model", requires_reextraction=True),
            2: Migration(2, "pure reshape", transform=lambda p: p),
        }
        with patch.dict(dossier_module.MIGRATIONS, chain, clear=True):
            _, needs_reextraction = migrate_payload(
                make_dossier(version=1).model_dump(mode="json"), target_version=3
            )
        self.assertTrue(needs_reextraction)

    def test_missing_migration_raises(self):
        with patch.dict(dossier_module.MIGRATIONS, {}, clear=True):
            with self.assertRaises(RuntimeError):
                migrate_payload(make_dossier(version=1).model_dump(mode="json"), target_version=2)


class LocalBlobStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = LocalBlobStore(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_put_is_content_addressed_and_sharded(self):
        data = b"image-bytes"
        digest = compute_sha256(data)
        uri = self.store.put(digest, data, "image/png")
        self.assertEqual(f"file://{digest[:2]}/{digest[2:4]}/{digest}.png", uri)
        self.assertEqual(data, self.store.get(uri))

    def test_put_is_idempotent_and_deduplicates(self):
        data = b"same-bytes"
        digest = compute_sha256(data)
        first = self.store.put(digest, data, "image/png")
        second = self.store.put(digest, data, "image/png")
        self.assertEqual(first, second)
        self.assertEqual(1, len(list(self.root.rglob("*.png"))))

    def test_no_partial_files_remain_after_a_failed_write(self):
        data = b"x" * 32
        digest = compute_sha256(data)
        with patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.put(digest, data, "image/png")
        self.assertEqual([], list(self.root.rglob("*.part")))
        self.assertFalse(self.store.exists(digest, "image/png"))

    def test_invalid_digest_is_rejected_before_touching_the_filesystem(self):
        for bad in ("../../etc/passwd", "", "zz" * 32, "abc"):
            with self.subTest(digest=bad):
                with self.assertRaises(ValueError):
                    self.store.put(bad, b"data", "image/png")

    def test_uri_escaping_the_root_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.get("file://../../../etc/passwd")

    def test_delete_removes_the_blob_and_prunes_empty_shards(self):
        data = b"to-delete"
        digest = compute_sha256(data)
        uri = self.store.put(digest, data, "image/png")
        self.assertTrue(self.store.delete(uri))
        self.assertFalse(self.store.exists(digest, "image/png"))
        self.assertFalse((self.root / digest[:2]).exists())
        self.assertFalse(self.store.delete(uri))

    def test_extension_mapping_falls_back_safely(self):
        self.assertEqual(".png", extension_for("image/png"))
        self.assertEqual(".jpg", extension_for("IMAGE/JPEG"))
        self.assertEqual(".bin", extension_for("application/x-unknown"))
        self.assertEqual(".bin", extension_for(""))


class AssetStoreTestCase(unittest.TestCase):
    """In-memory SQLite via StaticPool so every session shares one database."""

    def setUp(self):
        from backend.db.models import AssetExtraction, DocumentAsset

        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        DocumentAsset.__table__.create(self.engine)
        AssetExtraction.__table__.create(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

        self._tmp = TemporaryDirectory()
        self.blobs = LocalBlobStore(Path(self._tmp.name))
        # cache_enabled=False keeps Redis out of the unit tests entirely.
        self.store = AssetStore(
            session_factory=self.session_factory,
            blob_store=self.blobs,
            cache_enabled=False,
        )

    def tearDown(self):
        self._tmp.cleanup()
        self.engine.dispose()

    def _store_blob(self, data: bytes) -> tuple[str, str]:
        digest = compute_sha256(data)
        return digest, self.blobs.put(digest, data, "image/png")


class AssetStorePersistenceTests(AssetStoreTestCase):
    def test_record_and_get_round_trip(self):
        original = make_dossier()
        self.store.record(original)
        loaded = self.store.get(original.asset_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(original.asset_id, loaded.asset_id)
        self.assertEqual("Grade 5 fee schedule", loaded.extraction.text.caption)
        self.assertTrue(loaded.is_indexable)

    def test_record_is_idempotent_on_asset_id(self):
        """Re-ingesting a document must update occurrences, not duplicate them."""
        self.store.record(make_dossier(caption="first"))
        self.store.record(make_dossier(caption="second"))
        assets = self.store.list_by_filename("doc.pdf")
        self.assertEqual(1, len(assets))
        self.assertEqual("second", assets[0].extraction.text.caption)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get("nope"))
        self.assertIsNone(self.store.get(""))

    def test_get_many_preserves_requested_order_and_skips_missing(self):
        self.store.record_many([
            make_dossier(asset_id="doc.pdf::p1::img0"),
            make_dossier(asset_id="doc.pdf::p2::img0", page=2),
        ])
        found = self.store.get_many(["doc.pdf::p2::img0", "missing", "doc.pdf::p1::img0"])
        self.assertEqual(["doc.pdf::p2::img0", "doc.pdf::p1::img0"], [d.asset_id for d in found])

    def test_list_by_filename_indexable_only_filters_unusable_assets(self):
        self.store.record_many([
            make_dossier(asset_id="doc.pdf::p1::img0"),
            make_dossier(asset_id="doc.pdf::p1::img1", role=AssetRole.DECORATIVE),
            make_dossier(asset_id="doc.pdf::p1::img2", status=ExtractionStatus.PENDING, extraction=False),
        ])
        self.assertEqual(3, len(self.store.list_by_filename("doc.pdf")))
        indexable = self.store.list_by_filename("doc.pdf", indexable_only=True)
        self.assertEqual(["doc.pdf::p1::img0"], [d.asset_id for d in indexable])


class ExtractionCacheTests(AssetStoreTestCase):
    def test_cache_miss_then_hit(self):
        payload = make_dossier().extraction
        self.assertIsNone(self.store.find_extraction("a" * 64, "base"))
        self.store.save_extraction("a" * 64, "base", payload)
        cached = self.store.find_extraction("a" * 64, "base")
        self.assertIsNotNone(cached)
        self.assertEqual("Grade 5 fee schedule", cached.text.caption)

    def test_cache_is_keyed_by_profile_and_version(self):
        """The same bytes under a different profile or schema version is a different
        result and must not be served from cache."""
        self.store.save_extraction("a" * 64, "base", make_dossier().extraction)
        self.assertIsNone(self.store.find_extraction("a" * 64, "ecommerce"))
        self.assertIsNone(self.store.find_extraction("a" * 64, "base", dossier_version=99))

    def test_save_extraction_upserts(self):
        digest = "b" * 64
        self.store.save_extraction(digest, "base", make_dossier(caption="v1").extraction)
        self.store.save_extraction(digest, "base", make_dossier(caption="v2").extraction)
        self.assertEqual("v2", self.store.find_extraction(digest, "base").text.caption)

    def test_attach_cached_extraction_skips_the_expensive_path(self):
        digest = "c" * 64
        self.store.save_extraction(digest, "base", make_dossier().extraction)

        pending = make_dossier(
            asset_id="other.pdf::p1::img0",
            sha256=digest,
            filename="other.pdf",
            status=ExtractionStatus.PENDING,
            extraction=False,
        )
        self.assertTrue(self.store.attach_cached_extraction(pending))
        self.assertEqual(ExtractionStatus.EXTRACTED, pending.status)
        self.assertTrue(pending.is_indexable)
        # Already-extracted dossiers are left alone.
        self.assertFalse(self.store.attach_cached_extraction(pending))

    def test_attach_returns_false_on_cache_miss(self):
        pending = make_dossier(status=ExtractionStatus.PENDING, extraction=False)
        self.assertFalse(self.store.attach_cached_extraction(pending))
        self.assertEqual(ExtractionStatus.PENDING, pending.status)

    def test_corrupt_cached_payload_is_ignored_rather_than_raising(self):
        from backend.db.models import AssetExtraction

        session = self.session_factory()
        session.add(
            AssetExtraction(sha256="d" * 64, profile="base", dossier_version=DOSSIER_VERSION,
                            payload={"text": {"unknown_field": 1}})
        )
        session.commit()
        session.close()
        self.assertIsNone(self.store.find_extraction("d" * 64, "base"))


class DeletionTests(AssetStoreTestCase):
    def test_delete_removes_assets_and_orphaned_blobs(self):
        digest, uri = self._store_blob(b"only-in-one-doc")
        self.store.record(make_dossier(sha256=digest, uri=uri))

        result = self.store.delete_by_filename("doc.pdf")
        self.assertEqual(1, result.assets_deleted)
        self.assertEqual(1, result.blobs_deleted)
        self.assertFalse(self.blobs.exists(digest, "image/png"))
        self.assertEqual([], self.store.list_by_filename("doc.pdf"))

    def test_shared_blob_is_retained_while_another_document_references_it(self):
        """The dedup property has a deletion consequence: a logo shared across two
        documents must survive the deletion of one of them."""
        digest, uri = self._store_blob(b"shared-logo")
        self.store.record_many([
            make_dossier(asset_id="a.pdf::p1::img0", sha256=digest, filename="a.pdf", uri=uri),
            make_dossier(asset_id="b.pdf::p1::img0", sha256=digest, filename="b.pdf", uri=uri),
        ])

        result = self.store.delete_by_filename("a.pdf")
        self.assertEqual(1, result.assets_deleted)
        self.assertEqual(0, result.blobs_deleted)
        self.assertEqual(1, result.blobs_retained)
        self.assertTrue(self.blobs.exists(digest, "image/png"))
        self.assertEqual(1, len(self.store.list_by_filename("b.pdf")))

    def test_gc_can_be_disabled(self):
        digest, uri = self._store_blob(b"keep-me")
        self.store.record(make_dossier(sha256=digest, uri=uri))
        result = self.store.delete_by_filename("doc.pdf", gc_orphan_blobs=False)
        self.assertEqual(0, result.blobs_deleted)
        self.assertTrue(self.blobs.exists(digest, "image/png"))

    def test_extractions_survive_document_deletion(self):
        """Extractions are the expensive artifact and are keyed by an irreversible
        digest, so a re-upload must not have to pay for them again."""
        digest, uri = self._store_blob(b"expensive")
        self.store.save_extraction(digest, "base", make_dossier().extraction)
        self.store.record(make_dossier(sha256=digest, uri=uri))

        self.store.delete_by_filename("doc.pdf")
        self.assertIsNotNone(self.store.find_extraction(digest, "base"))

    def test_delete_of_unknown_document_is_a_noop(self):
        result = self.store.delete_by_filename("never-existed.pdf")
        self.assertEqual(0, result.assets_deleted)

    def test_blob_delete_failure_does_not_break_the_delete(self):
        digest, uri = self._store_blob(b"stubborn")
        self.store.record(make_dossier(sha256=digest, uri=uri))
        with patch.object(self.blobs, "delete", side_effect=OSError("locked")):
            result = self.store.delete_by_filename("doc.pdf")
        self.assertEqual(1, result.assets_deleted)
        self.assertEqual(0, result.blobs_deleted)


class BackfillTests(AssetStoreTestCase):
    def _seed(self, count, version=1):
        self.store.record_many([
            make_dossier(asset_id=f"doc.pdf::p{i}::img0", page=i, version=version)
            for i in range(count)
        ])

    def test_iter_stale_paginates_without_skipping_mutated_rows(self):
        self._seed(7, version=1)
        batches = list(self.store.iter_stale(target_version=2, batch_size=3))
        self.assertEqual([3, 3, 1], [len(batch) for batch in batches])
        seen = {d.asset_id for batch in batches for d in batch}
        self.assertEqual(7, len(seen))

    def test_iter_stale_ignores_current_version_rows(self):
        self._seed(3, version=DOSSIER_VERSION)
        self.assertEqual([], list(self.store.iter_stale(target_version=DOSSIER_VERSION)))

    def test_pure_migration_backfill_updates_rows(self):
        self._seed(4, version=1)
        migration = Migration(1, "reshape", transform=lambda p: {**p, "tier": "complex"})
        with patch.dict(dossier_module.MIGRATIONS, {1: migration}, clear=True):
            report = self.store.backfill(target_version=2)

        self.assertEqual(4, report.scanned)
        self.assertEqual(4, report.migrated)
        self.assertEqual(0, report.marked_stale)
        upgraded = self.store.get("doc.pdf::p0::img0")
        self.assertEqual(2, upgraded.dossier_version)
        self.assertEqual(AssetTier.COMPLEX, upgraded.tier)

    def test_reextraction_migration_marks_rows_stale(self):
        self._seed(2, version=1)
        migration = Migration(1, "new prompt", requires_reextraction=True)
        with patch.dict(dossier_module.MIGRATIONS, {1: migration}, clear=True):
            report = self.store.backfill(target_version=2)

        self.assertEqual(2, report.marked_stale)
        self.assertEqual(0, report.migrated)
        row = self.store.get("doc.pdf::p0::img0")
        self.assertEqual(ExtractionStatus.STALE, row.status)
        self.assertFalse(row.is_indexable)

    def test_dry_run_changes_nothing(self):
        self._seed(3, version=1)
        migration = Migration(1, "reshape", transform=lambda p: {**p, "tier": "complex"})
        with patch.dict(dossier_module.MIGRATIONS, {1: migration}, clear=True):
            report = self.store.backfill(target_version=2, dry_run=True)

        self.assertEqual(3, report.scanned)
        self.assertEqual(3, report.migrated)
        self.assertEqual(1, self.store.get("doc.pdf::p0::img0").dossier_version)

    def test_backfill_reports_failures_without_aborting_the_run(self):
        self._seed(3, version=1)
        broken = Migration(1, "bad", transform=lambda p: {**p, "role": "not_a_role"})
        with patch.dict(dossier_module.MIGRATIONS, {1: broken}, clear=True):
            report = self.store.backfill(target_version=2)
        self.assertEqual(3, report.scanned)
        self.assertEqual(3, report.failed)
        self.assertEqual(1, self.store.get("doc.pdf::p0::img0").dossier_version)


class StatsTests(AssetStoreTestCase):
    def test_dedup_ratio_reports_cache_leverage(self):
        shared = compute_sha256(b"logo")
        self.store.record_many([
            make_dossier(asset_id=f"doc.pdf::p{i}::img0", page=i, sha256=shared)
            for i in range(5)
        ])
        self.store.record(make_dossier(asset_id="doc.pdf::p9::img0", page=9, sha256=compute_sha256(b"unique")))
        self.store.save_extraction(shared, "base", make_dossier().extraction)

        stats = self.store.stats()
        self.assertEqual(6, stats["assets"])
        self.assertEqual(2, stats["distinct_images"])
        self.assertEqual(3.0, stats["dedup_ratio"])
        self.assertEqual(1, stats["cached_extractions"])
        self.assertEqual({"extracted": 6}, stats["by_status"])

    def test_stats_on_an_empty_store(self):
        stats = self.store.stats()
        self.assertEqual(0, stats["assets"])
        self.assertEqual(0.0, stats["dedup_ratio"])


class FakeCache:
    """Minimal stand-in for the Redis cache, recording calls so the read-through and
    invalidation paths can be asserted without a Redis instance."""

    def __init__(self):
        self.data = {}
        self.gets = 0
        self.sets = 0
        self.deletes = 0

    def get_json(self, key):
        self.gets += 1
        return self.data.get(key)

    def set_json(self, key, value, ttl=None):
        self.sets += 1
        self.data[key] = value

    def delete(self, key):
        self.deletes += 1
        self.data.pop(key, None)


class CachedStoreTests(AssetStoreTestCase):
    def setUp(self):
        super().setUp()
        self.fake_cache = FakeCache()
        self.store = AssetStore(
            session_factory=self.session_factory,
            blob_store=self.blobs,
            cache=self.fake_cache,
        )

    def test_record_populates_the_cache(self):
        self.store.record(make_dossier())
        self.assertIn("asset:doc.pdf::p1::img0", self.fake_cache.data)

    def test_get_is_served_from_cache_without_touching_the_database(self):
        self.store.record(make_dossier())
        with patch.object(self.store, "_models", side_effect=AssertionError("hit the database")):
            loaded = self.store.get("doc.pdf::p1::img0")
        self.assertEqual("Grade 5 fee schedule", loaded.extraction.text.caption)

    def test_cache_miss_falls_back_to_the_database_and_repopulates(self):
        self.store.record(make_dossier())
        self.fake_cache.data.clear()
        loaded = self.store.get("doc.pdf::p1::img0")
        self.assertIsNotNone(loaded)
        self.assertIn("asset:doc.pdf::p1::img0", self.fake_cache.data)

    def test_unreadable_cached_payload_falls_back_to_the_database(self):
        """A payload written by an older schema must not take down a read."""
        self.store.record(make_dossier())
        self.fake_cache.data["asset:doc.pdf::p1::img0"] = {"totally": "wrong"}
        loaded = self.store.get("doc.pdf::p1::img0")
        self.assertIsNotNone(loaded)
        self.assertEqual("doc.pdf::p1::img0", loaded.asset_id)

    def test_delete_invalidates_the_cache(self):
        self.store.record(make_dossier())
        self.store.delete_by_filename("doc.pdf")
        self.assertEqual({}, self.fake_cache.data)
        self.assertIsNone(self.store.get("doc.pdf::p1::img0"))


class StoreEdgeCaseTests(AssetStoreTestCase):
    def test_record_many_ignores_empty_input_and_id_less_dossiers(self):
        self.assertEqual(0, self.store.record_many([]))
        self.assertEqual(0, self.store.record_many([None]))

    def test_mixed_batch_writes_only_valid_rows(self):
        written = self.store.record_many([make_dossier(), None])
        self.assertEqual(1, written)

    def test_get_many_with_no_ids_returns_empty(self):
        self.assertEqual([], self.store.get_many([]))
        self.assertEqual([], self.store.get_many(["", "  "]))

    def test_list_by_filename_requires_a_filename(self):
        self.assertEqual([], self.store.list_by_filename(""))

    def test_same_image_in_two_documents_yields_two_indexable_occurrences(self):
        digest, uri = self._store_blob(b"one-image-two-docs")
        self.store.record_many([
            make_dossier(asset_id="a.pdf::p1::img0", sha256=digest, filename="a.pdf", uri=uri),
            make_dossier(asset_id="b.pdf::p4::img0", sha256=digest, filename="b.pdf", page=4, uri=uri),
        ])
        self.assertEqual(1, len(self.store.list_by_filename("a.pdf", indexable_only=True)))
        self.assertEqual(1, len(self.store.list_by_filename("b.pdf", indexable_only=True)))
        self.assertEqual(2.0, self.store.stats()["dedup_ratio"])

    def test_unicode_filenames_survive_a_round_trip(self):
        asset_id = build_asset_id("دليل القبول.pdf", 2, 0)
        self.store.record(make_dossier(asset_id=asset_id, filename="دليل القبول.pdf", page=2))
        self.assertEqual(asset_id, self.store.get(asset_id).asset_id)
        self.assertEqual(1, len(self.store.list_by_filename("دليل القبول.pdf")))

    def test_save_and_find_extraction_reject_a_blank_digest(self):
        self.store.save_extraction("", "base", make_dossier().extraction)
        self.assertIsNone(self.store.find_extraction("", "base"))

    def test_backfill_with_nothing_stale_is_a_noop(self):
        self.store.record_many([make_dossier(version=DOSSIER_VERSION)])
        report = self.store.backfill(target_version=DOSSIER_VERSION)
        self.assertEqual(0, report.scanned)
        self.assertEqual({}, report.by_version)

    def test_backfill_report_serialises_for_the_cli(self):
        report = self.store.backfill(target_version=DOSSIER_VERSION)
        self.assertEqual(
            {"scanned", "migrated", "marked_stale", "failed", "by_version"},
            set(report.as_dict()),
        )

    def test_iter_stale_batch_larger_than_the_table(self):
        self.store.record_many([make_dossier(version=1)])
        batches = list(self.store.iter_stale(target_version=2, batch_size=500))
        self.assertEqual([1], [len(batch) for batch in batches])

    def test_stats_reports_mixed_statuses(self):
        self.store.record_many([
            make_dossier(asset_id="doc.pdf::p1::img0"),
            make_dossier(asset_id="doc.pdf::p1::img1", status=ExtractionStatus.FAILED),
            make_dossier(asset_id="doc.pdf::p1::img2", status=ExtractionStatus.PENDING, extraction=False),
        ])
        by_status = self.store.stats()["by_status"]
        self.assertEqual({"extracted": 1, "failed": 1, "pending": 1}, by_status)


class DossierMetadataTests(unittest.TestCase):
    def test_is_stale_compares_against_the_current_version(self):
        self.assertTrue(make_dossier(version=1).is_stale(current_version=2))
        self.assertFalse(make_dossier(version=2).is_stale(current_version=2))
        self.assertFalse(make_dossier(version=3).is_stale(current_version=2))

    def test_touch_sets_created_at_once_and_updates_every_time(self):
        dossier = make_dossier()
        dossier.touch()
        created = dossier.created_at
        self.assertIsNotNone(created)
        dossier.touch()
        self.assertEqual(created, dossier.created_at)
        self.assertGreaterEqual(dossier.updated_at, created)

    def test_relations_round_trip(self):
        from backend.assets.dossier import Relation, RelationKind

        dossier = make_dossier()
        dossier.relations = [
            Relation(kind=RelationKind.DEPICTS, target="SKU-123"),
            Relation(kind=RelationKind.PART_OF, target="doc.pdf::p1::l3::4", note="figure 2"),
        ]
        restored = AssetDossier.model_validate(dossier.model_dump(mode="json"))
        self.assertEqual([RelationKind.DEPICTS, RelationKind.PART_OF], [r.kind for r in restored.relations])
        self.assertEqual("figure 2", restored.relations[1].note)

    def test_structured_surface_round_trips_typed_attributes_and_measures(self):
        from backend.assets.dossier import Measure, StructuredSurface

        dossier = make_dossier()
        dossier.extraction.structured = StructuredSurface(
            attributes={"color": "red", "in_stock": True, "price": 42.5, "count": 3},
            measures=[Measure(label="Q3", value=41.0, unit="%", series="2026")],
            entities=["Grade 5"],
        )
        restored = AssetDossier.model_validate(dossier.model_dump(mode="json"))
        attributes = restored.extraction.structured.attributes
        self.assertEqual("red", attributes["color"])
        self.assertIs(True, attributes["in_stock"])
        self.assertEqual(42.5, attributes["price"])
        self.assertEqual("%", restored.extraction.structured.measures[0].unit)

    def test_surrogate_works_with_transcription_only(self):
        dossier = make_dossier()
        dossier.extraction.text = TextSurface(transcription="Grade | Fee\n5 | 42000")
        surrogate = dossier.render_surrogate()
        self.assertTrue(surrogate.startswith("[Figure]"))
        self.assertIn("Grade | Fee", surrogate)

    def test_surrogate_omits_absent_sections(self):
        dossier = make_dossier()
        dossier.extraction.text = TextSurface(caption="Just a caption")
        dossier.extraction.answerable_questions = []
        surrogate = dossier.render_surrogate()
        self.assertEqual("[Figure] Just a caption", surrogate)

    def test_source_bbox_is_optional_and_round_trips(self):
        dossier = make_dossier()
        dossier.source = SourceRef(filename="doc.pdf", page_number=2, bbox=[10.0, 20.0, 100.0, 80.0])
        restored = AssetDossier.model_validate(dossier.model_dump(mode="json"))
        self.assertEqual([10.0, 20.0, 100.0, 80.0], restored.source.bbox)
        self.assertIsNone(make_dossier().source.bbox)


class S3BlobStoreTests(unittest.TestCase):
    """Key and URI mapping are pure functions and are tested without boto3 or a bucket."""

    def setUp(self):
        from backend.assets.blobs import S3BlobStore

        self.store = S3BlobStore(bucket="kb-assets", endpoint_url="http://localhost:9000")

    def test_key_is_sharded_like_the_local_backend(self):
        digest = "a" * 64
        self.assertEqual(f"aa/aa/{digest}.png", self.store._key_for(digest, "image/png"))

    def test_uri_round_trips_through_the_key_parser(self):
        digest = "b" * 64
        key = self.store._key_for(digest, "image/jpeg")
        self.assertEqual(key, self.store._key_from_uri(f"s3://kb-assets/{key}"))
        self.assertEqual(key, self.store._key_from_uri(key))

    def test_invalid_digest_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store._key_for("../escape", "image/png")

    def test_missing_boto3_produces_an_actionable_error(self):
        with patch.dict("sys.modules", {"boto3": None}):
            with self.assertRaises(RuntimeError) as ctx:
                self.store._get_client()
        self.assertIn("boto3", str(ctx.exception))
        self.assertIn("blob_backend", str(ctx.exception))


class ProfileIntegrationTests(unittest.TestCase):
    def test_every_shipped_profile_exposes_an_assets_section(self):
        from backend.profiles.registry import available_profiles, load_profile

        for name in available_profiles():
            with self.subTest(profile=name):
                assets = load_profile(name).assets
                self.assertIn(assets.blob_backend, ("local", "s3"))
                self.assertGreater(assets.triage.min_area, 0)

    def test_blob_store_is_built_from_the_profile(self):
        from backend.assets.blobs import build_blob_store
        from backend.profiles.registry import load_profile

        with TemporaryDirectory() as tmp:
            profile = load_profile("base")
            profile.assets.blob_root = tmp
            store = build_blob_store(profile.assets)
            self.assertIsInstance(store, LocalBlobStore)
            self.assertEqual(Path(tmp).resolve(), store.root)

    def test_unknown_blob_backend_fails_loudly(self):
        from backend.assets.blobs import build_blob_store
        from backend.profiles.registry import load_profile

        profile = load_profile("base")
        profile.assets.blob_backend = "gopher"
        with self.assertRaises(ValueError):
            build_blob_store(profile.assets)


if __name__ == "__main__":
    unittest.main()
