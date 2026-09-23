"""Item 31: the admin's view of what a vision model made of a document's images.

The point of these tests is that this route returns what NO other route has ever
returned. `AssetReference` — the public contract a chat client consumes — carries a
caption, alt text and tags. The description, the transcription, the model, its
confidence and the error behind a failure existed only on the dossier, and reviewing an
extraction without them is guesswork.
"""
import asyncio
import json
import unittest
from types import SimpleNamespace

from backend.api.routes.documents import list_document_assets, mark_asset_reviewed
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
)


def dossier(asset_id="kb.docx#p1#a", page=1, status=ExtractionStatus.EXTRACTED,
            role=AssetRole.FIGURE, tier=AssetTier.SIMPLE, caption="Fee table",
            description="A table of fees", transcription="Grade 1 | 50,000",
            tags=("table",), model="qwen-vl", confidence=0.9, needs_review=False,
            error="", extraction=True):
    return AssetDossier(
        asset_id=asset_id,
        sha256="d" * 64,
        role=role,
        tier=tier,
        status=status,
        source=SourceRef(filename="kb.docx", page_number=page),
        blob=BlobRef(content_type="image/png", byte_size=2048, width=800, height=600),
        extraction=ExtractionPayload(
            text=TextSurface(caption=caption, description=description,
                             transcription=transcription, tags=list(tags)),
            provenance=Provenance(tier=tier, pipeline="figure", model_used=model,
                                  confidence=confidence, needs_review=needs_review,
                                  error=error),
        ) if extraction else None,
    )


class _Milvus:
    def __init__(self, rows=(), error=None):
        self.rows = list(rows)
        self.error = error
        self.filters = []

    def init_collection(self):
        if self.error:
            raise self.error

    def query_all(self, filter_expr="", output_fields=()):
        self.filters.append(filter_expr)
        if self.error:
            raise self.error
        return list(self.rows)


def chunk_row(chunk_id, asset_ids, level=3, idx=0):
    return {"chunk_id": chunk_id, "asset_ids": json.dumps(asset_ids),
            "chunk_level": level, "chunk_idx": idx}


def review(dossiers, filename="kb.docx", chunks=(), milvus_error=None, store_error=None):
    class Store:
        def list_by_filename(_self, name):
            if store_error:
                raise store_error
            return list(dossiers)

    milvus = _Milvus(chunks, error=milvus_error)
    services = SimpleNamespace(asset_store=Store(), milvus=milvus)
    return asyncio.run(list_document_assets(filename, _=None, services=services)), milvus


class WhatItReturnsTests(unittest.TestCase):
    def test_the_whole_retrieval_surface_is_returned_not_just_the_caption(self):
        response, _ = review([dossier()])
        asset = response.assets[0]
        self.assertEqual("Fee table", asset.caption)
        self.assertEqual("A table of fees", asset.description)
        self.assertEqual("Grade 1 | 50,000", asset.transcription)
        self.assertEqual(["table"], asset.tags)

    def test_how_it_was_produced_is_returned_so_it_can_be_judged(self):
        response, _ = review([dossier(model="qwen-vl", confidence=0.31, needs_review=True)])
        asset = response.assets[0]
        self.assertEqual("qwen-vl", asset.model_used)
        self.assertAlmostEqual(0.31, asset.confidence)
        self.assertTrue(asset.needs_review)

    def test_a_failure_carries_its_reason(self):
        response, _ = review([dossier(status=ExtractionStatus.FAILED, error="429 from provider")])
        self.assertEqual("failed", response.assets[0].status)
        self.assertEqual("429 from provider", response.assets[0].error)

    def test_an_asset_with_no_extraction_at_all_is_still_listed(self):
        """A pending or failed asset is exactly what a reviewer is looking for; dropping
        it would hide the only rows that need attention."""
        response, _ = review([dossier(status=ExtractionStatus.PENDING, extraction=False)])
        self.assertEqual(1, response.total)
        self.assertEqual("", response.assets[0].caption)
        self.assertFalse(response.assets[0].needs_review)

    def test_the_image_itself_is_addressable_so_it_can_be_compared(self):
        response, _ = review([dossier(asset_id="kb.docx#p1#a")])
        self.assertTrue(response.assets[0].url.startswith("/media/"))
        # The '#' is percent-encoded rather than starting a URL fragment.
        self.assertIn("%23", response.assets[0].url)

    def test_indexable_says_whether_it_is_in_the_corpus_at_all(self):
        response, _ = review([dossier(), dossier(asset_id="b", role=AssetRole.DECORATIVE)])
        by_id = {asset.asset_id: asset for asset in response.assets}
        self.assertTrue(by_id["kb.docx#p1#a"].indexable)
        self.assertFalse(by_id["b"].indexable)

    def test_assets_are_ordered_by_page_so_they_read_like_the_document(self):
        response, _ = review([
            dossier(asset_id="c", page=3), dossier(asset_id="a", page=1),
            dossier(asset_id="b", page=2),
        ])
        self.assertEqual(["a", "b", "c"], [asset.asset_id for asset in response.assets])


class ReviewQueueTests(unittest.TestCase):
    """`needs_review` has been written on every extraction since the pipeline was
    built — a vision confidence under the profile's threshold, or a heuristic run that
    recovered nothing. Nothing had ever read it back."""

    def test_the_count_leads_with_the_number_that_matters(self):
        response, _ = review([
            dossier(asset_id="a", needs_review=True),
            dossier(asset_id="b", needs_review=False),
            dossier(asset_id="c", needs_review=True),
        ])
        self.assertEqual(3, response.total)
        self.assertEqual(2, response.needs_review_count)

    def test_nothing_needing_review_counts_zero_rather_than_omitting_the_field(self):
        response, _ = review([dossier()])
        self.assertEqual(0, response.needs_review_count)


class ChunkLinkTests(unittest.TestCase):
    """Item 32's half of this route: the link exists only chunk to asset, so the
    reverse is built by inverting it."""

    def test_an_asset_names_the_chunks_it_produced(self):
        response, _ = review(
            [dossier(asset_id="a")],
            chunks=[chunk_row("c1", ["a"]), chunk_row("c2", ["a"]), chunk_row("c3", ["other"])],
        )
        self.assertEqual(["c1", "c2"], response.assets[0].chunk_ids)

    def test_an_asset_that_produced_no_chunk_says_so_rather_than_guessing(self):
        response, _ = review([dossier(asset_id="a")], chunks=[chunk_row("c1", ["b"])])
        self.assertEqual([], response.assets[0].chunk_ids)

    def test_one_chunk_can_belong_to_several_assets(self):
        response, _ = review(
            [dossier(asset_id="a"), dossier(asset_id="b", page=2)],
            chunks=[chunk_row("c1", ["a", "b"])],
        )
        self.assertEqual(["c1"], response.assets[0].chunk_ids)
        self.assertEqual(["c1"], response.assets[1].chunk_ids)

    def test_it_asks_for_only_the_document_it_was_given(self):
        _response, milvus = review([dossier()], filename="fees_ar.docx", chunks=[])
        self.assertIn("fees_ar.docx", milvus.filters[0])


class WhenSomethingIsDownTests(unittest.TestCase):
    def test_an_unreadable_index_costs_the_chunk_ids_and_nothing_else(self):
        """A document's images are still worth reviewing when the vector store is down —
        that is precisely when an admin is trying to find out what went wrong."""
        response, _ = review([dossier()], milvus_error=RuntimeError("milvus is down"))
        self.assertEqual(1, response.total)
        self.assertEqual("Fee table", response.assets[0].caption)
        self.assertEqual([], response.assets[0].chunk_ids)

    def test_an_unreadable_asset_store_is_an_error_rather_than_an_empty_list(self):
        """Empty means this document has no images, which is a different fact."""
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as caught:
            review([dossier()], store_error=RuntimeError("postgres is down"))
        self.assertEqual(500, caught.exception.status_code)


class WhoMayReadItTests(unittest.TestCase):
    def test_the_route_is_admin_only(self):
        """It returns a model's confidence and its errors — operational detail that the
        public asset contract deliberately does not carry."""
        from backend.api.routes import documents
        from backend.infra.auth import require_admin

        route = next(
            r for r in documents.router.routes
            if getattr(r, "path", "") == "/documents/{filename}/assets"
        )
        guards = [dependency.call for dependency in route.dependant.dependencies]
        self.assertIn(require_admin, guards)


if __name__ == "__main__":
    unittest.main()


class MarkReviewedTests(unittest.TestCase):
    """Clearing the flag. Until this existed `needs_review` only ever went up, which
    makes it a permanent label rather than a queue."""

    def _services(self, dossier_out=None, error=None):
        calls = []

        class Store:
            def mark_reviewed(_self, asset_id):
                calls.append(asset_id)
                if error:
                    raise error
                return dossier_out

        return SimpleNamespace(asset_store=Store()), calls

    def test_it_returns_the_asset_as_it_now_reads(self):
        accepted = dossier(needs_review=False)
        services, _calls = self._services(accepted)
        result = asyncio.run(mark_asset_reviewed("kb.docx#p1#a", _=None, services=services))
        self.assertFalse(result.needs_review)
        self.assertEqual("kb.docx#p1#a", result.asset_id)

    def test_an_unknown_asset_is_a_404_rather_than_a_silent_success(self):
        from fastapi import HTTPException

        services, _calls = self._services(None)
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(mark_asset_reviewed("nope", _=None, services=services))
        self.assertEqual(404, caught.exception.status_code)

    def test_a_write_failure_is_reported_rather_than_read_as_accepted(self):
        from fastapi import HTTPException

        services, _calls = self._services(error=RuntimeError("postgres is down"))
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(mark_asset_reviewed("a", _=None, services=services))
        self.assertEqual(500, caught.exception.status_code)

    def test_the_route_is_admin_only(self):
        from backend.api.routes import documents
        from backend.infra.auth import require_admin

        route = next(
            r for r in documents.router.routes
            if getattr(r, "path", "") == "/documents/assets/{asset_id:path}/reviewed"
        )
        guards = [dependency.call for dependency in route.dependant.dependencies]
        self.assertIn(require_admin, guards)


class AssetIdRoutingTests(unittest.TestCase):
    """An asset id embeds a filename and a '#', and `:path` is greedy. The sibling
    media routes already carry a comment about exactly this hazard."""

    def _match(self, path):
        from backend.api.routes import documents

        route = next(
            r for r in documents.router.routes
            if getattr(r, "path", "") == "/documents/assets/{asset_id:path}/reviewed"
        )
        scope = {"type": "http", "method": "POST", "path": path, "headers": []}
        _match, child = route.matches(scope)
        return child.get("path_params", {}).get("asset_id")

    def test_a_plain_id_is_captured(self):
        self.assertEqual("abc", self._match("/documents/assets/abc/reviewed"))

    def test_an_id_holding_slashes_is_captured_whole(self):
        """`:path` is greedy, and must still stop at the suffix rather than eat it."""
        self.assertEqual("kb/fees.docx#p1#a",
                         self._match("/documents/assets/kb/fees.docx#p1#a/reviewed"))

    def test_the_suffix_is_never_swallowed_into_the_id(self):
        captured = self._match("/documents/assets/a/b/reviewed")
        self.assertEqual("a/b", captured)
        self.assertNotIn("reviewed", captured)
