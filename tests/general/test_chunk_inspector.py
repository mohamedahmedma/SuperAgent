# -*- coding: utf-8 -*-
"""The admin chunk inspector: the corpus as retrieval actually sees it.

The document list answers "how many chunks"; nothing answered "which ones, and why did
this question miss them". This endpoint does, and the two things it has to get right are
the HIERARCHY — a chunk's level and its parents, which is what the tree is drawn from —
and the FILTER.

The filter is the part worth testing hardest. It folds with `text_matching.fold`, the
same folding the sparse retrieval lane keys on, so an admin typing مدرسة finds مدرسه and
typing without diacritics finds text with them. Folding on the server is what keeps one
implementation of it: a second one in TypeScript would drift from this one the first time
either was edited, and the filter would quietly stop agreeing with retrieval.

Every field is read out of Milvus rather than recomputed. A value here that disagreed
with the index would be worse than not showing it at all, because the whole point of the
view is to be believed.
"""
import asyncio
import json
import unittest
from types import SimpleNamespace

from backend.api.routes.documents import (
    _CHUNK_PAGE_LIMIT,
    list_document_chunks,
)


class _Milvus:
    """Milvus as this route uses it: a filtered read of whole rows."""

    def __init__(self, rows):
        self.rows = rows
        self.filters = []
        self.fields = []
        self.initialised = 0

    def init_collection(self):
        self.initialised += 1

    def query_all(self, filter_expr="", output_fields=None):
        self.filters.append(filter_expr)
        self.fields.append(list(output_fields or []))
        return list(self.rows)


def row(chunk_id, text="", level=3, idx=0, parent="", root="", page=1,
        modality="text", asset_ids="[]"):
    return {
        "chunk_id": chunk_id, "text": text, "chunk_level": level, "chunk_idx": idx,
        "parent_chunk_id": parent, "root_chunk_id": root, "page_number": page,
        "modality": modality, "asset_ids": asset_ids,
    }


class _ParentChunks:
    """The other half of the corpus. Levels 1 and 2 live here, not in Milvus."""

    def __init__(self, rows=(), error=None):
        self.rows = list(rows)
        self.error = error
        self.asked = []

    def documents_by_filename(self, filename):
        self.asked.append(filename)
        if self.error:
            raise self.error
        return list(self.rows)


def inspect(rows, filename="kb.docx", q="", parents=(), parent_store=None):
    milvus = _Milvus(rows)
    parent_chunks = parent_store or _ParentChunks(parents)
    services = SimpleNamespace(milvus=milvus, parent_chunks=parent_chunks)
    response = asyncio.run(list_document_chunks(filename, q=q, _=None, services=services))
    return response, milvus


class WhatItReadsTests(unittest.TestCase):
    def test_it_asks_for_only_the_document_it_was_given(self):
        _, milvus = inspect([row("a")], filename="fees_ar.docx")
        self.assertEqual(['filename == "fees_ar.docx"'], milvus.filters)

    def test_a_filename_with_quotes_cannot_break_the_filter(self):
        """Filenames are admin-supplied. Asserted as a round trip rather than as exact
        bytes: whatever the quoting is, reading it back must give the name unchanged,
        which a naive f-string concatenation would not."""
        odd = 'od"d\\.docx'
        _, milvus = inspect([row("a")], filename=odd)
        prefix, _, quoted = milvus.filters[0].partition(" == ")
        self.assertEqual("filename", prefix)
        self.assertEqual(odd, json.loads(quoted))

    def test_an_arabic_filename_is_not_escaped_into_ascii(self):
        _, milvus = inspect([row("a")], filename="رسوم.docx")
        self.assertIn("رسوم.docx", milvus.filters[0])

    def test_it_never_reads_the_embeddings(self):
        """They are most of a chunk's bytes and none of its meaning."""
        _, milvus = inspect([row("a")])
        self.assertNotIn("dense_embedding", milvus.fields[0])
        self.assertNotIn("sparse_embedding", milvus.fields[0])

    def test_it_reads_every_field_the_tree_and_the_pins_need(self):
        _, milvus = inspect([row("a")])
        for field in ("chunk_id", "parent_chunk_id", "root_chunk_id", "chunk_level",
                      "chunk_idx", "page_number", "modality", "text", "asset_ids"):
            self.assertIn(field, milvus.fields[0])

    def test_the_collection_is_initialised_first(self):
        _, milvus = inspect([row("a")])
        self.assertEqual(1, milvus.initialised)

    def test_a_document_with_no_chunks(self):
        response, _ = inspect([])
        self.assertEqual(0, response.total)
        self.assertEqual([], response.chunks)


class BothHalvesOfTheCorpusTests(unittest.TestCase):
    """Only leaves are vectorised. The levels above them live in Postgres.

    `_process_upload_job` writes levels 1 and 2 to `parent_chunks` and level 3 to Milvus,
    so a view reading Milvus alone shows every leaf pointing at a parent that is not in
    the response. Measured on the shipped corpus before this was fixed: 175 leaves, 175
    orphans, 0 roots — a tree that was really a flat list claiming to be a tree.
    """

    LEAVES = [row("L3-a", "leaf", level=3, idx=0, parent="L2-a", root="L1-a")]
    PARENTS = [
        row("L1-a", "root", level=1, idx=0),
        row("L2-a", "mid", level=2, idx=0, parent="L1-a", root="L1-a"),
    ]

    def test_the_parent_store_is_asked_for_the_same_document(self):
        parents = _ParentChunks(self.PARENTS)
        inspect(self.LEAVES, filename="fees_en.docx", parent_store=parents)
        self.assertEqual(["fees_en.docx"], parents.asked)

    def test_the_levels_above_the_leaves_are_included(self):
        response, _ = inspect(self.LEAVES, parents=self.PARENTS)
        self.assertEqual([1, 2, 3], [chunk.chunk_level for chunk in response.chunks])
        self.assertEqual(3, response.total)

    def test_no_chunk_is_left_pointing_at_a_parent_that_is_absent(self):
        """The property the tree depends on, asserted directly."""
        response, _ = inspect(self.LEAVES, parents=self.PARENTS)
        present = {chunk.chunk_id for chunk in response.chunks}
        orphans = [c.chunk_id for c in response.chunks
                   if c.parent_chunk_id and c.parent_chunk_id not in present]
        self.assertEqual([], orphans)

    def test_the_filter_reaches_the_parent_levels_too(self):
        """A parent holds the text of all its children, so a term can match at L1 and
        the admin should see which level carries it."""
        response, _ = inspect(self.LEAVES, parents=self.PARENTS, q="root")
        matched = [chunk.chunk_id for chunk in response.chunks if chunk.matched]
        self.assertEqual(["L1-a"], matched)

    def test_a_parent_store_failure_is_reported_rather_than_half_a_tree(self):
        """Showing the leaves alone would draw a structure that is not the corpus's. An
        inspector that lies is worse than one that says it could not read."""
        from fastapi import HTTPException

        broken = _ParentChunks(error=RuntimeError("postgres is down"))
        with self.assertRaises(HTTPException) as raised:
            inspect(self.LEAVES, parent_store=broken)
        self.assertEqual(500, raised.exception.status_code)


class TheHierarchyTests(unittest.TestCase):
    """What the tree is drawn from."""

    def _corpus(self):
        return [
            row("L3-b", "leaf b", level=3, idx=1, parent="L2-a", root="L1-a"),
            row("L1-a", "root", level=1, idx=0),
            row("L3-a", "leaf a", level=3, idx=0, parent="L2-a", root="L1-a"),
            row("L2-a", "mid", level=2, idx=0, parent="L1-a", root="L1-a"),
        ]

    def test_chunks_come_back_in_reading_order(self):
        """Level then index, so the flat list reads the way the document does and the
        tree can be built from it without a second pass."""
        response, _ = inspect(self._corpus())
        self.assertEqual(["L1-a", "L2-a", "L3-a", "L3-b"],
                         [chunk.chunk_id for chunk in response.chunks])

    def test_the_parent_and_root_of_every_chunk_survive(self):
        response, _ = inspect(self._corpus())
        leaf = next(c for c in response.chunks if c.chunk_id == "L3-b")
        self.assertEqual("L2-a", leaf.parent_chunk_id)
        self.assertEqual("L1-a", leaf.root_chunk_id)
        self.assertEqual(3, leaf.chunk_level)

    def test_a_root_names_no_parent(self):
        """Which is how the tree finds its roots."""
        response, _ = inspect(self._corpus())
        root = next(c for c in response.chunks if c.chunk_id == "L1-a")
        self.assertEqual("", root.parent_chunk_id)

    def test_ties_break_on_the_chunk_id_so_the_order_is_stable(self):
        rows = [row("b", level=3, idx=0), row("a", level=3, idx=0)]
        first, _ = inspect(rows)
        second, _ = inspect(list(reversed(rows)))
        self.assertEqual([c.chunk_id for c in first.chunks],
                         [c.chunk_id for c in second.chunks])


class TheMetadataPinsTests(unittest.TestCase):
    """What a developer reads under each chunk."""

    def test_the_character_count_is_measured_server_side(self):
        """The size bound this corpus is chunked against is in characters, and the UI
        must not have to agree with Python about what one is."""
        response, _ = inspect([row("a", "مدرسة abc")])
        self.assertEqual(9, response.chunks[0].char_count)

    def test_asset_ids_arrive_as_a_list(self):
        response, _ = inspect([row("a", asset_ids='["img-1", "img-2"]')])
        self.assertEqual(["img-1", "img-2"], response.chunks[0].asset_ids)

    def test_a_chunk_with_no_assets(self):
        response, _ = inspect([row("a", asset_ids="[]")])
        self.assertEqual([], response.chunks[0].asset_ids)

    def test_malformed_asset_ids_do_not_fail_the_inspection(self):
        """A broken field is not worth refusing to show the document over."""
        response, _ = inspect([row("a", asset_ids="{not json")])
        self.assertEqual([], response.chunks[0].asset_ids)

    def test_asset_ids_already_decoded_are_accepted(self):
        response, _ = inspect([row("a", asset_ids=["img-1"])])
        self.assertEqual(["img-1"], response.chunks[0].asset_ids)

    def test_the_modality_is_carried_through(self):
        response, _ = inspect([row("a", modality="figure")])
        self.assertEqual("figure", response.chunks[0].modality)

    def test_a_row_missing_everything_optional_still_renders(self):
        response, _ = inspect([{"chunk_id": "a"}])
        chunk = response.chunks[0]
        self.assertEqual("a", chunk.chunk_id)
        self.assertEqual("text", chunk.modality)
        self.assertEqual(0, chunk.char_count)

    def test_a_null_text_is_empty_rather_than_the_word_none(self):
        response, _ = inspect([row("a", text=None)])
        self.assertEqual("", response.chunks[0].text)


class TheFilterFoldsTests(unittest.TestCase):
    """The filter matches on folded text, which is what makes it usable on Arabic.

    Each of these is a spelling an admin might type against a corpus written the other
    way. An exact-match filter fails every one of them.
    """

    CORPUS = [
        row("fees", "الرسوم الدراسية 105,000 جنيه", idx=0),
        row("hours", "المدرسة تفتح الساعة 7:45 صباحاً", idx=1),
        row("camp", "Summer Camp Fees Table", idx=2),
        row("staff", "إدارة المدرسة ومكتب القبول", idx=3),
    ]

    def _ids(self, q):
        """The chunks the filter MARKED. The response always carries the whole document —
        the bordered document view needs the structure to draw — so the hits are read off
        the flag rather than off the length of the list."""
        response, _ = inspect(self.CORPUS, q=q)
        self.assertEqual(4, len(response.chunks), "the document stays whole under a filter")
        return [chunk.chunk_id for chunk in response.chunks if chunk.matched]

    def test_an_empty_filter_marks_nothing(self):
        self.assertEqual([], self._ids(""))

    def test_a_whitespace_only_filter_marks_nothing(self):
        self.assertEqual([], self._ids("   "))

    def test_teh_marbuta_matches_heh(self):
        """Typed المدرسة, written المدرسه — or the other way round."""
        self.assertEqual(["hours"], self._ids("المدرسه تفتح"))

    def test_heh_matches_teh_marbuta(self):
        self.assertIn("staff", self._ids("اداره"))

    def test_alef_variants_match(self):
        """إدارة typed as ادارة."""
        self.assertIn("staff", self._ids("ادارة"))

    def test_diacritics_in_the_corpus_are_ignored(self):
        self.assertEqual(["hours"], self._ids("صباحا"))

    def test_diacritics_in_the_query_are_ignored(self):
        self.assertEqual(["hours"], self._ids("صَبَاحاً"))

    def test_latin_case_is_ignored(self):
        self.assertEqual(["camp"], self._ids("summer camp"))
        self.assertEqual(["camp"], self._ids("SUMMER CAMP"))

    def test_runs_of_whitespace_collapse(self):
        self.assertEqual(["camp"], self._ids("  Summer     Camp  "))

    def test_a_figure_still_matches_on_its_digits(self):
        self.assertEqual(["fees"], self._ids("105,000"))

    def test_a_substring_of_one_word_matches(self):
        self.assertEqual(["camp"], self._ids("umm"))

    def test_a_filter_matching_nothing_returns_nothing(self):
        self.assertEqual([], self._ids("swimming pool"))

    def test_a_filter_matching_several_returns_all_of_them(self):
        self.assertEqual(["hours", "staff"], self._ids("المدرسه"))

    def test_the_filter_is_not_stemmed(self):
        """`fold` and not `search_key`: a filter is a substring of what was typed, and a
        stemmer would make the result hard to predict from the box's contents."""
        self.assertEqual([], self._ids("رسم"))


class WhatItReportsTests(unittest.TestCase):
    def test_the_whole_document_comes_back_and_the_matches_are_counted(self):
        """So the UI can say "1 of 4 match" and still draw all four."""
        response, _ = inspect(TheFilterFoldsTests.CORPUS, q="camp")
        self.assertEqual(4, response.total)
        self.assertEqual(4, response.returned)
        self.assertEqual(1, response.match_count)

    def test_nothing_is_marked_when_nothing_is_typed(self):
        response, _ = inspect(TheFilterFoldsTests.CORPUS)
        self.assertEqual(0, response.match_count)
        self.assertTrue(all(not chunk.matched for chunk in response.chunks))

    def test_the_query_is_echoed_exactly_as_typed(self):
        """The server folds before matching; the box shows what the user wrote. Confusing
        the two would show them a query they did not type."""
        response, _ = inspect(TheFilterFoldsTests.CORPUS, q="  المدرسة  ")
        self.assertEqual("  المدرسة  ", response.query)

    def test_the_filename_is_echoed(self):
        response, _ = inspect([row("a")], filename="fees_en.docx")
        self.assertEqual("fees_en.docx", response.filename)

    def test_an_ordinary_document_is_not_truncated(self):
        response, _ = inspect([row(str(i), idx=i) for i in range(50)])
        self.assertFalse(response.truncated)
        self.assertEqual(50, response.returned)

    def test_a_match_beyond_the_ceiling_is_not_counted_as_shown(self):
        """`match_count` describes what came back, so it cannot promise a hit the UI was
        never given."""
        rows = [row(str(i), text="fees", idx=i) for i in range(_CHUNK_PAGE_LIMIT + 10)]
        response, _ = inspect(rows, q="fees")
        self.assertEqual(_CHUNK_PAGE_LIMIT, response.match_count)

    def test_a_document_past_the_ceiling_says_so(self):
        """The count stays truthful so the UI can report the cut rather than quietly
        showing less than the document holds."""
        response, _ = inspect([row(str(i), idx=i) for i in range(_CHUNK_PAGE_LIMIT + 25)])
        self.assertTrue(response.truncated)
        self.assertEqual(_CHUNK_PAGE_LIMIT + 25, response.total)
        self.assertEqual(_CHUNK_PAGE_LIMIT, len(response.chunks))


class WhoMayReadItTests(unittest.TestCase):
    def test_the_route_is_admin_only(self):
        """The inspector exposes the whole corpus verbatim, which is more than any
        chat answer would ever quote."""
        from backend.api.routes import documents
        from backend.infra.auth import require_admin

        route = next(
            r for r in documents.router.routes
            if getattr(r, "path", "") == "/documents/{filename}/chunks"
        )
        guards = [dependency.call for dependency in route.dependant.dependencies]
        self.assertIn(require_admin, guards)


if __name__ == "__main__":
    unittest.main()
