"""The corpus digest, the cached floor, and what the scope prompt now shows the model.

Three changes are pinned here, and each fixes something that could only be observed in
production:

  * `derive_floor` materialised an n x n matrix. At the 222-question catalogue it was
    written against that is 2 MB; at 20,000 questions it is 1.85 GB and at 40,000 it is
    4.3 GB and five minutes, most of it swapping — paid at boot, inside the process that
    already holds the embedder, by whichever user arrives first after a deploy.

  * The corpus description handed to the scope model was `", ".join(topics[:24])`,
    rebuilt per process from whatever rows loaded. It could describe a corpus that had
    since changed, and 24 labels are thin evidence for a decision whose false negative
    is an outright refusal.

  * Match scores were shown to the model with no scale. 0.31 reads as "nothing matched"
    unless you know what this corpus typically scores.
"""
import unittest
from unittest.mock import patch

from backend.indexing.section_summary import (
    DIGEST_INPUT_BUDGET,
    SectionRecord,
    build_corpus_digest,
    sections_fingerprint,
)
from backend.prompts import render
from backend.rag.scope_index import (
    build_index,
    derive_floor,
    floor_fingerprint,
)


def unit(index, dimension=8):
    return [1.0 if position == index else 0.0 for position in range(dimension)]


def record(chunk_id, answers=("q?",), summary="", content_sha256="h"):
    return SectionRecord(
        chunk_id=chunk_id,
        content_sha256=content_sha256,
        summary=summary,
        answers=list(answers),
    )


class BlockedFloorTests(unittest.TestCase):
    """The blocked pass must be arithmetically indistinguishable from the n x n one."""

    def reference(self, vectors, chunk_ids, point):
        """The original implementation, kept here as the oracle."""
        import numpy as np

        from backend.rag.scope_index import percentile

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return 0.0
        similarity = matrix @ matrix.T
        sections = np.asarray(chunk_ids)
        similarity[sections[:, None] == sections[None, :]] = -np.inf
        row_max = similarity.max(axis=1)
        maxima = [float(v) for v in row_max if np.isfinite(v)]
        return percentile(maxima, point) if maxima else 0.0

    def corpus(self, count, per_section=3):
        import numpy as np

        rng = np.random.default_rng(11)
        matrix = rng.random((count, 16)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix, [f"s{i // per_section}" for i in range(count)]

    def test_blocking_does_not_change_the_floor(self):
        for count, point in ((2, 10.0), (17, 10.0), (64, 50.0), (200, 90.0)):
            with self.subTest(questions=count, percentile=point):
                vectors, chunk_ids = self.corpus(count)
                self.assertAlmostEqual(
                    self.reference(vectors, chunk_ids, point),
                    derive_floor(vectors, chunk_ids, point),
                    places=6,
                )

    def test_a_block_boundary_does_not_split_a_row_maximum(self):
        """The failure this guards: taking a max per block instead of per row, which
        only shows up when a row's true maximum sits in a later block."""
        vectors, chunk_ids = self.corpus(97)
        expected = self.reference(vectors, chunk_ids, 10.0)
        for budget in (64, 512, 4096, 1 << 20):
            with self.subTest(block_bytes=budget):
                with patch("backend.rag.scope_index.FLOOR_BLOCK_BYTES", budget):
                    self.assertAlmostEqual(expected, derive_floor(vectors, chunk_ids, 10.0), places=6)

    def test_a_single_row_block_still_agrees(self):
        """The degenerate blocking: one row at a time."""
        vectors, chunk_ids = self.corpus(40)
        with patch("backend.rag.scope_index.FLOOR_BLOCK_BYTES", 1):
            self.assertAlmostEqual(
                self.reference(vectors, chunk_ids, 10.0),
                derive_floor(vectors, chunk_ids, 10.0),
                places=6,
            )

    def test_same_section_pairs_stay_excluded_across_blocks(self):
        """A question must never be calibrated against its own siblings, and blocking
        must not let a sibling in through the seam."""
        vectors = [unit(0), unit(0), unit(1)]
        with patch("backend.rag.scope_index.FLOOR_BLOCK_BYTES", 1):
            # s1's two identical questions would score 1.0 against each other.
            self.assertEqual(0.0, derive_floor(vectors, ["s1", "s1", "s2"], point=0))


class FloorCacheTests(unittest.TestCase):
    def records(self):
        return [record("s1", ["a?", "b?"]), record("s2", ["c?"])]

    def embed(self):
        table = {"a?": unit(0), "b?": unit(1), "c?": unit(2)}
        return lambda questions: [table[q] for q in questions]

    def test_a_matching_fingerprint_skips_the_derivation(self):
        index = build_index(
            self.records(), embed=self.embed(), embedding_model="m",
            cached_floor=("", 0.0),
        )
        with patch("backend.rag.scope_index.derive_floor") as derive:
            reused = build_index(
                self.records(), embed=self.embed(), embedding_model="m",
                cached_floor=(index.floor_sha256, 0.4242),
            )
        derive.assert_not_called()
        self.assertEqual(0.4242, reused.floor)

    def test_a_stale_fingerprint_is_ignored_not_trusted(self):
        fresh = build_index(self.records(), embed=self.embed(), embedding_model="m")
        stale = build_index(
            self.records(), embed=self.embed(), embedding_model="m",
            cached_floor=("not-the-right-hash", 0.9999),
        )
        self.assertEqual(fresh.floor, stale.floor)
        self.assertNotEqual(0.9999, stale.floor)

    def test_the_index_reports_the_fingerprint_it_derived_under(self):
        index = build_index(self.records(), embed=self.embed(), embedding_model="m")
        self.assertEqual(
            floor_fingerprint(index.questions, index.chunk_ids, "m", 10.0),
            index.floor_sha256,
        )


class FloorFingerprintTests(unittest.TestCase):
    """Every input to the floor has to be covered, or a cached floor outlives its corpus."""

    def test_question_order_does_not_matter(self):
        self.assertEqual(
            floor_fingerprint(["a", "b"], ["s1", "s2"], "m", 10.0),
            floor_fingerprint(["b", "a"], ["s2", "s1"], "m", 10.0),
        )

    def test_each_input_invalidates_it(self):
        base = floor_fingerprint(["a", "b"], ["s1", "s2"], "m", 10.0)
        for label, other in (
            ("edited question", floor_fingerprint(["a", "c"], ["s1", "s2"], "m", 10.0)),
            ("moved section", floor_fingerprint(["a", "b"], ["s1", "s3"], "m", 10.0)),
            ("new embedding model", floor_fingerprint(["a", "b"], ["s1", "s2"], "m2", 10.0)),
            ("new percentile", floor_fingerprint(["a", "b"], ["s1", "s2"], "m", 5.0)),
            ("extra question", floor_fingerprint(["a", "b", "c"], ["s1", "s2", "s2"], "m", 10.0)),
        ):
            with self.subTest(changed=label):
                self.assertNotEqual(base, other)

    def test_a_question_moving_between_sections_is_not_the_same_corpus(self):
        """Guards a separator bug: without one, ("ab","c") and ("a","bc") collide."""
        self.assertNotEqual(
            floor_fingerprint(["ab"], ["c"], "m", 10.0),
            floor_fingerprint(["a"], ["bc"], "m", 10.0),
        )


class SectionsFingerprintTests(unittest.TestCase):
    def test_reindexing_an_unchanged_corpus_is_the_same_corpus(self):
        one = [record("s1", content_sha256="x"), record("s2", content_sha256="y")]
        two = [record("s2", content_sha256="y"), record("s1", content_sha256="x")]
        self.assertEqual(sections_fingerprint(one), sections_fingerprint(two))

    def test_editing_adding_or_removing_a_section_changes_it(self):
        base = [record("s1", content_sha256="x"), record("s2", content_sha256="y")]
        edited = [record("s1", content_sha256="EDITED"), record("s2", content_sha256="y")]
        added = base + [record("s3", content_sha256="z")]
        removed = base[:1]
        for label, other in (("edited", edited), ("added", added), ("removed", removed)):
            with self.subTest(change=label):
                self.assertNotEqual(sections_fingerprint(base), sections_fingerprint(other))


class CorpusDigestTests(unittest.TestCase):
    """The paragraph must describe the WHOLE corpus, not the part that fit in one call."""

    def test_a_small_corpus_is_one_call(self):
        calls = []

        def invoke(prompt):
            calls.append(prompt)
            return "  The school's admissions and fees.  "

        digest = build_corpus_digest(
            [record("s1", summary="Admissions."), record("s2", summary="Fees.")],
            invoke=invoke,
        )
        self.assertEqual(1, len(calls))
        self.assertEqual("The school's admissions and fees.", digest)

    def test_every_section_reaches_the_model_when_it_does_not_fit_one_call(self):
        """The failure mode this exists for is silent: a corpus whose later half went
        undescribed refuses questions about it while looking perfectly healthy."""
        records = [record(f"s{i}", summary=f"Section {i} covers subject-{i}.") for i in range(40)]
        seen = []

        def invoke(prompt):
            seen.append(prompt)
            return "partial paragraph"

        digest = build_corpus_digest(records, invoke=invoke, budget=100)

        self.assertGreater(len(seen), 1, "a 100-char budget must force several batches")
        combined = "\n".join(seen)
        for index in range(40):
            self.assertIn(f"subject-{index}", combined, f"section {index} never reached the model")
        self.assertTrue(digest)

    def test_partial_rounds_are_reduced_to_one_paragraph(self):
        records = [record(f"s{i}", summary=f"Subject {i}.") for i in range(30)]
        rounds = []

        def invoke(prompt):
            rounds.append(prompt)
            return "a partial about several sections"

        digest = build_corpus_digest(records, invoke=invoke, budget=60)
        self.assertEqual("a partial about several sections", digest)
        # The last call must be a reduce over partials, not over raw section summaries.
        self.assertIn("a partial about several sections", rounds[-1])

    def test_the_reduce_converges_when_a_partial_barely_fits_the_budget(self):
        """Regression: batching purely by budget produced single-piece batches once the
        partials were as long as the budget. Each round then re-summarised one piece
        into one piece, made no progress, and only stopped at the round cap — spending
        a model call per section per round to arrive at concatenated partials.
        """
        records = [record(f"s{i}", summary=f"Subject {i}.") for i in range(30)]
        calls = []

        # A partial longer than the budget: two of them cannot fit, which is exactly the
        # case that used to stall.
        def invoke(prompt):
            calls.append(prompt)
            return "a partial paragraph that is longer than the whole budget allows"

        digest = build_corpus_digest(records, invoke=invoke, budget=60)

        self.assertEqual("a partial paragraph that is longer than the whole budget allows", digest)
        # 30 sections at 5 per batch is 6 first-round calls, then 3, 2, 1 as the
        # partials halve. Anything near 30-per-round means it stopped converging.
        self.assertLess(len(calls), 16, f"reduce did not converge: {len(calls)} calls")

    def test_a_failed_call_yields_no_digest_rather_than_half_a_corpus(self):
        def invoke(prompt):
            raise RuntimeError("provider down")

        self.assertEqual("", build_corpus_digest([record("s1", summary="Admissions.")], invoke=invoke))

    def test_sections_without_a_summary_are_not_described(self):
        self.assertEqual("", build_corpus_digest([record("s1", summary="  ")], invoke=lambda p: "x"))

    def test_the_default_budget_takes_a_realistic_corpus_in_one_call(self):
        """A 200-section corpus of one-sentence summaries should not need reducing."""
        records = [record(f"s{i}", summary="A section about school fees and payment dates.") for i in range(200)]
        calls = []
        build_corpus_digest(records, invoke=lambda p: calls.append(p) or "ok")
        self.assertLessEqual(sum(len(r.summary) for r in records), DIGEST_INPUT_BUDGET * 2)
        self.assertEqual(1, len(calls))


class ScopePromptTests(unittest.TestCase):
    """What the model is actually shown, since that is the whole of its evidence."""

    def prompt(self, **overrides):
        settings = {
            "question": "when does term two start?",
            "matches": [{"question": "when does the term begin?", "score": 0.42, "above_floor": True}],
            "catalogue": "Covers admissions, fees and term dates for the school.",
            "persona": "a school",
            "history": "",
            "personal_fields": ["child_name"],
            "floor": 0.38,
            "index_ready": True,
        }
        settings.update(overrides)
        return render("rag/scope_check.j2", **settings)

    def test_a_score_is_shown_against_the_floor(self):
        text = self.prompt()
        self.assertIn("0.42", text)
        self.assertIn("0.38", text)
        self.assertIn("above", text)

    def test_a_below_floor_match_is_labelled_as_such(self):
        text = self.prompt(matches=[{"question": "q", "score": 0.11, "above_floor": False}])
        self.assertIn("below", text)

    def test_the_paragraph_is_rendered_whole(self):
        text = self.prompt(catalogue="A paragraph naming term dates, uniform suppliers and bus routes.")
        self.assertIn("uniform suppliers and bus routes", text)

    def test_a_failed_search_is_reported_as_a_search_failure(self):
        """An empty match list means two opposite things, and the model cannot tell them
        apart unless the prompt does."""
        text = self.prompt(index_ready=False, matches=[])
        self.assertIn("could not run", text)
        self.assertNotIn("nothing comparable", text)

    def test_the_asymmetry_is_taught_by_example_in_both_languages(self):
        text = self.prompt()
        self.assertIn("in_domain", text)
        self.assertIn("out_of_domain", text)
        self.assertIn("الطقس", text)          # an Arabic out-of-domain example
        self.assertIn("الفصل الدراسي", text)   # an Arabic in-domain example

    def test_personal_fields_survive(self):
        self.assertIn("child_name", self.prompt())


if __name__ == "__main__":
    unittest.main()
