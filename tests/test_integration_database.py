"""Integration: the real Postgres, real drivers, real column types.

Unit tests for this layer run against SQLite, which silently forgives things Postgres
does not — JSON round-trips, NUL bytes, float precision, unique constraints under
concurrency, and the connection pool itself. Every test here exists because the SQLite
version of it would pass while production failed.

Everything written lives under a per-run profile or username and is deleted afterwards.
Nothing here deletes by a broad predicate.
"""
import threading
import unittest
import uuid

from sqlalchemy import text

from backend.indexing.section_summary import SectionRecord, sections_fingerprint
from backend.indexing.summary_store import (
    DigestRecord,
    delete_missing,
    existing_hashes,
    load_digest,
    load_records,
    save_digest,
    save_records,
)
from backend.infra.database import engine
from tests.integration_support import TEST_PREFIX, requires_postgres, temporary_profile


def record(chunk_id, answers=("what is it?",), vectors=None, summary="A section.", sha="h1"):
    return SectionRecord(
        chunk_id=chunk_id,
        content_sha256=sha,
        filename="doc.docx",
        chunk_level=1,
        summary=summary,
        answers=list(answers),
        topics=["fees"],
        question_vectors=list(vectors or []),
        embedding_model="BAAI/bge-m3",
        model_used="test",
    )


@requires_postgres
class SectionSummaryRoundTripTests(unittest.TestCase):
    """JSON columns are where SQLite and Postgres diverge most quietly."""

    def test_a_record_survives_a_round_trip_unchanged(self):
        with temporary_profile() as profile:
            vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            save_records(profile, [record("s1", ("q one?", "q two?"), vectors)])
            loaded = load_records(profile)

            self.assertEqual(1, len(loaded))
            got = loaded[0]
            self.assertEqual("s1", got.chunk_id)
            self.assertEqual(["q one?", "q two?"], got.answers)
            self.assertEqual(["fees"], got.topics)
            self.assertEqual(vectors, got.question_vectors)

    def test_float_precision_survives_the_json_column(self):
        """Vectors are stored as JSON. A float that does not come back bit-identical
        moves every score computed from it."""
        with temporary_profile() as profile:
            precise = [0.1234567890123456, -0.9876543210987654, 1e-8, -1e-8]
            save_records(profile, [record("s1", ("q?",), [precise])])
            got = load_records(profile)[0].question_vectors[0]
            for original, returned in zip(precise, got):
                self.assertEqual(original, returned)

    def test_a_1024_dimension_vector_round_trips(self):
        """The real width. Small vectors can hide a column-size problem."""
        with temporary_profile() as profile:
            vector = [i / 1024 for i in range(1024)]
            save_records(profile, [record("s1", ("q?",), [vector])])
            got = load_records(profile)[0].question_vectors[0]
            self.assertEqual(1024, len(got))
            self.assertEqual(vector, got)

    def test_arabic_and_emoji_survive(self):
        with temporary_profile() as profile:
            questions = ["متى يبدأ الفصل الدراسي الثاني؟", "Fees 💰 for grade 5?"]
            save_records(profile, [record("s1", questions, [[0.1], [0.2]])])
            self.assertEqual(questions, load_records(profile)[0].answers)

    def test_a_nul_byte_does_not_break_the_write(self):
        """Postgres rejects NUL in text columns outright, so the engine's
        before_cursor_execute listener strips it. SQLite accepts NUL happily and would
        hide this failure entirely — which is why it is tested here and not there."""
        with temporary_profile() as profile:
            save_records(profile, [record("s1", ("clean\x00question?",), [[0.1]], summary="a\x00b")])
            self.assertEqual(1, len(load_records(profile)))

    def test_text_columns_are_stripped_of_nul(self):
        with temporary_profile() as profile:
            save_records(profile, [record("s1", summary="a\x00b")])
            self.assertEqual("ab", load_records(profile)[0].summary)

    def test_json_columns_escape_nul_rather_than_stripping_it(self):
        """Documents a real asymmetry rather than asserting a tidier one.

        The sanitiser runs over bound parameters, but by then a JSON column's value is
        already a serialised string in which NUL has become the six characters \\u0000
        — which the regex does not match and Postgres's `json` type accepts happily.
        (`jsonb` would have rejected it.) Decoding on read turns it back into a real
        NUL, so a catalogued question can carry one all the way to the embedder.

        Harmless in practice and not worth a fix nobody asked for, but it is the kind
        of thing that should be written down where the next person will find it rather
        than rediscovered.
        """
        with temporary_profile() as profile:
            save_records(profile, [record("s1", ("clean\x00question?",), [[0.1]])])
            self.assertIn("\x00", load_records(profile)[0].answers[0])

            with engine.connect() as connection:
                raw = connection.execute(
                    text("SELECT answers::text FROM section_summaries WHERE profile = :p"),
                    {"p": profile},
                ).scalar()
            self.assertIn("u0000", raw)

    def test_saving_the_same_chunk_twice_updates_rather_than_duplicates(self):
        with temporary_profile() as profile:
            save_records(profile, [record("s1", ("first?",), [[0.1]])])
            save_records(profile, [record("s1", ("second?",), [[0.2]], sha="h2")])
            loaded = load_records(profile)
            self.assertEqual(1, len(loaded))
            self.assertEqual(["second?"], loaded[0].answers)
            self.assertEqual("h2", loaded[0].content_sha256)

    def test_profiles_do_not_see_each_others_rows(self):
        with temporary_profile() as first, temporary_profile() as second:
            save_records(first, [record("shared-id", ("from first?",), [[0.1]])])
            save_records(second, [record("shared-id", ("from second?",), [[0.2]])])
            self.assertEqual(["from first?"], load_records(first)[0].answers)
            self.assertEqual(["from second?"], load_records(second)[0].answers)

    def test_existing_hashes_reports_what_is_stored(self):
        with temporary_profile() as profile:
            save_records(profile, [record("s1", sha="aaa"), record("s2", sha="bbb")])
            self.assertEqual({"s1": "aaa", "s2": "bbb"}, existing_hashes(profile))

    def test_delete_missing_removes_only_absent_sections(self):
        with temporary_profile() as profile:
            save_records(profile, [record("s1"), record("s2"), record("s3")])
            removed = delete_missing(profile, ["s1", "s3"])
            self.assertEqual(1, removed)
            self.assertEqual({"s1", "s3"}, {r.chunk_id for r in load_records(profile)})

    def test_an_empty_save_is_a_no_op(self):
        with temporary_profile() as profile:
            self.assertEqual(0, save_records(profile, []))
            self.assertEqual([], load_records(profile))

    def test_a_long_chunk_id_is_accepted(self):
        """chunk_id is String(512); ids are built from file paths and can get long."""
        with temporary_profile() as profile:
            long_id = "d/" + ("x" * 480)
            save_records(profile, [record(long_id)])
            self.assertEqual([long_id], [r.chunk_id for r in load_records(profile)])


@requires_postgres
class CorpusDigestTests(unittest.TestCase):
    def test_a_digest_round_trips(self):
        with temporary_profile() as profile:
            digest = DigestRecord(
                paragraph="Covers admissions, fees and term dates.",
                sections_sha256="abc123",
                section_count=33,
                floor=0.5770054,
                floor_sha256="def456",
                question_count=226,
                model_used="test-model",
            )
            self.assertTrue(save_digest(profile, digest))
            got = load_digest(profile)

            self.assertEqual(digest.paragraph, got.paragraph)
            self.assertEqual(digest.sections_sha256, got.sections_sha256)
            self.assertEqual(33, got.section_count)
            self.assertEqual(226, got.question_count)
            self.assertAlmostEqual(0.5770054, got.floor, places=6)

    def test_an_absent_digest_reads_as_empty_not_an_error(self):
        with temporary_profile() as profile:
            got = load_digest(profile)
            self.assertEqual("", got.paragraph)
            self.assertEqual(0.0, got.floor)

    def test_saving_twice_updates_the_same_row(self):
        with temporary_profile() as profile:
            save_digest(profile, DigestRecord(paragraph="first", floor=0.1))
            save_digest(profile, DigestRecord(paragraph="second", floor=0.2))
            with engine.connect() as connection:
                count = connection.execute(
                    text("SELECT count(*) FROM corpus_digests WHERE profile = :p"),
                    {"p": profile},
                ).scalar()
            self.assertEqual(1, count)
            self.assertEqual("second", load_digest(profile).paragraph)

    def test_a_long_paragraph_is_not_truncated(self):
        """`paragraph` is Text, not String(n). A truncated corpus description would
        silently narrow the scope gate."""
        with temporary_profile() as profile:
            paragraph = ("The school covers admissions, fees, uniform and transport. " * 300).strip()
            save_digest(profile, DigestRecord(paragraph=paragraph))
            self.assertEqual(paragraph, load_digest(profile).paragraph)
            self.assertGreater(len(paragraph), 15000)

    def test_an_arabic_paragraph_round_trips(self):
        with temporary_profile() as profile:
            paragraph = "تغطي قاعدة المعرفة القبول والرسوم الدراسية ومواعيد الفصول الدراسية."
            save_digest(profile, DigestRecord(paragraph=paragraph))
            self.assertEqual(paragraph, load_digest(profile).paragraph)

    def test_the_fingerprint_detects_a_changed_corpus_through_the_database(self):
        with temporary_profile() as profile:
            first = [record("s1", sha="a"), record("s2", sha="b")]
            save_records(profile, first)
            stored = load_records(profile)
            save_digest(profile, DigestRecord(
                paragraph="p", sections_sha256=sections_fingerprint(stored)))

            self.assertEqual(
                load_digest(profile).sections_sha256, sections_fingerprint(load_records(profile))
            )

            save_records(profile, [record("s2", sha="EDITED")])
            self.assertNotEqual(
                load_digest(profile).sections_sha256, sections_fingerprint(load_records(profile))
            )

    def test_digests_are_per_profile(self):
        with temporary_profile() as first, temporary_profile() as second:
            save_digest(first, DigestRecord(paragraph="first corpus"))
            save_digest(second, DigestRecord(paragraph="second corpus"))
            self.assertEqual("first corpus", load_digest(first).paragraph)
            self.assertEqual("second corpus", load_digest(second).paragraph)


@requires_postgres
class ConnectionPoolTests(unittest.TestCase):
    """The pool was left at SQLAlchemy's 5+10 default while the server runs 40 worker
    threads, so concurrency above fifteen blocked and then raised. These pin the fix."""

    def test_the_pool_is_sized_above_the_default(self):
        self.assertGreaterEqual(engine.pool.size(), 20)

    def test_many_concurrent_queries_all_succeed(self):
        threads = 40
        results = []
        errors = []
        barrier = threading.Barrier(threads)

        def query():
            try:
                barrier.wait(timeout=30)
                with engine.connect() as connection:
                    results.append(connection.execute(text("SELECT 1")).scalar())
            except Exception as exc:
                errors.append(exc)

        workers = [threading.Thread(target=query) for _ in range(threads)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)

        self.assertEqual([], errors, f"pool exhausted under {threads} concurrent queries")
        self.assertEqual(threads, len(results))

    def test_concurrent_writes_to_distinct_profiles_do_not_collide(self):
        profiles = [f"{TEST_PREFIX}-conc-{i}" for i in range(12)]
        errors = []

        def write(profile):
            try:
                save_records(profile, [record(f"s-{profile}", ("q?",), [[0.5]])])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(p,)) for p in profiles]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)
            self.assertEqual([], errors)
            for profile in profiles:
                self.assertEqual(1, len(load_records(profile)))
        finally:
            with engine.begin() as connection:
                for profile in profiles:
                    connection.execute(
                        text("DELETE FROM section_summaries WHERE profile = :p"), {"p": profile}
                    )


@requires_postgres
class SchemaTests(unittest.TestCase):
    def test_the_live_schema_matches_the_models(self):
        """Drift is otherwise found by a user, mid-upload, as an UndefinedColumn."""
        from backend.db.migrate import detect_drift

        drift = detect_drift()
        self.assertFalse(drift.has_drift, drift.summary())

    def test_corpus_digests_table_exists_with_its_columns(self):
        from sqlalchemy import inspect

        columns = {c["name"] for c in inspect(engine).get_columns("corpus_digests")}
        for expected in (
            "profile", "paragraph", "sections_sha256", "section_count",
            "floor", "floor_sha256", "question_count", "model_used", "updated_at",
        ):
            self.assertIn(expected, columns)


if __name__ == "__main__":
    unittest.main()
