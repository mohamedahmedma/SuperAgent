"""Upload and delete job progress, kept in Postgres rather than in one process's memory.

What the move buys is what these pin: a job started by one worker is readable by any
other, a job whose process died stops claiming to run, and two threads reporting on the
same job cannot overwrite each other's step.
"""
import threading
import unittest
from datetime import UTC, datetime, timedelta

from backend.db.models import IngestJob
from backend.jobs.upload_jobs import DELETE_STEPS, RETAINED_FOR, STALLED_AFTER, IngestJobTracker
from tests.general.postgres_support import postgres_schema


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, delta):
        self.now += delta


class IngestJobTestCase(unittest.TestCase):
    def setUp(self):
        self.schema = postgres_schema(self, IngestJob)
        self.clock = Clock()
        self.uploads = self.tracker("upload")

    def tracker(self, kind):
        return IngestJobTracker(kind, self.schema.unit_of_work, clock=self.clock)


class LifecycleTests(IngestJobTestCase):
    def test_a_new_job_is_pending_with_every_step_pending(self):
        job = self.uploads.create_job("fees.pdf")
        self.assertEqual(("pending", "upload"), (job["status"], job["current_step"]))
        self.assertEqual(["pending"] * 5, [step["status"] for step in job["steps"]])

    def test_another_worker_reads_the_job_this_one_started(self):
        """The property the in-memory version lacked: a second tracker is a second process."""
        job = self.uploads.create_job("fees.pdf")
        self.uploads.update_step(job["job_id"], "parse", 40, message="Parsing")

        seen = self.tracker("upload").get_job(job["job_id"])
        self.assertEqual(("running", "parse", "Parsing"), (seen["status"], seen["current_step"], seen["message"]))

    def test_a_step_update_clamps_percent_and_records_totals(self):
        job_id = self.uploads.create_job("fees.pdf")["job_id"]
        job = self.uploads.update_step(job_id, "vector_store", 180, total_chunks=40, processed_chunks=12)

        step = next(step for step in job["steps"] if step["key"] == "vector_store")
        self.assertEqual((100, "running"), (step["percent"], step["status"]))
        self.assertEqual((40, 12), (job["total_chunks"], job["processed_chunks"]))

    def test_completing_marks_every_unfailed_step_done_and_lands_on_the_final_step(self):
        job_id = self.uploads.create_job("fees.pdf")["job_id"]
        job = self.uploads.complete_job(job_id, "Done")

        self.assertEqual(("completed", "vector_store", None), (job["status"], job["current_step"], job["error"]))
        self.assertEqual([100] * 5, [step["percent"] for step in job["steps"]])

    def test_failing_names_the_step_and_the_error(self):
        job_id = self.uploads.create_job("fees.pdf")["job_id"]
        job = self.uploads.fail_job(job_id, "parse", "could not extract content")

        self.assertEqual(("failed", "parse", "could not extract content"), (job["status"], job["current_step"], job["error"]))
        self.assertEqual("failed", next(s for s in job["steps"] if s["key"] == "parse")["status"])

    def test_unknown_jobs_and_steps_change_nothing(self):
        job_id = self.uploads.create_job("fees.pdf")["job_id"]
        self.assertIsNone(self.uploads.get_job("missing"))
        self.assertIsNone(self.uploads.update_step("missing", "parse", 10))
        self.assertIsNone(self.uploads.update_step(job_id, "no-such-step", 10))
        self.assertEqual("pending", self.uploads.get_job(job_id)["status"])

    def test_uploads_and_deletes_are_separate_and_listed_newest_first(self):
        deletes = self.tracker("delete")
        first = self.uploads.create_job("a.pdf")["job_id"]
        self.clock.advance(timedelta(minutes=1))
        second = self.uploads.create_job("b.pdf")["job_id"]
        removal = deletes.create_job("a.pdf", steps=DELETE_STEPS, current_step="prepare")["job_id"]

        self.assertEqual([second, first], [job["job_id"] for job in self.uploads.list_jobs()])
        self.assertEqual([removal], [job["job_id"] for job in deletes.list_jobs()])
        self.assertIsNone(self.uploads.get_job(removal))


class StalledJobTests(IngestJobTestCase):
    def test_a_job_that_stopped_reporting_reads_as_failed(self):
        job_id = self.uploads.create_job("fees.pdf")["job_id"]
        self.uploads.update_step(job_id, "parse", 10)

        self.clock.advance(STALLED_AFTER + timedelta(seconds=1))
        job = self.uploads.get_job(job_id)
        self.assertEqual("failed", job["status"])
        self.assertIn("stopped", job["error"])

    def test_a_job_that_reports_again_is_running_again(self):
        """Presented, not stored: a slow job that was never dead recovers on its next update."""
        job_id = self.uploads.create_job("fees.pdf")["job_id"]
        self.clock.advance(STALLED_AFTER + timedelta(minutes=5))
        self.assertEqual("failed", self.uploads.get_job(job_id)["status"])

        self.assertEqual("running", self.uploads.update_step(job_id, "parse", 50)["status"])

    def test_a_finished_job_never_goes_stale(self):
        job_id = self.uploads.create_job("fees.pdf")["job_id"]
        self.uploads.complete_job(job_id)
        self.clock.advance(timedelta(days=2))
        self.assertEqual("completed", self.uploads.get_job(job_id)["status"])


class RetentionTests(IngestJobTestCase):
    def test_creating_a_job_prunes_old_jobs_of_the_same_kind_only(self):
        old_upload = self.uploads.create_job("old.pdf")["job_id"]
        old_delete = self.tracker("delete").create_job("old.pdf", steps=DELETE_STEPS)["job_id"]

        self.clock.advance(RETAINED_FOR + timedelta(days=1))
        self.uploads.create_job("new.pdf")

        self.assertIsNone(self.uploads.get_job(old_upload))
        self.assertIsNotNone(self.tracker("delete").get_job(old_delete))


class ConcurrencyTests(IngestJobTestCase):
    def test_two_threads_updating_different_steps_lose_neither(self):
        """The vectoriser's progress callback and the job's own transitions run on
        different threads. Without the row lock, each read-modify-write can overwrite
        the other's step with the value it read before."""
        job_id = self.uploads.create_job("fees.pdf")["job_id"]
        rounds = 40
        errors = []

        def report(step_key):
            try:
                for percent in range(1, rounds + 1):
                    self.uploads.update_step(job_id, step_key, percent)
            except Exception as exc:  # pragma: no cover - surfaced by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=report, args=(key,)) for key in ("parse", "vector_store")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        self.assertEqual([], errors)
        steps = {step["key"]: step["percent"] for step in self.uploads.get_job(job_id)["steps"]}
        self.assertEqual((rounds, rounds), (steps["parse"], steps["vector_store"]))


if __name__ == "__main__":
    unittest.main()


class SubProgressTests(IngestJobTestCase):
    """A stage that runs INSIDE a step, drawn as a nested bar under it."""

    def _parse_step(self, job):
        return next(step for step in job["steps"] if step["key"] == "parse")

    def test_a_step_carries_no_sub_stage_until_one_reports(self):
        job = self.uploads.create_job("fees.pdf")
        self.assertEqual(0, self._parse_step(job)["sub_total"])

    def test_sub_progress_is_stored_and_read_back_by_another_worker(self):
        job = self.uploads.create_job("fees.pdf")
        self.uploads.update_step(
            job["job_id"], "parse", 25, message="Extracting images: 3 of 12",
            sub_label="Extracting images", sub_done=3, sub_total=12,
        )

        step = self._parse_step(self.tracker("upload").get_job(job["job_id"]))
        self.assertEqual(("Extracting images", 3, 12),
                         (step["sub_label"], step["sub_done"], step["sub_total"]))

    def test_an_ordinary_update_leaves_a_running_sub_stage_alone(self):
        """The nested bar is driven by one caller and the step by another. Defaulting
        the sub-fields to zero would have any other update erase the bar mid-extraction.
        """
        job = self.uploads.create_job("fees.pdf")
        self.uploads.update_step(
            job["job_id"], "parse", 25, sub_label="Extracting images", sub_done=3, sub_total=12,
        )
        self.uploads.update_step(job["job_id"], "parse", 30, message="still going")

        step = self._parse_step(self.uploads.get_job(job["job_id"]))
        self.assertEqual((3, 12), (step["sub_done"], step["sub_total"]))
        self.assertEqual("still going", step["message"])

    def test_sub_progress_on_one_step_never_touches_another(self):
        job = self.uploads.create_job("fees.pdf")
        self.uploads.update_step(
            job["job_id"], "parse", 25, sub_label="Extracting images", sub_done=1, sub_total=4,
        )
        steps = {s["key"]: s for s in self.uploads.get_job(job["job_id"])["steps"]}
        self.assertEqual(0, steps["vector_store"]["sub_total"])
        self.assertEqual(0, steps["upload"]["sub_total"])
