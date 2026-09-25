"""RAG_FIX_PLAN item 21: a turn's background save is not dropped, repeated, or raced.

The save stays in the background, so the parent is answered without waiting for it
(measured: the save takes 17-64 ms at p50 and up to ~750 ms at p95 under load, all of
which would otherwise hold the reply). What these pin is that it is:

  * **retried** through a transient database failure instead of dropped;
  * **idempotent**, so a retry after a commit whose acknowledgement was lost cannot store
    the answer twice;
  * **visible to every worker**, so the parent's next message waits for it wherever it lands.
"""
import threading
import time
import unittest
import uuid

from sqlalchemy import exc as sa_exc

from backend.chat.background import BackgroundJobs, SharedWriteBarrier
from backend.chat.storage import ConversationStorage, MessageToStore
from backend.infra.retry import is_transient, retry_transient
from tests.general.postgres_support import postgres_schema
from tests.general.test_asset_delivery import DictCache
from tests.general.test_turn_admission import _CLIENT, requires_redis


def _lost_connection():
    return sa_exc.OperationalError("COMMIT", {}, Exception("server closed the connection unexpectedly"))


class RetryTests(unittest.TestCase):
    def test_a_transient_failure_is_tried_again_until_it_succeeds(self):
        attempts, waits = [], []

        def save():
            attempts.append(1)
            if len(attempts) < 3:
                raise _lost_connection()
            return [7, 8]

        self.assertEqual([7, 8], retry_transient(save, sleep=waits.append))
        self.assertEqual(3, len(attempts))
        self.assertEqual(2, len(waits))
        self.assertLess(waits[0], waits[1] * 1.01 + 0.2)  # backing off, not hammering

    def test_a_refused_write_is_raised_at_once(self):
        attempts = []

        def save():
            attempts.append(1)
            raise sa_exc.IntegrityError("INSERT", {}, Exception("violates foreign key"))

        with self.assertRaises(sa_exc.IntegrityError):
            retry_transient(save, sleep=lambda _: None)
        self.assertEqual(1, len(attempts))

    def test_it_gives_up_after_the_last_attempt(self):
        attempts = []

        def save():
            attempts.append(1)
            raise _lost_connection()

        with self.assertRaises(sa_exc.OperationalError):
            retry_transient(save, attempts=4, sleep=lambda _: None)
        self.assertEqual(4, len(attempts))

    def test_what_counts_as_transient(self):
        self.assertTrue(is_transient(_lost_connection()))
        self.assertTrue(is_transient(sa_exc.TimeoutError("QueuePool limit reached")))
        self.assertFalse(is_transient(sa_exc.IntegrityError("INSERT", {}, Exception("dup"))))
        self.assertFalse(is_transient(ValueError("bad data")))


class IdempotentSaveTests(unittest.TestCase):
    def setUp(self):
        from backend.db.models import ChatMessage, ChatSession, User

        schema = postgres_schema(self, User, ChatSession, ChatMessage)
        self.factory = schema.sessionmaker()
        db = self.factory()
        db.add(User(username="parent", password_hash="x"))
        db.commit()
        db.close()
        self.schema = schema

    def _contents(self):
        from backend.db.models import ChatMessage

        db = self.factory()
        try:
            return [row.content for row in db.query(ChatMessage).order_by(ChatMessage.id)]
        finally:
            db.close()

    def test_the_same_messages_stored_twice_land_once_with_the_same_ids(self):
        storage = ConversationStorage(unit_of_work=self.schema.unit_of_work, cache=DictCache())
        turn = [MessageToStore("human", "fees?"), MessageToStore("ai", "The fees are...")]
        first = storage.append("parent", "s", turn)
        again = storage.append("parent", "s", turn)
        self.assertEqual(first, again)
        self.assertEqual(["fees?", "The fees are..."], self._contents())

    def test_a_retry_after_a_commit_whose_acknowledgement_was_lost_stores_nothing_twice(self):
        """The one failure a blind retry gets wrong: the database committed, the
        connection dropped before saying so, and the writer cannot tell."""
        commits = []

        def unit_of_work():
            uow = self.schema.unit_of_work()
            real_commit = uow.commit

            def commit():
                real_commit()
                commits.append(1)
                if len(commits) == 1:
                    raise _lost_connection()

            uow.commit = commit
            return uow

        storage = ConversationStorage(unit_of_work=unit_of_work, cache=DictCache())
        answer = [MessageToStore("ai", "Fees are reviewed each year.")]
        ids = retry_transient(lambda: storage.append("parent", "s", answer), sleep=lambda _: None)

        self.assertEqual(2, len(commits))
        self.assertEqual(["Fees are reviewed each year."], self._contents())
        self.assertEqual(1, len(ids))

    def test_messages_without_a_shared_key_are_stored_as_before(self):
        storage = ConversationStorage(unit_of_work=self.schema.unit_of_work, cache=DictCache())
        storage.append("parent", "s", [MessageToStore("human", "hello")])
        storage.append("parent", "s", [MessageToStore("human", "hello")])
        self.assertEqual(["hello", "hello"], self._contents())


@requires_redis
class SharedWriteBarrierTests(unittest.TestCase):
    """Two job runners over one Redis are two workers."""

    def setUp(self):
        self.prefix = f"test-swb-{uuid.uuid4().hex[:8]}"
        self.runners = []

    def tearDown(self):
        for runner in self.runners:
            runner.shutdown(5.0)
        for key in _CLIENT.scan_iter(f"{self.prefix}:*"):
            _CLIENT.delete(key)

    def worker(self, **options):
        runner = SharedWriteBarrier(BackgroundJobs(name=f"w{len(self.runners)}"), redis=lambda: _CLIENT,
                                    key=lambda name: f"{self.prefix}:{name}", **options)
        self.runners.append(runner)
        return runner

    def test_the_next_turn_on_another_worker_waits_for_this_workers_save(self):
        saving, reading = self.worker(), self.worker()
        gate, landed = threading.Event(), []
        saving.submit("conversation:u:s", lambda: (gate.wait(5), landed.append("answer")))

        waiter = threading.Thread(target=lambda: landed.append(("flushed", reading.flush("conversation:u:s", timeout=5))))
        waiter.start()
        time.sleep(0.2)
        self.assertEqual([], landed, "the other worker went ahead before the save landed")
        gate.set()
        waiter.join(5)
        self.assertEqual(["answer", ("flushed", True)], landed)

    def test_another_conversation_is_not_held_up(self):
        saving, reading = self.worker(), self.worker()
        gate = threading.Event()
        saving.submit("conversation:u:a", lambda: gate.wait(5))
        started = time.perf_counter()
        self.assertTrue(reading.flush("conversation:u:b", timeout=5))
        self.assertLess(time.perf_counter() - started, 0.5)
        gate.set()

    def test_a_save_that_fails_still_releases_the_wait(self):
        saving, reading = self.worker(), self.worker()

        def fail():
            raise RuntimeError("database refused")

        saving.submit("conversation:u:s", fail)
        self.assertTrue(reading.flush("conversation:u:s", timeout=5))

    def test_a_count_left_by_a_dead_worker_expires_instead_of_hanging_the_conversation(self):
        _CLIENT.set(f"{self.prefix}:pending_writes:conversation:u:s", 1, ex=1)  # a crash mid-save
        started = time.perf_counter()
        self.assertTrue(self.worker().flush("conversation:u:s", timeout=5))
        self.assertLess(time.perf_counter() - started, 2.5)

    def test_a_wait_past_the_timeout_says_so(self):
        _CLIENT.set(f"{self.prefix}:pending_writes:conversation:u:s", 1, ex=30)
        self.assertFalse(self.worker().flush("conversation:u:s", timeout=0.2))


class FailOpenTests(unittest.TestCase):
    def test_without_redis_saves_still_run_and_flush_waits_for_this_process(self):
        import redis

        def unreachable():
            raise redis.ConnectionError("down")

        runner = SharedWriteBarrier(BackgroundJobs(name="fo"), redis=unreachable)
        try:
            done = []
            runner.submit("conversation:u:s", lambda: done.append(1))
            self.assertTrue(runner.flush("conversation:u:s", timeout=5))
            self.assertEqual([1], done)
        finally:
            runner.shutdown(5.0)


if __name__ == "__main__":
    unittest.main()
