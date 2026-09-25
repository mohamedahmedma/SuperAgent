"""RAG_FIX_PLAN item 35: several workers, without multiplying database connections.

One process is bound by the GIL at ~4.6 turns/s whatever the load. Measured with the
stub model, 80 parents: 1 worker 4.64 turns/s (first word p50 13.9 s), 2 workers 9.07
(6.8 s), 4 workers 10.09 (5.3 s). Workers are separate processes, so each opens its own
database pool: the pool is therefore a share of one budget, not a fixed size per process.
"""
import unittest
from unittest.mock import patch


class PoolShareTests(unittest.TestCase):
    def share(self, **env):
        import backend.infra.database as database

        with patch.dict("os.environ", env, clear=False):
            return database._pool_share()

    def test_one_worker_keeps_the_pool_it_always_had(self):
        self.assertEqual(20, self.share(WEB_CONCURRENCY="1", DB_CONNECTION_BUDGET="40"))

    def test_workers_split_the_budget_instead_of_multiplying_it(self):
        for workers in (2, 4):
            with self.subTest(workers=workers):
                share = self.share(WEB_CONCURRENCY=str(workers), DB_CONNECTION_BUDGET="40")
                self.assertLessEqual(workers * share * 2, 40)

    def test_no_worker_is_left_with_a_useless_pool(self):
        self.assertEqual(5, self.share(WEB_CONCURRENCY="16", DB_CONNECTION_BUDGET="40"))

    def test_the_budget_is_the_operators_to_raise(self):
        self.assertEqual(10, self.share(WEB_CONCURRENCY="4", DB_CONNECTION_BUDGET="80"))


class LocalModelWarningTests(unittest.TestCase):
    def warned(self, **env):
        from backend.indexing.embedding import warn_if_every_worker_loads_the_model

        with patch.dict("os.environ", env, clear=False):
            return warn_if_every_worker_loads_the_model()

    def test_several_workers_each_loading_the_local_model_is_called_out(self):
        with self.assertLogs("backend.indexing.embedding", "WARNING"):
            self.assertTrue(self.warned(WEB_CONCURRENCY="4", EMBEDDING_BACKEND="local"))

    def test_a_hosted_endpoint_or_one_worker_is_fine(self):
        self.assertFalse(self.warned(WEB_CONCURRENCY="4", EMBEDDING_BACKEND="openai"))
        self.assertFalse(self.warned(WEB_CONCURRENCY="1", EMBEDDING_BACKEND="local"))


if __name__ == "__main__":
    unittest.main()
