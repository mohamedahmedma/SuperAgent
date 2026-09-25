"""RAG_FIX_PLAN item 36: the thread pool a turn's blocking work runs on is sized on purpose."""
import asyncio
import threading
import unittest
from unittest.mock import patch

from backend.infra import executor


class SizingTests(unittest.TestCase):
    def test_the_default_is_the_measured_one_not_pythons(self):
        with patch.dict("os.environ", {"TURN_EXECUTOR_WORKERS": ""}, clear=False):
            self.assertEqual(executor.DEFAULT_WORKERS, executor.turn_executor_workers())
        self.assertEqual(32, executor.DEFAULT_WORKERS)

    def test_a_deployment_can_set_it(self):
        with patch.dict("os.environ", {"TURN_EXECUTOR_WORKERS": "48"}, clear=False):
            self.assertEqual(48, executor.turn_executor_workers())

    def test_nonsense_falls_back_to_the_default_rather_than_failing_startup(self):
        with patch.dict("os.environ", {"TURN_EXECUTOR_WORKERS": "lots"}, clear=False):
            self.assertEqual(executor.DEFAULT_WORKERS, executor.turn_executor_workers())

    def test_it_is_never_below_one(self):
        with patch.dict("os.environ", {"TURN_EXECUTOR_WORKERS": "0"}, clear=False):
            self.assertEqual(1, executor.turn_executor_workers())


class InstallTests(unittest.TestCase):
    def test_to_thread_and_sync_tools_land_on_the_installed_pool(self):
        """`asyncio.to_thread` and LangChain's `run_in_executor(None, ...)` both use the
        loop's default executor — which is the whole point of installing one."""

        async def main():
            pool = executor.install_turn_executor()
            name_via_to_thread = await asyncio.to_thread(lambda: threading.current_thread().name)
            name_via_run_in_executor = await asyncio.get_running_loop().run_in_executor(
                None, lambda: threading.current_thread().name
            )
            return pool, name_via_to_thread, name_via_run_in_executor

        with patch.dict("os.environ", {"TURN_EXECUTOR_WORKERS": "5"}, clear=False):
            pool, a, b = asyncio.run(main())
        self.assertEqual(5, pool._max_workers)
        self.assertTrue(a.startswith("turn"), a)
        self.assertTrue(b.startswith("turn"), b)


if __name__ == "__main__":
    unittest.main()
