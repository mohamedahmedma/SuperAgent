"""RAG_FIX_PLAN item 44: a Redis that stalls costs a caller a short wait and a miss.

Unstated, redis-py 8 waits 5 s per call (measured against the server below: 5.0 s, then
TimeoutError), and a turn makes several cache calls before its first word: the
conversation's messages, the session list, parent chunks, asset dossiers. These run the
real client against a socket that accepts and never answers.
"""
import socket
import threading
import time
import unittest
from unittest.mock import patch

from backend.infra.cache import RedisCache


class _SilentServer:
    """Accepts connections, reads what it is sent, and never replies."""

    def __init__(self):
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.url = f"redis://127.0.0.1:{self._sock.getsockname()[1]}/0"
        self._held = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self._held.append(conn)

    def close(self):
        self._sock.close()
        for conn in self._held:
            conn.close()


class StalledRedisTests(unittest.TestCase):
    def cache(self, url, **env):
        with patch.dict("os.environ", {"REDIS_URL": url, **env}, clear=False):
            return RedisCache()

    def test_a_read_from_a_stalled_redis_is_a_miss_after_the_timeout(self):
        server = _SilentServer()
        try:
            cache = self.cache(server.url, REDIS_SOCKET_TIMEOUT_SECONDS="0.3")
            started = time.perf_counter()
            self.assertIsNone(cache.get_json("session"))
            self.assertLess(time.perf_counter() - started, 2.0)
        finally:
            server.close()

    def test_a_write_to_a_stalled_redis_gives_up_after_the_timeout(self):
        server = _SilentServer()
        try:
            cache = self.cache(server.url, REDIS_SOCKET_TIMEOUT_SECONDS="0.3")
            started = time.perf_counter()
            cache.set_json("session", {"a": 1})
            self.assertLess(time.perf_counter() - started, 2.0)
        finally:
            server.close()

    def test_the_shipped_timeouts_are_a_second(self):
        cache = self.cache("redis://127.0.0.1:1/0")
        self.assertEqual(1.0, cache.socket_timeout)
        self.assertEqual(1.0, cache.connect_timeout)
        kwargs = cache.client().connection_pool.connection_kwargs
        self.assertEqual(1.0, kwargs["socket_timeout"])
        self.assertEqual(1.0, kwargs["socket_connect_timeout"])

    def test_keys_are_namespaced_by_the_profile(self):
        cache = self.cache("redis://127.0.0.1:1/0")
        self.assertEqual(f"{cache.key_prefix}:turns:rate:parent", cache.key("turns:rate:parent"))


if __name__ == "__main__":
    unittest.main()
