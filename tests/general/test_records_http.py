"""RAG_FIX_PLAN item 41: calls to the records facade share kept-alive connections."""
import socket
import threading
import unittest

import requests

import backend.records_http as records_http


class _KeepAliveServer:
    """An HTTP/1.1 server answering `{}` to every GET, counting the connections it accepts."""

    def __init__(self):
        self.connections = 0
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    @staticmethod
    def _serve(conn):
        buffer = b""
        try:
            while True:
                while b"\r\n\r\n" not in buffer:
                    data = conn.recv(65536)
                    if not data:
                        return
                    buffer += data
                _, _, buffer = buffer.partition(b"\r\n\r\n")
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                             b"Content-Length: 2\r\n\r\n{}")
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self._sock.close()


class PooledFacadeTests(unittest.TestCase):
    def setUp(self):
        records_http._session = None  # a pool of this test's own
        self.server = _KeepAliveServer()

    def tearDown(self):
        self.server.close()
        records_http._session = None

    def test_requests_get_on_its_own_opens_a_connection_per_call(self):
        """What the roster and the records tool did; pinned so the comparison stays honest."""
        for _ in range(5):
            requests.get(f"{self.server.url}/v1/guardians/G-1/students", timeout=5)
        self.assertEqual(5, self.server.connections)

    def test_the_shared_pool_serves_every_call_on_one_connection(self):
        for path in ("/v1/guardians/G-1/students", "/v1/students/S-1/grades") * 3:
            self.assertEqual(200, records_http.get(f"{self.server.url}{path}", timeout=5).status_code)
        self.assertEqual(1, self.server.connections)

    def test_the_pool_is_as_deep_as_the_turn_threads(self):
        from backend.infra.executor import turn_executor_workers

        adapter = records_http._pooled().get_adapter("http://records:8100")
        self.assertEqual(turn_executor_workers(), adapter._pool_maxsize)


if __name__ == "__main__":
    unittest.main()
