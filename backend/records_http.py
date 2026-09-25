"""HTTP to the records facade: one pooled session for the whole process.

RAG_FIX_PLAN item 41. The records tool and the child roster each called `requests.get`,
which builds a new Session for the call, opens a new TCP connection, and closes it once
the answer is read. A parent's turn about their child makes one to three such calls,
and under load each one leaves a socket in TIME_WAIT. Measured against the local facade,
100+ calls each way: p50 7.7 -> 5.6 ms, p95 23 -> 10 ms. This machine reaches the facade
through Docker Desktop's port proxy, which makes connection setup dearer than inside
the compose network, so production's gain per call is smaller. The socket saved is the
same everywhere.

`get` is the one way either caller reaches the facade, which also makes it the single
seam a test replaces: patching it catches the roster and the records calls together, as
patching `requests.get` used to.
"""
from __future__ import annotations

import threading

import requests
from requests.adapters import HTTPAdapter

_session: requests.Session | None = None
_lock = threading.Lock()


def _pooled() -> requests.Session:
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                from backend.infra.executor import turn_executor_workers

                # As many kept-alive connections as there are threads that can be calling
                # at once; below that, concurrent turns wait for a socket.
                adapter = HTTPAdapter(pool_connections=1, pool_maxsize=turn_executor_workers())
                session = requests.Session()
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                _session = session
    return _session


def get(url: str, **kwargs) -> requests.Response:
    """`requests.get`, over the shared pool."""
    return _pooled().get(url, **kwargs)


__all__ = ["get"]
