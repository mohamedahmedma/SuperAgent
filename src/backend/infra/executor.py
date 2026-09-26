"""The thread pool a chat turn's blocking work runs on — sized on purpose, not by accident.

Everything a streamed turn cannot do on the event loop goes to the loop's DEFAULT
executor: every `asyncio.to_thread` in `chat_with_agent_stream` (opening the
conversation, settling a child, planning, waiting for the save) and every agent tool,
because the tools are sync `def` and LangChain runs a sync tool with
`run_in_executor(None, ...)`. Nothing ever sized that pool, so it was Python's default,
`min(32, os.cpu_count() + 4)` — 16 threads on a 12-core laptop, 20 on the 16-vCPU
production VPS, and on Python 3.12 `os.cpu_count()` reports the HOST's cores rather than
the container's limit, so the number moved with the machine.

Those threads spend nearly all their time waiting on a socket — a model provider, the
embedding provider, Milvus, Postgres — not on a CPU. A turn holds one for seconds, so the
pool, not the hardware, bounded how many turns could be in flight (RAG_FIX_PLAN item 36).
It turned out to be the smaller of two ceilings — see `DEFAULT_WORKERS`.

`TURN_EXECUTOR_WORKERS` sets it. Threads blocked on I/O are cheap, but not free, and they
are not the only limit: each may hold a database connection (the pool is sized in
backend/infra/database.py) and each may be a concurrent call against a provider's quota
(item 37). Raise it with those in view.
"""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

#: Chosen by measurement (tests/load/chat_load.py, stub model, 12-core dev machine):
#:
#:     threads   40 parents    60 parents
#:        16     4.11 turns/s  4.23 turns/s     (Python's default on that machine)
#:        32     4.48          4.64
#:        64     4.57-4.67     4.43-4.81        (four runs)
#:
#: 16 -> 32 buys ~10%; past 32 nothing more, because the ceiling is then the GIL — the
#: process sat at ~100% of ONE core with the machine at 33%. More threads only queue for
#: that core, while pressing harder on the database pool and the provider's quota. The
#: next throughput is more processes (item 35) and less Python per turn, not more threads.
DEFAULT_WORKERS = 32


def turn_executor_workers() -> int:
    configured = (os.getenv("TURN_EXECUTOR_WORKERS") or "").strip()
    if not configured:
        return DEFAULT_WORKERS
    try:
        return max(1, int(configured))
    except ValueError:
        logger.warning("TURN_EXECUTOR_WORKERS=%r is not a number; using %d", configured, DEFAULT_WORKERS)
        return DEFAULT_WORKERS


def install_turn_executor(loop: asyncio.AbstractEventLoop | None = None) -> ThreadPoolExecutor:
    """Make a sized pool the running loop's default executor, and return it.

    Installed at startup, before the first request, so every `to_thread` and every sync
    tool call lands on it. asyncio shuts the default executor down itself when the loop
    closes (`loop.shutdown_default_executor()`), so nothing else has to.
    """
    loop = loop or asyncio.get_running_loop()
    workers = turn_executor_workers()
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="turn")
    loop.set_default_executor(executor)
    logger.info("turn executor: %d threads", workers)
    return executor
