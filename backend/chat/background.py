"""Work a turn hands off so the parent is not kept waiting for it.

Saving the turn changes nothing the parent is shown, and it used to run on the request,
after the last event. The streamed reply held its connection open for it, and a browser
that left once the answer was on screen cancelled the request before the save had run, so
the answer the parent had just read was never stored.

`BackgroundJobs` runs that work on worker threads of its own, detached from the request
that queued it. Three properties make it safe to hand a conversation's writes to:

  * **Jobs queued under one key run in the order they were queued.** A conversation's key
    is its `(user, session)`, so its question is stored before its answer, and this turn's
    rows land before the next turn's, whatever the thread pool does. A job waiting its
    turn holds no thread: it is started by the completion of the job before it.
  * **A caller can wait for a key.** The next turn's first step is to load the
    conversation, which must see the previous turn's save — `flush` is that barrier.
  * **Each lane is its own pool.** Work that is slower than a row insert gets a lane of
    its own, so one lane's backlog is never another's.

One instance per process, owned by the composition root, drained at shutdown so a restart
cannot drop a save that was queued. The ordering is per process: a deployment running
several workers behind one address would need the barrier in Redis. Today there is one.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout, wait
from typing import Protocol

logger = logging.getLogger(__name__)

#: The lane for database writes — the default, and the one the barrier is usually about.
WRITES = "writes"

_DEFAULT_LANES: Mapping[str, int] = {WRITES: 4}


class JobRunner(Protocol):
    """What a turn needs from whatever runs its deferred work.

    The turn pipeline depends on this and not on a thread pool, so a test — or a script
    that wants its writes done when the call returns — hands it `InlineJobs` instead.
    """

    def submit(
        self, key: str, work: Callable[[], object], *, lane: str = WRITES, describe: str = ""
    ) -> Future:
        """Queue `work` to run after everything already queued under `key`, on `lane`."""
        ...

    def flush(self, key: str, timeout: float | None = None) -> bool:
        """Wait until everything queued under `key` has run. False on timeout."""
        ...

    def drain(self, timeout: float | None = None) -> bool:
        """Wait for every queued job, under every key. False on timeout."""
        ...

    def shutdown(self, timeout: float = 30.0) -> None:
        """Refuse new work, finish what is queued, release whatever runs it."""
        ...


class BackgroundJobs:
    """Ordered, keyed background work on small pools of threads, one per lane."""

    def __init__(self, *, lanes: Mapping[str, int] | None = None, name: str = "chat-background") -> None:
        self._executors = {
            lane: ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{name}-{lane}")
            for lane, workers in (lanes or _DEFAULT_LANES).items()
        }
        self._lock = threading.Lock()
        # The last job queued under each key. A new job under the same key is started by
        # its completion, which is what makes a key's jobs run in order.
        self._tails: dict[str, Future] = {}
        self._closed = False

    def submit(
        self, key: str, work: Callable[[], object], *, lane: str = WRITES, describe: str = ""
    ) -> Future:
        """Queue `work` after everything already queued under `key`.

        Returns the future so a caller that wants the result — or the exception — can wait
        for it. Most callers do not: a failed save is logged here with what it was, and the
        parent has already been answered.
        """
        label = describe or getattr(work, "__name__", "job")
        try:
            executor = self._executors[lane]
        except KeyError:
            raise ValueError(f"no background lane named {lane!r}; known: {sorted(self._executors)}") from None
        future: Future = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError(f"background jobs are shut down; refusing {label!r}")
            previous = self._tails.get(key)
            self._tails[key] = future
            if len(self._tails) > 256:
                self._prune()

        def start(_previous: Future | None = None) -> None:
            try:
                executor.submit(self._execute, key, label, work, future)
            except RuntimeError as exc:  # the pool was shut down between queueing and starting
                future.set_exception(exc)

        if previous is None or previous.done():
            start()
        else:
            # Its failure is its own, already logged; this job still runs.
            previous.add_done_callback(start)
        return future

    @staticmethod
    def _execute(key: str, label: str, work: Callable[[], object], future: Future) -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = work()
        except BaseException as exc:
            logger.exception("background job failed: %s (key=%s)", label, key)
            future.set_exception(exc)
        else:
            future.set_result(result)

    def _prune(self) -> None:
        for key in [key for key, future in self._tails.items() if future.done()]:
            del self._tails[key]

    def flush(self, key: str, timeout: float | None = None) -> bool:
        """Wait until everything queued under `key` has run. False on timeout."""
        with self._lock:
            tail = self._tails.get(key)
        if tail is None:
            return True
        try:
            tail.result(timeout=timeout)
        except FutureTimeout:
            return False
        except Exception:
            # Already logged by `_execute`; the caller asked whether the queue is drained,
            # which it is.
            pass
        return True

    def drain(self, timeout: float | None = None) -> bool:
        """Wait for every queued job. False when some were still running at the timeout."""
        with self._lock:
            tails = list(self._tails.values())
        _done, pending = wait(tails, timeout=timeout)
        return not pending

    @property
    def pending(self) -> int:
        with self._lock:
            return sum(1 for future in self._tails.values() if not future.done())

    def shutdown(self, timeout: float = 30.0) -> None:
        """Refuse new work, finish what is queued, then release the threads.

        Called from the application's lifespan on the way down. `timeout` bounds how long a
        stopping process waits for a save that is stuck on an unreachable database; what is
        still running after it is logged and abandoned rather than holding the restart.
        """
        with self._lock:
            self._closed = True
        if not self.drain(timeout=timeout):
            logger.error("background jobs still running after %.0fs; abandoning them", timeout)
        for executor in self._executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            for future in self._tails.values():
                future.cancel()


class InlineJobs:
    """The same interface, run at once on the calling thread.

    For tests and one-off scripts that want the write to have happened by the time the
    call returns, and a failure to raise where it is caused rather than in a log.
    """

    def submit(
        self, key: str, work: Callable[[], object], *, lane: str = WRITES, describe: str = ""
    ) -> Future:
        future: Future = Future()
        future.set_result(work())
        return future

    def flush(self, key: str, timeout: float | None = None) -> bool:
        return True

    def drain(self, timeout: float | None = None) -> bool:
        return True

    @property
    def pending(self) -> int:
        return 0

    def shutdown(self, timeout: float = 30.0) -> None:
        return None


__all__ = ["WRITES", "BackgroundJobs", "InlineJobs", "JobRunner"]
